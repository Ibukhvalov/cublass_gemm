#include "kernel.hpp"
#include "macro.hpp"
#include "utils.hpp"
#include "kernels/tf32_tensor_core_ptx_async.cuh"

#include <cuda_pipeline_primitives.h>
#include <mma.h>

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 16;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kStages = 2;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;

constexpr int kWarpTileM = 32;
constexpr int kWarpTileN = 32;
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 8;

constexpr int kAStageElems = kBlockM * kBlockK;
constexpr int kBStageElems = kBlockK * kBlockN;
constexpr int kSharedBytes = (kStages * kAStageElems + kStages * kBStageElems) * sizeof(float);

__device__ inline float* a_stage_ptr(float* smem, int stage, int row, int col) {
    return smem + stage * kAStageElems + row * kBlockK + col;
}

__device__ inline float* b_stage_ptr(float* smem, int stage, int row, int col) {
    float* b_base = smem + kStages * kAStageElems;
    return b_base + stage * kBStageElems + row * kBlockN + col;
}

__device__ inline void issue_async_copy_stage(
    float* smem,
    const float* A,
    const float* B, int m,
    int n,
    int k,
    int block_row,
    int block_col,
    int k0,
    int stage,
    int linear_tid) {
    constexpr int kAVecCols = kBlockK / 4;
    for (int vec_idx = linear_tid; vec_idx < kBlockM * kAVecCols; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / kAVecCols;
        const int col = (vec_idx % kAVecCols) * 4;
        void* dst = static_cast<void*>(a_stage_ptr(smem, stage, row, col));
        const int global_row = block_row + row;
        const int global_col = k0 + col;

        if (global_row < m) {
            const int remaining = k - global_col;
            if (remaining >= 4) {
                __pipeline_memcpy_async(dst, &A[global_row * k + global_col], 16);
            } else if (remaining > 0) {
                __pipeline_memcpy_async(dst, &A[global_row * k + global_col], 16, 16 - remaining * static_cast<int>(sizeof(float)));
            } else {
                __pipeline_memcpy_async(dst, &A[0], 16, 16);
            }
        } else {
            __pipeline_memcpy_async(dst, &A[0], 16, 16);
        }
    }

    constexpr int kBVecCols = kBlockN / 4;
    for (int vec_idx = linear_tid; vec_idx < kBlockK * kBVecCols; vec_idx += kThreadsPerBlock) {
        const int row = vec_idx / kBVecCols;
        const int col = (vec_idx % kBVecCols) * 4;
        void* dst = static_cast<void*>(b_stage_ptr(smem, stage, row, col));
        const int global_row = k0 + row;
        const int global_col = block_col + col;

        if (global_row < k) {
            const int remaining = n - global_col;
            if (remaining >= 4) {
                __pipeline_memcpy_async(dst, &B[global_row * n + global_col], 16);
            } else if (remaining > 0) {
                __pipeline_memcpy_async(dst, &B[global_row * n + global_col], 16, 16 - remaining * static_cast<int>(sizeof(float)));
            } else {
                __pipeline_memcpy_async(dst, &B[0], 16, 16);
            }
        } else {
            __pipeline_memcpy_async(dst, &B[0], 16, 16);
        }
    }
}

__device__ inline void mma_tf32_1688(
    float& d0,
    float& d1,
    float& d2,
    float& d3,
    unsigned a0,
    unsigned a1,
    unsigned a2,
    unsigned a3,
    unsigned b0,
    unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ inline void load_a_frag(
    unsigned (&a)[4],
    float* smem,
    int stage,
    int row_base,
    int kk,
    int group_id,
    int thread_id_in_group) {
    const float a0 = nvcuda::wmma::__float_to_tf32(*a_stage_ptr(smem, stage, row_base + group_id, kk + thread_id_in_group));
    const float a1 = nvcuda::wmma::__float_to_tf32(*a_stage_ptr(smem, stage, row_base + group_id + 8, kk + thread_id_in_group));
    const float a2 = nvcuda::wmma::__float_to_tf32(*a_stage_ptr(smem, stage, row_base + group_id, kk + thread_id_in_group + 4));
    const float a3 = nvcuda::wmma::__float_to_tf32(*a_stage_ptr(smem, stage, row_base + group_id + 8, kk + thread_id_in_group + 4));
    a[0] = __float_as_uint(a0);
    a[1] = __float_as_uint(a1);
    a[2] = __float_as_uint(a2);
    a[3] = __float_as_uint(a3);
}

__device__ inline void load_b_frag(
    unsigned (&b)[2],
    float* smem,
    int stage,
    int col_base,
    int kk,
    int group_id,
    int thread_id_in_group) {
    const float b0 = nvcuda::wmma::__float_to_tf32(*b_stage_ptr(smem, stage, kk + thread_id_in_group, col_base + group_id));
    const float b1 = nvcuda::wmma::__float_to_tf32(*b_stage_ptr(smem, stage, kk + thread_id_in_group + 4, col_base + group_id));
    b[0] = __float_as_uint(b0);
    b[1] = __float_as_uint(b1);
}

__global__ void tf32_tensor_core_ptx_async_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m,
    int n,
    int k) {
    extern __shared__ float smem[];

    const int warp_idx = threadIdx.y;
    const int lane = threadIdx.x;
    const int linear_tid = warp_idx * kWarpSize + lane;
    const int group_id = lane >> 2;
    const int thread_id_in_group = lane & 3;

    const int block_row = blockIdx.y * kBlockM;
    const int block_col = blockIdx.x * kBlockN;
    const int warp_row = warp_idx / 4;
    const int warp_col = warp_idx % 4;
    const int warp_row_base = warp_row * kWarpTileM;
    const int warp_col_base = warp_col * kWarpTileN;

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            #pragma unroll
            for (int ci = 0; ci < 4; ++ci) {
                acc[mi][ni][ci] = 0.0f;
            }
        }
    }

    int stage = 0;
    issue_async_copy_stage(smem, A, B, m, n, k, block_row, block_col, 0, stage, linear_tid);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    for (int k0 = 0; k0 < k; k0 += kBlockK) {
        const int next_k0 = k0 + kBlockK;
        const int next_stage = stage ^ 1;
        if (next_k0 < k) {
            issue_async_copy_stage(smem, A, B, m, n, k, block_row, block_col, next_k0, next_stage, linear_tid);
            __pipeline_commit();
        }

        unsigned a_regs[2][4];
        unsigned next_a_regs[2][4];
        unsigned b_regs[4][2];
        unsigned next_b_regs[4][2];

        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            load_a_frag(a_regs[mi], smem, stage, warp_row_base + mi * 16, 0, group_id, thread_id_in_group);
        }
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            load_b_frag(b_regs[ni], smem, stage, warp_col_base + ni * 8, 0, group_id, thread_id_in_group);
        }

        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += kMmaK) {
            const bool has_next = kk + kMmaK < kBlockK;
            if (has_next) {
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    load_a_frag(next_a_regs[mi], smem, stage, warp_row_base + mi * 16, kk + kMmaK, group_id, thread_id_in_group);
                }
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    load_b_frag(next_b_regs[ni], smem, stage, warp_col_base + ni * 8, kk + kMmaK, group_id, thread_id_in_group);
                }
            }

            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    mma_tf32_1688(
                        acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                        a_regs[mi][0], a_regs[mi][1], a_regs[mi][2], a_regs[mi][3],
                        b_regs[ni][0], b_regs[ni][1]);
                }
            }

            if (has_next) {
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    #pragma unroll
                    for (int ai = 0; ai < 4; ++ai) {
                        a_regs[mi][ai] = next_a_regs[mi][ai];
                    }
                }
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    #pragma unroll
                    for (int bi = 0; bi < 2; ++bi) {
                        b_regs[ni][bi] = next_b_regs[ni][bi];
                    }
                }
            }
        }

        if (next_k0 < k) {
            __pipeline_wait_prior(0);
            __syncthreads();
            stage = next_stage;
        }
    }

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int tile_row = block_row + warp_row_base + mi * 16;
            const int tile_col = block_col + warp_col_base + ni * 8;
            const int row0 = tile_row + group_id;
            const int row1 = tile_row + group_id + 8;
            const int col0 = tile_col + thread_id_in_group * 2;
            const int col1 = col0 + 1;

            if (row0 < m) {
                if (col0 < n) C[row0 * n + col0] = acc[mi][ni][0];
                if (col1 < n) C[row0 * n + col1] = acc[mi][ni][1];
            }
            if (row1 < m) {
                if (col0 < n) C[row1 * n + col0] = acc[mi][ni][2];
                if (col1 < n) C[row1 * n + col1] = acc[mi][ni][3];
            }
        }
    }
}
}

void Tf32TensorCorePtxAsyncKernel::launch(tf32_ptx_async_fp_t* dA, tf32_ptx_async_fp_t* dB, tf32_ptx_async_fp_t* dC, int m, int n, int k) {
    CHECK_CUDA(cudaFuncSetAttribute(
        tf32_tensor_core_ptx_async_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes));
    CHECK_CUDA(cudaFuncSetAttribute(
        tf32_tensor_core_ptx_async_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));

    dim3 block_size(kWarpSize, kWarpsPerBlock);
    dim3 grid_size(::ceil_div(n, kBlockN), ::ceil_div(m, kBlockM));
    tf32_tensor_core_ptx_async_kernel<<<grid_size, block_size, kSharedBytes>>>(dA, dB, dC, m, n, k);
}
