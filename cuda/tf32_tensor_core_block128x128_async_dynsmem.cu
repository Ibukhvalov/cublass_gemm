#include "kernel.hpp"
#include "utils.hpp"
#include "kernels/tf32_tensor_core_block128x128_async_dynsmem.cuh"

#include <cuda_pipeline_primitives.h>
#include <mma.h>

using namespace nvcuda::wmma;

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 16;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kStages = 2;
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;
constexpr int kSkew = 8;
constexpr int kSharedStrideA = kBlockK + kSkew;
constexpr int kSharedStrideB = kBlockN + kSkew;
constexpr int kWarpTileM = 32;
constexpr int kWarpTileN = 32;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;
constexpr int kAStageElems = kBlockM * kSharedStrideA;
constexpr int kBStageElems = kBlockK * kSharedStrideB;
constexpr int kSharedBytes = (kStages * kAStageElems + kStages * kBStageElems) * sizeof(float);

__device__ inline float* a_stage_ptr(float* smem, int stage, int row, int col) {
    return smem + stage * kAStageElems + row * kSharedStrideA + col;
}

__device__ inline float* b_stage_ptr(float* smem, int stage, int row, int col) {
    float* b_base = smem + kStages * kAStageElems;
    return b_base + stage * kBStageElems + row * kSharedStrideB + col;
}

__device__ void issue_async_copy_stage(
    float* smem,
    const float* A,
    const float* B,
    int m,
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
                __pipeline_memcpy_async(dst, &A[global_row * k + global_col], 16, 16 - remaining * sizeof(float));
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
                __pipeline_memcpy_async(dst, &B[global_row * n + global_col], 16, 16 - remaining * sizeof(float));
            } else {
                __pipeline_memcpy_async(dst, &B[0], 16, 16);
            }
        } else {
            __pipeline_memcpy_async(dst, &B[0], 16, 16);
        }
    }
}

__global__ void tf32_tensor_core_block128x128_async_dynsmem_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m,
    int n,
    int k) {
    extern __shared__ float smem[];

    const int warp_idx = threadIdx.y;
    const int lane_idx = threadIdx.x;
    const int linear_tid = warp_idx * kWarpSize + lane_idx;
    const int block_row = blockIdx.y * kBlockM;
    const int block_col = blockIdx.x * kBlockN;
    const int warp_row = warp_idx / 4;
    const int warp_col = warp_idx % 4;
    const int warp_row_base = warp_row * kWarpTileM;
    const int warp_col_base = warp_col * kWarpTileN;

    fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc[2][2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            fill_fragment(acc[i][j], 0.0f);
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

        for (int kk = 0; kk < kBlockK; kk += kWmmaK) {
            fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major> a_frag[2];
            fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major> b_frag[2];

            load_matrix_sync(a_frag[0], a_stage_ptr(smem, stage, warp_row_base + 0, kk), kSharedStrideA);
            load_matrix_sync(a_frag[1], a_stage_ptr(smem, stage, warp_row_base + kWmmaM, kk), kSharedStrideA);
            load_matrix_sync(b_frag[0], b_stage_ptr(smem, stage, kk, warp_col_base + 0), kSharedStrideB);
            load_matrix_sync(b_frag[1], b_stage_ptr(smem, stage, kk, warp_col_base + kWmmaN), kSharedStrideB);

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        if (next_k0 < k) {
            __pipeline_wait_prior(0);
            __syncthreads();
            stage = next_stage;
        }
    }

    if (block_row + warp_row_base + kWarpTileM <= m && block_col + warp_col_base + kWarpTileN <= n) {
        store_matrix_sync(C + (block_row + warp_row_base + 0) * n + (block_col + warp_col_base + 0), acc[0][0], n, mem_row_major);
        store_matrix_sync(C + (block_row + warp_row_base + 0) * n + (block_col + warp_col_base + kWmmaN), acc[0][1], n, mem_row_major);
        store_matrix_sync(C + (block_row + warp_row_base + kWmmaM) * n + (block_col + warp_col_base + 0), acc[1][0], n, mem_row_major);
        store_matrix_sync(C + (block_row + warp_row_base + kWmmaM) * n + (block_col + warp_col_base + kWmmaN), acc[1][1], n, mem_row_major);
    }
}
}

void Tf32TensorCoreBlock128x128AsyncDynamicSmemKernel::launch(tf32_block128x128_async_dynsmem_fp_t* dA, tf32_block128x128_async_dynsmem_fp_t* dB, tf32_block128x128_async_dynsmem_fp_t* dC, int m, int n, int k) {
    CHECK_CUDA(cudaFuncSetAttribute(
        tf32_tensor_core_block128x128_async_dynsmem_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes));
    CHECK_CUDA(cudaFuncSetAttribute(
        tf32_tensor_core_block128x128_async_dynsmem_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));

    dim3 block_size(kWarpSize, kWarpsPerBlock);
    dim3 grid_size(::ceil_div(n, kBlockN), ::ceil_div(m, kBlockM));
    tf32_tensor_core_block128x128_async_dynsmem_kernel<<<grid_size, block_size, kSharedBytes>>>(dA, dB, dC, m, n, k);
}
