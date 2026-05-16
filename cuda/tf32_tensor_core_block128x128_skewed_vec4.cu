#include "kernel.hpp"
#include "utils.hpp"
#include "kernels/tf32_tensor_core_block128x128_skewed_vec4.cuh"

#include <mma.h>

using namespace nvcuda::wmma;

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 16;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
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

__device__ inline void store_tf32_vec4(float* smem_row, int col, const float4& vec) {
    smem_row[col + 0] = __float_to_tf32(vec.x);
    smem_row[col + 1] = __float_to_tf32(vec.y);
    smem_row[col + 2] = __float_to_tf32(vec.z);
    smem_row[col + 3] = __float_to_tf32(vec.w);
}

__global__ void tf32_tensor_core_block128x128_skewed_vec4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m,
    int n,
    int k) {
    __shared__ float As[kBlockM][kSharedStrideA];
    __shared__ float Bs[kBlockK][kSharedStrideB];

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

    for (int k0 = 0; k0 < k; k0 += kBlockK) {
        constexpr int kAVecCols = kBlockK / 4;
        for (int vec_idx = linear_tid; vec_idx < kBlockM * kAVecCols; vec_idx += kThreadsPerBlock) {
            const int row = vec_idx / kAVecCols;
            const int col = (vec_idx % kAVecCols) * 4;
            const int global_row = block_row + row;
            const int global_col = k0 + col;
            float* smem_row = &As[row][0];

            if (global_row < m && global_col + 3 < k) {
                const float4 vec = *reinterpret_cast<const float4*>(&A[global_row * k + global_col]);
                store_tf32_vec4(smem_row, col, vec);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int cur_col = global_col + i;
                    const float value = (global_row < m && cur_col < k) ? A[global_row * k + cur_col] : 0.0f;
                    smem_row[col + i] = __float_to_tf32(value);
                }
            }
        }

        constexpr int kBVecCols = kBlockN / 4;
        for (int vec_idx = linear_tid; vec_idx < kBlockK * kBVecCols; vec_idx += kThreadsPerBlock) {
            const int row = vec_idx / kBVecCols;
            const int col = (vec_idx % kBVecCols) * 4;
            const int global_row = k0 + row;
            const int global_col = block_col + col;
            float* smem_row = &Bs[row][0];

            if (global_row < k && global_col + 3 < n) {
                const float4 vec = *reinterpret_cast<const float4*>(&B[global_row * n + global_col]);
                store_tf32_vec4(smem_row, col, vec);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int cur_col = global_col + i;
                    const float value = (global_row < k && cur_col < n) ? B[global_row * n + cur_col] : 0.0f;
                    smem_row[col + i] = __float_to_tf32(value);
                }
            }
        }

        __syncthreads();

        for (int kk = 0; kk < kBlockK; kk += kWmmaK) {
            fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major> a_frag[2];
            fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major> b_frag[2];

            load_matrix_sync(a_frag[0], &As[warp_row_base + 0][kk], kSharedStrideA);
            load_matrix_sync(a_frag[1], &As[warp_row_base + kWmmaM][kk], kSharedStrideA);
            load_matrix_sync(b_frag[0], &Bs[kk][warp_col_base + 0], kSharedStrideB);
            load_matrix_sync(b_frag[1], &Bs[kk][warp_col_base + kWmmaN], kSharedStrideB);

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    if (block_row + warp_row_base + kWarpTileM <= m && block_col + warp_col_base + kWarpTileN <= n) {
        store_matrix_sync(C + (block_row + warp_row_base + 0) * n + (block_col + warp_col_base + 0), acc[0][0], n, mem_row_major);
        store_matrix_sync(C + (block_row + warp_row_base + 0) * n + (block_col + warp_col_base + kWmmaN), acc[0][1], n, mem_row_major);
        store_matrix_sync(C + (block_row + warp_row_base + kWmmaM) * n + (block_col + warp_col_base + 0), acc[1][0], n, mem_row_major);
        store_matrix_sync(C + (block_row + warp_row_base + kWmmaM) * n + (block_col + warp_col_base + kWmmaN), acc[1][1], n, mem_row_major);
    }
}
}

void Tf32TensorCoreBlock128x128SkewedVec4Kernel::launch(tf32_block128x128_skewed_vec4_fp_t* dA, tf32_block128x128_skewed_vec4_fp_t* dB, tf32_block128x128_skewed_vec4_fp_t* dC, int m, int n, int k) {
    dim3 block_size(kWarpSize, kWarpsPerBlock);
    dim3 grid_size(::ceil_div(n, kBlockN), ::ceil_div(m, kBlockM));
    tf32_tensor_core_block128x128_skewed_vec4_kernel<<<grid_size, block_size>>>(dA, dB, dC, m, n, k);
}
