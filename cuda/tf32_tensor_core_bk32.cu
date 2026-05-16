#include "kernel.hpp"
#include "utils.hpp"
#include "kernels/tf32_tensor_core_bk32.cuh"

#include <mma.h>

using namespace nvcuda::wmma;

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 32;
constexpr int kWarpTileM = 32;
constexpr int kWarpTileN = 32;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;

__global__ void tf32_tensor_core_bk32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m,
    int n,
    int k) {
    __shared__ float As[kBlockM][kBlockK];
    __shared__ float Bs[kBlockK][kBlockN];
    __shared__ float Ctile[kBlockM][kBlockN];

    const int warp_idx = threadIdx.y;
    const int lane_idx = threadIdx.x;
    const int linear_tid = warp_idx * kWarpSize + lane_idx;
    const int block_row = blockIdx.y * kBlockM;
    const int block_col = blockIdx.x * kBlockN;
    const int warp_row = warp_idx / 2;
    const int warp_col = warp_idx % 2;
    const int warp_row_base = warp_row * kWarpTileM;
    const int warp_col_base = warp_col * kWarpTileN;

    fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc[2][2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            fill_fragment(acc[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < k; k0 += kBlockK) {
        for (int idx = linear_tid; idx < kBlockM * kBlockK; idx += kWarpSize * kWarpsPerBlock) {
            const int row = idx / kBlockK;
            const int col = idx % kBlockK;
            const int global_row = block_row + row;
            const int global_col = k0 + col;
            const float value = (global_row < m && global_col < k) ? A[global_row * k + global_col] : 0.0f;
            As[row][col] = __float_to_tf32(value);
        }

        for (int idx = linear_tid; idx < kBlockK * kBlockN; idx += kWarpSize * kWarpsPerBlock) {
            const int row = idx / kBlockN;
            const int col = idx % kBlockN;
            const int global_row = k0 + row;
            const int global_col = block_col + col;
            const float value = (global_row < k && global_col < n) ? B[global_row * n + global_col] : 0.0f;
            Bs[row][col] = __float_to_tf32(value);
        }

        __syncthreads();

        for (int kk = 0; kk < kBlockK; kk += kWmmaK) {
            fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major> a_frag[2];
            fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major> b_frag[2];

            load_matrix_sync(a_frag[0], &As[warp_row_base + 0][kk], kBlockK);
            load_matrix_sync(a_frag[1], &As[warp_row_base + kWmmaM][kk], kBlockK);
            load_matrix_sync(b_frag[0], &Bs[kk][warp_col_base + 0], kBlockN);
            load_matrix_sync(b_frag[1], &Bs[kk][warp_col_base + kWmmaN], kBlockN);

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    store_matrix_sync(&Ctile[warp_row_base + 0][warp_col_base + 0], acc[0][0], kBlockN, mem_row_major);
    store_matrix_sync(&Ctile[warp_row_base + 0][warp_col_base + kWmmaN], acc[0][1], kBlockN, mem_row_major);
    store_matrix_sync(&Ctile[warp_row_base + kWmmaM][warp_col_base + 0], acc[1][0], kBlockN, mem_row_major);
    store_matrix_sync(&Ctile[warp_row_base + kWmmaM][warp_col_base + kWmmaN], acc[1][1], kBlockN, mem_row_major);
    __syncthreads();

    for (int idx = linear_tid; idx < kBlockM * kBlockN; idx += kWarpSize * kWarpsPerBlock) {
        const int row = idx / kBlockN;
        const int col = idx % kBlockN;
        const int global_row = block_row + row;
        const int global_col = block_col + col;
        if (global_row < m && global_col < n) {
            C[global_row * n + global_col] = Ctile[row][col];
        }
    }
}
}

void Tf32TensorCoreBk32Kernel::launch(tf32_bk32_fp_t* dA, tf32_bk32_fp_t* dB, tf32_bk32_fp_t* dC, int m, int n, int k) {
    dim3 block_size(kWarpSize, kWarpsPerBlock);
    dim3 grid_size(::ceil_div(n, kBlockN), ::ceil_div(m, kBlockM));
    tf32_tensor_core_bk32_kernel<<<grid_size, block_size>>>(dA, dB, dC, m, n, k);
}
