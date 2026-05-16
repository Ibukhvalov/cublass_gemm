#include "kernel.hpp"
#include "utils.hpp"
#include "kernels/tf32_tensor_core_register_tiling.cuh"

#include <mma.h>

using namespace nvcuda::wmma;

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockM = 32;
constexpr int kBlockN = 64;
constexpr int kBlockK = 8;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;

__global__ void tf32_tensor_core_register_tiling_kernel(
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

    fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major> a_frag;
    fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major> b_frag0;
    fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, precision::tf32, row_major> b_frag1;
    fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag0;
    fragment<accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag1;
    fill_fragment(acc_frag0, 0.0f);
    fill_fragment(acc_frag1, 0.0f);

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

        const int tile_row = warp_idx / 2;
        const int tile_col_group = warp_idx % 2;
        const int col0 = tile_col_group * 2 * kWmmaN;
        const int col1 = col0 + kWmmaN;

        load_matrix_sync(a_frag, &As[tile_row * kWmmaM][0], kBlockK);
        load_matrix_sync(b_frag0, &Bs[0][col0], kBlockN);
        load_matrix_sync(b_frag1, &Bs[0][col1], kBlockN);
        mma_sync(acc_frag0, a_frag, b_frag0, acc_frag0);
        mma_sync(acc_frag1, a_frag, b_frag1, acc_frag1);

        __syncthreads();
    }

    const int tile_row = warp_idx / 2;
    const int tile_col_group = warp_idx % 2;
    const int col0 = tile_col_group * 2 * kWmmaN;
    const int col1 = col0 + kWmmaN;
    store_matrix_sync(&Ctile[tile_row * kWmmaM][col0], acc_frag0, kBlockN, mem_row_major);
    store_matrix_sync(&Ctile[tile_row * kWmmaM][col1], acc_frag1, kBlockN, mem_row_major);
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

void Tf32TensorCoreRegisterTilingKernel::launch(tf32_reg_fp_t* dA, tf32_reg_fp_t* dB, tf32_reg_fp_t* dC, int m, int n, int k) {
    dim3 block_size(kWarpSize, kWarpsPerBlock);
    dim3 grid_size(::ceil_div(n, kBlockN), ::ceil_div(m, kBlockM));
    tf32_tensor_core_register_tiling_kernel<<<grid_size, block_size>>>(dA, dB, dC, m, n, k);
}
