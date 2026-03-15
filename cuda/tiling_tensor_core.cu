#include "kernel.hpp"
#include "utils.hpp"
#include "kernels/tiling_tensor_core.cuh"
#include <cuda_fp16.h>
#include <mma.h>

#include <iostream>

using namespace nvcuda::wmma;

constexpr int WARP_SIZE = 32;
constexpr int TILE_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int FRAG_SIZE = 16;

__global__ void tensor_core_tiling_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, const int m, const int n, const int k) {

    // (col, row)
    const uint2 blockStart = {blockIdx.y * 32, blockIdx.x * 32};
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;

    fill_fragment(acc_frag, 0.0f);

    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];

    assert(blockDim.x == WARP_SIZE);
    int warpIdx = threadIdx.y;
    int warpThreadIdx = threadIdx.x;

    // (row, col)
    const int2 tileIdx = {warpIdx / 2, warpIdx % 2};
    for(int k0 = 0; k0 < k; k0 += TILE_SIZE) {
        for(int localRow = warpIdx; localRow < TILE_SIZE; localRow += WARPS_PER_BLOCK) {

            int rowA = blockStart.y + localRow;
            int colA = k0 + warpThreadIdx;
            As[localRow][warpThreadIdx] =
                (rowA < m && colA < k)
                ? A[rowA * k + colA]
                : __float2half(0.0f);

            int rowB = k0 + localRow;
            int colB = blockStart.x + warpThreadIdx;
            Bs[localRow][warpThreadIdx] =
                (rowB < k && colB < n)
                    ? B[rowB * n + colB]
                    : __float2half(0.0f);
        }

        __syncthreads();

        for(int blockEntry = 0; blockEntry < TILE_SIZE; blockEntry += FRAG_SIZE) {
            load_matrix_sync(a_frag, &As[tileIdx.x * FRAG_SIZE][blockEntry], 32);
            load_matrix_sync(b_frag, &Bs[blockEntry][tileIdx.y * FRAG_SIZE], 32);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }
    __shared__ float cTile[TILE_SIZE][TILE_SIZE];
    store_matrix_sync(&cTile[tileIdx.x * FRAG_SIZE][tileIdx.y * FRAG_SIZE], acc_frag, 32, mem_row_major);
    __syncthreads();

    for(int localRow = warpIdx; localRow < TILE_SIZE; localRow += WARPS_PER_BLOCK) {
        int globalCol = blockStart.x + warpThreadIdx;
        int globalRow = blockStart.y + localRow;

        if(globalRow < m && globalCol < n) {
            C[globalRow * n + globalCol] = cTile[localRow][warpThreadIdx];
        }
    }
}

 void TilingTensorCoreKernel::launch(i_fp_t* dA, i_fp_t* dB, o_fp_t* dC, int m, int n, int k) {
     dim3 blockSize(WARP_SIZE, WARPS_PER_BLOCK);
     dim3 gridSize(::ceil_div(n, 32), ::ceil_div(m, 32));
     tensor_core_tiling_kernel<<<gridSize, blockSize>>>(dA, dB, dC, m, n, k);
};
