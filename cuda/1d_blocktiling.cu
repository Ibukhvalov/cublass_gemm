#include "kernel.hpp"
#include "utils.hpp"
#include "kernels/1d_blocktiling.cuh"
#include <vector_types.h>


constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;


 using fp_t = Kernel::fp_t;

__global__ void block_tiling_1d_kernel(const fp_t* __restrict__ A, const fp_t* __restrict__ B, fp_t* __restrict__ C, const int m, const int n, const int k) {
    __shared__ fp_t As[BM][BK];
    __shared__ fp_t Bs[BK][BN];

    // (row, col) is a upper-left corner of the current block
    const uint2 blockStart = {blockIdx.y * BM, blockIdx.x * BN};

    fp_t acc[TM] = {0};
    int tilesNb = ((k - 1) / BK) + 1;
    for(int tileIdx = 0; tileIdx < tilesNb; ++tileIdx) {

        int tileShift = tileIdx * BK;
        // load As
        int colA = threadIdx.y + tileShift;
        int rowA = blockStart.x + threadIdx.x;
        if(rowA < m && colA < k)
            As[threadIdx.x][threadIdx.y] = A[rowA * k + colA];
        else
            As[threadIdx.x][threadIdx.y] = 0.0f;


        // load Bs
        int colB = blockStart.y + threadIdx.x;
        int rowB = threadIdx.y + tileShift;
        if (rowB < k && colB < n)
            Bs[threadIdx.y][threadIdx.x] = B[rowB * n + colB];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for(int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            fp_t b = Bs[dotIdx][threadIdx.x];
            for(int resIdx = 0; resIdx < TM; ++resIdx) {
                fp_t a = As[threadIdx.y * TM + resIdx][dotIdx];
                acc[resIdx] += a * b;
            }
        }

        __syncthreads();

    }

    for(int resIdx = 0; resIdx < TM; ++resIdx) {
        int row = blockStart.x + resIdx + threadIdx.y * TM;
        int col = blockStart.y + threadIdx.x;
        if(row < m && col < n) {
            C[row * n + col] = acc[resIdx];
        }
    }
}

 void BlockTiling1DKernel::launch(fp_t* dA, fp_t* dB, fp_t* dC, int m, int n, int k) {
    assert(BN == BM);
    assert(BM % TM == 0);
    assert(BK == BM / TM);
    dim3 blockSize(BN, BK);
    dim3 gridSize(::ceil_div(n, BN), ::ceil_div(m, BM));
    block_tiling_1d_kernel<<<gridSize, blockSize>>>(dA, dB, dC, m, n, k);
};
