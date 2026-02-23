#include "kernel.hpp"
#include "utils.hpp"
#include "kernels/2d_blocktiling.cuh"
#include <vector_types.h>


constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

using uint = unsigned int;

using fp_t = Kernel::fp_t;

__global__ void block_tiling_2d_kernel(const fp_t* __restrict__ A, const fp_t* __restrict__ B, fp_t* __restrict__ C, const int m, const int n, const int k) {
    __shared__ fp_t As[BM][BK];
    __shared__ fp_t Bs[BK][BN];

    // (row, col) is a upper-left corner of the current block
    const uint2 blockStart = {blockIdx.y * BM, blockIdx.x * BN};

    uint threadsNb = blockDim.x;
    uint curThread = threadIdx.x;

    // (row, col)
    const uint2 startInnerA = {curThread / BK, curThread % BK};
    const uint2 startInnerB = {curThread / BN, curThread % BN};

    const uint loadsNb = BM * BK / threadsNb;
    const uint strideA = threadsNb / BK;
    const uint strideB = threadsNb / BN;

    const uint rowThreadsNb = BN / TN;

    // (x, y)
    const uint2 foldedThreadIdx = {curThread % rowThreadsNb, curThread / rowThreadsNb};

    fp_t acc[TM][TN] = {0};
    fp_t regM[TM] = {0};
    fp_t regN[TN] = {0};

    int tilesNb = ((k - 1) / BK) + 1;
    for (int tileIdx = 0; tileIdx < tilesNb; ++tileIdx) {
        int tileShift = tileIdx * BK;

         // load As
        int colA = startInnerA.y + tileShift;
        int startRowA = blockStart.x + startInnerA.x;
        for (int loadIdx = 0; loadIdx < loadsNb; ++loadIdx) {
            const uint shift = loadIdx * strideA;
            const uint rowA = startRowA + shift;

            if(rowA < m && colA < k)
                As[startInnerA.x + shift][startInnerA.y] = A[rowA * k + colA];
            else
                As[startInnerA.x + shift][startInnerA.y] = 0.0f;
        }

        // load Bs
        int colB = blockStart.y + startInnerB.y;
        int startRowB = startInnerB.x + tileShift;
        for (int loadIdx = 0; loadIdx < loadsNb; ++loadIdx) {
            const uint shift = loadIdx * strideB;
            const uint rowB = startRowB + shift;

            if(rowB < k && colB < n)
                Bs[startInnerB.x + shift][startInnerB.y] = B[rowB * n + colB];
            else
                Bs[startInnerB.x + shift][startInnerB.y] = 0.0f;
        }
        __syncthreads();
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load from SMEM to registers
            for (int regLoadIdx = 0; regLoadIdx < TM; ++regLoadIdx) {
                regM[regLoadIdx] = As[foldedThreadIdx.y * TM + regLoadIdx][dotIdx];
            }
            for (int regLoadIdx = 0; regLoadIdx < TN; ++regLoadIdx) {
                regN[regLoadIdx] = Bs[dotIdx][foldedThreadIdx.x * TN + regLoadIdx];
            }

            // Perform calculations from register
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    acc[resIdxM][resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }


        }

        __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int row = blockStart.x + (foldedThreadIdx.y * TM) + resIdxM;
            int col = blockStart.y + (foldedThreadIdx.x * TN) + resIdxN;
            if(row < m && col < n) {
                C[row * n + col] = acc[resIdxM][resIdxN];
            }
        }
    }
}

 void BlockTiling2DKernel::launch(fp_t* dA, fp_t* dB, fp_t* dC, int m, int n, int k) {
    static_assert(BN == BM, "Tile must be a square");
    static_assert(BM % TM == 0, "Tile rows must be a multiple of row-elements calculated by a thread");
    static_assert(BK == BM / TM);
    static_assert(BK == BN / TN);
    static_assert(TM * TN == BN);
    dim3 blockSize((BN * BM) / (TN * TM)); // blockSize unfolded for manual mapping smem load, since we would like to load each row of A and B as a single load.
    dim3 gridSize(::ceil_div(n, BN), ::ceil_div(m, BM));
    block_tiling_2d_kernel<<<gridSize, blockSize>>>(dA, dB, dC, m, n, k);
    //                       gridDim,  blockDim
};
