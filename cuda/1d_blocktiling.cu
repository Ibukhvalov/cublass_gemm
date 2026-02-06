#include "kernel.hpp"
#include "utils.hpp"
#include "kernels/smem_tiling.cuh"


#define BM 64
#define BN 64
#define BK 8
#define TM 8



using fp_t = Kernel::fp_t;

__global__ void shared_memory_tiling_kernel(const fp_t* __restrict__ A, const fp_t* __restrict__ B, fp_t* __restrict__ C, const int m, const int n, const int k) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    fp_t acc[TM]{0};
    int tilesNb = ((k - 1) / TILE_SIZE) + 1;
    for(int tileIdx = 0; tileIdx < tilesNb; ++tileIdx) {
        int tileShift = tileIdx * TILE_SIZE;

        // load As
        int colA = threadIdx.x + tileShift;
        if(row < m && colA < k)
            As[threadIdx.y][threadIdx.x] = A[row * k + colA];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // load Bs
        int rowB = threadIdx.y + tileShift;
        if(rowB < k && col < n)
            Bs[threadIdx.y][threadIdx.x] = B[rowB * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        //#pragma unroll
        for(int i = 0; i < TILE_SIZE; ++i)
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if(row < m && col < n)
        C[row * n + col] = acc;
}

 void SharedMemoryTilingKernel::launch(fp_t* dA, fp_t* dB, fp_t* dC, int m, int n, int k) {
    assert(BM % TM == 0);
    dim3 blockSize(BN, BM / TM);
    dim3 gridSize(::ceil_div(n, BN), ::ceil_div(m, BM));
    shared_memory_tiling_kernel<<<gridSize, blockSize>>>(dA, dB, dC, m, n, k);
};
