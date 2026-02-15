#include <utils.hpp>
#include "kernels/naive.cuh"

using fp_t = Kernel::fp_t;

__global__ void naive_kernel(const fp_t* __restrict__ A, const fp_t* __restrict__ B, fp_t* __restrict__ C, int m, int n, int k) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < m && col < n) {
        fp_t acc{0};
        for(int i = 0; i < k; ++i) {
            acc += A[row * k + i] * B[i * n + col];
        }
        C[row*n+col] = acc;
    }
}

void NaiveKernel::launch(fp_t* dA, fp_t* dB, fp_t* dC, int m, int n, int k) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize(::ceil_div(m, 32), ::ceil_div(n, 32));
    naive_kernel<<<gridSize, blockSize>>>(dA, dB, dC, m, n, k);
};
