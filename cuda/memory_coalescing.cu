#include "kernel.hpp"
#include "kernels/memory_coalescing.cuh"

using fp_t = Kernel::fp_t;

__global__ void memory_coalescing_kernel(fp_t* A, fp_t* B, fp_t* C, int m, int n, int k) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < m && col < n) {
        fp_t acc{0};
        for(int i = 0; i < k; ++i) {
            acc += A[row * k + i] * B[i * n + col];
        }
        C[row*n+col] = acc;
    }
}

/*
 * Transposing cuda-basis (meaning that in naive, x corresponds to the row, y - col,
 * while memory_coalescing uses x for col and y for row). Thus, cuda-x-axis aligns with a row direction of a matrix,
 * which is row-major (cuda wraps are cuda-x-axis major as well). So, at each iteration, memory load from matrix A is consecutive,
 * so hardware-accelerated parallel load is applied
 */
 void MemoryCoalesingKernel::launch(fp_t* dA, fp_t* dB, fp_t* dC, int m, int n, int k) {
    dim3 blockSize(32, 32, 1);
    dim3 gridSize(ceil(n / 32.f), ceil(m / 32.f));
    memory_coalescing_kernel<<<gridSize, blockSize>>>(dA, dB, dC, m, n, k);
};
