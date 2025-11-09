 #include "kernel.cuh"
#include <cuda_runtime.h>

__global__ void incrementKernel(float* A, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        A[idx] += 1.0f;
    }
}

void IncrementVec(float* A, int N) {
    float* d_A;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    incrementKernel<<<blocks, threads>>>(d_A, N);
    cudaDeviceSynchronize();

    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
}
