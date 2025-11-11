#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "macro.hpp"

using fp_t = float;

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 4096;
    std::cout << "Running cuBLAS SGEMM with size " << n << std::endl;

    size_t elements_nb = n * n;
    size_t bytes_nb = elements_nb * sizeof(fp_t);

    std::vector<fp_t> hA(rand_vector<fp_t>(elements_nb)),
                      hB(rand_vector<fp_t>(elements_nb)),
                      hC(rand_vector<fp_t>(elements_nb));

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes_nb));
    CHECK_CUDA(cudaMalloc(&dB, bytes_nb));
    CHECK_CUDA(cudaMalloc(&dC, bytes_nb));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes_nb, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes_nb, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH))


    const float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    auto run = [&] () {
        CHECK_CUBLAS(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        dB, CUDA_R_32F, n,
        dA, CUDA_R_32F, n,
        &beta,
        dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // Warming up the GPU
    for(int i=0; i<10; ++i) run();
    cudaDeviceSynchronize();

    const int reps = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for(int i=0; i<reps; ++i) run();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    double ops_nb = 2.0 * n * n * n * reps;
    std::cout << "Time: " << elapsed_ms << " ms\n"
    << "TFLOPS: " << ops_nb / (elapsed_ms * 1e9);

    CHECK_CUBLAS(cublasDestroy(handle));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
