#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "macro.hpp"

using fp_t = float;

int main(int argc, char** argv) {
    const size_t n = (argc > 1) ? atoi(argv[1]) : 1 << 13;
    const size_t reps = 10;
    std::cout << "Running cuBLAS SGEMM with size " << n << std::endl;

    size_t elements_nb = n * n;
    size_t bytes_nb = elements_nb * sizeof(fp_t);

    std::vector<fp_t> hA(rand_vector<fp_t>(elements_nb)),
                      hB(rand_vector<fp_t>(elements_nb)),
                      hC(elements_nb, -1.f);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes_nb));
    CHECK_CUDA(cudaMalloc(&dB, bytes_nb));
    CHECK_CUDA(cudaMalloc(&dC, bytes_nb));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes_nb, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes_nb, cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    auto run = [&] () {
        CHECK_CUBLAS(cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &alpha,
            dA, CUDA_R_32F, n,
            dB, CUDA_R_32F, n,
            &beta,
            dC, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // Warming up the GPU
    for (int i=0; i<10; ++i) run();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i=0; i<reps; ++i) run();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    double ops_nb = 2.0 * n * n * n * reps;
    double elapsed_s = elapsed_ms / 1e3;
    std::cout << "Time: " << elapsed_ms << "ms (" << reps << " runs)\n"
    << "TFLOPS: " << ops_nb / (elapsed_s * 1e12);

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytes_nb, cudaMemcpyDeviceToHost));

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
