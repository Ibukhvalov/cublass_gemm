#include "kernels/cublass.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "macro.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "macro.hpp"
#include "kernel.hpp"


using fp_t = Kernel::fp_t;

void CublassKernel::launch(fp_t* dA, fp_t* dB, fp_t* dC, int m, int n, int k) {
    const float alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    CHECK_CUBLAS(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        dB, CUDA_R_32F, n,
        dA, CUDA_R_32F, k,
        &beta,
        dC, CUDA_R_32F, n,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
};
