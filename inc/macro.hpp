#pragma once

#include <iostream>

#define CHECK_CUDA(val) do {   \
cudaError_t err = (val);       \
if(err != cudaSuccess) {       \
    std::cerr                  \
    << "CUDA Error: "          \
    << cudaGetErrorString(err) \
    << std::endl;              \
    exit(EXIT_FAILURE);        \
}                              \
} while(0)

#define CHECK_CUBLAS(val) do {   \
cublasStatus_t s = (val);        \
if(s != CUBLAS_STATUS_SUCCESS) { \
    std::cerr                    \
    << "cuBLAS Error: " << s     \
    << std::endl;                \
    exit(EXIT_FAILURE);          \
}                                \
} while(0)
