#pragma once

#include "matrix_data.hpp"
#include "macro.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace device {

template <typename T>
struct Allocator  {
    using fp_t = T;

    static fp_t* allocate(int elements_nb) {
        fp_t* data;
        CHECK_CUDA(cudaMalloc(&data, elements_nb * sizeof(fp_t)));
        return data;
    }
    static void deallocate(fp_t* data) {
        CHECK_CUDA(cudaFree(data));
    }
};

template <typename fp_t = float>
class Matrix : public MatrixData<Allocator<fp_t>> {
protected:
    using MatrixData<Allocator<fp_t>>::MatrixData;
};

};
