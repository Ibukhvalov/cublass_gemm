#pragma once

#include "matrix_data.hpp"
#include "macro.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace host {
class Matrix;
};

namespace device {

struct Allocator : public FPType {
    static fp_t* allocate(int elements_nb) {
        fp_t* data;
        CHECK_CUDA(cudaMalloc(&data, elements_nb * sizeof(fp_t)));
        return data;
    }
    static void deallocate(fp_t* data) {
        CHECK_CUDA(cudaFree(data));
    }
};

class Matrix : public MatrixData<Allocator> {
public:
    static Matrix CopyFromHost(const host::Matrix& hostMatrix);

protected:
    using MatrixData<Allocator>::MatrixData;
};

};
