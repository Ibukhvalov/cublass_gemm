 #pragma once

#include "device/matrix.hpp"
#include "matrix_data.hpp"
#include "host/matrix.hpp"
#include <macro.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace device {

Matrix Matrix::CopyFromHost(const host::Matrix& hostMatrix) {
    Matrix matrix(hostMatrix.shape);
    CHECK_CUDA(cudaMemcpy(matrix.data, hostMatrix.data, hostMatrix.shape.elements_nb() * sizeof(fp_t), cudaMemcpyHostToDevice));
    return matrix;
};

};
