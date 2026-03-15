#include <macro.hpp>
#include "host/matrix.hpp"
#include "device/matrix.hpp"

constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}


template <typename fp_t>
host::Matrix<fp_t> CopyFromDeviceToHost(const device::Matrix<fp_t>& deviceMatrix) {
    host::Matrix<fp_t> matrix(deviceMatrix.shape);
    CHECK_CUDA(cudaMemcpy(matrix.data, deviceMatrix.data, deviceMatrix.bytes_size(), cudaMemcpyDeviceToHost));
    return matrix;
}

template <typename fp_t>
device::Matrix<fp_t> CopyFromHostToDevice(const host::Matrix<fp_t>& hostMatrix) {
    device::Matrix<fp_t> matrix(hostMatrix.shape);
    CHECK_CUDA(cudaMemcpy(matrix.data, hostMatrix.data, hostMatrix.shape.elements_nb() * sizeof(fp_t), cudaMemcpyHostToDevice));
    return matrix;
}
