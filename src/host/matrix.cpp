#pragma once

#include "host/matrix.hpp"
#include "device/matrix.hpp"
#include "macro.hpp"

#include <iomanip>
#include <numeric>
#include <random>
#include <cassert>

#include <cuda_runtime.h>

namespace host {

Matrix Matrix::CreateRandom(Shape shape, fp_t deviation) {
    static unsigned int seed = 27;
    static std::mt19937 eng(seed);

    Matrix matrix(shape);

    std::uniform_real_distribution<fp_t> dist(-deviation, deviation);

    std::generate(matrix.data, matrix.data + matrix.shape.elements_nb(), [&]() {
        return dist(eng);
    });

    return matrix;
};

Matrix Matrix::CopyFromDevice(const device::Matrix &deviceMatrix) {
    Matrix matrix(deviceMatrix.shape);
    CHECK_CUDA(cudaMemcpy(matrix.data, deviceMatrix.data, deviceMatrix.bytes_size(), cudaMemcpyDeviceToHost));
    return matrix;
};

};

std::ostream& operator<<(std::ostream& out, const host::Matrix& mat) {
    out << "( " << mat.shape.rows << ", " << mat.shape.cols << " )\n";
    out << std::fixed << std::setprecision(2);
    for(int i = 0; i < mat.shape.rows; ++i) {
        for(int j = 0; j < mat.shape.cols; ++j) {
            out << std::setw(8) << mat.at(i,j);
        }
        out << std::endl;
    }
    return out;
}

bool operator==(const host::Matrix& left, const host::Matrix& right) {
    if(left.shape != right.shape)
        return false;

    auto rows = left.shape.rows;
    auto cols = left.shape.cols;
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            const double precision = 1e-2;
            auto diff = std::abs(left.at(i,j) - right.at(i,j));
            if(diff > precision) {
                std::cerr << "too high diff at (" << i << ", " << j << ") = " << std::fixed << diff << std::endl;
                return false;
            }
        }
    }
    return true;
}

host::Matrix operator*(const host::Matrix& left, const host::Matrix& right) {
    assert(left.shape.cols == right.shape.rows);

    int m = left.shape.rows;
    int n = left.shape.cols;
    int k = right.shape.cols;
    host::Matrix res({m,k});
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            auto& acc = res.at(i, j);
            acc = 0;
            for(int t = 0; t < n; ++t)
                acc += left.at(i, t) * right.at(t, j);
        }
    }
    return res;
};
