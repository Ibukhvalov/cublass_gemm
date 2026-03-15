#pragma once

#include "matrix_data.hpp"
#include <ostream>
#include <iomanip>
#include <cassert>
#include <random>

#include <cuda_fp16.h>


namespace host {

template <typename T>
struct Allocator {
    using fp_t = T;

    static fp_t* allocate(int elements_nb) {
        return new T[elements_nb];
    }
    static void deallocate(fp_t* data) {
        delete[] data;
    }
};
/*
template <>
struct Allocator<half> {
    using fp_t = half;

    static fp_t* allocate(int elements_nb) {
        const size_t alignment = 16; // 16-byte alignment for WMMA
        void* ptr = _aligned_malloc(elements_nb * sizeof(fp_t), alignment);
        if (!ptr) return nullptr; // allocation failed
        return reinterpret_cast<fp_t*>(ptr);
    }

    static void deallocate(fp_t* data) {
        _aligned_free(data);
    }
};
*/
template <typename fp_t = float>
class Matrix : public MatrixData<Allocator<fp_t>> {
public:

    static Matrix CreateRandom(Shape shape, float deviation = 1.f) {
        static unsigned int seed = 27;
        static std::mt19937 eng(seed);

        Matrix matrix(shape);

        using dist_t = std::conditional_t<
            std::is_same_v<fp_t, half>,
            float,
            fp_t
        >;

        std::uniform_real_distribution<dist_t> dist(-deviation, deviation);

        std::generate(matrix.data,
                       matrix.data + matrix.shape.elements_nb(),
                       [&]() {
                           return static_cast<fp_t>(dist(eng));
                       });

        return matrix;
    }

    static Matrix CreateConsecutive(Shape shape) {
        Matrix matrix(shape);
        fp_t i = 1;
        std::generate(matrix.data, matrix.data + matrix.shape.elements_nb(), [&]() {
            fp_t r = i;
            i += fp_t(1);
            return r;
            });

        return matrix;
    }

    template<typename other_fp_t>
    Matrix(const Matrix<other_fp_t>& other)
        : MatrixData<Allocator<fp_t>>(other.shape)
    {
        std::transform(
            other.data,
            other.data + other.shape.elements_nb(),
            this->data,
            [](other_fp_t v) {
                return static_cast<fp_t>(v);
            });
    }

protected:
    using MatrixData<Allocator<fp_t>>::MatrixData;
};

};

template <typename fp_t>
std::ostream& operator<<(std::ostream& out, const host::Matrix<fp_t>& mat) {
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

template <typename fp_t>
bool operator==(const host::Matrix<fp_t>& left, const host::Matrix<fp_t>& right) {
    if(left.shape != right.shape)
        return false;

    fp_t maxDiff = 0.;
    const double precision = 1e-2;
    const double diffThreshold = 10.;

    auto rows = left.shape.rows;
    auto cols = left.shape.cols;
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            auto diff = std::abs(left.at(i,j) - right.at(i,j));
            maxDiff = std::max(diff, maxDiff);
            if(diff > diffThreshold) {
                std::cerr << "too high diff at (" << i << ", " << j << ") = " << std::fixed << diff << std::endl;
                return false;
            }
        }
    }
    if(maxDiff > precision) {
        std::cerr << "diff is high: " << maxDiff << std::endl;
    }
    return true;
}

template <typename fp_t>
host::Matrix<fp_t> operator*(const host::Matrix<fp_t>& left, const host::Matrix<fp_t>& right) {
    assert(left.shape.cols == right.shape.rows);

    int m = left.shape.rows;
    int n = left.shape.cols;
    int k = right.shape.cols;
    host::Matrix<fp_t> res({m,k});
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            auto& acc = res.at(i, j);
            acc = 0;
            for(int t = 0; t < n; ++t)
                acc += left.at(i, t) * right.at(t, j);
        }
    }
    return res;
}
