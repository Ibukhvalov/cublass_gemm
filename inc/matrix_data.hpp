#pragma once
#include <iostream>


struct Shape {
    int elements_nb() const {
        return rows * cols;
    }

    int rows, cols;
};

bool operator==(const Shape& left, const Shape& right);
bool operator!=(const Shape& left, const Shape& right);

struct FPType {
    using fp_t = float;
};

// Row-Major order
template <typename Allocator>
struct MatrixData {
    using fp_t = typename Allocator::fp_t;

    MatrixData(): data(nullptr) {};
    MatrixData(Shape shape): shape(shape), data(Allocator::allocate(shape.elements_nb())) {};

    fp_t& at(int row, int col) {
        return data[row * shape.cols + col];
    }

    fp_t at(int row, int col) const {
        return data[row * shape.cols + col];
    }

    ~MatrixData() {
        if(data)
            Allocator::deallocate(data);
    }

    MatrixData(MatrixData&& other) noexcept
        : shape(other.shape), data(other.data) {
        other.data = nullptr;
    }

    MatrixData& operator=(MatrixData&& other) noexcept {
        if (this != &other) {
            if(data)
                Allocator::deallocate(data);
            shape = other.shape;
            data = other.data;
            other.data = nullptr;
        }

        return *this;
    }

    MatrixData(const MatrixData&) = delete;
    MatrixData& operator=(const MatrixData&) = delete;

    int bytes_size() const {
        return shape.elements_nb() * sizeof(fp_t);
    }

    Shape shape;
    fp_t* data;
};
