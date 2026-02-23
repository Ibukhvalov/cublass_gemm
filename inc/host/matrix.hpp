#pragma once

#include "matrix_data.hpp"
#include <ostream>

namespace device {
class Matrix;
};

namespace host {

struct Allocator : public FPType {
    static fp_t* allocate(int elements_nb) {
        return new fp_t[elements_nb];
    }
    static void deallocate(fp_t* data) {
        delete[] data;
    }
};

class Matrix : public MatrixData<Allocator> {
public:
    static Matrix CreateRandom(Shape shape, fp_t deviation = 10);
    static Matrix CreateConsecutive(Shape shape);
    static Matrix CopyFromDevice(const device::Matrix& deviceMatrix);
protected:
    using MatrixData<Allocator>::MatrixData;
};

};

std::ostream& operator<<(std::ostream& out, const host::Matrix& mat);

bool operator==(const host::Matrix& left, const host::Matrix& right);

host::Matrix operator*(const host::Matrix& left, const host::Matrix& right);
