#pragma once

#include "device/matrix.hpp"
#include <cassert>


// MxK @ KxN = MxN
class Kernel {
public:
    using fp_t = device::Allocator::fp_t;

    virtual ~Kernel() = default;

    void launch(device::Matrix& dA, device::Matrix& dB, device::Matrix& dC) {
        assert(dA.shape.rows == dC.shape.rows);
        assert(dB.shape.cols == dC.shape.cols);
        assert(dA.shape.cols == dB.shape.rows);

        std::cout << "o\n";
        launch(dA.data, dB.data, dC.data, dC.shape.rows, dC.shape.cols, dA.shape.cols);
        std::cout << "o\n";
    }

private:
    virtual void launch(fp_t* dA, fp_t* dB, fp_t* dC, int m, int n, int k) = 0;
};
