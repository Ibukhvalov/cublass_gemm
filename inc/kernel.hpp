#pragma once

#include "device/matrix.hpp"
#include <cassert>


// MxK @ KxN = MxN
template <typename I_FP_T, typename O_FP_T = I_FP_T>
class Kernel {
public:
    virtual ~Kernel() = default;

    void launch(device::Matrix<I_FP_T>& dA, device::Matrix<I_FP_T>& dB, device::Matrix<O_FP_T>& dC) {
        assert(dA.shape.rows == dC.shape.rows);
        assert(dB.shape.cols == dC.shape.cols);
        assert(dA.shape.cols == dB.shape.rows);
        launch(dA.data, dB.data, dC.data, dC.shape.rows, dC.shape.cols, dA.shape.cols);
    }

private:
    virtual void launch(I_FP_T* dA, I_FP_T* dB, O_FP_T* dC, int m, int n, int k) = 0;
};
