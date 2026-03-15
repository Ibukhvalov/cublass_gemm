#pragma once

#include "kernel.hpp"

using i_fp_t = half;
using o_fp_t = float;

class TilingTensorCoreKernel : public Kernel<i_fp_t, o_fp_t> {
    virtual void launch(i_fp_t* dA, i_fp_t* dB, o_fp_t* dC, int m, int n, int k) override;
};
