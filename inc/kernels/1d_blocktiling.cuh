#pragma once

#include "kernel.hpp"

using fp_t = float;

class BlockTiling1DKernel : public Kernel<fp_t> {
    virtual void launch(fp_t* dA, fp_t* dB, fp_t* dC, int m, int n, int k) override;
};
