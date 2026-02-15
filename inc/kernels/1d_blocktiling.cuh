#pragma once

#include "kernel.hpp"

class BlockTiling1DKernel : public Kernel {
    virtual void launch(fp_t* dA, fp_t* dB, fp_t* dC, int m, int n, int k) override;
};
