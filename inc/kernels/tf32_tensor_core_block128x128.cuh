#pragma once

#include "kernel.hpp"

using tf32_block128x128_fp_t = float;

class Tf32TensorCoreBlock128x128Kernel : public Kernel<tf32_block128x128_fp_t> {
private:
    void launch(tf32_block128x128_fp_t* dA, tf32_block128x128_fp_t* dB, tf32_block128x128_fp_t* dC, int m, int n, int k) override;
};
