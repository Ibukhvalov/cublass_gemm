#pragma once

#include "kernel.hpp"

using tf32_ptx_async_fp_t = float;

class Tf32TensorCorePtxAsyncKernel : public Kernel<tf32_ptx_async_fp_t> {
private:
    void launch(tf32_ptx_async_fp_t* dA, tf32_ptx_async_fp_t* dB, tf32_ptx_async_fp_t* dC, int m, int n, int k) override;
};
