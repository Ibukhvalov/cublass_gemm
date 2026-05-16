#pragma once

#include "kernel.hpp"

using tf32_ptx_async_block128x256_warp32x64_aligned_fp_t = float;

class Tf32TensorCorePtxAsyncBlock128x256Warp32x64AlignedKernel : public Kernel<tf32_ptx_async_block128x256_warp32x64_aligned_fp_t> {
private:
    void launch(tf32_ptx_async_block128x256_warp32x64_aligned_fp_t* dA, tf32_ptx_async_block128x256_warp32x64_aligned_fp_t* dB, tf32_ptx_async_block128x256_warp32x64_aligned_fp_t* dC, int m, int n, int k) override;
};
