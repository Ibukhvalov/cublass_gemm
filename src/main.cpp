#include "benchmark_collector.hpp"
#include "kernels/cublas.cuh"
#include "kernels/naive.cuh"
#include "kernels/memory_coalescing.cuh"
#include "kernels/1d_blocktiling.cuh"
#include "kernels/2d_blocktiling.cuh"
#include "kernels/tf32_tensor_core_block128x128.cuh"
#include "kernels/tf32_tensor_core_ptx_async.cuh"
#include "kernels/tf32_tensor_core_ptx_async_warp32x64.cuh"
#include "kernels/tf32_tensor_core_ptx_async_block128x256_warp32x64.cuh"
#include "kernels/tf32_tensor_core_ptx_async_block128x256_warp32x64_aligned.cuh"
#include "kernels/tf32_tensor_core_ptx_async_block128x256_warp32x64_bskew_aligned.cuh"
#include <memory>
#include <vector>

#include "kernels/smem_tiling.cuh"


using FloatBenchmarkCollector = BenchmarkCollector<float, float>;


int main() {
    const std::vector<int> tiny_sizes = {512, 1024};
    const std::vector<int> cuda_core_sizes = {512, 1024, 2048, 4096};
    const std::vector<int> tensor_core_sizes = {512, 1024, 2048, 4096, 8192, 16384};

    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<NaiveKernel>(), "./results/naive.md", tiny_sizes);
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<MemoryCoalesingKernel>(), "./results/memory_coalescing.md", cuda_core_sizes);
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<BlockTiling1DKernel>(), "./results/1d_blocktiling.md", cuda_core_sizes);
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<BlockTiling2DKernel>(), "./results/2d_blocktiling.md", cuda_core_sizes);
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<Tf32TensorCoreBlock128x128Kernel>(), "./results/tf32_tensor_block128x128.md", tensor_core_sizes);
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<Tf32TensorCorePtxAsyncKernel>(), "./results/tf32_tensor_ptx_async.md", tensor_core_sizes);
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<Tf32TensorCorePtxAsyncWarp32x64Kernel>(), "./results/tf32_tensor_ptx_async_warp32x64.md", tensor_core_sizes);
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<Tf32TensorCorePtxAsyncBlock128x256Warp32x64Kernel>(), "./results/tf32_tensor_ptx_async_block128x256_warp32x64.md", tensor_core_sizes);
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<Tf32TensorCorePtxAsyncBlock128x256Warp32x64AlignedKernel>(), "./results/tf32_tensor_ptx_async_block128x256_warp32x64_aligned.md", tensor_core_sizes);

    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<SharedMemoryTilingKernel>(), "./results/smem_tiling.md", tiny_sizes);
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<Tf32TensorCorePtxAsyncBlock128x256Warp32x64BskewAlignedKernel>(), "./results/tf32_tensore_async_bskew_aligned.md", tensor_core_sizes);

    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<CublasKernel>(), "./results/cublas.md", tensor_core_sizes);
    return 0;
}
