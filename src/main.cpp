#include "benchmark_collector.hpp"
#include "kernels/1d_blocktiling.cuh"
#include "kernels/cublas.cuh"
#include "kernels/naive.cuh"
#include "kernels/memory_coalescing.cuh"
#include "kernels/smem_tiling.cuh"
#include "kernels/1d_blocktiling.cuh"
#include "kernels/2d_blocktiling.cuh"
#include "kernels/tiling_tensor_core.cuh"
#include <memory>


using FloatBenchmarkCollector = BenchmarkCollector<float, float>;
using HalfBenchmarkCollector = BenchmarkCollector<half, float>;


int main(int argc, char** argv) {
    // FloatBenchmarkCollector::PerformAndFormat(std::make_shared<NaiveKernel>(), "./results/naive.md");
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<CublasKernel>(), "./results/cublas.md");
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<MemoryCoalesingKernel>(),  "./results/memory_coalescing.md");
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<SharedMemoryTilingKernel>(),  "./results/smem_tiling.md");
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<BlockTiling1DKernel>(), "./results/1d_blocktiling.md");
    FloatBenchmarkCollector::PerformAndFormat(std::make_shared<BlockTiling2DKernel>(), "./results/2d_blocktiling.md");
    HalfBenchmarkCollector::PerformAndFormat(std::make_shared<TilingTensorCoreKernel>(),  "./results/tiling_tensor_core.md");
    return 0;
}
