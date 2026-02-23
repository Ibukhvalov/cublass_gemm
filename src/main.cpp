#include "benchmark_collector.hpp"
#include "kernel_runner.hpp"
#include "kernels/1d_blocktiling.cuh"
#include "kernels/cublas.cuh"
#include "kernels/naive.cuh"
#include "kernels/memory_coalescing.cuh"
#include "kernels/smem_tiling.cuh"
#include "kernels/1d_blocktiling.cuh"
#include "kernels/2d_blocktiling.cuh"
#include <memory>


int main(int argc, char** argv) {
    BenchmarckCollector::PerformAndFormat(std::make_shared<CublasKernel>(), "./results/cublass.md");
    BenchmarckCollector::PerformAndFormat(std::make_shared<NaiveKernel>(), "./results/naive.md");
    BenchmarckCollector::PerformAndFormat(std::make_shared<MemoryCoalesingKernel>(),  "./results/memory_coalescing.md");
    BenchmarckCollector::PerformAndFormat(std::make_shared<SharedMemoryTilingKernel>(),  "./results/smem_tiling.md");
    BenchmarckCollector::PerformAndFormat(std::make_shared<BlockTiling1DKernel>(), "./results/1d_blocktiling.md");
    BenchmarckCollector::PerformAndFormat(std::make_shared<BlockTiling2DKernel>(), "./results/2d_blocktiling.md");

    return 0;
}
