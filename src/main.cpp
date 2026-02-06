#include "benchmark_collector.hpp"
#include "kernel_runner.hpp"
#include "kernels/cublas.cuh"
#include "kernels/naive.cuh"
#include "kernels/memory_coalescing.cuh"
#include "kernels/smem_tiling.cuh"


int main(int argc, char** argv) {
    BenchmarckCollector::PerformAndFormat(std::make_shared<CublasKernel>(), "./results/cublass.txt");
    BenchmarckCollector::PerformAndFormat(std::make_shared<NaiveKernel>(), "./results/naive.txt");
    BenchmarckCollector::PerformAndFormat(std::make_shared<MemoryCoalesingKernel>(),  "./results/memory_coalescing.txt");
    BenchmarckCollector::PerformAndFormat(std::make_shared<SharedMemoryTilingKernel>(),  "./results/smem_tiling.txt");

    return 0;
}
