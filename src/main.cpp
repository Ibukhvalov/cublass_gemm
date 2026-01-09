#include "benchmark_collector.hpp"
#include "kernels/cublass.cuh"
#include "kernels/naive.cuh"
#include "kernels/memory_coalescing.cuh"
#include <iostream>


int main(int argc, char** argv) {
    BenchmarckCollector::PerformAndFormat(std::make_shared<CublassKernel>(), "./results/cublass.txt");
    BenchmarckCollector::PerformAndFormat(std::make_shared<NaiveKernel>(), "./results/naive.txt");
    BenchmarckCollector::PerformAndFormat(std::make_shared<MemoryCoalesingKernel>(),  "./results/memory_coalescing.txt");
}
