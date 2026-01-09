#include "benchmark_collector.hpp"
#include "kernel.hpp"
#include <algorithm>
#include <ios>
#include <iterator>
#include <kernel_runner.hpp>

#include <fstream>
#include <iomanip>
#include <numeric>
#include <vector>


namespace {
struct BenchmarkResult {
    double flops_mean;
    double flops_deviation;
};

BenchmarkResult DoPerformBenchmark(size_t n, std::shared_ptr<Kernel> kernel) {
    KernelRunner runner;
    runner.SetUpDeviceData(n);
    auto runs_res = runner.PerformBenchmark(kernel);

    std::vector<double> gflops_samples;
    std::transform(runs_res.cbegin(), runs_res.cend(), std::back_inserter(gflops_samples), [&](float delta) {
        const size_t flop_nb = (n*n*n) * size_t(2);
        return static_cast<double>(flop_nb) / delta * 1e3 / 1e9;
    });

    BenchmarkResult res;
    res.flops_mean = std::accumulate(gflops_samples.begin(), gflops_samples.end(), 0.0, [] (double acc, double gflops) {
        return acc + gflops;
    }) / runs_res.size();
    double variance = std::accumulate(gflops_samples.begin(), gflops_samples.end(), 0.0, [&] (double acc, double gflops) {
        double d = gflops - res.flops_mean;
        return acc + d*d;
    }) / (runs_res.size() - 1);
    res.flops_deviation = std::sqrt(variance);
    return res;
}

bool DoPerformCheck(int n, std::shared_ptr<Kernel> kernel) {
    KernelRunner runner;
    runner.SetUpDeviceData(n);
    return runner.PerformCheck(kernel);
}
}
void BenchmarckCollector::PerformAndFormat(std::shared_ptr<Kernel> kernel, std::ostream& output) {
    const int test_size = 1 << 8;
    if(!DoPerformCheck(test_size, kernel)) {
        output << "Kernel result is differ from the expected one\n\n";
    } else {
        output << "Kernel has been testes, results are within a precision\n\n";
    }

    std::vector<int> sizes;
    for (int i = 9; i <= 12; ++i) {
        sizes.push_back(1 << i);
    }

    output << std::setw(13)
    << std::left << "Matrix size" << " | " << "GFLOPS" << std::endl;
    for(int size : sizes) {
        auto res = DoPerformBenchmark(size, kernel);
        output << std::setw(13) << std::fixed << std::setprecision(1)
        << std::left << size << " | " << res.flops_mean;
        output << " ±" << res.flops_deviation << std::endl;
    }
}

void BenchmarckCollector::PerformAndFormat(std::shared_ptr<Kernel> kernel, const std::string& filename) {
    std::ofstream output(filename);
    PerformAndFormat(kernel, output);

    if(output.good()) {
        std::cout << filename << " has benchmarked successfully\n";
    } else {
        std::cerr << filename << " has failed\n";
    }

    output.close();
 }
