#pragma once

#include <kernel.hpp>
#include <kernel_runner.hpp>
#include "kernels/cublas.cuh"
#include "kernels/naive.cuh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>


namespace {
struct BenchmarkResult {
    double flops_mean;
    double flops_deviation;
    ErrorMetrics error_metrics;
};

constexpr size_t kMaxNaiveReferenceSize = 4096;

inline void WriteMetric(std::ostream& output, double value, bool available) {
    if (!available) {
        output << "n/a";
        return;
    }
    output << std::scientific << std::setprecision(3) << value;
}

template <typename I_FP_T, typename O_FP_T>
BenchmarkResult DoPerformBenchmark(
    size_t n,
    std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel,
    std::shared_ptr<Kernel<I_FP_T, O_FP_T>> reference_kernel) {
    KernelRunner<I_FP_T, O_FP_T> runner;
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
    if (reference_kernel && n <= kMaxNaiveReferenceSize) {
        res.error_metrics = runner.PerformErrorMetrics(kernel, reference_kernel);
    }
    return res;
}

template <typename I_FP_T, typename O_FP_T>
bool DoPerformCheck(int m, int n, int k, std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel) {
    KernelRunner<I_FP_T, O_FP_T> runner;
    runner.SetUpDeviceData(m, n, k);
    return runner.PerformCheck(kernel);
}

template <typename I_FP_T, typename O_FP_T>
bool DoPerformCheck(int n, std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel) {
    return DoPerformCheck(n, n, n, kernel);
}
}

template <typename I_FP_T, typename O_FP_T>
class BenchmarkCollector {
public:
    static std::vector<int> DefaultSizes() {
        return {512, 1024, 2048, 4096, 8192, 16384};
    }

    static void PerformAndFormat(
        std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel,
        std::ostream& output,
        const std::vector<int>& sizes) {
        std::shared_ptr<Kernel<I_FP_T, O_FP_T>> reference_kernel;
        if constexpr (std::is_same_v<I_FP_T, float> && std::is_same_v<O_FP_T, float>) {
            reference_kernel = std::make_shared<NaiveKernel>();
        }

        const int test_size = 1 << 8;
        if(!DoPerformCheck(test_size, kernel)) {
            output << "Kernel result is differ from the expected one\n\n";
            return;
        // } else if (!DoPerformCheck(test_size / 2, test_size, test_size * 2, kernel)) {
        //     output << "Kernel failed for non-squared matrix\n\n";
        //     return;
        } else {
            output << "Kernel has been tested, results are within a precision\n\n";
        }

        output << "| Matrix size | GFLOPS | Max abs err | Rel L-inf err | Rel L2 err |\n";
        output << "|-------------|--------|-------------|---------------|------------|\n";
        for (int size : sizes) {
            auto res = DoPerformBenchmark(size, kernel, reference_kernel);

            output << "| "
                   << std::setw(11) << std::left << size
                   << " | "
                   << std::fixed << std::setprecision(1)
                   << res.flops_mean << " ± " << res.flops_deviation
                   << " | ";
            WriteMetric(output, res.error_metrics.max_abs_error, res.error_metrics.available);
            output << " | ";
            WriteMetric(output, res.error_metrics.relative_linf_error, res.error_metrics.available);
            output << " | ";
            WriteMetric(output, res.error_metrics.relative_l2_error, res.error_metrics.available);
            output << " |\n";
        }
    }

    static void PerformAndFormat(std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel, std::ostream& output) {
        PerformAndFormat(kernel, output, DefaultSizes());
    }

    static void PerformAndFormat(
        std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel,
        const std::string& filename,
        const std::vector<int>& sizes) {
        std::ofstream output(filename);
        if (!output.good()) {
            std::cerr << filename << " failed to open\n";
            return;
        }
        PerformAndFormat(kernel, output, sizes);

        if(output.good()) {
            std::cout << filename << " has benchmarked successfully\n";
        } else {
            std::cerr << filename << " has failed\n";
        }

        output.close();
    }

    static void PerformAndFormat(std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel, const std::string& filename) {
        PerformAndFormat(kernel, filename, DefaultSizes());
    }
};
