#pragma once

#include "device/matrix.hpp"
#include "utils.hpp"
#include "host/matrix.hpp"
#include "kernel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>


namespace {
using MillisecondsVec = std::vector<float>;
struct ErrorMetrics {
    bool available = false;
    double max_abs_error = 0.0;
    double relative_linf_error = 0.0;
    double relative_l2_error = 0.0;
};

template <typename fp_t>
ErrorMetrics ComputeErrorMetrics(const host::Matrix<fp_t>& expected, const host::Matrix<fp_t>& actual) {
    ErrorMetrics metrics;
    metrics.available = true;
    double max_abs_expected = 0.0;
    double squared_error_sum = 0.0;
    double squared_expected_sum = 0.0;

    const int elements_nb = expected.shape.elements_nb();
    for (int idx = 0; idx < elements_nb; ++idx) {
        const double expected_value = static_cast<double>(expected.data[idx]);
        const double actual_value = static_cast<double>(actual.data[idx]);
        const double diff = std::abs(actual_value - expected_value);

        metrics.max_abs_error = std::max(metrics.max_abs_error, diff);
        max_abs_expected = std::max(max_abs_expected, std::abs(expected_value));
        squared_error_sum += diff * diff;
        squared_expected_sum += expected_value * expected_value;
    }

    const double eps = std::numeric_limits<double>::epsilon();
    metrics.relative_linf_error = metrics.max_abs_error / std::max(max_abs_expected, eps);
    metrics.relative_l2_error = std::sqrt(squared_error_sum) / std::sqrt(std::max(squared_expected_sum, eps));
    return metrics;
}

template <typename F>
MillisecondsVec BenchmarkCudaKernel (
    int warmupIterationsNb,
    int iterationsNb,
    F&& kernelCall) {

        for (int i=0; i < warmupIterationsNb; ++i)
            kernelCall();

        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, end;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&end));
        MillisecondsVec aRes(iterationsNb);
        for(int i=0; i < iterationsNb; ++i) {
            CHECK_CUDA(cudaEventRecord(start));
            kernelCall();
            CHECK_CUDA(cudaEventRecord(end));
            CHECK_CUDA(cudaEventSynchronize(end));
            CHECK_CUDA(cudaEventElapsedTime(aRes.data() + i, start, end));
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(end));
        return aRes;
    }
}

template <typename I_FP_T, typename O_FP_T>
class KernelRunner {
public:

    void SetUpDeviceData(int m, int n, int k) {
        auto hA = host::Matrix<I_FP_T>::CreateRandom({m, k});
        auto hB = host::Matrix<I_FP_T>::CreateRandom({k, n});

        dA = CopyFromHostToDevice(hA);
        dB = CopyFromHostToDevice(hB);
    }

    void SetUpDeviceData(int n) {
        SetUpDeviceData(n, n, n);
    }

    std::vector<float> PerformBenchmark(std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel) {
        device::Matrix<O_FP_T> dC({ dA.shape.rows, dB.shape.cols });
        int warmupNb = 20;
        int iterationsNb = 100;
        return BenchmarkCudaKernel(warmupNb, iterationsNb,
            [&] {
                kernel->launch(dA, dB, dC);
            });
    }

    void PerformAndPrint(std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel) {
        device::Matrix<O_FP_T> dC({ dA.shape.rows, dB.shape.cols });
        int warmupNb = 0;
        int iterationsNb = 1;
        BenchmarkCudaKernel(warmupNb, iterationsNb,
            [&] {
                kernel->launch(dA, dB, dC);
            });

        host::Matrix<O_FP_T> expected_C = CopyFromDeviceToHost(dA) * CopyFromDeviceToHost(dB);
        host::Matrix<O_FP_T> performed_C = CopyFromDeviceToHost(dC);

        std::cout << "Expected:\n" << expected_C << std::endl;
        std::cout << "Performed:\n" << performed_C << std::endl;
    }

    bool PerformCheck(std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel) {
        device::Matrix<O_FP_T> dC({ dA.shape.rows, dB.shape.cols });
        int warmupNb = 0;
        int iterationsNb = 1;
        BenchmarkCudaKernel(warmupNb, iterationsNb,
            [&] {
                kernel->launch(dA, dB, dC);
            });

        host::Matrix<O_FP_T> expected_C = CopyFromDeviceToHost(dA) * CopyFromDeviceToHost(dB);
        host::Matrix<O_FP_T> performed_C = CopyFromDeviceToHost(dC);

        return expected_C == performed_C;
    }

    ErrorMetrics PerformErrorMetrics(
        std::shared_ptr<Kernel<I_FP_T, O_FP_T>> kernel,
        std::shared_ptr<Kernel<I_FP_T, O_FP_T>> reference_kernel) {
        device::Matrix<O_FP_T> dC({ dA.shape.rows, dB.shape.cols });
        device::Matrix<O_FP_T> dReferenceC({ dA.shape.rows, dB.shape.cols });

        kernel->launch(dA, dB, dC);
        reference_kernel->launch(dA, dB, dReferenceC);
        CHECK_CUDA(cudaDeviceSynchronize());

        host::Matrix<O_FP_T> expected_C = CopyFromDeviceToHost(dReferenceC);
        host::Matrix<O_FP_T> performed_C = CopyFromDeviceToHost(dC);
        return ComputeErrorMetrics(expected_C, performed_C);
    }

private:
    device::Matrix<I_FP_T> dA, dB;
};
