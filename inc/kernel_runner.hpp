#pragma once

#include "device/matrix.hpp"
#include "utils.hpp"
#include "host/matrix.hpp"
#include "kernel.hpp"

#include <vector>


namespace {
using MillisecondsVec = std::vector<float>;
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

private:
    device::Matrix<I_FP_T> dA, dB;
};
