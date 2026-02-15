#include "kernel_runner.hpp"
#include "cuda_runtime_api.h"
#include "kernel.hpp"
#include "device/matrix.hpp"
#include "host/matrix.hpp"
#include "macro.hpp"
#include <chrono>
#include <vector>


void KernelRunner::SetUpDeviceData(int n) {
    SetUpDeviceData(n, n, n);
};

void KernelRunner::SetUpDeviceData(int m, int n, int k) {
    auto hA = host::Matrix::CreateRandom({m, k});
    auto hB = host::Matrix::CreateRandom({k, n});

    dA = device::Matrix::CopyFromHost(hA);
    dB = device::Matrix::CopyFromHost(hB);
};

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

std::vector<float> KernelRunner::PerformBenchmark(std::shared_ptr<Kernel> kernel) {
    device::Matrix dC({ dA.shape.rows, dB.shape.cols });
    int warmupNb = 20;
    int iterationsNb = 100;
    return BenchmarkCudaKernel(warmupNb, iterationsNb,
        [&] {
            kernel->launch(dA, dB, dC);
        });
};

bool KernelRunner::PerformCheck(std::shared_ptr<Kernel> kernel) {
    device::Matrix dC({ dA.shape.rows, dB.shape.cols });
    int warmupNb = 0;
    int iterationsNb = 1;
    BenchmarkCudaKernel(warmupNb, iterationsNb,
        [&] {
            kernel->launch(dA, dB, dC);
        });


    auto expected_C = host::Matrix::CopyFromDevice(dA) * host::Matrix::CopyFromDevice(dB);
    auto performed_C = host::Matrix::CopyFromDevice(dC);

    return expected_C == performed_C;
};

void KernelRunner::PerformAndPrint(std::shared_ptr<Kernel> kernel) {
    device::Matrix dC({ dA.shape.rows, dB.shape.cols });
    int warmupNb = 0;
    int iterationsNb = 1;
    BenchmarkCudaKernel(warmupNb, iterationsNb,
        [&] {
            kernel->launch(dA, dB, dC);
        });

    auto expected_C = host::Matrix::CopyFromDevice(dA) * host::Matrix::CopyFromDevice(dB);
    auto performed_C = host::Matrix::CopyFromDevice(dC);
    std::cout << "Expected:\n" << expected_C << std::endl;
    std::cout << "Performed:\n" << performed_C << std::endl;
};
