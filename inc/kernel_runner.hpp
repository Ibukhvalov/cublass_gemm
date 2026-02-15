#pragma once

#include "device/matrix.hpp"
#include "host/matrix.hpp"
#include "kernel.hpp"

#include <optional>
#include <vector>

class KernelRunner {
public:
    void SetUpDeviceData(int m, int n, int k);
    void SetUpDeviceData(int n);

    std::vector<float> PerformBenchmark(std::shared_ptr<Kernel> kernel);
    void PerformAndPrint(std::shared_ptr<Kernel> kernel);
    bool PerformCheck(std::shared_ptr<Kernel> kernel);

private:
    device::Matrix dA, dB;
};
