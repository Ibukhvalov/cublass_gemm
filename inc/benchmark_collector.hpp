#pragma once

#include <iostream>
#include <kernel.hpp>

class BenchmarckCollector {
public:
    static void PerformAndFormat(std::shared_ptr<Kernel> kernel, std::ostream& output);
    static void PerformAndFormat(std::shared_ptr<Kernel> kernel, const std::string& filename);
};
