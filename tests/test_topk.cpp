#include <alpaka/alpaka.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "../kernels/topk.hpp"

// Test domain parameters
constexpr std::size_t NumDims = 2;
using Dim = alpaka::DimInt<NumDims>;
using Idx = std::size_t;

// Define the accelerator
using Acc = alpaka::AccCpuThreads<Dim, Idx>;

// Define the platform types
using PlatAcc = alpaka::Platform<Acc>;
using PlatHost = alpaka::PlatformCpu;

int main() { return 0; }
