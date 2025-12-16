#include <alpaka/alpaka.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../kernels/transpose.hpp"

// Test domain parameters
constexpr std::size_t NumDims = 2;
using Dim = alpaka::DimInt<NumDims>;
using Idx = std::size_t;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using DevAcc = alpaka::DevCudaRt;
using QueueAcc = alpaka::Queue<DevAcc, alpaka::NonBlocking>;
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;

#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
using DevAcc = alpaka::DevCpu;
using QueueAcc = alpaka::Queue<DevAcc, alpaka::Blocking>;
using Acc = alpaka::AccCpuThreads<Dim, Idx>;

#else
#error Please define a single one of ALPAKA_ACC_GPU_CUDA_ENABLED, ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#endif

using DevHost = alpaka::DevCpu;
using PlatAcc = alpaka::Platform<DevAcc>;
using PlatHost = alpaka::PlatformCpu;

auto now() { return std::chrono::high_resolution_clock::now(); }

int main(int argc, char* argv[]) {
    using namespace alpaka_kernels;
    using T = float;

    // Random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib_int(50, 500);
    std::uniform_real_distribution<float> distrib_real(-1.0f, 1.0f);

    // Input matrix dimensions
    std::size_t rows = distrib_int(gen);
    std::size_t cols = distrib_int(gen);

    if (argc >= 2) {
        rows = std::atoi(argv[1]);
        cols = rows;
        std::cout << "Using input dimensions " << rows << "x" << cols << "\n";
    }
    else {
        std::cout << "Using random dimensions " << rows << "x" << cols << "\n";
    }

    const std::size_t numElems = rows * cols;

    std::vector<T> INPUT(numElems);
    for (auto& val : INPUT) val = distrib_real(gen);

    // Setup the accelerator, host and queue
    auto devAcc = alpaka::getDevByIdx(PlatAcc{}, 0u);
    auto devHost = alpaka::getDevByIdx(PlatHost{}, 0u);
    alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

    // Allocate buffers
    auto extentIn = alpaka::Vec<Dim, Idx>(rows, cols);
    auto extentOut = alpaka::Vec<Dim, Idx>(cols, rows);

    // 1) Accelerator buffers
    auto aIn = alpaka::allocBuf<T, Idx>(devAcc, extentIn);
    auto aOut = alpaka::allocBuf<T, Idx>(devAcc, extentOut);

    // 2) Host buffers
    auto hIn = alpaka::allocBuf<T, Idx>(devHost, extentIn);
    auto hOut = alpaka::allocBuf<T, Idx>(devHost, extentOut);

    // Prepare kernel arguments
    auto input_strides = alpaka::Vec<Dim, Idx>(cols, 1);
    auto output_strides = alpaka::Vec<Dim, Idx>(rows, 1);

    // output axis i corresponds to input axis perm[i]
    // For transpose out[j,i] = in[i,j], so perm = {1,0}
    auto perm = alpaka::Vec<Dim, Idx>(1, 0);

    // Work division: 2D mapping of threads to elements
    const std::size_t threadsX = 16, threadsY = 16;
    const std::size_t blocksX = (cols + threadsX - 1) / threadsX;
    const std::size_t blocksY = (rows + threadsY - 1) / threadsY;

    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{
        alpaka::Vec<Dim, Idx>(blocksX, blocksY),
        alpaka::Vec<Dim, Idx>(threadsX, threadsY), extentOut};

    // Initial data transfer
    // 1) INPUT -> host buffer (safe via raw pointer)
    {
        T* pHost = alpaka::getPtrNative(hIn);
        for (Idx i = 0; i < numElems; ++i) pHost[i] = INPUT[i];
    }

    // 2) host -> accelerator
    auto start_total = now();
    alpaka::memcpy(queue, aIn, hIn);
    alpaka::wait(queue);

    // Launch kernel
    TransposeKernel kernel;

    auto start_kernel = now();

    alpaka::exec<Acc>(queue, workDiv, kernel, alpaka::getPtrNative(aIn),
                      alpaka::getPtrNative(aOut), input_strides, output_strides,
                      extentOut, perm);

    alpaka::wait(queue);
    auto end_kernel = now();

    // Final data transfer: accelerator -> host
    alpaka::memcpy(queue, hOut, aOut);
    alpaka::wait(queue);
    auto end_total = now();

    // Print result
    std::cout << "Output is of shape " << cols << "x" << rows << "\n";

    {
        T* pHost = alpaka::getPtrNative(hOut);
        for (std::size_t i = 0; i < cols; ++i) {
            for (std::size_t j = 0; j < rows; ++j) {
                T valOut = pHost[i * rows + j];
                T valIn = INPUT[j * cols + i];

                if (valIn != valOut) {
                    std::cerr << "Failed!\n";
                    return 1;
                }
            }
        }
    }

    std::cout << "Correct!\n";

    std::chrono::duration<double, std::milli> kernel_ms =
        end_kernel - start_kernel;
    std::chrono::duration<double, std::milli> total_ms =
        end_total - start_total;

    std::cout << "TIME_KERNEL_MS: " << kernel_ms.count() << std::endl;
    std::cout << "TIME_TOTAL_MS: " << total_ms.count() << std::endl;
    return 0;
}
