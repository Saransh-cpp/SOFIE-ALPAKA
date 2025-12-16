#include <alpaka/alpaka.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "../kernels/where.hpp"

// Test domain parameters
constexpr std::size_t NumDims = 2;
using Dim = alpaka::DimInt<NumDims>;
using Idx = std::size_t;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using DevAcc = alpaka::DevCudaRt;
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
using QueueAcc = alpaka::Queue<Acc, alpaka::NonBlocking>;

#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
using DevAcc = alpaka::DevCpu;
using Acc = alpaka::AccCpuThreads<Dim, Idx>;
using QueueAcc = alpaka::Queue<Acc, alpaka::Blocking>;

#else
#error Please define a single one of ALPAKA_ACC_GPU_CUDA_ENABLED, ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#endif

using DevHost = alpaka::DevCpu;
using PlatAcc = alpaka::Platform<DevAcc>;
using PlatHost = alpaka::PlatformCpu;

int main() {
    using namespace alpaka_kernels;
    using T = float;
    using TCond = bool;

    // Random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib_int(50, 500);
    std::uniform_real_distribution<float> distrib_real(-1.0f, 1.0f);
    std::bernoulli_distribution distrib_bool(0.5);

    // Input matrix dimensions
    const std::size_t rows = distrib_int(gen);
    const std::size_t cols = distrib_int(gen);
    const std::size_t numElems = rows * cols;

    std::cout << "Inputs are of shape " << rows << "x" << cols << "\n";

    std::vector<T> INPUT_X(numElems), INPUT_Y(numElems);
    std::vector<TCond> INPUT_COND(numElems);

    for (auto& val : INPUT_X) val = distrib_real(gen) * 100.0;
    for (auto& val : INPUT_Y) val = distrib_real(gen);
    for (std::size_t i = 0; i < numElems; ++i) INPUT_COND[i] = distrib_bool(gen);

    // Setup the accelerator, host and queue
    auto devAcc = alpaka::getDevByIdx(PlatAcc{}, 0u);
    auto devHost = alpaka::getDevByIdx(PlatHost{}, 0u);
    QueueAcc queue{devAcc};

    // Allocate buffers
    auto extent = alpaka::Vec<Dim, Idx>(rows, cols);

    // 1) Accelerator buffers
    auto aIn_X = alpaka::allocBuf<T, Idx>(devAcc, extent);
    auto aIn_Y = alpaka::allocBuf<T, Idx>(devAcc, extent);
    auto aIn_Cond = alpaka::allocBuf<T, Idx>(devAcc, extent);
    auto aOut = alpaka::allocBuf<T, Idx>(devAcc, extent);

    // 2) Host buffers
    auto hIn_X = alpaka::allocBuf<T, Idx>(devHost, extent);
    auto hIn_Y = alpaka::allocBuf<T, Idx>(devHost, extent);
    auto hIn_Cond = alpaka::allocBuf<T, Idx>(devHost, extent);
    auto hOut = alpaka::allocBuf<T, Idx>(devHost, extent);

    // Initial data transfer
    // 1) INPUT -> host buffer (safe via raw pointer)
    {
        T* pHost_X = alpaka::getPtrNative(hIn_X);
        for (Idx i = 0; i < numElems; ++i) pHost_X[i] = INPUT_X[i];

        T* pHost_Y = alpaka::getPtrNative(hIn_Y);
        for (Idx i = 0; i < numElems; ++i) pHost_Y[i] = INPUT_Y[i];

        T* pHost_Cond = alpaka::getPtrNative(hIn_Cond);
        for (Idx i = 0; i < numElems; ++i) pHost_Cond[i] = INPUT_COND[i];
    }

    // 2) host -> accelerator
    {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // For GPU, use cudaMemcpy directly
        T* pAIn_X = alpaka::getPtrNative(aIn_X);
        T* pAIn_Y = alpaka::getPtrNative(aIn_Y);
        T* pAIn_Cond = alpaka::getPtrNative(aIn_Cond);
        T* pHIn_X = alpaka::getPtrNative(hIn_X);
        T* pHIn_Y = alpaka::getPtrNative(hIn_Y);
        T* pHIn_Cond = alpaka::getPtrNative(hIn_Cond);
        cudaMemcpy(pAIn_X, pHIn_X, numElems * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(pAIn_Y, pHIn_Y, numElems * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(pAIn_Cond, pHIn_Cond, numElems * sizeof(T), cudaMemcpyHostToDevice);
#else
        // For CPU, use memcpy
        alpaka::memcpy(queue, aIn_X, hIn_X);
        alpaka::memcpy(queue, aIn_Y, hIn_Y);
        alpaka::memcpy(queue, aIn_Cond, hIn_Cond);
#endif
    }

    // Prepare kernel arguments
    auto strides = alpaka::Vec<Dim, Idx>(cols, 1);

    // Work division: 2D mapping of threads to elements
    const std::size_t threadsX = 16, threadsY = 16;
    const std::size_t blocksX = (cols + threadsX - 1) / threadsX;
    const std::size_t blocksY = (rows + threadsY - 1) / threadsY;

    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{alpaka::Vec<Dim, Idx>(blocksX, blocksY),
                                                          alpaka::Vec<Dim, Idx>(threadsX, threadsY), extent};

    // Launch kernel
    WhereKernel kernel;

    alpaka::exec<Acc>(queue, workDiv, kernel, alpaka::getPtrNative(aIn_Cond), alpaka::getPtrNative(aIn_X),
                      alpaka::getPtrNative(aIn_Y), alpaka::getPtrNative(aOut), strides, strides, strides, strides,
                      extent);

    alpaka::wait(queue);

    // Final data transfer: accelerator -> host
    {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        T* pAOut = alpaka::getPtrNative(aOut);
        T* pHOut = alpaka::getPtrNative(hOut);
        cudaMemcpy(pHOut, pAOut, numElems * sizeof(T), cudaMemcpyDeviceToHost);
#else
        alpaka::memcpy(queue, hOut, aOut);
#endif
    }

    // Print result
    std::cout << "Output is of shape " << rows << "x" << cols << "\n";

    {
        T* pHost = alpaka::getPtrNative(hOut);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                T valOut = pHost[i * cols + j];
                T valIn = INPUT_COND[i * cols + j] ? INPUT_X[i * cols + j] : INPUT_Y[i * cols + j];

                if (valIn != valOut) {
                    std::cerr << "Failed!\n";
                    return 1;
                }
            }
        }
    }

    std::cout << "Correct!\n";
    return 0;
}
