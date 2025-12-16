#include <alpaka/alpaka.hpp>
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

    // Random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib_int(50, 500);
    std::uniform_real_distribution<float> distrib_real(-1.0f, 1.0f);

    // Input matrix dimensions
    const std::size_t rows = distrib_int(gen);
    const std::size_t cols = distrib_int(gen);
    const std::size_t numElems = rows * cols;

    std::cout << "Input is of shape " << rows << "x" << cols << "\n";

    std::vector<T> INPUT(numElems);
    for (auto& val : INPUT) val = distrib_real(gen);

    // Setup devices and queue
    auto devAcc = alpaka::getDevByIdx(PlatAcc{}, 0);
    auto devHost = alpaka::getDevByIdx(PlatHost{}, 0);
    QueueAcc queue{devAcc};

    // Create extents
    auto extentIn = alpaka::Vec<Dim, Idx>(rows, cols);
    auto extentOut = alpaka::Vec<Dim, Idx>(cols, rows);

    // 1) Accelerator buffers
    auto aIn = alpaka::allocBuf<T, Idx>(devAcc, extentIn);
    auto aOut = alpaka::allocBuf<T, Idx>(devAcc, extentOut);

    // 2) Host buffers
    // Note that host and accelerator may coincide when using CPU backend,
    // still it's better to allocate buffers separately for portability and
    // because this ensures memory is pinned and not paged
    auto hIn = alpaka::allocBuf<T, Idx>(devHost, extentIn);
    auto hOut = alpaka::allocBuf<T, Idx>(devHost, extentOut);

    // Initial data transfer
    // 1) INPUT -> host buffer (safe via raw pointer)
    {
        T* pHost = alpaka::getPtrNative(hIn);
        for (Idx i = 0; i < numElems; ++i) pHost[i] = INPUT[i];
    }

    // 2) host -> accelerator
    {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        T* pAIn = alpaka::getPtrNative(aIn);
        T* pHIn = alpaka::getPtrNative(hIn);
        // For GPU, use cudaMemcpy directly
        cudaMemcpy(pAIn, pHIn, numElems * sizeof(T), cudaMemcpyHostToDevice);
#else
        // For CPU, use memcpy
        alpaka::memcpy(queue, aIn, hIn);
#endif
    }

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

    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{alpaka::Vec<Dim, Idx>(blocksY, blocksX),
                                                          alpaka::Vec<Dim, Idx>(threadsY, threadsX), extentOut};

    // Launch kernel
    TransposeKernel kernel;

    alpaka::exec<Acc>(queue, workDiv, kernel, alpaka::getPtrNative(aIn), alpaka::getPtrNative(aOut), input_strides,
                      output_strides, extentOut, perm);

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

    // Print results
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
    return 0;
}
