#include <alpaka/alpaka.hpp>
#include <array>
#include <iostream>
#include <random>
#include <vector>

#include "../kernels/topk.hpp"

// Test domain parameters
constexpr std::size_t NumDims = 2;
constexpr std::size_t TopkAxis = 1;
constexpr std::size_t K = 4;
constexpr std::size_t MaxRegisters = 64;
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

    // Setup the accelerator, host and queue
    auto devAcc = alpaka::getDevByIdx(PlatAcc{}, 0u);
    auto devHost = alpaka::getDevByIdx(PlatHost{}, 0u);
    QueueAcc queue{devAcc};

    // Allocate buffers
    auto extentIn = alpaka::Vec<Dim, Idx>(rows, cols);
    auto extentOut = alpaka::Vec<Dim, Idx>(rows, K);

    // 1) Accelerator buffers
    auto aIn = alpaka::allocBuf<T, Idx>(devAcc, extentIn);
    auto aOut = alpaka::allocBuf<T, Idx>(devAcc, extentOut);

    // 2) Host buffers
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
        T* pAIn = alpaka::getPtrNative(aIn);
        T* pHIn = alpaka::getPtrNative(hIn);

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        // For GPU, use cudaMemcpy directly
        cudaMemcpy(pAIn, pHIn, numElems * sizeof(T), cudaMemcpyHostToDevice);
#else
        // For CPU, use memcpy
        std::memcpy(pAIn, pHIn, numElems * sizeof(T));
#endif
    }

    // Prepare kernel arguments
    T const padding_value = -1.0;
    auto input_strides = alpaka::Vec<Dim, Idx>(cols, 1);
    auto output_strides = alpaka::Vec<Dim, Idx>(K, 1);

    // Work division: 2D mapping of threads to elements
    auto grid_elements = extentOut;
    grid_elements[TopkAxis] = 1;

    alpaka::Vec<Dim, Idx> threadsPerBlock;
    alpaka::Vec<Dim, Idx> blocksPerGrid;
    Idx const TARGET_BLOCK_SIZE = 16;

    for (std::size_t d = 0; d < Dim::value; ++d) {
        if (d == TopkAxis) {
            threadsPerBlock[d] = 1;
            blocksPerGrid[d] = 1;
        } else {
            threadsPerBlock[d] = TARGET_BLOCK_SIZE;
            blocksPerGrid[d] = (grid_elements[d] + threadsPerBlock[d] - 1) / threadsPerBlock[d];
        }
    }

    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{blocksPerGrid, threadsPerBlock, grid_elements};

    // Launch kernel
    TopKKernel<K, MaxRegisters> kernel;

    alpaka::exec<Acc>(queue, workDiv, kernel, alpaka::getPtrNative(aIn), alpaka::getPtrNative(aOut), input_strides,
                      output_strides, grid_elements, TopkAxis, extentIn[TopkAxis], padding_value);

    alpaka::wait(queue);

    // Final data transfer: accelerator -> host
    {
        T* pAOut = alpaka::getPtrNative(aOut);
        T* pHOut = alpaka::getPtrNative(hOut);

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        cudaMemcpy(pHOut, pAOut, numElems * sizeof(T), cudaMemcpyDeviceToHost);
#else
        std::memcpy(pHOut, pAOut, numElems * sizeof(T));
#endif
    }

    // Print result
    std::cout << "Output is of shape " << rows << "x" << K << "\n";

    {
        T* pHost = alpaka::getPtrNative(hOut);
        for (std::size_t i = 0; i < rows; ++i) {
            std::array<T, K> top_vals = {padding_value};
            std::size_t count = 0;

            for (std::size_t j = 0; j < extentIn[TopkAxis]; ++j) {
                T const val = INPUT[i * cols + j];
                if (count == K && val <= top_vals[K - 1]) continue;

                Idx insert_pos = 0;
                while (insert_pos < count) {
                    if (val > top_vals[insert_pos]) break;
                    insert_pos++;
                }

                if (insert_pos < K) {
                    std::size_t const last = std::min(count, K - 1);
                    for (std::size_t s = last; s > insert_pos; --s) {
                        top_vals[s] = top_vals[s - 1];
                    }

                    top_vals[insert_pos] = val;
                    if (count < K) count++;
                }
            }

            for (std::size_t j = 0; j < K; ++j) {
                T const valOut = pHost[i * K + j];
                T const valIn = top_vals[j];

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
