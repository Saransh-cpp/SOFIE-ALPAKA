#include <alpaka/alpaka.hpp>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../kernels/concat.hpp"

// Test domain parameters
constexpr std::size_t NumDims = 2;
constexpr std::size_t NumInputs = 3;
constexpr std::size_t ConcatAxis = 0;
using Dim = alpaka::DimInt<NumDims>;
using Idx = std::size_t;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
using DevAcc = alpaka::DevCudaRt;
using QueueAcc = alpaka::Queue<DevAcc, alpaka::NonBlocking>;
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;

#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
using DevAcc = alpaka::DevCpu;
using QueueAcc = alpaka::Queue<DevAcc, alpaka::Blocking>;
using Acc = alpaka::AccCpuTbbBlocks<Dim, Idx>;

#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
using DevAcc = alpaka::DevCpu;
using QueueAcc = alpaka::Queue<DevAcc, alpaka::Blocking>;
using Acc = alpaka::AccCpuSerial<Dim, Idx>;

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
    std::array<std::size_t, NumInputs> in_rows;
    std::size_t cols = distrib_int(gen);
    std::size_t total_rows = 0;

    std::cout << "Number of inputs: " << NumInputs << "\n";

    if (argc >= 2) {
        std::cout << "Using input dimensions ";
        cols = std::atoi(argv[1]);
        for (auto& val : in_rows) {
            val = cols;
            total_rows += val;
        }
    } else {
        std::cout << "Using random dimensions ";
        for (auto& val : in_rows) {
            val = distrib_int(gen);
            total_rows += val;
        }
    }

    for (std::size_t k = 0; k < NumInputs; ++k)
        std::cout << in_rows[k] << "x" << cols << ((k < NumInputs - 1) ? ", " : "\n");

    std::array<std::vector<T>, NumInputs> INPUT;
    for (std::size_t k = 0; k < NumInputs; ++k) {
        INPUT[k].resize(in_rows[k] * cols);
        for (auto& val : INPUT[k]) val = distrib_real(gen);
    }

    // Setup the accelerator, host and queue
    auto devAcc = alpaka::getDevByIdx(PlatAcc{}, 0u);
    auto devHost = alpaka::getDevByIdx(PlatHost{}, 0u);
    alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

    // Allocate buffers
    using BufAcc = decltype(alpaka::allocBuf<T, Idx>(devAcc, alpaka::Vec<Dim, Idx>{}));
    using BufHost = decltype(alpaka::allocBuf<T, Idx>(devHost, alpaka::Vec<Dim, Idx>{}));

    std::vector<BufAcc> aIn_bufs;
    aIn_bufs.reserve(NumInputs);

    std::vector<BufHost> hIn_bufs;
    hIn_bufs.reserve(NumInputs);

    const std::size_t out_rows = total_rows;
    const std::size_t out_cols = cols;
    auto extentOut = alpaka::Vec<Dim, Idx>(out_rows, out_cols);

    for (std::size_t k = 0; k < NumInputs; ++k) {
        auto extentIn = alpaka::Vec<Dim, Idx>(in_rows[k], cols);

        // Allocate input buffers
        // 1) Accelerator input buffers
        aIn_bufs.push_back(alpaka::allocBuf<T, Idx>(devAcc, extentIn));

        // 2) Host input buffers
        hIn_bufs.push_back(alpaka::allocBuf<T, Idx>(devHost, extentIn));

        // INPUT to host buffer data transfer (safe via raw pointers)
        T* pHost = alpaka::getPtrNative(hIn_bufs.back());
        for (std::size_t i = 0; i < INPUT[k].size(); ++i) pHost[i] = INPUT[k][i];
    }

    // Allocate output buffers
    auto aOut = alpaka::allocBuf<T, Idx>(devAcc, extentOut);
    auto hOut = alpaka::allocBuf<T, Idx>(devHost, extentOut);

    // Prepare kernel arguments
    std::array<T const*, NumInputs> aIn_ptrs;
    for (std::size_t k = 0; k < NumInputs; ++k) {
        aIn_ptrs[k] = alpaka::getPtrNative(aIn_bufs[k]);
    }

    std::array<alpaka::Vec<Dim, Idx>, NumInputs> input_strides_vec;
    for (std::size_t k = 0; k < NumInputs; ++k) {
        input_strides_vec[k] = alpaka::Vec<Dim, Idx>(cols, 1);
    }

    std::array<Idx, NumInputs> axis_sizes;
    for (std::size_t k = 0; k < NumInputs; ++k) {
        axis_sizes[k] = in_rows[k];
    }

    auto output_strides = alpaka::Vec<Dim, Idx>(out_cols, 1);

    // Work division: 2D mapping of threads to elements
    std::size_t threadsX = 16, threadsY = 16;
    std::size_t blocksX = (out_rows + threadsX - 1) / threadsX;
    std::size_t blocksY = (out_cols + threadsY - 1) / threadsY;

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)

    threadsX = 1;
    threadsY = 1;
    blocksX = 64;
    blocksY = 1;
#endif

    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{alpaka::Vec<Dim, Idx>(blocksX, blocksY),
                                                          alpaka::Vec<Dim, Idx>(threadsX, threadsY), extentOut};

    // Warmup run
    ConcatKernel kernel;

    alpaka::exec<Acc>(queue, workDiv, kernel, aIn_ptrs, alpaka::getPtrNative(aOut), input_strides_vec, output_strides,
                      extentOut, axis_sizes, ConcatAxis);

    alpaka::wait(queue);

    // Host to accelerator data transfer
    auto start_total = now();

    for (std::size_t k = 0; k < NumInputs; ++k) {
        alpaka::memcpy(queue, aIn_bufs[k], hIn_bufs[k]);
    }

    alpaka::wait(queue);

    // Launch kernel
    auto start_kernel = now();

    alpaka::exec<Acc>(queue, workDiv, kernel, aIn_ptrs, alpaka::getPtrNative(aOut), input_strides_vec, output_strides,
                      extentOut, axis_sizes, ConcatAxis);

    alpaka::wait(queue);
    auto end_kernel = now();

    // Final data transfer: accelerator -> host
    alpaka::memcpy(queue, hOut, aOut);
    alpaka::wait(queue);
    auto end_total = now();

    // Print result
    std::cout << "Output is of shape " << out_rows << "x" << out_cols << "\n";

    std::vector<T> expected;
    for (const auto& vec : INPUT) expected.insert(expected.end(), vec.begin(), vec.end());

    {
        T* pHost = alpaka::getPtrNative(hOut);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (pHost[i] != expected[i]) {
                std::cerr << "Failed!\n";
                return 1;
            }
        }
    }

    std::cout << "Correct!\n";

    std::chrono::duration<double, std::milli> kernel_ms = end_kernel - start_kernel;
    std::chrono::duration<double, std::milli> total_ms = end_total - start_total;

    std::cout << "TIME_KERNEL_MS: " << kernel_ms.count() << std::endl;
    std::cout << "TIME_TOTAL_MS: " << total_ms.count() << std::endl;
    return 0;
}
