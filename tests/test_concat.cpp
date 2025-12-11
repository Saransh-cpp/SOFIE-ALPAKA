#include <alpaka/alpaka.hpp>
#include <array>
#include <iostream>
#include <random>
#include <vector>

#include "../kernels/concat.hpp"

// Architecture configuration
constexpr std::size_t NumDims = 2;
using Dim = alpaka::DimInt<NumDims>;
using Idx = std::size_t;

// Define the accelerator
using Acc = alpaka::AccCpuThreads<Dim, Idx>;

// Define the platform types
using PlatAcc = alpaka::Platform<Acc>;
using PlatHost = alpaka::PlatformCpu;

int main() {
    using namespace alpaka_kernels;
    using T = float;

    // Random engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib_int(100, 1000);
    std::uniform_real_distribution<float> distrib_real(-1.0f, 1.0f);

    // Number of inputs
    constexpr std::size_t num_inputs = 3;

    // Input matrix dimensions
    const std::size_t cols = distrib_int(gen);
    std::size_t total_rows = 0;

    std::vector<std::size_t> in_rows(num_inputs);
    for (auto& val : in_rows) {
        val = distrib_int(gen);
        total_rows += val;
    }

    std::cout << "Number of inputs: " << num_inputs << "\n";
    std::cout << "Inputs are of shape: ";
    for (std::size_t k = 0; k < num_inputs; ++k)
        std::cout << in_rows[k] << "x" << cols
                  << ((k < num_inputs - 1) ? ", " : "\n");

    std::vector<std::vector<T>> INPUT(num_inputs);
    for (std::size_t k = 0; k < num_inputs; ++k) {
        INPUT[k].resize(in_rows[k] * cols);
        for (auto& val : INPUT[k]) val = distrib_real(gen);
    }

    // Setup the accelerator, host and queue
    auto devAcc = alpaka::getDevByIdx(PlatAcc{}, 0u);
    auto devHost = alpaka::getDevByIdx(PlatHost{}, 0u);
    alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

    using BufAcc =
        decltype(alpaka::allocBuf<T, Idx>(devAcc, alpaka::Vec<Dim, Idx>{}));
    using BufHost =
        decltype(alpaka::allocBuf<T, Idx>(devHost, alpaka::Vec<Dim, Idx>{}));

    std::vector<BufAcc> acc_input_bufs;
    acc_input_bufs.reserve(num_inputs);

    std::vector<BufHost> host_input_bufs;
    host_input_bufs.reserve(num_inputs);

    const std::size_t out0 = total_rows;
    const std::size_t out1 = cols;
    auto extent_out = alpaka::Vec<Dim, Idx>(out0, out1);

    for (std::size_t k = 0; k < num_inputs; ++k) {
        auto extent = alpaka::Vec<Dim, Idx>(in_rows[k], cols);

        acc_input_bufs.push_back(alpaka::allocBuf<T, Idx>(devAcc, extent));
        host_input_bufs.push_back(alpaka::allocBuf<T, Idx>(devHost, extent));

        // Fill Host Buffer
        T* p = alpaka::getPtrNative(host_input_bufs.back());
        for (std::size_t i = 0; i < INPUT[k].size(); ++i) p[i] = INPUT[k][i];

        // Copy Host -> Acc
        alpaka::memcpy(queue, acc_input_bufs.back(), host_input_bufs.back());
    }

    auto acc_out_buf = alpaka::allocBuf<T, Idx>(devAcc, extent_out);
    auto host_out_buf = alpaka::allocBuf<T, Idx>(devHost, extent_out);

    alpaka::wait(queue);

    std::array<T const*, num_inputs> acc_input_ptrs;
    for (std::size_t k = 0; k < num_inputs; ++k) {
        acc_input_ptrs[k] = alpaka::getPtrNative(acc_input_bufs[k]);
    }

    std::array<alpaka::Vec<Dim, Idx>, num_inputs> input_strides_vec;
    for (std::size_t k = 0; k < num_inputs; ++k) {
        input_strides_vec[k] = alpaka::Vec<Dim, Idx>(cols, 1);
    }

    std::array<Idx, num_inputs> axis_sizes;
    for (std::size_t k = 0; k < num_inputs; ++k) {
        axis_sizes[k] = in_rows[k];
    }

    auto output_shape = alpaka::Vec<Dim, Idx>(out0, out1);
    auto output_strides = alpaka::Vec<Dim, Idx>(out1, 1);

    /*
    auto computeStrides = [](const std::vector<std::size_t>& shape) {
        std::vector<std::size_t> strides(shape.size());
        if (shape.empty()) return strides;
        strides[shape.size() - 1] = 1;

        for (std::size_t i = shape.size() - 1; i != 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }

        return strides;
    };
    */

    std::size_t const concat_axis = 0;

    const std::size_t threadsX = 16, threadsY = 16;
    const std::size_t blocksX = (out0 + threadsX - 1) / threadsX;
    const std::size_t blocksY = (out1 + threadsY - 1) / threadsY;

    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{
        alpaka::Vec<Dim, Idx>(blocksX, blocksY),
        alpaka::Vec<Dim, Idx>(threadsX, threadsY), extent_out};

    ConcatKernel kernel;

    alpaka::exec<Acc>(queue, workDiv, kernel, acc_input_ptrs,
                      alpaka::getPtrNative(acc_out_buf), input_strides_vec,
                      axis_sizes, num_inputs, concat_axis, output_strides,
                      output_shape);

    alpaka::wait(queue);

    alpaka::memcpy(queue, host_out_buf, acc_out_buf, extent_out);
    alpaka::wait(queue);

    std::vector<T> expected;
    for (const auto& vec : INPUT)
        expected.insert(expected.end(), vec.begin(), vec.end());

    {
        T* p = alpaka::getPtrNative(host_out_buf);

        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (p[i] != expected[i]) {
                std::cerr << "Failed!\n";
                return 1;
            }
        }
    }

    std::cout << "Correct!\n";
    return 0;
}
