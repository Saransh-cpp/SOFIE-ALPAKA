#include <alpaka/alpaka.hpp>
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

    // Input matrix dimensions
    const std::size_t num_inputs = 3;
    const std::size_t cols = distrib_int(gen);
    std::size_t acc = 0;

    std::vector<std::size_t> in_rows(num_inputs);
    for (auto& val : in_rows) {
        val = distrib_int(gen);
        acc += val;
    }

    std::cout << "Number of inputs: " << num_inputs << "\n";
    std::cout << "Inputs are of shape: ";
    for (std::size_t k = 0; k < num_inputs; ++k)
        std::cout << in_rows[k] << "x" << cols
                  << ((k < num_inputs - 1) ? ", " : "\n");

    std::vector<std::vector<T>> INPUT(num_inputs);
    for (auto& val : in_rows) INPUT.push_back(std::vector<T>(val * cols));

    for (auto& vec : INPUT) {
        for (auto& val : vec) val = distrib_real(gen);
    }

    // Setup the accelerator, host and queue
    auto devAcc = alpaka::getDevByIdx(PlatAcc{}, 0u);
    auto devHost = alpaka::getDevByIdx(PlatHost{}, 0u);
    alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

    const std::size_t out0 = acc;   // total along axis 0
    const std::size_t out1 = cols;  // axis 1 stays same

    // strides
    auto computeStrides = [](const std::vector<std::size_t>& shape) {
        std::vector<std::size_t> strides(shape.size());
        if (shape.empty()) return strides;
        strides[shape.size() - 1] = 1;

        for (std::size_t i = shape.size() - 1; i != 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }

        return strides;
    };

    std::vector<std::size_t> in_strides = computeStrides({in_rows[0], cols});
    std::vector<std::size_t> out_strides = computeStrides({out0, out1});

    // ------------------ allocate device buffers: one buffer per input
    // ------------------ 2D extent for each input
    std::vector<alpaka::Vec<Dim, Idx>> extent_in;
    for (std::size_t i = 0; i < num_inputs; ++i)
        extent_in.push_back(alpaka::Vec<Dim, Idx>(in_rows[i], cols));
    auto extent_out = alpaka::Vec<Dim, Idx>(out0, out1);

    // allocate device buffers for inputs and output
    std::vector<decltype(alpaka::allocBuf<T, Idx>(devAcc, extent_in[0]))>
        acc_input_bufs;
    acc_input_bufs.reserve(num_inputs);
    for (std::size_t k = 0; k < num_inputs; ++k) {
        acc_input_bufs.push_back(
            alpaka::allocBuf<T, Idx>(devAcc, extent_in[k]));
    }
    auto acc_out_buf = alpaka::allocBuf<T, Idx>(devAcc, extent_out);

    // host staging buffers (use device as host device for CPU backend)
    std::vector<decltype(alpaka::allocBuf<T, Idx>(devHost, extent_in[0]))>
        host_input_bufs;
    host_input_bufs.reserve(num_inputs);
    for (std::size_t k = 0; k < num_inputs; ++k) {
        host_input_bufs.push_back(
            alpaka::allocBuf<T, Idx>(devHost, extent_in[k]));
    }
    auto host_out_buf = alpaka::allocBuf<T, Idx>(devHost, extent_out);

    // copy host input vectors into host_input_bufs
    for (std::size_t k = 0; k < num_inputs; ++k) {
        T* p = alpaka::getPtrNative(host_input_bufs[k]);
        for (std::size_t i = 0; i < cols * in_rows[k]; ++i) p[i] = INPUT[k][i];
    }

    // host -> accelerator
    for (std::size_t k = 0; k < num_inputs; ++k) {
        alpaka::memcpy(queue, acc_input_bufs[k], host_input_bufs[k],
                       extent_in[k]);
    }

    alpaka::wait(queue);

    // Build arrays of device pointers and device stride-pointers
    alpaka::Array<T const*, num_inputs> acc_input_ptrs;

    for (std::size_t k = 0; k < num_inputs; ++k) {
        acc_input_ptrs[k] = alpaka::getPtrNative(acc_input_bufs[k]);
    }
    // prepare per-input strides storage
    alpaka::Array<std::array<std::size_t, NumDims>, num_inputs>
        input_strides_vec;
    for (std::size_t k = 0; k < num_inputs; ++k) {
        for (std::size_t i = 0; i < NumDims; ++i) {
            input_strides_vec[k][i] = in_strides[i];
        }
    }
    auto output_shape = alpaka::Vec<Dim, Idx>(out0, out1);
    auto output_strides = alpaka::Vec<Dim, Idx>(out1, 1);
    auto axis_sizes =
        alpaka::Vec<alpaka::DimInt<3>, Idx>(in_rows[0], in_rows[1], in_rows[2]);
    std::size_t const concat_axis = 0;

    // ------------------ work division (2D) ------------------
    const std::size_t threadsX = 16, threadsY = 16;
    const std::size_t blocksX = (out0 + threadsX - 1) / threadsX;
    const std::size_t blocksY = (out1 + threadsY - 1) / threadsY;

    auto const extThreads = alpaka::Vec<Dim, Idx>(threadsX, threadsY);
    auto const extBlocks = alpaka::Vec<Dim, Idx>(blocksX, blocksY);
    auto const elemsPerThread = alpaka::Vec<Dim, Idx>(1, 1);

    auto const workDiv =
        alpaka::WorkDivMembers<Dim, Idx>{extBlocks, extThreads, elemsPerThread};

    // ------------------ launch kernel ------------------
    ConcatKernel kernel;

    alpaka::exec<Acc>(queue, workDiv, kernel, acc_input_ptrs,
                      alpaka::getPtrNative(acc_out_buf), input_strides_vec,
                      axis_sizes, num_inputs, concat_axis, output_strides,
                      output_shape);
    alpaka::wait(queue);

    // D2H: copy device output to host_out_buf
    alpaka::memcpy(queue, host_out_buf, acc_out_buf, extent_out);
    alpaka::wait(queue);

    // verify expected
    std::vector<T> expected;
    expected.reserve(out0 * out1);
    for (std::size_t k = 0; k < num_inputs; ++k) {
        expected.insert(expected.end(), INPUT[k].begin(), INPUT[k].end());
    }
    bool ok = true;
    {
        T* p = alpaka::getPtrNative(host_out_buf);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (p[i] != expected[i]) {
                ok = false;
                break;
            }
        }
    }
    return 0;
}
