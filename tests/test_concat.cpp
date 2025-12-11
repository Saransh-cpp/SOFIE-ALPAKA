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

    std::vector<std::size_t> in_rows(num_inputs);
    for (auto& val : in_rows) val = distrib_int(gen);

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

    // ------------------ platform / device / queue ------------------
    auto plat = alpaka::Platform<Acc>{};
    auto dev = alpaka::getDevByIdx(plat, 0u);
    alpaka::Queue<Acc, alpaka::Blocking> queue{dev};

    // --- compute output shape: concat on axis 0 -> output_shape[0] = sum of
    // input sizes along axis 0
    std::vector<std::size_t> axis_sizes(num_inputs);
    for (std::size_t k = 0; k < num_inputs; ++k) axis_sizes[k] = in0;
    std::vector<std::size_t> axis_offsets(num_inputs);
    std::size_t acc = 0;
    for (std::size_t k = 0; k < num_inputs; ++k) {
        axis_offsets[k] = acc;
        acc += axis_sizes[k];
    }
    const std::size_t out0 = acc;  // total along axis 0
    const std::size_t out1 = in1;  // axis 1 stays same
    std::vector<std::size_t> output_shape = {out0, out1};

    // strides (row-major C-order)
    std::vector<std::size_t> input_shape = {in0, in1};  // input dims
    auto computeRowMajorStrides = [](const std::vector<std::size_t>& shape) {
        std::vector<std::size_t> strides(shape.size());
        if (shape.empty()) return strides;
        strides[shape.size() - 1] = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2;
             i >= 0; --i)
            strides[static_cast<std::size_t>(i)] =
                strides[static_cast<std::size_t>(i + 1)] *
                shape[static_cast<std::size_t>(i + 1)];
        return strides;
    };

    std::vector<std::size_t> in_strides =
        computeRowMajorStrides(input_shape);  // {in1, 1}
    std::vector<std::size_t> out_strides =
        computeRowMajorStrides(output_shape);  // {out1, 1}

    // ------------------ allocate device buffers: one buffer per input
    // ------------------ 2D extent for each input
    auto extent_in = alpaka::Vec<Dim, Idx>(in0, in1);
    auto extent_out = alpaka::Vec<Dim, Idx>(out0, out1);

    // allocate device buffers for inputs and output
    std::vector<decltype(alpaka::allocBuf<T, Idx>(dev, extent_in))>
        dev_input_bufs;
    dev_input_bufs.reserve(num_inputs);
    for (std::size_t k = 0; k < num_inputs; ++k) {
        dev_input_bufs.push_back(alpaka::allocBuf<T, Idx>(dev, extent_in));
    }
    auto dev_out_buf = alpaka::allocBuf<T, Idx>(dev, extent_out);

    // host staging buffers (use device as host device for CPU backend)
    std::vector<decltype(alpaka::allocBuf<T, Idx>(dev, extent_in))>
        host_input_bufs;
    host_input_bufs.reserve(num_inputs);
    for (std::size_t k = 0; k < num_inputs; ++k) {
        host_input_bufs.push_back(alpaka::allocBuf<T, Idx>(dev, extent_in));
    }
    auto host_out_buf = alpaka::allocBuf<T, Idx>(dev, extent_out);

    // copy host input vectors into host_input_bufs
    for (std::size_t k = 0; k < num_inputs; ++k) {
        T* p = alpaka::getPtrNative(host_input_bufs[k]);
        for (Idx i = 0; i < in0 * in1; ++i) p[i] = INPUT[k][i];
    }

    // H2D: copy each host_input_buf to device input buf
    for (std::size_t k = 0; k < num_inputs; ++k) {
        alpaka::memcpy(queue, dev_input_bufs[k], host_input_bufs[k], extent_in);
    }
    alpaka::wait(queue);

    // Build arrays of device pointers and device stride-pointers
    std::vector<T const*> input_ptrs_host(num_inputs);
    std::vector<std::size_t const*> input_strides_ptrs_host(num_inputs);

    for (std::size_t k = 0; k < num_inputs; ++k) {
        input_ptrs_host[k] = alpaka::getPtrNative(dev_input_bufs[k]);
    }
    // prepare per-input strides storage
    std::vector<std::vector<std::size_t>> input_strides_vecs(num_inputs,
                                                             in_strides);
    for (std::size_t k = 0; k < num_inputs; ++k)
        input_strides_ptrs_host[k] = input_strides_vecs[k].data();

    // pointers for kernel args
    T const* const* dev_input_ptrs = input_ptrs_host.data();
    std::size_t const* const* dev_input_strides_ptrs =
        input_strides_ptrs_host.data();
    std::size_t const* axis_sizes_ptr = axis_sizes.data();
    std::size_t const* output_strides_ptr = out_strides.data();
    std::size_t const* output_shape_ptr = output_shape.data();
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

    alpaka::exec<Acc>(queue, workDiv, kernel, dev_input_ptrs,
                      alpaka::getPtrNative(dev_out_buf), dev_input_strides_ptrs,
                      axis_sizes_ptr, num_inputs, concat_axis,
                      output_strides_ptr, output_shape_ptr);
    alpaka::wait(queue);

    // D2H: copy device output to host_out_buf
    alpaka::memcpy(queue, host_out_buf, dev_out_buf, extent_out);
    alpaka::wait(queue);

    // print output
    std::cout << "Output (" << out0 << "x" << out1 << "):\n";
    {
        T* p = alpaka::getPtrNative(host_out_buf);
        for (std::size_t r = 0; r < out0; ++r) {
            for (std::size_t c = 0; c < out1; ++c)
                std::cout << p[r * out1 + c] << " ";
            std::cout << "\n";
        }
    }

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
