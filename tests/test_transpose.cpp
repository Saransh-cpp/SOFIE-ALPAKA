#include <alpaka/alpaka.hpp>
#include <iostream>
#include <vector>

#include "../kernels/transpose.hpp"

using Dim = alpaka::DimInt<2>;
using Idx = std::size_t;
using Acc = alpaka::AccCpuThreads<Dim, Idx>;

int main() {
    using namespace alpaka_kernels;
    using T = float;

    // Input matrix dimensions (rows x cols)
    const std::size_t rows = 2;
    const std::size_t cols = 3;
    const std::size_t numElems = rows * cols;
    std::vector<T> INPUT = {1, 2, 3, 4, 5, 6};

    std::cout << "Input (" << rows << "x" << cols << "):\n";
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) std::cout << INPUT[i * cols + j] << " ";
        std::cout << "\n";
    }

    // ------------------ platform / device / queue ------------------
    auto plat = alpaka::Platform<Acc>{};
    auto dev = alpaka::getDevByIdx(plat, 0u);
    alpaka::Queue<Acc, alpaka::Blocking> queue{dev};

    // ------------------ allocate buffers ------------------
    // Alpaka extent for linear allocation (1D)
    auto extent = alpaka::Vec<Dim, Idx>(cols, rows);

    auto dIn = alpaka::allocBuf<T, Idx>(dev, extent);
    auto dOut = alpaka::allocBuf<T, Idx>(dev, extent);

    // host buffers (device may be host when using CPU backend)
    auto hIn = alpaka::allocBuf<T, Idx>(alpaka::getDevByIdx(plat, 0u), extent);
    auto hOut = alpaka::allocBuf<T, Idx>(alpaka::getDevByIdx(plat, 0u), extent);

    // copy INPUT -> host buffer
    {
        T* p = alpaka::getPtrNative(hIn);
        for (Idx i = 0; i < numElems; ++i) p[i] = INPUT[i];
    }

    // copy host -> device
    alpaka::memcpy(queue, dIn, hIn, extent);
    alpaka::wait(queue);

    // ------------------ prepare kernel arguments ------------------
    // For row-major (C-order) 2D shape [rows, cols]:
    // stride[0] = cols, stride[1] = 1  (stride for axis 0 is product of dims
    // after it)
    std::array<std::size_t, 2> input_strides = {cols, 1};
    // Output shape is transposed: [cols, rows]
    // constexpr std::size_t NDIM = 2;
    std::array<std::size_t, 2> input_shape = {rows, cols};
    std::array<std::size_t, 2> output_shape = {cols, rows};
    std::array<std::size_t, 2> output_strides = {rows, 1};

    // perm: output axis i corresponds to input axis perm[i]
    // For transpose y[j,i] = x[i,j], output axis 0 (rows of y) comes from input
    // axis 1 (cols of x), so perm = {1,0}
    std::array<std::size_t, 2> perm = {1, 0};

    const std::size_t ndim = 2;

    // can pass pointers to host arrays directly on CPU backend
    std::size_t const* input_strides_ptr = input_strides.data();
    std::size_t const* output_strides_ptr = output_strides.data();
    std::size_t const* perm_ptr = perm.data();
    std::size_t const* output_shape_ptr = output_shape.data();
    std::size_t const* input_shape_ptr = input_shape.data();

    // ------------------ work division ------------------
    // A 2D mapping of threads to elements
    using DimT = alpaka::DimInt<2>;
    const std::size_t threadsX = 16, threadsY = 16;
    const std::size_t blocksX = (cols + threadsX - 1) / threadsX;
    const std::size_t blocksY = (rows + threadsY - 1) / threadsY;
    auto const extThreads = alpaka::Vec<DimT, Idx>(threadsX, threadsY);
    auto const extBlocks = alpaka::Vec<DimT, Idx>(blocksX, blocksY);
    auto const workDiv = alpaka::WorkDivMembers<DimT, Idx>{extBlocks, extThreads, extent};

    // ------------------ launch kernel ------------------
    TransposeKernel kernel;

    alpaka::exec<Acc>(queue, workDiv, kernel, alpaka::getPtrNative(dIn), alpaka::getPtrNative(dOut), input_strides_ptr,
                      output_strides_ptr, input_shape_ptr, output_shape_ptr, perm_ptr, ndim);
    alpaka::wait(queue);

    // ------------------ device -> host copy ------------------
    alpaka::memcpy(queue, hOut, dOut, extent);
    alpaka::wait(queue);

    // ------------------ print result ------------------
    std::cout << "Output (" << cols << "x" << rows << "):\n";
    {
        T* p = alpaka::getPtrNative(hOut);
        for (std::size_t i = 0; i < cols; ++i) {
            for (std::size_t j = 0; j < rows; ++j) std::cout << p[i * rows + j] << " ";
            std::cout << "\n";
        }
    }

    return 0;
}
