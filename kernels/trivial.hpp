#ifndef TRIVIAL_KERNEL_HPP
#define TRIVIAL_KERNEL_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

/**
 * @brief A trivial kernel that performs no operation.
 *
 * @param TAcc Alpaka accelerator type.
 * @param T Data type of the tensor elements.
 * @param Dim Dimensionality of the tensors.
 * @param Idx Index type for tensor dimensions.
 * @param input Pointer to the input tensor.
 * @param output Pointer to the output tensor.
 * @param output_shape Shape vector for the tensors.
 */
struct TrivialKernel {
    template <typename TAcc, typename T, typename Dim, typename Idx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output, alpaka::Vec<Dim, Idx> /*output_strides*/,
                                  alpaka::Vec<Dim, Idx> output_shape) const {
        // Get global thread index
        auto const threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const threadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        // Convert to linear thread index
        Idx global_thread_idx = 0;
        Idx stride = 1;
        for (std::size_t d = 0; d < Dim::value; ++d) {
            global_thread_idx += threadIdx[d] * stride;
            stride *= threadExtent[d];
        }

        // Total number of elements
        Idx total_elements = output_shape.prod();

        // Simple grid-stride copy for contiguous memory
        for (Idx i = global_thread_idx; i < total_elements; i += threadExtent.prod()) {
            output[i] = input[i];
        }
    }
};

}  // namespace alpaka_kernels

#endif  // TRIVIAL_KERNEL_HPP
