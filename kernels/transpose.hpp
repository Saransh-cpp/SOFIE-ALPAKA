#ifndef TRANSPOSE_KERNEL_HPP
#define TRANSPOSE_KERNEL_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

/**
 * @brief Kernel to transpose a tensor according to a given permutation of axes.
 *
 * @param TAcc Alpaka accelerator type.
 * @param T Data type of the tensor elements.
 * @param Dim Dimensionality of the tensors.
 * @param Idx Index type for tensor dimensions.
 * @param input Pointer to the input tensor.
 * @param output Pointer to the output tensor.
 * @param input_strides Stride vector for the input tensor.
 * @param output_strides Stride vector for the output tensor.
 * @param output_shape Shape vector for the output tensor.
 * @param perm Permutation vector defining the transpose operation.
 */
struct TransposeKernel {
    template <typename TAcc, typename T, typename Dim, typename Idx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output, alpaka::Vec<Dim, Idx> input_strides,
                                  alpaka::Vec<Dim, Idx> output_strides, alpaka::Vec<Dim, Idx> output_shape,
                                  alpaka::Vec<Dim, Idx> perm) const {
        constexpr std::size_t D = Dim::value;

        // Get global thread index (maps to output element)
        auto const threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const threadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        // Convert to linear index
        Idx global_thread_idx = 0;
        Idx stride = 1;
        for (std::size_t d = 0; d < D; ++d) {
            global_thread_idx += threadIdx[d] * stride;
            stride *= threadExtent[d];
        }

        // Total number of output elements
        Idx total_elements = 1;
        for (std::size_t d = 0; d < D; ++d) {
            total_elements *= output_shape[d];
        }

        // Process elements with stride equal to total threads (grid-stride loop)
        for (Idx elem_idx = global_thread_idx; elem_idx < total_elements; elem_idx += threadExtent.prod()) {
            // Convert linear index to multi-dimensional output index
            Idx remaining = elem_idx;
            alpaka::Vec<Dim, Idx> out_idx;
            for (int d = D - 1; d >= 0; --d) {
                out_idx[d] = remaining % output_shape[d];
                remaining /= output_shape[d];
            }

            // Compute output linear index
            Idx output_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                output_idx += out_idx[d] * output_strides[d];
            }

            // Compute input linear index using permutation
            Idx input_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                input_idx += out_idx[d] * input_strides[perm[d]];
            }

            output[output_idx] = input[input_idx];
        }
    }
};

}  // namespace alpaka_kernels

#endif  // TRANSPOSE_KERNEL_HPP
