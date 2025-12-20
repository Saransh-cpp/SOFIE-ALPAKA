#ifndef WHERE_KERNEL_HPP
#define WHERE_KERNEL_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

/**
 * @brief Kernel to perform the "where" operation on tensors.
 *
 * @param TAcc Alpaka accelerator type.
 * @param T Data type of the tensor elements.
 * @param TCond Data type of the condition tensor elements (should be boolean).
 * @param Dim Dimensionality of the tensors.
 * @param Idx Index type for tensor dimensions.
 * @param condition Pointer to the condition tensor.
 * @param x Pointer to the tensor from which values are taken when condition is true.
 * @param y Pointer to the tensor from which values are taken when condition is false.
 * @param output Pointer to the output tensor.
 * @param cond_strides Stride vector for the condition tensor.
 * @param x_strides Stride vector for the x tensor.
 * @param y_strides Stride vector for the y tensor.
 * @param out_strides Stride vector for the output tensor.
 * @param output_shape Shape vector for the output tensor.
 */
struct WhereKernel {
    template <typename TAcc, typename T, typename TCond, typename Dim, typename Idx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TCond const* condition, T const* x, T const* y, T* output,
                                  alpaka::Vec<Dim, Idx> cond_strides, alpaka::Vec<Dim, Idx> x_strides,
                                  alpaka::Vec<Dim, Idx> y_strides, alpaka::Vec<Dim, Idx> out_strides,
                                  alpaka::Vec<Dim, Idx> output_shape) const {
        constexpr std::size_t D = Dim::value;

        // Get global thread index and total threads
        auto const threadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const threadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        // Convert to linear thread index
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

        // Grid-stride loop: each thread processes multiple elements if needed
        for (Idx elem_idx = global_thread_idx; elem_idx < total_elements; elem_idx += threadExtent.prod()) {
            // Convert linear index to multi-dimensional output index
            Idx remaining = elem_idx;
            alpaka::Vec<Dim, Idx> out_idx;
            for (int d = D - 1; d >= 0; --d) {
                out_idx[d] = remaining % output_shape[d];
                remaining /= output_shape[d];
            }

            // Compute linear indices for all arrays
            Idx cond_idx = 0;
            Idx x_idx = 0;
            Idx y_idx = 0;
            Idx output_idx = 0;

            for (std::size_t d = 0; d < D; ++d) {
                Idx const coord = out_idx[d];
                cond_idx += coord * cond_strides[d];
                x_idx += coord * x_strides[d];
                y_idx += coord * y_strides[d];
                output_idx += coord * out_strides[d];
            }

            // Perform the where operation
            output[output_idx] = condition[cond_idx] ? x[x_idx] : y[y_idx];
        }
    }
};

}  // namespace alpaka_kernels

#endif  // WHERE_KERNEL_HPP
