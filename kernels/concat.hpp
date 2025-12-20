#ifndef CONCAT_KERNEL_HPP
#define CONCAT_KERNEL_HPP

#include <alpaka/alpaka.hpp>
#include <array>

namespace alpaka_kernels {

/**
 * @brief Kernel to concatenate multiple input tensors along a specified axis.
 *
 * @param TAcc Alpaka accelerator type.
 * @param T Data type of the tensor elements.
 * @param Dim Dimensionality of the tensors.
 * @param Idx Index type for tensor dimensions.
 * @param N Number of input tensors to concatenate.
 * @param input_ptrs Array of pointers to input tensors.
 * @param output Pointer to the output tensor.
 * @param input_strides_vec Array of stride vectors for each input tensor.
 * @param output_strides Stride vector for the output tensor.
 * @param output_shape Shape vector for the output tensor.
 * @param axis_sizes Sizes of each input tensor along the concatenation axis.
 * @param concat_axis Axis along which to concatenate the input tensors.
 */
struct ConcatKernel {
    template <typename TAcc, typename T, typename Dim, typename Idx, std::size_t N>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, std::array<T const*, N> input_ptrs, T* output,
                                  std::array<alpaka::Vec<Dim, Idx>, N> input_strides_vec,
                                  alpaka::Vec<Dim, Idx> output_strides, alpaka::Vec<Dim, Idx> output_shape,
                                  std::array<Idx, N> axis_sizes, std::size_t concat_axis) const {
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

        // Grid-stride loop
        for (Idx elem_idx = global_thread_idx; elem_idx < total_elements; elem_idx += threadExtent.prod()) {
            // Convert linear index to multi-dimensional output index
            Idx remaining = elem_idx;
            alpaka::Vec<Dim, Idx> out_idx;
            for (int d = D - 1; d >= 0; --d) {
                out_idx[d] = remaining % output_shape[d];
                remaining /= output_shape[d];
            }

            // Determine which input tensor this element comes from
            Idx concat_coord = out_idx[concat_axis];
            std::size_t chosen = 0;
            Idx offset = 0;

            // Find the input tensor that contains this coordinate
            for (std::size_t k = 0; k < N; ++k) {
                Idx const sz = axis_sizes[k];
                if (concat_coord < offset + sz) {
                    chosen = k;
                    break;
                }
                offset += sz;
            }

            // Compute output linear index
            Idx output_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                output_idx += out_idx[d] * output_strides[d];
            }

            // Compute input linear index (adjust for concat axis offset)
            alpaka::Vec<Dim, Idx> in_idx = out_idx;
            in_idx[concat_axis] = concat_coord - offset;

            Idx input_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                input_idx += in_idx[d] * input_strides_vec[chosen][d];
            }

            // Copy the element
            output[output_idx] = input_ptrs[chosen][input_idx];
        }
    }
};

}  // namespace alpaka_kernels

#endif  // CONCAT_KERNEL_HPP
