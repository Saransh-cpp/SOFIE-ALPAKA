#ifndef TOPK_KERNEL_HPP
#define TOPK_KERNEL_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

// K: the number of elements to select along the specified axis
// MaxRegisters: threshold where we switch from registers to global memory
template <std::size_t K, std::size_t MaxRegisters = 64>
struct TopKKernel {
    template <typename TAcc, typename T, typename Dim, typename Idx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output, alpaka::Vec<Dim, Idx> input_strides,
                                  alpaka::Vec<Dim, Idx> output_strides, alpaka::Vec<Dim, Idx> output_shape,
                                  Idx topk_axis, Idx topk_axis_size, T padding_value) const {
        if constexpr (K == 0) return;

        constexpr std::size_t D = Dim::value;

        // Total number of output positions (excluding the K dimension)
        // Each thread handles one position and finds top K along the topk_axis
        Idx total_positions = 1;
        for (std::size_t d = 0; d < D; ++d) {
            if (d != static_cast<std::size_t>(topk_axis)) {
                total_positions *= output_shape[d];
            }
        }

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

        // Grid-stride loop over positions (each position is a slice along topk_axis)
        for (Idx pos_idx = global_thread_idx; pos_idx < total_positions; pos_idx += threadExtent.prod()) {
            // Convert linear position index to multi-dimensional index (excluding topk_axis)
            Idx remaining = pos_idx;
            alpaka::Vec<Dim, Idx> full_idx;

            // Initialize full index
            for (std::size_t d = 0; d < D; ++d) {
                full_idx[d] = 0;
            }

            // Fill in coordinates for dimensions except topk_axis
            for (int d = D - 1; d >= 0; --d) {
                if (static_cast<std::size_t>(d) != static_cast<std::size_t>(topk_axis)) {
                    Idx dim_size = output_shape[d];
                    full_idx[d] = remaining % dim_size;
                    remaining /= dim_size;
                }
            }

            // Compute base indices for input and output
            Idx input_base_idx = 0;
            Idx output_base_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                Idx const coord = full_idx[d];
                input_base_idx += coord * input_strides[d];
                output_base_idx += coord * output_strides[d];
            }

            Idx const input_topk_axis_stride = input_strides[topk_axis];
            Idx const output_topk_axis_stride = output_strides[topk_axis];

            // Use registers for small K
            if constexpr (K <= MaxRegisters && K > 0) {
                T top_vals[K];
                Idx count = 0;

                // Initialize with padding value
                for (Idx i = 0; i < K; ++i) {
                    top_vals[i] = padding_value;
                }

                // Process all elements along the topk_axis
                for (Idx j = 0; j < topk_axis_size; ++j) {
                    Idx const input_idx = input_base_idx + (j * input_topk_axis_stride);
                    T const val = input[input_idx];

                    if (count < K) {
                        // Fill the array first
                        Idx insert_pos = count;
                        while (insert_pos > 0 && val < top_vals[insert_pos - 1]) {
                            top_vals[insert_pos] = top_vals[insert_pos - 1];
                            insert_pos--;
                        }
                        top_vals[insert_pos] = val;
                        count++;
                    } else if (val > top_vals[0]) {
                        // Replace smallest element
                        Idx insert_pos = 0;
                        while (insert_pos < K - 1 && val > top_vals[insert_pos + 1]) {
                            top_vals[insert_pos] = top_vals[insert_pos + 1];
                            insert_pos++;
                        }
                        top_vals[insert_pos] = val;
                    }
                }

                // Write results to output (largest to smallest)
                for (Idx i = 0; i < K; ++i) {
                    Idx const output_idx = output_base_idx + (i * output_topk_axis_stride);
                    output[output_idx] = top_vals[K - 1 - i];  // Reverse to get largest first
                }
            }
            // Use global memory for large K
            else {
                // Use output buffer as temporary storage
                for (Idx i = 0; i < K; ++i) {
                    Idx const output_idx = output_base_idx + (i * output_topk_axis_stride);
                    output[output_idx] = padding_value;
                }

                Idx count = 0;

                // Process all elements along the topk_axis
                for (Idx j = 0; j < topk_axis_size; ++j) {
                    Idx const input_idx = input_base_idx + (j * input_topk_axis_stride);
                    T const val = input[input_idx];

                    if (count < K) {
                        // Fill the output first
                        Idx insert_pos = count;
                        while (insert_pos > 0 &&
                               val < output[output_base_idx + (insert_pos - 1) * output_topk_axis_stride]) {
                            Idx src_idx = output_base_idx + (insert_pos - 1) * output_topk_axis_stride;
                            Idx dst_idx = output_base_idx + insert_pos * output_topk_axis_stride;
                            output[dst_idx] = output[src_idx];
                            insert_pos--;
                        }
                        output[output_base_idx + insert_pos * output_topk_axis_stride] = val;
                        count++;
                    } else if (val > output[output_base_idx]) {
                        // Replace smallest element (at position 0 since we store ascending)
                        Idx insert_pos = 0;
                        while (insert_pos < K - 1 &&
                               val > output[output_base_idx + (insert_pos + 1) * output_topk_axis_stride]) {
                            Idx src_idx = output_base_idx + (insert_pos + 1) * output_topk_axis_stride;
                            Idx dst_idx = output_base_idx + insert_pos * output_topk_axis_stride;
                            output[dst_idx] = output[src_idx];
                            insert_pos++;
                        }
                        output[output_base_idx + insert_pos * output_topk_axis_stride] = val;
                    }
                }

                // Reverse to get largest first
                for (Idx i = 0; i < K / 2; ++i) {
                    Idx idx1 = output_base_idx + i * output_topk_axis_stride;
                    Idx idx2 = output_base_idx + (K - 1 - i) * output_topk_axis_stride;
                    T temp = output[idx1];
                    output[idx1] = output[idx2];
                    output[idx2] = temp;
                }
            }
        }
    }
};

}  // namespace alpaka_kernels

#endif  // TOPK_KERNEL_HPP
