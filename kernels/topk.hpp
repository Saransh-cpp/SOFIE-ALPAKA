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
        using DimAcc = alpaka::Dim<TAcc>;
        static_assert(DimAcc::value == Dim::value, "Accelerator and data dimensions must match!");

        if constexpr (K == 0) return;

        constexpr std::size_t D = Dim::value;
        auto elements = alpaka::uniformElementsND(acc, output_shape);

        for (auto const& idx : elements) {
            Idx input_base_idx = 0;
            Idx output_base_idx = 0;

            for (std::size_t d = 0; d < D; ++d) {
                Idx const coord = idx[d];
                input_base_idx += coord * input_strides[d];
                output_base_idx += coord * output_strides[d];
            }

            Idx const input_topk_axis_stride = input_strides[topk_axis];
            Idx const output_topk_axis_stride = output_strides[topk_axis];

            // Use registers
            if constexpr (K <= MaxRegisters && K > 0) {
                T top_vals[K];
                Idx count = 0;

                for (Idx j = 0; j < topk_axis_size; ++j) {
                    Idx const input_idx = input_base_idx + (j * input_topk_axis_stride);
                    T const val = input[input_idx];

                    if (count == K && val <= top_vals[K - 1]) continue;

                    Idx insert_pos = 0;
                    while (insert_pos < count && val <= top_vals[insert_pos]) {
                        insert_pos++;
                    }

                    if (insert_pos < K) {
                        Idx const last = std::min(count, K - 1);
                        for (Idx s = last; s > insert_pos; --s) {
                            top_vals[s] = top_vals[s - 1];
                        }

                        top_vals[insert_pos] = val;
                        if (count < K) count++;
                    }
                }

                for (Idx i = 0; i < K; ++i) {
                    Idx const output_idx = output_base_idx + (i * output_topk_axis_stride);
                    output[output_idx] = (i < count) ? top_vals[i] : padding_value;
                }
            }
            // Use global memory
            else {
                Idx count = 0;
                for (Idx j = 0; j < topk_axis_size; ++j) {
                    Idx const input_idx = input_base_idx + (j * input_topk_axis_stride);
                    T const val = input[input_idx];

                    if (count == K && val <= output[output_base_idx + (K - 1) * output_topk_axis_stride]) continue;

                    Idx insert_pos = 0;
                    while (insert_pos < count) {
                        if (val > output[output_base_idx + insert_pos * output_topk_axis_stride]) break;
                        insert_pos++;
                    }

                    if (insert_pos < K) {
                        Idx const last = std::min(count, K - 1);
                        for (Idx s = last; s > insert_pos; --s) {
                            output[output_base_idx + s * output_topk_axis_stride] =
                                output[output_base_idx + (s - 1) * output_topk_axis_stride];
                        }

                        output[output_base_idx + insert_pos * output_topk_axis_stride] = val;
                        if (count < K) count++;
                    }
                }

                for (Idx i = count; i < K; ++i) {
                    output[output_base_idx + i * output_topk_axis_stride] = padding_value;
                }
            }
        }
    }
};

}  // namespace alpaka_kernels

#endif  // TOPK_KERNEL_HPP
