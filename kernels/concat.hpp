#ifndef CONCAT_KERNEL_HPP
#define CONCAT_KERNEL_HPP

#include <alpaka/alpaka.hpp>
#include <array>

namespace alpaka_kernels {

struct ConcatKernel {
    template <typename TAcc, typename T, typename Dim, typename Idx, std::size_t N>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, std::array<T const*, N> input_ptrs, T* output,
                                  std::array<alpaka::Vec<Dim, Idx>, N> input_strides_vec,
                                  alpaka::Vec<Dim, Idx> output_strides, alpaka::Vec<Dim, Idx> output_shape,
                                  std::array<Idx, N> axis_sizes, std::size_t concat_axis) const {
        using DimAcc = alpaka::Dim<TAcc>;
        static_assert(DimAcc::value == Dim::value, "Accelerator and data dims must match");

        constexpr std::size_t D = Dim::value;
        auto elements = alpaka::uniformElementsND(acc, output_shape);

        for (auto const& idx : elements) {
            Idx concat_coord = idx[concat_axis];
            std::size_t chosen = 0;
            Idx offset = 0;

            // Find which input matrix this pixel belongs to
            for (std::size_t k = 0; k < N; ++k) {
                Idx const sz = axis_sizes[k];
                if (concat_coord < offset + sz) {
                    chosen = k;
                    break;
                }

                offset += sz;
            }

            // Compute input and output indexes
            Idx input_idx = 0;
            Idx output_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                Idx const out_coord = idx[d];
                output_idx += out_coord * output_strides[d];

                Idx const in_coord = out_coord - offset * (d == concat_axis);
                input_idx += in_coord * input_strides_vec[chosen][d];
            }

            output[output_idx] = input_ptrs[chosen][input_idx];
        }
    }
};

}  // namespace alpaka_kernels

#endif  // CONCAT_KERNEL_HPP
