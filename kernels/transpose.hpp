#ifndef TRANSPOSE_KERNEL_HPP
#define TRANSPOSE_KERNEL_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

struct TransposeKernel {
    template <typename TAcc, typename T, typename Dim, typename Idx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output,
                                  alpaka::Vec<Dim, Idx> input_strides,
                                  alpaka::Vec<Dim, Idx> output_strides,
                                  alpaka::Vec<Dim, Idx> output_shape,
                                  alpaka::Vec<Dim, Idx> perm) const {
        using DimAcc = alpaka::Dim<TAcc>;
        static_assert(DimAcc::value == Dim::value,
                      "Accelerator and data dimensions must match!");

        constexpr std::size_t D = Dim::value;
        auto elements = alpaka::uniformElementsND(acc, output_shape);

        for (auto const& idx : elements) {
            Idx input_idx = 0;
            Idx output_idx = 0;

            // Compute input and output indexes
            for (std::size_t d = 0; d < D; ++d) {
                Idx const out_coord = idx[d];
                output_idx += out_coord * output_strides[d];
                input_idx += out_coord * input_strides[perm[d]];
            }

            output[output_idx] = input[input_idx];
        }
    }
};

}  // namespace alpaka_kernels

#endif  // TRANSPOSE_KERNEL_HPP
