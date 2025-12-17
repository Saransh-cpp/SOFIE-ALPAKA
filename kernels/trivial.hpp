#ifndef TRIVIAL_KERNEL_HPP
#define TRIVIAL_KERNEL_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

struct TrivialKernel {
    template <typename TAcc, typename T, typename Dim, typename Idx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output, alpaka::Vec<Dim, Idx> output_strides,
                                  alpaka::Vec<Dim, Idx> output_shape) const {
        using DimAcc = alpaka::Dim<TAcc>;
        static_assert(DimAcc::value == Dim::value, "Accelerator and data dimensions must match!");

        constexpr std::size_t D = Dim::value;
        auto elements = alpaka::uniformElementsND(acc, output_shape);

        for (auto const& idx : elements) {
            Idx linear_idx = 0;

            // Compute input and output indexes
            for (std::size_t d = 0; d < D; ++d) {
                Idx const coord = idx[d];
                linear_idx += coord * output_strides[d];
            }

            output[linear_idx] = input[linear_idx];
        }
    }
};

}  // namespace alpaka_kernels

#endif  // TRIVIAL_KERNEL_HPP
