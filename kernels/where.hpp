#ifndef ALPAKA_KERNELS_WHERE_HPP
#define ALPAKA_KERNELS_WHERE_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

struct WhereKernel {
    template <typename TAcc, typename T, typename TCond, typename Dim,
              typename Idx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TCond const* condition,
                                  T const* x, T const* y, T* output,
                                  alpaka::Vec<Dim, Idx> output_shape,
                                  alpaka::Vec<Dim, Idx> cond_strides,
                                  alpaka::Vec<Dim, Idx> x_strides,
                                  alpaka::Vec<Dim, Idx> y_strides,
                                  alpaka::Vec<Dim, Idx> out_strides) const {
        using DimAcc = alpaka::Dim<TAcc>;
        static_assert(DimAcc::value == Dim::value,
                      "Accelerator and data dims must match");

        constexpr std::size_t D = Dim::value;
        auto elements = alpaka::uniformElementsND(acc, output_shape);

        for (auto const& idx : elements) {
            // Compute input and output indexes
            Idx cond_idx = 0;
            Idx x_idx = 0;
            Idx y_idx = 0;
            Idx output_idx = 0;

            for (std::size_t d = 0; d < D; ++d) {
                Idx const out_coord = idx[d];

                cond_idx += out_coord * cond_strides[d];
                x_idx += out_coord * x_strides[d];
                y_idx += out_coord * y_strides[d];
                output_idx += out_coord * out_strides[d];
            }

            output[output_idx] = condition[cond_idx] ? x[x_idx] : y[y_idx];
        }
    }
};

}  // namespace alpaka_kernels
