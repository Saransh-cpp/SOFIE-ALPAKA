#ifndef ALPAKA_KERNELS_WHERE_HPP
#define ALPAKA_KERNELS_WHERE_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

struct WhereKernel {
    template <typename Acc, typename T, typename TCond, typename Dim,
              typename Idx>
    ALPAKA_FN_ACC void operator()(Acc const& acc, TCond const* condition,
                                  T const* x, T const* y, T* output,
                                  alpaka::Vec<Dim, Idx> extent,
                                  alpaka::Vec<Dim, Idx> cond_strides,
                                  alpaka::Vec<Dim, Idx> x_strides,
                                  alpaka::Vec<Dim, Idx> y_strides,
                                  alpaka::Vec<Dim, Idx> out_strides) const {
        // Get the number of dimensions
        constexpr std::size_t D = Dim::value;

        // Iterate over every element in the N-D grid
        auto elements = alpaka::uniformElementsND(acc, extent);

        for (auto const& idx : elements) {
            // Compute Linear Indices for all buffers
            Idx idx_cond = 0;
            Idx idx_x = 0;
            Idx idx_y = 0;
            Idx idx_out = 0;

            for (std::size_t d = 0; d < D; ++d) {
                Idx const coord = idx[d];

                idx_cond += coord * cond_strides[d];
                idx_x += coord * x_strides[d];
                idx_y += coord * y_strides[d];
                idx_out += coord * out_strides[d];
            }

            output[idx_out] = condition[idx_cond] ? x[idx_x] : y[idx_y];
        }
    }
};

}  // namespace alpaka_kernels
