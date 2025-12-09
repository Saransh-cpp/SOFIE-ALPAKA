#ifndef ALPAKA_KERNELS_CONCAT_HPP
#define ALPAKA_KERNELS_CONCAT_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

struct ConcatKernel {
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* const* input_ptrs, T* output,
                                  std::size_t const* const* input_strides_ptrs, std::size_t const* axis_sizes,
                                  std::size_t num_inputs, std::size_t axis, std::size_t const* output_strides,
                                  std::size_t const* output_shape) const {
        using DimAcc = alpaka::Dim<TAcc>;
        using IdxAcc = alpaka::Idx<TAcc>;
        constexpr std::size_t D = static_cast<std::size_t>(DimAcc::value);

        // Build shapeVec from output_shape to call uniformElementsND
        alpaka::Vec<DimAcc, IdxAcc> shapeVec{};
        for (std::size_t d = 0; d < D; ++d) shapeVec[d] = output_shape[d];

        auto elements = alpaka::uniformElementsND(acc, shapeVec);

        for (auto const& idx : elements) {
            // compute linear output index
            std::size_t out_idx = 0;
            for (std::size_t d = 0; d < D; ++d) out_idx += idx[d] * output_strides[d];

            // find chosen input by scanning axis_sizes, computing cumulative offset
            std::size_t axis_coord = idx[axis];

            std::size_t chosen = 0;
            std::size_t offset = 0;
            for (std::size_t k = 0; k < num_inputs; ++k) {
                std::size_t sz = axis_sizes[k];
                if (axis_coord < offset + sz) {
                    chosen = k;
                    break;
                }
                offset += sz;
            }

            // compute input linear index: subtract offset for concat axis
            std::size_t in_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                std::size_t coord_out = idx[d];
                std::size_t coord_in = (d == axis) ? (coord_out - offset) : coord_out;
                in_idx += coord_in * input_strides_ptrs[chosen][d];
            }

            // copy
            T const* src = input_ptrs[chosen];
            output[out_idx] = src[in_idx];
        }
    }
};

}  // namespace alpaka_kernels

#endif  // ALPAKA_KERNELS_CONCAT_HPP
