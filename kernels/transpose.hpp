#ifndef TRANSPOSE_KERNEL_HPP
#define TRANSPOSE_KERNEL_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

struct TransposeKernel {
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output, const std::size_t* input_strides,
                                  const std::size_t* output_strides, const std::size_t* input_shape,
                                  const std::size_t* output_shape, const std::size_t* perm,
                                  const std::size_t ndim) const {
        using DimAcc = alpaka::Dim<TAcc>;
        using IdxAcc = alpaka::Idx<TAcc>;
        constexpr std::size_t D = static_cast<std::size_t>(DimAcc::value);
        alpaka::Vec<DimAcc, IdxAcc> shapeVec{};
        for (std::size_t d = 0; d < D; ++d) shapeVec[d] = output_shape[d];
        auto elements = alpaka::uniformElementsND(acc, shapeVec);
        for (auto const& idx : elements) {
            std::size_t input_idx = 0;
            std::size_t output_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                std::size_t out_coord = idx[d];
                std::size_t in_axis = perm[d];
                input_idx += out_coord * input_strides[in_axis];
                output_idx += out_coord * output_strides[d];
            }
            output[output_idx] = input[input_idx];
        }  // end elements loop
    }
};

}  // namespace alpaka_kernels

#endif  // TRANSPOSE_KERNEL_HPP
