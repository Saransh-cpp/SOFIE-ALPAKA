cmake_minimum_required(VERSION 3.21)
project(AlpakaTest LANGUAGES CXX CUDA)

# --- User-configurable options ---
set(CUDA_BASE "/usr/local/cuda-13.1" CACHE PATH "CUDA base path")
set(TBB_BASE "/usr" CACHE PATH "TBB base path")
set(ALPAKA_BASE "../../../alpaka" CACHE PATH "Alpaka base path")

find_package(ROOT COMPONENTS)
# Add ROOT compile options
include(${ROOT_USE_FILE})

# Path to SOFIE headers (only SOFIE_common.hxx needed)
set(SOFIE_INCLUDE "../../../SOFIE/install/include")
set(SOFIE_BLAS_INCLUDE "../../../sofieBLAS/include")

# --- Compiler flags ---
set(CXXFLAGS -O2 -g -DALPAKA_HAS_STD_ATOMIC_REF)
set(CXX_HOST_FLAGS -fPIC -pthread)
set(CUDA_ARCH "sm_75")
set(CXX_CUDA_FLAGS -arch=${CUDA_ARCH} -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc" CACHE PATH "Cuda compiler path")

# --- Executable ---
set(TEST_SRC main.cxx)

add_executable(test_cuda ${TEST_SRC})
set_source_files_properties(${TEST_SRC} PROPERTIES LANGUAGE CUDA)
enable_language(CUDA)

set_target_properties(test_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(test_cuda PUBLIC cxx_std_20)

target_compile_options(test_cuda PRIVATE
    ${CXXFLAGS}
    ${CXX_CUDA_FLAGS}
    ${CXX_HOST_FLAGS}
)

target_compile_definitions(test_cuda PRIVATE
    ALPAKA_ACC_GPU_CUDA_ENABLED
)

target_include_directories(test_cuda PRIVATE
    ${ALPAKA_BASE}/include
    ${SOFIE_INCLUDE}
    ${ROOT_INCLUDE_DIRS}
    ${SOFIE_BLAS_INCLUDE}
    ${CUDA_BASE}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(test_cuda PRIVATE ${CUDA_BASE}/lib64)

target_link_libraries(test_cuda
    PRIVATE
        cublas
        cublasLt
        cudart
        nvidia-ml
        ${ROOT_LIBRARIES}
)

# Optional clean
add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} -E rm -f test_cuda *.d *.o *.so
  COMMENT "Cleaning all generated files"
)
