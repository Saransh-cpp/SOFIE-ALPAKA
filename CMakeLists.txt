cmake_minimum_required (VERSION 3.25.2)
project(SODIE-ALPAKA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- User-configurable options ---
set(CUDA_BASE "/usr/local/cuda-13.1" CACHE PATH "CUDA base path")
set(TBB_BASE "/usr" CACHE PATH "TBB base path")
set(ALPAKA_BASE "external/alpaka" CACHE PATH "Alpaka base path")

# --- Compiler flags ---
set(CXXFLAGS -O2 -g -DALPAKA_HAS_STD_ATOMIC_REF)
set(CXX_HOST_FLAGS -fPIC -pthread)
set(CUDA_ARCH "sm_75")
set(CXX_CUDA_FLAGS -arch=${CUDA_ARCH} -Wno-deprecated-gpu-targets --extended-lambda --expt-relaxed-constexpr)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc" CACHE PATH "Cuda compiler path")

# --- Executables ---
add_executable(test_transpose tests/test_transpose.cpp)
set_source_files_properties(tests/test_transpose.cpp PROPERTIES LANGUAGE CUDA)
enable_language(CUDA)

set_target_properties(test_transpose PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(test_transpose PUBLIC cxx_std_20)

target_compile_options(test_transpose PRIVATE
    ${CXXFLAGS}
    ${CXX_CUDA_FLAGS}
    ${CXX_HOST_FLAGS}
)

target_compile_definitions(test_transpose PRIVATE
    ALPAKA_ACC_GPU_CUDA_ENABLED
)

target_include_directories(test_transpose PRIVATE
    ${ALPAKA_BASE}/include
    ${CUDA_BASE}/include
)

target_link_directories(test_transpose PRIVATE ${CUDA_BASE}/lib64)

target_link_libraries(test_transpose
    PRIVATE
        cublas
        cublasLt
        cudart
        nvidia-ml
)

add_executable(test_concat tests/test_concat.cpp)
set_source_files_properties(tests/test_concat.cpp PROPERTIES LANGUAGE CUDA)
enable_language(CUDA)

set_target_properties(test_concat PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(test_concat PUBLIC cxx_std_20)

target_compile_options(test_concat PRIVATE
    ${CXXFLAGS}
    ${CXX_CUDA_FLAGS}
    ${CXX_HOST_FLAGS}
)

target_compile_definitions(test_concat PRIVATE
    ALPAKA_ACC_GPU_CUDA_ENABLED
)

target_include_directories(test_concat PRIVATE
    ${ALPAKA_BASE}/include
    ${CUDA_BASE}/include
)

target_link_directories(test_concat PRIVATE ${CUDA_BASE}/lib64)

target_link_libraries(test_concat
    PRIVATE
        cublas
        cublasLt
        cudart
        nvidia-ml
)

add_executable(test_where tests/test_where.cpp)
set_source_files_properties(tests/test_where.cpp PROPERTIES LANGUAGE CUDA)
enable_language(CUDA)

set_target_properties(test_where PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(test_where PUBLIC cxx_std_20)

target_compile_options(test_where PRIVATE
    ${CXXFLAGS}
    ${CXX_CUDA_FLAGS}
    ${CXX_HOST_FLAGS}
)

target_compile_definitions(test_where PRIVATE
    ALPAKA_ACC_GPU_CUDA_ENABLED
)

target_include_directories(test_where PRIVATE
    ${ALPAKA_BASE}/include
    ${CUDA_BASE}/include
)

target_link_directories(test_where PRIVATE ${CUDA_BASE}/lib64)

target_link_libraries(test_where
    PRIVATE
        cublas
        cublasLt
        cudart
        nvidia-ml
)

add_executable(test_topk tests/test_topk.cpp)
set_source_files_properties(tests/test_topk.cpp PROPERTIES LANGUAGE CUDA)
enable_language(CUDA)

set_target_properties(test_topk PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(test_topk PUBLIC cxx_std_20)

target_compile_options(test_topk PRIVATE
    ${CXXFLAGS}
    ${CXX_CUDA_FLAGS}
    ${CXX_HOST_FLAGS}
)

target_compile_definitions(test_topk PRIVATE
    ALPAKA_ACC_GPU_CUDA_ENABLED
)

target_include_directories(test_topk PRIVATE
    ${ALPAKA_BASE}/include
    ${CUDA_BASE}/include
)

target_link_directories(test_topk PRIVATE ${CUDA_BASE}/lib64)

target_link_libraries(test_topk
    PRIVATE
        cublas
        cublasLt
        cudart
        nvidia-ml
)

# Optional clean
add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} -E rm -f test_transpose *.d *.o *.so
  COMMENT "Cleaning all generated files"
)
