# SOFIE-ALPAKA

[![Build and Test on CPU](https://github.com/Saransh-cpp/SOFIE-ALPAKA/actions/workflows/build_and_test.yml/badge.svg?branch=main)](https://github.com/Saransh-cpp/SOFIE-ALPAKA/actions/workflows/build_and_test.yml)

Kernels for heterogeneous architectures written in [Alpaka](https://alpaka.readthedocs.io/en/stable/) (An Abstraction Library for Parallel Kernel Acceleration) for [SOFIE](https://github.com/ML4EP/SOFIE) (System for Optimised Fast Inference code Emit).

This repository does not depend on SOFIE, but these kernels will eventually go into SOFIE.

Submission for CS-433: Machine Learning; hopefully, will not stay just as a random course project, but will become a part of the actual ML code written at CERN.

## Dependencies

- `Alpaka` (`1.2.0`): for heterogenous kernels; present as a git submodule in `external/`
- `Boost` (`libboost-all-dev` on Debian): for Alpaka
- `cmake` and `make`: for building and testing the project

## Usage

Clone the repository with all the submodules (dependencies):

```
git clone https://github.com/Saransh-cpp/SOFIE-ALPAKA --recursive
```

### Building kernels and tests on a threaded CPU

To build all kernels and tests in `bin/`:

```
make all -j10
```

### Running tests on a threaded CPU

To run all kernel tests (and build if not built before):

```
make test -j10
```

### Building kernels and tests on an NVIDIA GPU

To build all the kernels and tests in `build/`

```
cmake -S. -Bbuild
cmake --build build -j10
```

where the following flags can be configured by the user:
- `CUDA_BASE` (default: "/usr/local/cuda-13.1"): CUDA base path
- `ALPAKA_BASE` (default: "external/alpaka"): Alpaka base path
- `CUDA_ARCH` (default: "sm_75"): CUDA architecture
- `CMAKE_CUDA_COMPILER` (default: "/usr/local/cuda-12.5/bin/nvcc"): Cuda compiler path

To run the tests, simply execute `test_*` executables produced in `build/`.

### Running integration tests on an NVIDIA GPU

1. Port a kernel to [SOFIE](https://github.com/ML4EP/SOFIE) on a stand-alone branch (against the `gpu/alpaka` branch) (see https://github.com/ML4EP/SOFIE/pull/7 and https://github.com/ML4EP/SOFIE/pull/8 for reference).
2. Make sure there is a corrresponding `onnx` model in `SOFIE/src/SOFIE_core/test/input_models/`.
3. Make sure there is a reference output in `SOFIE/src/SOFIE_core/test/input_models/references`.
4. Follow instructions in SOFIE's README to build and run tests with CUDA (remember to set `-DCUDA_ARCH` as per your GPU's architecture).

The relevant header and DAT files will be generated in `SOFIE/build/src/SOFIE_core/test/`.

#### Kernels already ported to SOFIE

`Transpose` and `Concat` kernels have already been ported to SOFIE (pull requests not merged yet). This repository has an updated implementation for both of these kernels, and two other kernels, which much be ported in the future.
