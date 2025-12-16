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
make all
```

### Running tests on a threaded CPU

To run all kernel tests (and build if not built before):

```
make test
```

### Building kernels and tests on an NVIDIA GPU

To build all the kernels and tests in `build/`

```
cmake -S. -Bbuild
cmake --build build
```

where the following flags can be configured by the user:
- `CUDA_BASE` (default: "/usr/local/cuda-13.1"): CUDA base path
- `TBB_BASE` (default: "/usr"): TBB base path
- `ALPAKA_BASE` (default: "external/alpaka"): Alpaka base path
- `CUDA_ARCH` (default: "sm_75"): CUDA architecture
- `CMAKE_CUDA_COMPILER` (default: "/usr/local/cuda-13.1/bin/nvcc"): Cuda compiler path

### Running integration tests on an NVIDIA GPU

To run SOFIE integration tests:

```
cd tests/sofie_integration
cmake -S. -Bbuild
cmake --build build
```

with the same configurable flags listed in the section above.
