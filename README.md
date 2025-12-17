# SOFIE-ALPAKA

[![Build and Test on CPU](https://github.com/Saransh-cpp/SOFIE-ALPAKA/actions/workflows/build_and_test.yml/badge.svg?branch=main)](https://github.com/Saransh-cpp/SOFIE-ALPAKA/actions/workflows/build_and_test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Saransh-cpp/SOFIE-ALPAKA/main.svg)](https://results.pre-commit.ci/latest/github/Saransh-cpp/SOFIE-ALPAKA/main)

Kernels for heterogeneous architectures written in [Alpaka](https://alpaka.readthedocs.io/en/stable/) (An Abstraction Library for Parallel Kernel Acceleration) for [SOFIE](https://github.com/ML4EP/SOFIE) (System for Optimised Fast Inference code Emit).

This repository does not depend on SOFIE, but these kernels will eventually go into SOFIE.

Submission for CS-433: Machine Learning; hopefully, will not stay just as a random course project, but will become a part of the actual ML code written at CERN.

## Decription

High Energy Physics (HEP) experiments, such as the Large Hadron Collider (LHC) at the European Organization for Nuclear Research (CERN), produce petabytes of data every second. Physicists are now actively integrating Machine Learning techniques in various parts of the pipeline to collect and analyze this data. Given the massive scale of these experiments, and the upcoming High Luminosity upgrade to LHC (HL-LHC), it's becoming increasingly important to accelerate the inference of ML models beyond the supported capabilities of present day frameworks. The System for Optimized Fast Inference code Emit (SOFIE), is a C++ library developed by CERN for fast ML inference. SOFIE allows to parse a trained ML model into a highly-optimized C++ function which can be run without any overhead and minimal dependencies, making the inference process blazingly fast. This paper benchmarks SOFIE to evaluate its readiness to support inference for HEP experiments. Furthermore, CERN's Machine Learning For Experimental Physics team has recently been experimenting with adding heterogeneous computing support to SOFIE using the Alpaka library, allowing it to run inference on any device (including GPUs) while maintaining a single codebase. This repository extends SOFIE's Alpaka backend with four new kernels, and adds related tests and documentation, allowing SOFIE to support inference on GPUs for more ML models. It further benchmarks the the newly added operators against PyTorch implementations to showcase an increase in performance and the readiness to be used at scale.

```
.
├── .clang-format            # formatting configuration for CXX and HXX files
├── .github
│   └── workflows            # automated CI pipeline to run tests on GitHub Actions
├── .gitignore               # ignore files from git
├── .gitmodules              # to manage dependencies as git submodules
├── .pre-commit-config.yaml  # pre-commit hooks to format and lint all files
├── CMakeLists.txt           # build file for CUDA
├── Makefile                 # build file for CPU
├── README.md                # this file
├── external                 # included dependencies
│   └── alpaka
├── kernels                  # the kernels / operators
│   ├── concat.hpp
│   ├── topk.hpp
│   ├── transpose.hpp
│   ├── trivial.hpp
│   └── where.hpp
├── run.py                   # automated script to run benchmarks
└── tests                    # kernel / operator tests
    ├── sofie_integration
    ├── test_concat.cpp
    ├── test_topk.cpp
    ├── test_transpose.cpp
    ├── test_trivial.cpp
    └── test_where.cpp
```

## Dependencies

- `Alpaka` (`1.2.0`): for heterogenous kernels; present as a git submodule in `external/`
- `Boost` (`libboost-all-dev` on Debian): for Alpaka
- `cmake` and `make`: for building and testing the project

## Usage

Clone the repository with all the submodules (dependencies):

```
git clone https://github.com/Saransh-cpp/SOFIE-ALPAKA --recursive
```

### Building, testing, and benchmarking kernels

The repository contains a wrapper script to build, test, and benchmark all kernels on CPU, GPU, and against CPU/GPU enabled PyTorch.

To run benchmarks on CPU:
```
python run.py
```

To run benchmarks on GPU, update `CMakeLists.txt` with your CUDA compiler path and architecture, and run:
```
python run.py --gpu
```

The script will automatically find PyTorch (and if it was compiled for CPU or GPU) and benchmark against it (or skip if PyTorch is not installed).

### Building and running kernels and tests on a threaded CPU

To build all kernels and tests in `bin/`:

```
make all -j10
```

To run all kernel tests (and build if not built before):

```
make test -j10
```

### Building and running kernels and tests on an NVIDIA GPU

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

## Results

Please refer to our final report for the final results.
