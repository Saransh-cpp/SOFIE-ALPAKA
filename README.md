# SOFIE-ALPAKA

Kernels for heterogeneous architectures written in [Alpaka](https://alpaka.readthedocs.io/en/stable/) (An Abstraction Library for Parallel Kernel Acceleration) for [SOFIE](https://github.com/ML4EP/SOFIE) (System for Optimised Fast Inference code Emit).

This repository does not depend on SOFIE, but these kernels will eventually go into SOFIE.

Submission for CS-433: Machine Learning; hopefully, will not stay just as a random course project, but will become a part of the actual ML code written at CERN.

## Usage

To run kernel tests for a particular kernel (say `TransposeKernel`):

```
g++ ./tests/test_transpose.cpp \
-I/path/to/alpaka/headers/ \
-DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED \
-std=c++17
```

where `-DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED` enables the CPU threaded backend (change if required).
