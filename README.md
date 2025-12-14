# SOFIE-ALPAKA

Kernels for heterogeneous architectures written in [Alpaka](https://alpaka.readthedocs.io/en/stable/) (An Abstraction Library for Parallel Kernel Acceleration) for [SOFIE](https://github.com/ML4EP/SOFIE) (System for Optimised Fast Inference code Emit).

This repository does not depend on SOFIE, but these kernels will eventually go into SOFIE.

Submission for CS-433: Machine Learning; hopefully, will not stay just as a random course project, but will become a part of the actual ML code written at CERN.

## Dependencies

- `Alpaka` (`v2.1.0`): for heterogenous kernels; present as a git submodule in `external/`

## Usage

Clone the repository with all the submodules (dependencies):

```
git clone https://github.com/Saransh-cpp/SOFIE-ALPAKA --recursive
```

To build all kernels in `bin/`:

```
make all ALPAKA_ACCELERATOR_FLAG=enable_an_alpaka_accelerator
```

where `ALPAKA_ACCELERATOR_FLAG` defaults to `ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED` (enables the CPU threaded backend)

To build (in `bin/`) and run all kernel tests:

```
make test ALPAKA_DIR=/path/to/alpaka/headers CPLUS_INCLUDE_PATH=/path/to/any/other/headers ALPAKA_ACCELERATOR_FLAG=enable_an_alpaka_accelerator
```

To clean the outputs:

```
make clean
```

To run SOFIE integration tests:

```
cd tests/sofie_integration
cmake -S. -Bbuild
cmake --build build
```
