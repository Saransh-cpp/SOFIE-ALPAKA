"""
Benchmark Runner Script

This script builds and runs benchmarks for various tensor operations implemented in C++.
It supports both CPU and GPU benchmarks based on user input.
It compares the performance of these implementations against PyTorch (on the same device)
equivalents if PyTorch is available.
"""

import subprocess
import sys
import os
import re
import argparse
import time

try:
    import torch

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
    print("PyTorch not found, running only C++.\n")

# Configuration
EXECUTABLE_PATHS_CPU = [
    "./bin/test_trivial.out",
    "./bin/test_concat.out",
    "./bin/test_topk.out",
    "./bin/test_transpose.out",
    "./bin/test_where.out",
]

EXECUTABLE_PATHS_GPU = [
    "./build/test_trivial",
    "./build/test_transpose",
    "./build/test_concat",
    "./build/test_where",
    "./build/test_topk",
]

BENCHMARK_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]


def build_kernel_tests_cpu():
    """
    Calls the Makefile to build the kernel tests.
    Returns True if successful, False otherwise.
    """
    print("Building kernel tests with Make...")
    try:
        # Check if Makefile exists
        if not os.path.exists("Makefile"):
            print("Error: Makefile not found in current directory")
            return False

        # Run 'make'.
        subprocess.run(["make", "-j8"], check=True)

        print("Build successful\n")
        return True

    except subprocess.CalledProcessError:
        print("Build failed. Please fix C++ errors before running benchmarks")
        return False
    except FileNotFoundError:
        print("Error: 'make' command not found. Is it installed?")
        return False


def build_kernel_tests_gpu():
    """
    Runs cmake to build the kernel tests.
    Returns True if successful, False otherwise.
    """
    print("Building kernel tests with CMake...")
    try:
        # Check if Makefile exists
        if not os.path.exists("Makefile"):
            print("Error: Makefile not found in current directory")
            return False

        # Run 'make'.
        subprocess.run(["cmake", "-S.", "-Bbuild"], check=True)
        subprocess.run(["cmake", "--build", "build", "-j8"], check=True)

        print("Build successful\n")
        return True

    except subprocess.CalledProcessError:
        print("Build failed. Please fix C++ errors before running benchmarks")
        return False
    except FileNotFoundError:
        print("Error: 'cmake' command not found. Is it installed?")
        return False


def get_op_name(executable_path):
    """
    Infers the operation name from the executable path.
    """
    if "trivial" in executable_path:
        return "trivial"
    if "transpose" in executable_path:
        return "transpose"
    if "concat" in executable_path:
        return "concat"
    if "where" in executable_path:
        return "where"
    if "topk" in executable_path:
        return "topk"
    return "unknown"

def run_pytorch_benchmark(op_name, N, num_repeats=10, warmup=10):
    """
    Runs the equivalent operation in PyTorch and measures execution time.
    Compatible with both CPU and GPU.

    Parameters:
    - op_name: Name of the operation to benchmark.
    - N: Size parameter for the operation.
    - num_repeats: Number of times to repeat the operation for averaging.
    - warmup: Number of warmup iterations before timing.

    Returns:
    - Average execution time in milliseconds, or None if operation is unknown.
    """
    if not HAS_TORCH:
        return None

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Data
    if op_name == "trivial":
        x = torch.randn(N, N, device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        op = lambda: y.copy_(x)  # noqa: E731

    elif op_name == "transpose":
        x = torch.randn(N, N, device=device, dtype=torch.float32)
        y = torch.empty(N, N, device=device, dtype=torch.float32)
        op = lambda: y.copy_(x.t())  # noqa: E731

    elif op_name == "concat":
        t1 = torch.randn(N, N, device=device, dtype=torch.float32)
        t2 = torch.randn(N, N, device=device, dtype=torch.float32)
        t3 = torch.randn(N, N, device=device, dtype=torch.float32)
        out_tensor = torch.empty(N, 3 * N, device=device, dtype=torch.float32)
        op = lambda: torch.cat((t1, t2, t3), dim=1, out=out_tensor)  # noqa: E731

    elif op_name == "where":
        cond = torch.randint(0, 2, (N, N), device=device, dtype=torch.bool)
        x = torch.randn(N, N, device=device, dtype=torch.float32)
        y = torch.randn(N, N, device=device, dtype=torch.float32)
        out_tensor = torch.empty_like(x)
        op = lambda: torch.where(cond, x, y, out=out_tensor)  # noqa: E731

    elif op_name == "topk":
        k = 4
        x = torch.randn(N, N, device=device, dtype=torch.float32)
        values = torch.empty(N, k, device=device, dtype=torch.float32)
        indices = torch.empty(N, k, device=device, dtype=torch.long)
        op = lambda: torch.topk(x, k, out=(values, indices))  # noqa: E731
    else:
        return None

    # Warmup
    for _ in range(warmup):
        op()

    if device.type == "cuda":
        torch.cuda.synchronize()

    #  Benchmarking
    if device.type == "cuda":
        # GPU Timing (Asynchronous)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_repeats):
            op()
        end_event.record()
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)

    else:
        # CPU Timing (Synchronous)
        start_time = time.perf_counter()
        for _ in range(num_repeats):
            op()
        end_time = time.perf_counter()
        total_ms = (end_time - start_time) * 1000.0

    return total_ms / num_repeats


def run_cpp_benchmark(executable_path, args):
    """
    Runs the compiled executable with arguments.

    Parameters:
    - executable_path: Path to the compiled C++ executable.
    - args: List of arguments to pass to the executable.
    """
    if not os.path.exists(executable_path):
        print(f"Error: Executable '{executable_path}' not found after build")
        return

    N = args[0]

    try:
        # Construct the command
        cmd = [executable_path] + [str(a) for a in args]

        # Run and capture output for parsing
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        kernel_match = re.search(r"TIME_KERNEL_MS:\s+(\d+\.?\d*)", output)
        total_match = re.search(r"TIME_TOTAL_MS:\s+(\d+\.?\d*)", output)

        if kernel_match and total_match:
            return float(kernel_match.group(1)), float(total_match.group(1))
        else:
            print(f"Output parsing failed for size {N}x{N}.")
            print("Printing raw output from the cpp executable")
            print(output)
            return None

    except subprocess.CalledProcessError as e:
        print(f"Execution failed with return code {e.returncode}")
        print("Stderr:", e.stderr)


def main(gpu=False):
    """
    Main function to build and run benchmarks.
    1. Builds the kernel tests (CPU or GPU based on flag).
    2. Runs benchmarks for each operation and size.
    3. Compares C++ results with PyTorch if available.

    Parameters:
    - gpu: Boolean flag to indicate whether to benchmark GPU executables.
    """
    if gpu:
        print(f"\n{'Build System':^100}")
        print("-" * 100)

        # Build Phase
        if not build_kernel_tests_gpu():
            sys.exit(1)

        # Benchmarking System
        device_name = "CPU"
        if HAS_TORCH and torch.cuda.is_available():
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"

        print(f"\n{'Benchmarking System':^100}")
        if HAS_TORCH:
            print(f"{f'PyTorch Device: {device_name}':^100}")
        print("-" * 100)

        for EXECUTABLE_PATH in EXECUTABLE_PATHS_GPU:
            op_name = get_op_name(EXECUTABLE_PATH)
            print(f"Operation: {op_name.upper()}")

            # Flexible Headers
            # K = Kernel Time, T = Total Time
            if HAS_TORCH:
                header = (
                    f"{'SIZE':<6} | {'CUDA(K)':<9} | {'CUDA(T)':<9} | {'TORCH':<9} | "
                    f"{'CUDA GB/s':<9} | {'TORCH GB/s':<11} | {'SPEEDUP':<8}"
                )
            else:
                header = f"{'SIZE':<6} | {'CUDA(K)':<10} | {'CUDA(T)':<10} | {'CUDA GB/s':<12}"

            print(header)
            print("-" * len(header))

            for N in BENCHMARK_SIZES:
                # 1. Run C++ Benchmark
                cpp_res = run_cpp_benchmark(EXECUTABLE_PATH, [N])
                if cpp_res:
                    cpp_k_ms, cpp_t_ms = cpp_res
                else:
                    cpp_k_ms, cpp_t_ms = None, None

                # 2. Run PyTorch Benchmark
                torch_ms = run_pytorch_benchmark(op_name, N) if HAS_TORCH else None

                # 3. Calculate Bandwidth (Using Kernel Time)
                total_bytes = 0.0
                if op_name == "trivial":
                    total_bytes = 8 * N * N
                elif op_name == "transpose":
                    total_bytes = 8 * N * N
                elif op_name == "concat":
                    total_bytes = 24 * N * N
                elif op_name == "where":
                    total_bytes = 13 * N * N
                elif op_name == "topk":
                    total_bytes = 4 * N * N + 16 * N

                # GB/s = (Bytes/1e9) / (ms/1000)
                cpp_bw = (
                    (total_bytes / 1e9) / (cpp_k_ms / 1000.0)
                    if (cpp_k_ms and cpp_k_ms > 0)
                    else 0.0
                )
                torch_bw = (
                    (total_bytes / 1e9) / (torch_ms / 1000.0)
                    if (torch_ms and torch_ms > 0)
                    else 0.0
                )

                # Formatting
                c_k_str = f"{cpp_k_ms:.4f}" if cpp_k_ms else "ERR"
                c_t_str = f"{cpp_t_ms:.4f}" if cpp_t_ms else "ERR"
                c_bw_str = f"{cpp_bw:.2f}" if cpp_k_ms else "-"

                if HAS_TORCH:
                    t_ms_str = f"{torch_ms:.4f}" if torch_ms else "ERR"
                    t_bw_str = f"{torch_bw:.2f}" if torch_ms else "-"

                    # Compare PyTorch Time vs C++ Kernel Time
                    speedup_str = "-"
                    if cpp_k_ms and torch_ms and cpp_k_ms > 0:
                        ratio = torch_ms / cpp_k_ms
                        speedup_str = f"{ratio:.2f}x"

                    print(
                        f"{N:<6} | {c_k_str:<9} | {c_t_str:<9} | {t_ms_str:<9} | "
                        f"{c_bw_str:<9} | {t_bw_str:<11} | {speedup_str:<8}"
                    )
                else:
                    print(f"{N:<6} | {c_k_str:<10} | {c_t_str:<10} | {c_bw_str:<12}")

            print("-" * len(header))
    else:
        # Build System
        print(f"\n{'Build System':^100}")
        print("-" * 100)
        if not build_kernel_tests_cpu():
            sys.exit(1)

        # Benchmarking System
        device_name = "CPU"
        if HAS_TORCH and torch.cuda.is_available():
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"

        print(f"\n{'Benchmarking System':^100}")
        if HAS_TORCH:
            print(f"{f'PyTorch Device: {device_name}':^100}")
        print("-" * 100)

        for EXECUTABLE_PATH in EXECUTABLE_PATHS_CPU:
            op_name = get_op_name(EXECUTABLE_PATH)
            print(f"Operation: {op_name.upper()}")

            # Flexible Headers
            # K = Kernel Time, T = Total Time
            if HAS_TORCH:
                header = (
                    f"{'SIZE':<6} | {'CPU(K)':<9} | {'CPU(T)':<9} | {'TORCH':<9} | "
                    f"{'CPU GB/s':<9} | {'TORCH GB/s':<11} | {'SPEEDUP':<8}"
                )
            else:
                header = (
                    f"{'SIZE':<6} | {'CPU(K)':<10} | {'CPU(T)':<10} | {'CPU GB/s':<12}"
                )

            print(header)
            print("-" * len(header))

            for N in BENCHMARK_SIZES:
                # 1. Run C++ Benchmark
                cpp_res = run_cpp_benchmark(EXECUTABLE_PATH, [N])
                if cpp_res:
                    cpp_k_ms, cpp_t_ms = cpp_res
                else:
                    cpp_k_ms, cpp_t_ms = None, None

                # 2. Run PyTorch Benchmark
                torch_ms = run_pytorch_benchmark(op_name, N) if HAS_TORCH else None

                # 3. Calculate Bandwidth (Using Kernel Time)
                total_bytes = 0.0
                if op_name == "trivial":
                    total_bytes = 8 * N * N
                elif op_name == "transpose":
                    total_bytes = 8 * N * N
                elif op_name == "concat":
                    total_bytes = 24 * N * N
                elif op_name == "where":
                    total_bytes = 13 * N * N
                elif op_name == "topk":
                    total_bytes = 4 * N * N + 16 * N

                # GB/s = (Bytes/1e9) / (ms/1000)
                cpp_bw = (
                    (total_bytes / 1e9) / (cpp_k_ms / 1000.0)
                    if (cpp_k_ms and cpp_k_ms > 0)
                    else 0.0
                )
                torch_bw = (
                    (total_bytes / 1e9) / (torch_ms / 1000.0)
                    if (torch_ms and torch_ms > 0)
                    else 0.0
                )

                # Formatting
                c_k_str = f"{cpp_k_ms:.4f}" if cpp_k_ms else "ERR"
                c_t_str = f"{cpp_t_ms:.4f}" if cpp_t_ms else "ERR"
                c_bw_str = f"{cpp_bw:.2f}" if cpp_k_ms else "-"

                if HAS_TORCH:
                    t_ms_str = f"{torch_ms:.4f}" if torch_ms else "ERR"
                    t_bw_str = f"{torch_bw:.2f}" if torch_ms else "-"

                    # Compare PyTorch Time vs C++ Kernel Time
                    speedup_str = "-"
                    if cpp_k_ms and torch_ms and cpp_k_ms > 0:
                        ratio = torch_ms / cpp_k_ms
                        speedup_str = f"{ratio:.2f}x"

                    print(
                        f"{N:<6} | {c_k_str:<9} | {c_t_str:<9} | {t_ms_str:<9} | "
                        f"{c_bw_str:<9} | {t_bw_str:<11} | {speedup_str:<8}"
                    )
                else:
                    print(f"{N:<6} | {c_k_str:<10} | {c_t_str:<10} | {c_bw_str:<12}")

            print("-" * len(header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark runner")
    parser.add_argument(
        "--gpu",
        help="Flag to indicate whether to benchmark GPU executables",
        action="store_true",
    )
    args = parser.parse_args()

    main(args.gpu)
