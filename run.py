import subprocess
import sys
import os
import re
import argparse

# Configuration
EXECUTABLE_PATHS_CPU = [
    "./bin/test_transpose.out",
    "./bin/test_concat.out",
    "./bin/test_where.out",
    "./bin/test_topk.out"
]

EXECUTABLE_PATHS_GPU = [
    "./build/test_transpose",
    "./build/test_concat",
    "./build/test_where",
    "./build/test_topk"
]

BENCHMARK_SIZES = [
    512,
    1024,
    2048,
    4096,
]

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


def run_benchmark(executable_path, args):
    """
    Runs the compiled executable with arguments.
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


def main(gpu = False):
    if gpu:
        # Build Phase
        if not build_kernel_tests_gpu():
            sys.exit(1)

        print("Bandwidth calculated based on kernel execution time only")

        # Benchmark Phase
        for EXECUTABLE_PATH in EXECUTABLE_PATHS_GPU:
            print(f"Benchmarking {EXECUTABLE_PATH}")
            print(f"{'SIZE (NxN)':<12} | {'KERNEL (ms)':<12} | {'TOTAL (ms)':<12} | {'BANDWIDTH (GB/s)':<18}")
            print("-" * 65)

            for N in BENCHMARK_SIZES:
                res = run_benchmark(EXECUTABLE_PATH, [N])
                if res:
                    k_ms, t_ms = res

                    # Bandwidth Calculation (approximate)
                    total_bytes = 0.0

                    if EXECUTABLE_PATH == "./build/test_transpose":
                        total_bytes = 8 * N * N
                    elif EXECUTABLE_PATH == "./build/test_concat":
                        total_bytes = 24 * N * N
                    elif EXECUTABLE_PATH == "./build/test_where":
                        total_bytes = 13 * N * N
                    elif EXECUTABLE_PATH == "./build/test_topk":
                        k = 4
                        total_bytes = 4 * N * N + 4 * N * k

                    # GB/s = (Bytes / 1e9) / (Seconds)
                    # Time is in ms, so divide by 1000.0
                    if k_ms > 0:
                        bandwidth_gbs = (total_bytes / 1e9) / (k_ms / 1000.0)
                    else:
                        bandwidth_gbs = 0.0

                    print(f"{N:<12} | {k_ms:<12.4f} | {t_ms:<12.4f} | {bandwidth_gbs:<18.4f}")

            print("-" * 65)

            if EXECUTABLE_PATH != EXECUTABLE_PATHS_GPU[-1]:
                print("")
    else:
        # Build Phase
        if not build_kernel_tests_cpu():
            sys.exit(1)

        print("Bandwidth calculated based on kernel execution time only")

        # Benchmark Phase
        for EXECUTABLE_PATH in EXECUTABLE_PATHS_CPU:
            print(f"Benchmarking {EXECUTABLE_PATH}")
            print(f"{'SIZE (NxN)':<12} | {'KERNEL (ms)':<12} | {'TOTAL (ms)':<12} | {'BANDWIDTH (GB/s)':<18}")
            print("-" * 65)

            for N in BENCHMARK_SIZES:
                res = run_benchmark(EXECUTABLE_PATH, [N])
                if res:
                    k_ms, t_ms = res

                    # Bandwidth Calculation (approximate)
                    total_bytes = 0.0

                    if EXECUTABLE_PATH == "./bin/test_transpose.out":
                        total_bytes = 8 * N * N
                    elif EXECUTABLE_PATH == "./bin/test_concat.out":
                        total_bytes = 24 * N * N
                    elif EXECUTABLE_PATH == "./bin/test_where.out":
                        total_bytes = 13 * N * N
                    elif EXECUTABLE_PATH == "./bin/test_topk.out":
                        k = 4
                        total_bytes = 4 * N * N + 4 * N * k

                    # GB/s = (Bytes / 1e9) / (Seconds)
                    # Time is in ms, so divide by 1000.0
                    if k_ms > 0:
                        bandwidth_gbs = (total_bytes / 1e9) / (k_ms / 1000.0)
                    else:
                        bandwidth_gbs = 0.0

                    print(f"{N:<12} | {k_ms:<12.4f} | {t_ms:<12.4f} | {bandwidth_gbs:<18.4f}")

            print("-" * 65)

            if EXECUTABLE_PATH != EXECUTABLE_PATHS_CPU[-1]:
                print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark runner')
    parser.add_argument('--gpu', help='Description for foo argument', action='store_true')
    args = parser.parse_args()

    main(args.gpu)
