#include <cuda_runtime.h>
#include <nvml.h>
#include <unistd.h>

#include <alpaka/alpaka.hpp>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

#include "gemm_simple_alpaka.hxx"

// Function to get host memory usage in KB (Linux)
long getHostMemoryKB() {
    std::ifstream statm("/proc/self/statm");
    long size = 0, resident = 0;
    statm >> size >> resident;
    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    return resident * page_size_kb;
}

// Function to get current GPU usage in MB
size_t getGpuUsedMB() {
    nvmlDevice_t device;
    nvmlMemory_t memInfo;
    nvmlDeviceGetHandleByIndex(0, &device);
    nvmlDeviceGetMemoryInfo(device, &memInfo);
    return memInfo.used / (1024 * 1024);
}

int main() {
    using Idx = std::size_t;
    using Dim = alpaka::DimInt<1>;
    using Ext1D = alpaka::Vec<Dim, Idx>;

    // Init NVML once
    nvmlInit();

    // session
    SOFIE_gemm_simple::Session<alpaka::TagGpuCudaRt> session;

    // Host device
    alpaka::PlatformCpu hostPlatform{};
    auto host = alpaka::getDevByIdx(hostPlatform, 0u);

    // Allocate host buffer
    auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{4}));
    float* A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

    // Fill host buffer with random floats
    for (Idx i = 0; i < 4; ++i) {
        A_ptr[i] = 1.0;
    }

    // GPU device + queue
    alpaka::PlatformCudaRt platform{};
    alpaka::DevCudaRt device = alpaka::getDevByIdx(platform, 0u);
    alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue{device};

    // Allocate device buffer
    auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{4}));
    alpaka::memcpy(queue, A_d, A);
    alpaka::wait(queue);

    // Force CUDA context init before measuring
    cudaDeviceSynchronize();

    // Measure host memory before inference
    long host_mem_before = getHostMemoryKB();

    // Track GPU usage in background
    std::atomic<bool> running{true};
    size_t peak_usage = getGpuUsedMB();
    std::thread monitor([&]() {
        while (running) {
            size_t used = getGpuUsedMB();
            if (used > peak_usage) {
                peak_usage = used;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // initial warm-up
    auto result = session.infer(A_d);

    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    result = session.infer(A_d);
    alpaka::wait(queue);  // ensure GPU finishes
    auto end = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();

    // printing output
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{4}));
    alpaka::memcpy(queue, result_h, result);
    alpaka::wait(queue);

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));

    std::cout << "Model output:\n";
    for (Idx i = 0; i < 4; ++i) {
        std::cout << res_ptr[i] << " ";
    }
    std::cout << "\n---------------------------------\n";

    // Stop monitoring
    running = false;
    monitor.join();

    // Measure host memory after inference
    long host_mem_after = getHostMemoryKB();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Inference time: " << elapsed.count() << " ms\n";

    std::cout << "Peak GPU memory used by program: " << peak_usage << " MB\n";
    std::cout << "Host memory change during inference: " << (host_mem_after - host_mem_before) << " KB\n";

    // Shutdown NVML
    nvmlShutdown();

    return 0;
}
