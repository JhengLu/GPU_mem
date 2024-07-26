import os
import numpy as np
from numba import cuda
import time

# Define the kernel that accesses memory on the GPU
@cuda.jit
def access_memory_gpu(data):
    idx = cuda.grid(1)
    if idx < data.size:
        value = data[idx]

# Function to measure latency of accessing CPU (local DRAM) memory
def measure_cpu_memory_latency(data):
    start_time = time.time()
    for i in range(data.size):
        value = data[i]
    end_time = time.time()
    latency = (end_time - start_time) / data.size * 1e9  # Latency in nanoseconds
    return latency

# Function to measure latency of accessing GPU memory
def measure_gpu_memory_latency(data, use_gpu_memory=True):
    threads_per_block = 256
    blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block  # Ensure full utilization

    # Ensure a minimum grid size for better utilization
    min_blocks = 128  # You can adjust this based on your GPU's SM count
    blocks_per_grid = max(blocks_per_grid, min_blocks)

    # Allocate memory on the GPU if specified
    if use_gpu_memory:
        d_data = cuda.to_device(data)
    else:
        d_data = cuda.pinned_array(data.shape, dtype=data.dtype)
        d_data[:] = data[:]

    # Check if data is on GPU memory
    if cuda.is_cuda_array(d_data):
        print("Data is on GPU memory.")
    else:
        print("Data is on CPU (pinned) memory.")

    # Warm-up: Perform a dummy kernel launch to initialize the GPU
    access_memory_gpu[blocks_per_grid, threads_per_block](d_data)
    cuda.synchronize()

    # Measure latency using CUDA events
    start = cuda.event()
    end = cuda.event()

    start.record()
    access_memory_gpu[blocks_per_grid, threads_per_block](d_data)
    end.record()
    end.synchronize()

    latency = cuda.event_elapsed_time(start, end) * 1e6  # Latency in nanoseconds
    return latency

if __name__ == '__main__':
    cuda.select_device(0)
    device = cuda.get_current_device()
    print(f"Compute Capability: {device.compute_capability}")

    # Test with different values of n
    for n in [100, 1000, 10000, 100000, 1000000]:
        print(f"\nNumber of elements: {n}")
        data = np.random.random(n).astype(np.float32)

        # Measure latency for accessing CPU memory
        print("Measuring CPU memory access latency...")
        cpu_latency = measure_cpu_memory_latency(data)
        print(f"CPU Memory Access Latency: {cpu_latency:.6f} ns")

        # Measure latency for accessing GPU memory
        print("Measuring GPU memory access latency...")
        gpu_latency = measure_gpu_memory_latency(data, use_gpu_memory=True)
        print(f"GPU Memory Access Latency: {gpu_latency:.6f} ns")

        # Measure latency for accessing CPU pinned memory with GPU
        print("Measuring GPU access to CPU (pinned) memory latency...")
        pinned_cpu_latency = measure_gpu_memory_latency(data, use_gpu_memory=False)
        print(f"GPU Access to CPU (pinned) Memory Latency: {pinned_cpu_latency:.6f} ns")
