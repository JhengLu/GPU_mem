import numpy as np
from numba import cuda
import time

# Define the kernel that accesses memory on the GPU using pointer chasing
@cuda.jit
def pointer_chase(ptr, iters, out):
    idx = cuda.grid(1)
    if idx < ptr.size:
        value = 0
        for i in range(iters):
            value = ptr[value]
        out[idx] = value

def prepare_pointer_chase_data(n):
    data = np.arange(n, dtype=np.int32)
    np.random.shuffle(data)
    return data

def measure_gpu_memory_latency(data, iters):
    threads_per_block = 256
    blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block

    # Allocate memory on the GPU
    d_data = cuda.to_device(data)
    d_out = cuda.device_array(data.size, dtype=np.int32)

    # Check if data is on GPU memory
    if cuda.is_cuda_array(d_data):
        print("Data is on GPU memory.")
    else:
        print("Data is on CPU (pinned) memory.")

    # Warm-up: Perform a dummy kernel launch to initialize the GPU
    pointer_chase[blocks_per_grid, threads_per_block](d_data, iters, d_out)
    cuda.synchronize()

    # Measure latency using CUDA events
    start = cuda.event()
    end = cuda.event()

    start.record()
    pointer_chase[blocks_per_grid, threads_per_block](d_data, iters, d_out)
    end.record()
    end.synchronize()

    latency = cuda.event_elapsed_time(start, end) * 1e6  # Latency in nanoseconds
    return latency / iters  # Average latency per access

def measure_gpu_access_cpu_pinned_latency(data, iters):
    threads_per_block = 256
    blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block

    # Allocate pinned memory on the CPU
    d_data = cuda.pinned_array(data.shape, dtype=data.dtype)
    d_data[:] = data[:]
    d_out = cuda.device_array(data.size, dtype=np.int32)

    # Check if data is on GPU memory
    if cuda.is_cuda_array(d_data):
        print("Data is on GPU memory.")
    else:
        print("Data is on CPU (pinned) memory.")

    # Warm-up: Perform a dummy kernel launch to initialize the GPU
    pointer_chase[blocks_per_grid, threads_per_block](d_data, iters, d_out)
    cuda.synchronize()

    # Measure latency using CUDA events
    start = cuda.event()
    end = cuda.event()

    start.record()
    pointer_chase[blocks_per_grid, threads_per_block](d_data, iters, d_out)
    end.record()
    end.synchronize()

    latency = cuda.event_elapsed_time(start, end) * 1e6  # Latency in nanoseconds
    return latency / iters  # Average latency per access

if __name__ == '__main__':
    cuda.select_device(0)
    device = cuda.get_current_device()
    print(f"Compute Capability: {device.compute_capability}")

    iters = 10000000  # Number of iterations for pointer chasing

    # Test with different values of n
    for n in [100, 1000, 10000, 100000, 1000000]:
        print(f"\nNumber of elements: {n}")
        data = prepare_pointer_chase_data(n)

        # Measure latency for accessing GPU memory
        print("Measuring GPU memory access latency...")
        gpu_latency = measure_gpu_memory_latency(data, iters)
        print(f"GPU Memory Access Latency: {gpu_latency:.6f} ns")

        # # Measure latency for GPU accessing CPU pinned memory
        # print("Measuring GPU access to CPU (pinned) memory latency...")
        # cpu_pinned_latency = measure_gpu_access_cpu_pinned_latency(data, iters)
        # print(f"GPU Access to CPU (pinned) Memory Latency: {cpu_pinned_latency:.6f} ns")
