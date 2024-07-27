import numpy as np
from numba import cuda
import pycuda.driver as drv
import pycuda.autoinit
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    # blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block
    blocks_per_grid = 84

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

    # Run the kernel multiple times to average out the kernel launch overhead
    num_runs = 10
    latencies = []
    for _ in range(num_runs):
        start.record()
        pointer_chase[blocks_per_grid, threads_per_block](d_data, iters, d_out)
        end.record()
        end.synchronize()
        latency = cuda.event_elapsed_time(start, end) * 1e6  # Latency in nanoseconds
        latencies.append(latency / iters)  # Average latency per access

    avg_latency = np.mean(latencies)
    return avg_latency

def measure_latency_and_plot(iters):
    cuda.select_device(0)
    device = cuda.get_current_device()
    num_sms = device.MULTIPROCESSOR_COUNT
    l1_cache_size_kb = device.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR / 1024

    # Use pycuda to get the L2 cache size
    l2_cache_size_bytes = drv.Context.get_device().get_attribute(drv.device_attribute.L2_CACHE_SIZE)
    l2_cache_size_kb = l2_cache_size_bytes / 1024

    print(f"Compute Capability: {device.compute_capability}")
    print(f"Number of SMs: {num_sms}")
    print(f"L1 Cache Size: {l1_cache_size_kb:.1f} KB per SM")
    print(f"L2 Cache Size: {l2_cache_size_kb:.1f} KB total")
    print(f"Iteration number: {iters}")

    # sizes = [100, 1000, 10000, 20000, 30000, 100000, 1000000, 1500000, 2000000, 3000000, 9000000, 12000000,24000000,36000000]
    sizes = [100, 1000, 10000,20000, 30000, 100000, 1000000,1500000, 2000000,3000000]
    
    latencies = []
    data_sizes = []

    for n in sizes:
        print(f"\nNumber of elements: {n}")
        data = prepare_pointer_chase_data(n)
        
        # Calculate data size in KB
        data_size_kb = data.nbytes / 1024
        data_sizes.append(data_size_kb)
        print(f"Data size: {data_size_kb:.2f} KB")

        # Determine where the data fits
        if data_size_kb <= l1_cache_size_kb:
            print("Data fits within L1 cache per SM")
        elif data_size_kb <= l2_cache_size_kb:
            print("Data fits within L2 cache")
        else:
            print("Data exceeds L2 cache size")

        # Measure latency for accessing GPU memory
        print("Measuring GPU memory access latency...")
        gpu_latency = measure_gpu_memory_latency(data, iters)
        latencies.append(gpu_latency)
        print(f"GPU Memory Access Latency: {gpu_latency:.6f} ns")

        # Calculate and print the number of SMs utilized
        threads_per_block = 256
        # blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block
        blocks_per_grid = 84
        sm_occupancy = min(blocks_per_grid, num_sms)
        print(f"SMs Utilized: {sm_occupancy} out of {num_sms}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, latencies, marker='o', linestyle='-', color='b', label='GPU Memory Latency')
    plt.axvline(x=l1_cache_size_kb, color='r', linestyle='--', label='L1 Cache Size')
    plt.axvline(x=l2_cache_size_kb, color='g', linestyle='--', label='L2 Cache Size')
    plt.xlabel('Data Size (KB)')
    plt.ylabel('Latency (ns)')
    plt.title('GPU Memory Access Latency')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('gpu_memory_latency.png')  # Save the figure to a file
    plt.show()

if __name__ == '__main__':
    measure_latency_and_plot(iters=30000)
