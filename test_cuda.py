from numba import cuda
import numpy as np

# Define a simple CUDA kernel
@cuda.jit
def add_kernel(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

# Set the compute capability (adjust for your GPU if necessary)
cuda.select_device(0)
device = cuda.get_current_device()
cc = device.compute_capability

# Check the compute capability and set options accordingly
if cc == (8, 6):  # For NVIDIA A40
    cuda.jit(target='cuda', ptxas_flags='-arch=sm_86')

N = 1000
a = np.ones(N, dtype=np.float32)
b = np.ones(N, dtype=np.float32)
c = np.zeros(N, dtype=np.float32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

threads_per_block = 128
blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

d_c.copy_to_host(c)
print(c[:10])
