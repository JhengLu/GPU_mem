import os
import numpy as np
import faiss
import time

# Generate some random query vectors
d = 128  # dimension
nq = 10  # number of queries
np.random.seed(1234)

# Query vectors
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

try:
    # Load GPU Index
    res = faiss.StandardGpuResources()
    index_gpu = faiss.read_index('./index_storage/index_gpu.faiss')
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_gpu)

    # Number of nearest neighbors to retrieve
    k = 5

    while True:
        # Perform GPU search and measure latency
        start_time = time.time()
        D_gpu, I_gpu = index_gpu.search(xq, k)
        end_time = time.time()

        # Output the results and latency
        print("GPU Distances:\n", D_gpu)
        print("GPU Indices:\n", I_gpu)
        print(f"GPU Search Latency: {end_time - start_time:.6f} seconds")

     

except AttributeError:
    print("FAISS GPU resources not available. Ensure you have faiss-gpu installed.")
except Exception as e:
    print(f"An error occurred while performing GPU search: {e}")

