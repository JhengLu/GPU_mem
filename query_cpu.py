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

# Load CPU Index
index_cpu = faiss.read_index('./index_storage/index_cpu.faiss')

# Number of nearest neighbors to retrieve
k = 5

while True:
    # Perform CPU search and measure latency
    start_time = time.time()
    D_cpu, I_cpu = index_cpu.search(xq, k)
    end_time = time.time()

    # Output the results and latency
    print("CPU Distances:\n", D_cpu)
    print("CPU Indices:\n", I_cpu)
    print(f"CPU Search Latency: {end_time - start_time:.6f} seconds")

    
