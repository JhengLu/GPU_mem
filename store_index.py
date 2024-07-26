import os
import numpy as np
import faiss
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Create and store FAISS index of specified size.')
parser.add_argument('--size_gb', type=int, required=True, help='Size of the index in GB')
args = parser.parse_args()

# Set up directories
os.makedirs('./index_storage', exist_ok=True)

# Parameters
d = 128  # dimension
size_gb = args.size_gb
nb = (size_gb * 10**9) // (d * 4)  # number of vectors

# Generate some random data
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.

# CPU Index
index_cpu = faiss.IndexFlatL2(d)
index_cpu.add(xb)
faiss.write_index(index_cpu, './index_storage/index_cpu.faiss')

try:
    # Check if GPU is available and create GPU resources
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    faiss.write_index(faiss.index_gpu_to_cpu(index_gpu), './index_storage/index_gpu.faiss')
    print("GPU index stored successfully.")
except AttributeError:
    print("FAISS GPU resources not available. Ensure you have faiss-gpu installed.")
except Exception as e:
    print(f"An error occurred while creating the GPU index: {e}")

print(f"Indexing completed and stored. Number of vectors: {nb}, Approximate size: {size_gb}GB.")
