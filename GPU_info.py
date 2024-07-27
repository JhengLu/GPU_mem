import pycuda.driver as cuda
import pycuda.autoinit

# Initialize CUDA
cuda.init()

# Get the number of devices
num_devices = cuda.Device.count()
print(f"Number of CUDA devices: {num_devices}")

# Iterate through devices and get properties
for i in range(num_devices):
    device = cuda.Device(i)
    name = device.name()
    compute_capability = device.compute_capability()
    total_memory = device.total_memory()
    num_sm = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    
    print(f"\nDevice {i}: {name}")
    print(f"  Compute Capability: {compute_capability}")
    print(f"  Total Memory: {total_memory / (1024 ** 2)} MB")
    print(f"  Number of SMs: {num_sm}")

    # Get device properties
    attributes = device.get_attributes()
    
    # L2 Cache Size
    l2_cache_size = attributes.get(cuda.device_attribute.L2_CACHE_SIZE, 'N/A')
    if l2_cache_size != 'N/A':
        l2_cache_size = l2_cache_size / 1024  # Convert to KB
    print(f"  L2 Cache Size: {l2_cache_size} KB")

    # Shared Memory Size
    max_shared_memory_per_block = attributes.get(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK, 'N/A')
    max_shared_memory_per_multiprocessor = attributes.get(cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, 'N/A')
    
    # For compute capabilities 2.x and higher, there might be specific attributes for L1 cache
    l1_cache_size = 'N/A'
    if compute_capability[0] >= 2:
        shared_memory_per_block_optin = attributes.get(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 'N/A')
        if shared_memory_per_block_optin != 'N/A':
            shared_memory_per_block_optin = shared_memory_per_block_optin / 1024  # Convert to KB
        print(f"  Max Shared Memory Per Block (Opt-in): {shared_memory_per_block_optin} KB")

        l1_cache_size = max_shared_memory_per_multiprocessor / 1024  # Assuming shared memory and L1 cache share the same space
        print(f"  Max Shared Memory Per Block: {max_shared_memory_per_block / 1024} KB")
        print(f"  Max Shared Memory Per Multiprocessor: {l1_cache_size} KB")

    # Calculate total L1 cache size
    if l1_cache_size != 'N/A':
        total_l1_cache_size = l1_cache_size * num_sm
        print(f"  Total L1 Cache Size: {total_l1_cache_size} KB")

    if cuda.device_attribute.GLOBAL_L1_CACHE_SUPPORTED in attributes:
        print(f"  Global L1 Cache Supported: {attributes[cuda.device_attribute.GLOBAL_L1_CACHE_SUPPORTED]}")
        
    if cuda.device_attribute.LOCAL_L1_CACHE_SUPPORTED in attributes:
        print(f"  Local L1 Cache Supported: {attributes[cuda.device_attribute.LOCAL_L1_CACHE_SUPPORTED]}")
