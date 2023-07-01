import numpy as np
import cupy as cp
import quflow as qf
from cupyx.profiler import benchmark
import json


test_data = {}

# ------- Testing block size for skew-herm cuThomas --------- #

current_device_prop = cp.cuda.runtime.getDeviceProperties(0)
test_data["device_name"] = f"Device 0: {str(current_device_prop['name'])}:"
test_data["sm_count"] = f"Multiprocessor count: {current_device_prop['multiProcessorCount']}"

print(test_data["device_name"])
print(test_data["sm_count"])

N = 4096 # Size of matrices <---- Needs 
#N = 2048 # Size of matrix
n_repeat = 1000

# Cupy arrays
W0_cp = qf.gpu.utils.get_random_mat_cp(N)

P0_c = cp.zeros_like(W0_cp)
ham_c = qf.gpu.gpu_core.solve_poisson_interleaved_cp(N)


block_sizes = np.array([32,64,96,128,256,384,512,768,1024])
block_timings = np.zeros_like(block_sizes)

for (i,b_size) in enumerate(block_sizes):
    block_timings[i] = benchmark(ham_c.solve_poisson,(W0_cp,P0_c,b_size),n_repeat=n_repeat).gpu_times[0].mean()*1e6

print(block_timings)
print(f"Best sm/block : {block_sizes[np.argmin(block_timings)]}")

test_data["block_sizes"] = block_sizes.tolist()
test_data["block_timings"] = block_timings.tolist()

with open("results/sm_test.json","w") as outfile:
    json.dump(test_data,outfile,indent=4)