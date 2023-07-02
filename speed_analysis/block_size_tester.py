import numpy as np
import cupy as cp
import quflow as qf
from cupyx.profiler import benchmark
import json


test_data = {}

def test_block_size():
    # ------- Testing block size for skew-herm cuThomas --------- #
    block_size_data = {}
    current_device_prop = cp.cuda.runtime.getDeviceProperties(0)
    block_size_data["device_name"] = f"Device 0: {str(current_device_prop['name'])}:"
    block_size_data["sm_count"] = f"Multiprocessor count: {current_device_prop['multiProcessorCount']}"

    print(block_size_data["device_name"])
    print(block_size_data["sm_count"])

    N_list = np.array([4096,5120,6144]) # Size of matrices <---- Needs
    #N_list = np.array([100,200,300])
    #N = 2048 # Size of matrix
    n_repeat = 100

    # Cupy arrays

    block_sizes = np.array([32,64,96,128,256,384,512,768,1024])
    block_timings = np.zeros((len(N_list),len(block_sizes)))

    for (i,N) in enumerate(N_list):
        W0_cp = qf.gpu.utils.get_random_mat_cp(N)
        P0_cp = cp.zeros_like(W0_cp)

        ham_cp = qf.gpu.gpu_core.solve_poisson_interleaved_cp(N)
        for (j,b_size) in enumerate(block_sizes):
            block_timings[i,j] = benchmark(ham_cp.solve_poisson,(W0_cp,P0_cp,b_size),n_repeat=n_repeat).gpu_times[0].mean()*1e6

        del W0_cp
        del P0_cp
        del ham_cp

        print(f"Best sm/block for N = {N} : {block_sizes[np.argmin(block_timings[i,:])]}")

    print(block_timings)
    

    block_size_data["block_sizes"] = block_sizes.tolist()
    block_size_data["block_timings"] = block_timings.tolist()
    block_size_data["N_list"] = N_list.tolist()
    return block_size_data


test_data["block_size"]  = test_block_size()
with open("results/block_size_test.json","w") as outfile:
    json.dump(test_data,outfile,indent=4)