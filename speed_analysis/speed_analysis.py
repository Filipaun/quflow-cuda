import quflow as qf
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import csv
import time as time_pack
import json

test_data = {}

# Size of matrices
N_samples = np.array([30, 40,60,80,100,120,140,160,180,200,250,300,350,400])  
timings = np.zeros((2,len(N_samples)),dtype = float)

time = 3.0 # in second
inner_time = 0.5 # in seconds
qstepsize = 0.2 # in qtime

steps = 2200 
inner_steps = 200

lmax = 10  # How many spherical harmonics (SH) coefficients to include
np.random.seed(42)  # For reproducability
omega0 = np.random.randn(lmax**2)  # Array with SH coefficients

print(f"N to run: {N_samples}")
print("################")

for (i,N) in enumerate(N_samples):
    print(f"Running N={N}")

    W0 = qf.shr2mat(omega0, N=N)  # Convert SH coefficients to matrix

    # GPU
    print(f"GPU, N = {N}")
    solver_gpu = qf.gpu.isomp_gpu_skewherm_solver(W0)
    ham = qf.gpu.solve_poisson_cp(N)
    method_kwargs = {"hamiltonian": ham, "verbatim":False, "maxit":7, "tol":1e-8}

    start_time_ns = time_pack.time_ns()

    qf.solve(W_gpu, qstepsize=qstepsize, time=time, inner_time=inner_time,
         method=solver_gpu.solve_step, method_kwargs=method_kwargs)
    
    end_time_ns = time_pack.time_ns()
    timings[0,i] = (end_time_ns - start_time_ns)*1e-9

    # CPU
    print(f"CPU, N = {N}")
    W_cpu = W0.copy()
    solver_cpu = qf.isomp_fixedpoint
    dt = qf.qtime2seconds(qstepsize, N)

    start_time_ns = time_pack.time_ns()

    qf.solve(W_cpu, qstepsize=qstepsize, time=time, inner_time=inner_time,
         method=solver_cpu, method_kwargs={"verbatim":False, "maxit":7, "tol":1e-8})
    
    end_time_ns = time_pack.time_ns()
    timings[1,i] = (end_time_ns - start_time_ns)*1e-9

test_data["n_samples"] = N_samples.tolist()
test_data["timings"] = timings.tolist()
test_data["time"] = time
test_data["inner_time"] = inner_time
test_data["qstepsize"] = qstepsize

test_data_json = json.dumps(test_data, indent=4)

# TODO: Write to file underway, incase of crash
with open("speed_test.json","w") as outfile:
    outfile.write(test_data_json)