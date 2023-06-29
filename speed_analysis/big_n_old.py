import numpy as np
import cupy as cp
import h5py
import quflow as qf
import argparse
import time as time_pack

# ---------- Plotting ------------ #
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ------------------- DEFAULT PARAMETERS ----------------- #
# Size of matrices

default_N = 2048
#default_N = 4096  # <---- Allgedly needs 171 GiB of memory for computing basis

# Time parameters
default_time = 60 # in second
default_inner_time = 0.5 # in seconds
default_qstepsize = 0.25 # in qtime


def init_argparse() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Runs Quflow-cuda script"
    )
    parser.add_argument('-n','--matrix-dim-n', dest='N', type=int, nargs=1,
                        default=[default_N],
                        help='Dimension N of matrix W')
    parser.add_argument('-t','--time', dest='time', type=float, nargs=1,
                        default=[default_time],
                        help='Time in seconds')
    parser.add_argument('-it','--inner-time', dest='inner_time', type=float, nargs=1,
                        default=[default_inner_time],
                        help='Inner time in second')
    parser.add_argument('-qs','--qstepsize', dest='qstepsize', type=float, nargs=1,
                        default=[default_qstepsize],
                        help='Qtime of steps')

    return parser

def main() -> None:
    """ 
    Runs a simulation with quflow using GPU
    
    When running from terminal the file has options:
        -n: int, dimension of W,
        -t: float, time duration of sim,
        -it: time step of simulation,
        -qs: qtime of steps
    """
    # ------------------- ARGPARSE ----------------- #
    parser = init_argparse()
    args = parser.parse_args()

    N = args.N[0]
    time = args.time[0]
    inner_time = args.inner_time[0]
    qstepsize = args.qstepsize[0]

    print(args)

    #dt = qf.qtime2seconds(qstepsize, N)
    #print("The physical stepsize is {:.3e} seconds.".format(dt))
    #print("{} steps per output, in total {} steps.".format(round(inner_time/dt), round(time/dt)))

    # Simulation settings
    lmax = 10  # How many spherical harmonics (SH) coefficients to include
    np.random.seed(42)  # For reproducability
    omega0 = np.random.randn(lmax**2)  # Array with SH coefficients

    shr2mat_start_time_ns = time_pack.time_ns()
    W0 = qf.shr2mat(omega0, N=N)  # Convert SH coefficients to matrix
    shr2mat_elapsed_time_ns = (time_pack.time_ns() - shr2mat_start_time_ns)*1e-9

    filename_gpu = "gpu_test_sim_N_{}.hdf5".format(str(N))

    # Callback data object
    mysim_gpu = qf.QuData(filename_gpu)

    # Save initial conditions if file does not exist already, otherwise load from last step
    try:
        f = h5py.File(filename_gpu, "r")
    except IOError or KeyError:
        W = W0.copy()
        mysim_gpu(W, 0.0)
    else:
        W = qf.shr2mat(f['state'][-1,:], N=N)
        assert W.shape[0] == N, "Looks like the saved data use N = {} whereas you specified N = {}.".format(W.shape[0], N)
        f.close()

    # Select solver
    # The cupy based isomp solver and poisson solver needs to be initialized 
    method = qf.gpu.isomp_gpu_skewherm_solver(W)
    ham = qf.gpu.solve_poisson_interleaved_cp(N)

    # Method kwargs are the same, since we still use the same qf.solve from dynamics.py
    method_kwargs = {"hamiltonian": ham.solve_poisson, "verbatim":False, "maxit":7, "tol":1e-8}

    # Time sim
    sim_start_time_ns = time_pack.time_ns()

    # Run simulation
    qf.solve(W, qstepsize=qstepsize, time=time, inner_time=inner_time, callback=mysim_gpu,
            method=method.solve_step, method_kwargs=method_kwargs)
    
    sim_elapsed_time_ns = (time_pack.time_ns() - sim_start_time_ns)*1e-9
    
    # Flush cache data
    mysim_gpu.flush()

    with open("elapsed_time.txt",'w') as time_file:
        time_file.write(f"{shr2mat_elapsed_time_ns}\n{sim_elapsed_time_ns}")

        # Plot last state
    with h5py.File(filename_gpu, 'r') as data:
        plt.figure()
        omega = data['state'][0]
        qf.plot2(omega, projection='hammer', N=520, colorbar=True)
        plt.savefig(f"sim_{N}_initial.pdf", bbox_inches='tight')

        plt.figure()
        omega = data['state'][-1]
        qf.plot2(omega, projection='hammer', N=520, colorbar=True)
        plt.savefig(f"sim_{N}_end.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main()