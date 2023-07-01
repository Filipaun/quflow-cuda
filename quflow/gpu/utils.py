import cupy as cp

def get_random_mat_cp(N=5,seed=None):
    if seed != None:
        cp.random.seed(seed)

    W = cp.random.randn(N, N) + 1j*cp.random.randn(N, N)
    W -= W.conj().T
    W -= cp.eye(N)*cp.trace(W)/N
    return W