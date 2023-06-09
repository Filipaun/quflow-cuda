import numpy as np
import cupy as cp
import tensorflow as tf
import pkgutil

_tridiagonal_laplacian_cp_cache = dict()

# Check Tensorflow :
gpu_devices = tf.config.list_physical_devices('GPU')
if (len(gpu_devices) > 0):
    #print("Found GPU device!")
    #print(gpu_devices)
    pass
else:
    raise RuntimeError("No GPU found by Tensorflow")

def check_version():  
    #print(f"Tensorflow version: {tf.__version__}")
    print(f"CuPy version: {cp.__version__}")



# Import cuda source code

tridiag_source = pkgutil.get_data(__package__,'tridiag.cu')
tridiag_module = cp.RawModule(code=tridiag_source.decode('utf-8'))

del tridiag_source
# Get non interleaved kernels
mat2diagh_ker = tridiag_module.get_function('mat2diagh')
diagh2mat_ker = tridiag_module.get_function('diagh2mat')

def mat2diagh_cp(lowdiag,dense,N):

    # Wrapper for mat2diagh CUDA kernel
    # Simple grid and block size

    grid_dim = N//44 + 1
    block_dim_x = 44
    block_dim_y = 23

    mat2diagh_ker((grid_dim,grid_dim),(block_dim_x,block_dim_y),(lowdiag,dense,N))
    #cp.cuda.runtime.deviceSynchronize()

    return 0


def diagh2mat_cp(dense,lowdiag,N):

    # Wrapper for mat2diagh CUDA kernel
    # Simple grid and block size

    grid_dim = N//44 + 1
    block_dim_x = 44
    block_dim_y = 23

    diagh2mat_ker((grid_dim,grid_dim),(block_dim_x,block_dim_y),(dense,lowdiag,N))
    #cp.cuda.runtime.deviceSynchronize()

    return 0

#### -------------- Non interleaved using Tensorflow --------- ###
def laplacian_cp(N, bc=False):
    """
    Return quantized laplacian (as a tridiagonal laplacian).

    Parameters
    ----------
    N: int
    bc: bool (optional)
        Whether to include boundary conditions.

    Returns
    -------
    lap : ndarray(shape=(2, N*(N+1)/2), dtype=flaot)
    """
    global _tridiagonal_laplacian_cp_cache

    if (N, bc) not in _tridiagonal_laplacian_cp_cache:
        lap = compute_tridiagonal_laplacian_cp(N, bc=bc)
        _tridiagonal_laplacian_cp_cache[(N, bc)] = lap

    return _tridiagonal_laplacian_cp_cache[(N, bc)]

def dot_tridiagonal_cp(lap, P):
    """
    Dot product for tridiagonal operator.

    Parameters
    ----------
    lap: cdarray(shape(N//2+1, 2, N), dtype=complex128)
        Tridiagonal operator (typically laplacian).
    P: cdarray(shape=(N,N), dtype=complex128)
        Input matrix.

    Returns
    -------
    W: ndarray(shape=(N,N), dtype=complex)
        Output matrix.
    """
    N = P.shape[0]
    W = cp.zeros_like(P)
    Pdiagh = cp.empty((N//2+1,N),dtype='complex128')
    mat2diagh_cp(Pdiagh,P,N)

    Wdiagh = lap[:, 1, :]*Pdiagh
    Wdiagh[:, 1:] += lap[:, 0, :-1]*Pdiagh[:, :-1]
    Wdiagh[:, :-1] += lap[:, 0, :-1]*Pdiagh[:, 1:]

    diagh2mat_cp(W,Wdiagh,N)

    return W

def laplace_cp(P):
    """
    Return quantized laplacian applied to stream function `P`.

    Parameters
    ----------
    P: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    W: ndarray(shape=(N, N), dtype=complex)
    """
    N = P.shape[0]
    lap = laplacian_cp(N)

    # Apply dot product
    W = dot_tridiagonal_cp(lap, P)

    return W


def solve_tridiagonal_cp(lap, W, P, Wdiagh):
    """
    Function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` array). Uses NUMBA to accelerate the
    tridiagonal solver calculations.

    Parameters
    ----------
    lap: ndarray(shape=(N//2+1, 3, N), dtype=float)
        Tridiagonal laplacian.
    W: ndarray(shape=(N, N), dtype=complex)
        Input matrix.
    P: ndarray(shape=(N, N), dtype=complex)
        Output matrix.

    Returns
    -------
    void 
    """
    N = W.shape[0]
    
    # Convert laplacian to tensorflow tensor with dlpack
    # Should point to same memory adress as cupy array
    lap_tf = tf.experimental.dlpack.from_dlpack(lap.toDlpack())

    # Swap values into Wdiagh
    # Pass by reference faster than returning ?
    mat2diagh_cp(Wdiagh,W,N)

    # Convert to tensor
    Wdiagh_tf = tf.experimental.dlpack.from_dlpack(Wdiagh.toDlpack())

    # For each double-tridiagonal, solve a tridiagonal system
    cp.cuda.runtime.deviceSynchronize()
    Pdiagh = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(tf.linalg.tridiagonal_solve(lap_tf, Wdiagh_tf, partial_pivoting = False)))
    cp.cuda.runtime.deviceSynchronize()

    # Make sure we preserve trace of W (corresponds to bc for laplacian)
    trP = Pdiagh[0, :].sum()/N
    Pdiagh[0, :] -= trP 

    # Convert back to dense matrix
    diagh2mat_cp(P,Pdiagh,N)


def compute_tridiagonal_laplacian_cp(N, bc=False):
    """
    Compute tridiagonal laplacian.

    Parameters
    ----------
    N: int
    bc: bool (optional)
        Whether boundary conditions should be added to make the laplacian non-singular.
        Notice that this bc is different from the one used for sparse laplacians.

    Returns
    -------
    lap: cparray, shape (N//2+1, 3, N)
        Outer index: system for diagonal m and N-m.
        Middle index: which diagonal, stored according to tensorflow format (0 upper, 1 main, 2 lower)
        Inner index: entries on the diagonals
    """

    # Tensorflow needs same types on both sides of Ax=B, so lap is complex128
    # Tensorflow needs all 3 diagonals, so axis 1 has size = 3.
    lap = np.zeros((N//2+1, 3, N), dtype=np.complex128)
    i_full = np.arange(N)
    for m in range(N//2+1):

        # Global diagonal m (of length N-m)
        i = i_full[:N - m]
        lap[m, 1, 0:N - m] = -((N - 1)*(2*i + 1 + m) - 2*i*(i + m))
        i = i_full[1:N - m]
        lap[m, 0, 0:N - m - 1] = np.sqrt(((i + m)*(N - i - m))*(i*(N - i)))

        # Global diagonal N-m (of length m)
        i = i_full[:m]
        lap[m, 1, N - m:] = -((N - 1)*(2*i + 1 + N - m) - 2*i*(i + N - m))
        i = i_full[1:m]
        lap[m, 0, N - m:-1] = np.sqrt(((i + N - m)*(m - i))*(i*(N - i)))

        np.copyto(lap[m, 2, 1:],lap[m , 0, 0:-1])

    if bc:
        lap[0, 1, 0] -= 0.5
        pass

    return lap

class solve_poisson_cp:
    def __init__(self,N, bc = True) -> None:
        self.N = N
        self.lap = cp.asarray(laplacian_cp(N, bc = bc))
        self.Wdiagh = cp.empty((N//2+1,N),dtype='complex128')
        #self.Pdiagh = cp.empty((N//2+1,N),dtype='complex128')

    def solve_poisson(self,W,P) -> None:
        """
        Gives stream matrix `P` for `W`.
        #Return stream matrix `P` for `W`.

        Parameters
        ----------
        W: ndarray(shape=(N, N), dtype=complex)

        Returns
        -------
        none
        ##P: ndarray(shape=(N, N), dtype=complex)
        """

        solve_tridiagonal_cp(self.lap, W, P, self.Wdiagh)