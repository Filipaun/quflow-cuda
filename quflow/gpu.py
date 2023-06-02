import numpy as np
import cupy as cp
import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU')
if (len(gpu_devices) > 0):
    #print("Found GPU device!")
    #print(gpu_devices)
    pass
else:
    raise RuntimeError("No GPU found by Tensorflow")

_tridiagonal_laplacian_cp_cache = dict()

def check_version():  
    print(f"Tensorflow version: {tf.__version__}")
    print(f"CuPy version: {cp.__version__}")

mat_tofrom_diag_module = r'''
#include <cupy/complex.cuh>
extern "C" {

__global__ void mat2diagh(complex<double>* lowdiag_matrix, const complex<double>* dense_matrix,\
                         unsigned int N)
{
    /*
    C based CUDA Kernel equivalent of mat2diagh.
    Return lower diagonal format for skew hermitian matrix W.

    Parameters
    ----------
    lowdiag_matrix : complex<double>*: size (N//2+1)* N
        Lower diagonal form output
    dense_matrix :  const complex<double>*: size N * N
        Dense skew hermitian input matrix.

    Returns
    -------
    void
    */

    unsigned int x_tridiag = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y_tridiag = blockDim.y * blockIdx.y + threadIdx.y;

    int x_dense;
    int y_dense;

    if ((x_tridiag < N) && (y_tridiag < (int)(N/2) + 1))
    {
        if( x_tridiag > N-y_tridiag-1){
            x_dense = x_tridiag - (N - y_tridiag);
            y_dense = x_dense + N - y_tridiag;
        }   
        else {
            x_dense = x_tridiag;
            y_dense = x_dense + y_tridiag;
        }
            
        lowdiag_matrix[x_tridiag + y_tridiag*N] = dense_matrix[x_dense + y_dense*N];
    }
}

__global__ void diagh2mat(complex<double>* dense_matrix, const complex<double>* lowdiag_matrix,\
                        unsigned int N) 
{
    /*
    C based CUDA Kernel equivalent of diagh2mat.
    Gives skewhermitian matrix W from lower diagonal format.


    Parameters
    ----------
    dense_matrix :  complex<double>*: size N * N
        Dense skew hermitian output matrix.
    lowdiag_matrix : const complex<double>*: size (N//2+1)* N
        Lower diagonal form input
    N : unsigned int,
        Dimension of output matrix

    Returns
    -------
    void
    */

    unsigned int x_tridiag = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y_tridiag = blockDim.y * blockIdx.y + threadIdx.y;

    int x_dense;
    int y_dense;

    if ((x_tridiag < N) && (y_tridiag < (int)(N/2) + 1))
    {
        if( x_tridiag > N-y_tridiag-1){
            x_dense = x_tridiag - (N - y_tridiag);
            y_dense = x_dense + N - y_tridiag;
        }   
        else {
            x_dense = x_tridiag;
            y_dense = x_dense + y_tridiag;
        }
            
        dense_matrix[x_dense + y_dense*N] = lowdiag_matrix[x_tridiag + y_tridiag*N];
        dense_matrix[y_dense + x_dense*N] = -conj(lowdiag_matrix[x_tridiag + y_tridiag*N]);
    }
}
}
'''

module = cp.RawModule(code=mat_tofrom_diag_module)
mat2diagh_ker = module.get_function('mat2diagh')
diagh2mat_ker = module.get_function('diagh2mat')

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


def solve_tridiagonal_cp(lap, W, P, Wdiagh, Pdiagh):
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
    lap : cparray(shape=(2, N*(N+1)/2), dtype=complex128)
    """
    global _tridiagonal_laplacian_cp_cache

    if (N, bc) not in _tridiagonal_laplacian_cp_cache:
        lap = compute_tridiagonal_laplacian_cp(N, bc=bc)
        _tridiagonal_laplacian_cp_cache[(N, bc)] = lap

    return _tridiagonal_laplacian_cp_cache[(N, bc)]

class solve_poisson_cp:
    def __init__(self,N, bc = True) -> None:
        self.N = N
        self.lap = cp.asarray(laplacian_cp(N, bc = bc))
        self.Wdiagh = cp.empty((N//2+1,N),dtype='complex128')
        self.Pdiagh = cp.empty((N//2+1,N),dtype='complex128')

    def __call__(self,W,P) -> None:
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

        solve_tridiagonal_cp(self.lap, W, P, self.Wdiagh, self.Pdiagh)

class isomp_gpu_skewherm_solver:
    
    def __init__(self,h_W) -> None:
        """
        Initialize the CuPy based isospectral midpoint second order method for skew-Hermitian W.
        Parameters.
        This init will allocate memory on the device and copy the initial value for W.

        Note that dW is not for CUDA device, but rather the same name as in the CPU-implementation
        Parameters
        ------
        W: ndarray
            Initial skew-Hermitian vorticity matrix
        """
        self.W = cp.array(h_W,copy=True)
        self.dW = cp.zeros_like(self.W)
        self.dW_old = cp.zeros_like(self.W)
        self.Whalf = cp.zeros_like(self.W)
        self.Phalf = cp.zeros_like(self.W)
        self.PWcomm = cp.zeros_like(self.W)
    
    def solve_step(self, h_W, stepsize=0.1, steps=5, hamiltonian=solve_poisson_cp,
                       tol=1e-8, maxit=5, verbatim=True, skewherm_proj_freq=3000, forcing=None) -> np.ndarray:
        """
        Time-stepping by isospectral midpoint second order method for skew-Hermitian W.

        Parameters
        ----------
        h_W: ndarray
            Initial skew-Hermitian vorticity matrix (overwritten and returned).
        stepsize: float
            Time step length.
        steps: int
            Number of steps to take.
        hamiltonian: function
            The Hamiltonian returning a stream matrix.
        tol: float
            Tolerance for iterations.
        maxit: int
            Maximum number of iterations.
        verbatim: bool
            Print extra information if True. Default is False.
        skewherm_proj_freq: int
            Project onto skew-Hermitian every skewherm_proj_freq step.
        forcing: function(P, W) or None (default)
            Extra force function (to allow non-isospectral perturbations).

        Returns
        -------
        h_W: ndarray
        """
        assert maxit >= 1, "maxit must be at least 1."


        total_iterations = 0

        # Set dW to zero
        self.dW.fill(0)

        for k in range(steps):

            # --- Beginning of step ---

            for i in range(maxit):

                # --- Beginning of iterations ---

                # Update iterations
                total_iterations += 1

                # Compute Wtilde
                cp.copyto(self.Whalf, self.W)

                self.Whalf += self.dW

                # Update Ptilde
                hamiltonian(self.Whalf, self.Phalf)

                self.Phalf *= stepsize/2.0
                # Update dW
                cp.copyto(self.dW_old, self.dW)

                # Compute middle variables
                cp.matmul(self.Phalf,self.Whalf, out = self.PWcomm)

                cp.matmul(self.PWcomm,self.Phalf, out = self.dw)
                self.PWcomm -= self.PWcomm.conj().T
                #cp.copyto(self.dW, PWcomm)
                self.dW =  self.dW + self.PWcomm
                # Add forcing if needed

                if forcing is not None:
                    # Compute forcing if needed
                    FW = forcing(self.Phalf/(stepsize/2.0), self.Whalf)
                    FW *= stepsize/2.0
                    self.dW += FW

                # Compute error
                resnorm = cp.linalg.norm(self.dW - self.dW_old, cp.inf)
                # Check error
                if resnorm < tol:
                    break

            else:
                # We used maxit iterations
                if verbatim:
                    print("Max iterations {} reached at step {}.".format(maxit, k))

                # --- End of iterations ---

            # Update W
            self.W += 2.0*self.PWcomm
            cp.cuda.runtime.deviceSynchronize()
            # Check if projection needed
            if k+1 % skewherm_proj_freq == 0:
                self.W /= 2.0
                self.W -= self.W.conj().T

            # --- End of step ---

        if verbatim:
            print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))

        cp.cuda.runtime.deviceSynchronize()
        h_W = self.W.get(out = h_W)
        return h_W