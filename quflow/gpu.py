import numpy as np
import cupy as cp
import pkgutil

_tridiagonal_laplacian_interleaved_cp_cache = dict()

def check_version():  
    print(f"CuPy version: {cp.__version__}")

# Import cuda source code

tridiag_source = pkgutil.get_data(__package__,'tridiag.cu')
tridiag_module = cp.RawModule(code=tridiag_source.decode('utf-8'))

del tridiag_source

# Get non interleaved kernels, not needed for non-tf 
#mat2diagh_ker = tridiag_module.get_function('mat2diagh')
#diagh2mat_ker = tridiag_module.get_function('diagh2mat')

# Kernels for reformating skewhermitian matrix into interleaved format
mat2diagh_interleaved_ker = tridiag_module.get_function('mat2diagh_interleaved')
diagh2mat_interleaved_ker = tridiag_module.get_function('diagh2mat_interleaved')

# Tridiagonal solvers
ker_tridiag_solve = tridiag_module.get_function('solve_tridiag_skewh_cached')
ker_tridiag_solve_lessmemory = tridiag_module.get_function('solve_tridiag_skewh_lessmemory')

def mat2diagh_interleaved_cp(lowdiag,dense,N):

    # Wrapper for mat2diagh CUDA kernel
    # Simple grid and block size

    grid_dim = N//44 + 1
    block_dim_x = 23
    block_dim_y = 44

    mat2diagh_interleaved_ker((grid_dim,grid_dim),(block_dim_x,block_dim_y),(lowdiag,dense,N))
    #cp.cuda.runtime.deviceSynchronize()

    return 0


def diagh2mat_interleaved_cp(dense,lowdiag,N):

    # Wrapper for mat2diagh CUDA kernel
    # Simple grid and block size

    grid_dim = N//44 + 1
    block_dim_x = 23
    block_dim_y = 44

    diagh2mat_interleaved_ker((grid_dim,grid_dim),(block_dim_x,block_dim_y),(dense,lowdiag,N))
    #cp.cuda.runtime.deviceSynchronize()

    return 0

def tridiag_solve(N, lap, Wdiagh, Pdiagh, gamma_tmp):
    grid_dim = N//640 + 1
    block_dim_x = 640

    ker_tridiag_solve((grid_dim,),(block_dim_x,),(N, lap , Wdiagh, Pdiagh, gamma_tmp))

def tridiag_solve_lessmemory(N, lap, Wdiagh, Pdiagh):
    grid_dim = N//640 + 1
    block_dim_x = 640

    ker_tridiag_solve_lessmemory((grid_dim,),(block_dim_x,),(N, lap , Wdiagh, Pdiagh))


##### -------------- Interleaved using kernel ------------ ###

def laplacian_interleaved_cp(N, bc=False):
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
    global _tridiagonal_laplacian_interleaved_cp_cache

    if (N, bc) not in _tridiagonal_laplacian_interleaved_cp_cache:
        lap = compute_tridiagonal_laplacian_interleaved_cp(N, bc=bc)
        _tridiagonal_laplacian_interleaved_cp_cache[(N, bc)] = lap

    return _tridiagonal_laplacian_interleaved_cp_cache[(N, bc)]


def solve_tridiagonal_interleaved_cp(lap, W, P, Wdiagh, Pdiagh, gamma_tmp):
    """
    Function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` array). Uses NUMBA to accelerate the
    tridiagonal solver calculations.

    Parameters
    ----------
    lap: ndarray(shape=(2, N, N//2+1), dtype=float)
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

    # Swap values into Wdiagh
    # Pass by reference faster than returning ?
    mat2diagh_interleaved_cp(Wdiagh,W,N)

    # Convert to tensor

    # For each double-tridiagonal, solve a tridiagonal system
    cp.cuda.runtime.deviceSynchronize()
    tridiag_solve(N,lap, Wdiagh, Pdiagh, gamma_tmp)
    cp.cuda.runtime.deviceSynchronize()

    # Make sure we preserve trace of W (corresponds to bc for laplacian)
    trP = Pdiagh[:, 0].sum()/N
    Pdiagh[:, 0] -= trP 

    # Convert back to dense matrix
    diagh2mat_interleaved_cp(P,Pdiagh,N)

def solve_tridiagonal_interleaved_lessmemory_cp(lap, W, P, Wdiagh, Pdiagh):
    """
    Function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` array). Uses NUMBA to accelerate the
    tridiagonal solver calculations.

    Parameters
    ----------
    lap: ndarray(shape=(2, N, N//2+1), dtype=float)
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

    # Swap values into Wdiagh
    # Pass by reference faster than returning ?
    mat2diagh_interleaved_cp(Wdiagh,W,N)

    # Convert to tensor

    # For each double-tridiagonal, solve a tridiagonal system
    cp.cuda.runtime.deviceSynchronize()
    tridiag_solve_lessmemory(N,lap, Wdiagh, Pdiagh)
    cp.cuda.runtime.deviceSynchronize()

    # Make sure we preserve trace of W (corresponds to bc for laplacian)
    trP = Pdiagh[:, 0].sum()/N
    Pdiagh[:, 0] -= trP 

    # Convert back to dense matrix
    diagh2mat_interleaved_cp(P,Pdiagh,N)



def compute_tridiagonal_laplacian_interleaved_cp(N, bc=False):
    """
    Compute tridiagonal laplacian with interleaved elements

    Parameters
    ----------
    N: int
    bc: bool (optional)
        Whether boundary conditions should be added to make the laplacian non-singular.
        Notice that this bc is different from the one used for sparse laplacians.

    Returns
    -------
    lap: float64 nparray, shape (2, N, N//2+1)
        Outer index: diagonal index, 0 main diag, 1 lower
        Inner index: entries on the diagonals
        Middle index: which diagonal, stored according to tensorflow format
        
    """

    # Tensorflow needs same types on both sides of Ax=B, so lap is complex128
    # Tensorflow needs all 3 diagonals, so axis 1 has size = 3.
    ##lap = np.zeros((N//2+1, 3, N), dtype=np.complex128)
    lap = np.zeros((2, N, N//2+1), dtype=np.float64)
    i_full = np.arange(N)
    for m in range(N//2+1):

        # Global diagonal m (of length N-m)
        i = i_full[:N - m]
        lap[0, 0:N - m, m] = -((N - 1)*(2*i + 1 + m) - 2*i*(i + m))
        i = i_full[1:N - m]
        lap[1,0:N - m - 1,m] = np.sqrt(((i + m)*(N - i - m))*(i*(N - i)))

        # Global diagonal N-m (of length m)
        i = i_full[:m]
        lap[0, N - m:, m] = -((N - 1)*(2*i + 1 + N - m) - 2*i*(i + N - m))
        i = i_full[1:m]
        lap[1, N - m:-1, m] = np.sqrt(((i + N - m)*(m - i))*(i*(N - i)))


    if bc:
        lap[0, 0, 0] -= 0.5
        pass

    return lap

class solve_poisson_interleaved_cp:
    def __init__(self,N, bc = True) -> None:
        self.N = N
        self.lap = cp.asarray(laplacian_interleaved_cp(N, bc = bc))
        self.Wdiagh = cp.empty((N,N//2+1),dtype='complex128')
        self.Pdiagh = cp.empty((N,N//2+1),dtype='complex128')
        self.gamma_tmp = cp.empty((N,N//2+1),dtype='float64')

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
        ##P: ndarray(shape=(N, N), dtype
        =complex)
        """

        solve_tridiagonal_interleaved_cp(self.lap, W, P, self.Wdiagh, self.Pdiagh, self.gamma_tmp)

class solve_poisson_interleaved_lessmemory_cp:
    def __init__(self,N, bc = True) -> None:
        self.N = N
        self.lap = cp.asarray(laplacian_interleaved_cp(N, bc = bc))
        self.Wdiagh = cp.empty((N,N//2+1),dtype='complex128')
        self.Pdiagh = cp.empty((N,N//2+1),dtype='complex128')

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
        ##P: ndarray(shape=(N, N), dtype
        =complex)
        """

        solve_tridiagonal_interleaved_lessmemory_cp(self.lap, W, P, self.Wdiagh, self.Pdiagh)
    
# ----------- GPU Solver ---------------

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
    
    def solve_step(self, h_W, stepsize=0.1, steps=5, hamiltonian=solve_poisson_interleaved_cp,
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

                cp.matmul(self.PWcomm,self.Phalf, out = self.dW)
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
        self.W.get(out = h_W)
        return h_W