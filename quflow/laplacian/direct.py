import numpy as np
from numba import njit, prange

# ----------------
# GLOBAL VARIABLES
# ----------------

_direct_laplacian_cache = dict()
_direct_heat_cache = dict()
_direct_viscdamp_cache = dict()


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

@njit
def compute_direct_laplacian(N, bc=False):
    """
    Compute direct laplacian.

    Parameters
    ----------
    N: int
    bc: bool (optional)
        Whether boundary conditions should be added to make the laplacian non-singular.
        Notice that this bc is different from the one used for sparse laplacians.

    Returns
    -------
    lap: ndarray
    """
    s = (N - 1)/2
    mvals = np.linspace(-s, s, N)
    lap = np.zeros((2, N*(N+1)//2), dtype=np.float64)

    for m1 in mvals:
        for m2 in mvals:
            coeff1 = 2*(s*(s+1)-m1*m2)
            if abs(coeff1) > 1e-10:
                m = round(m1-m2)
                if m < 0:
                    continue
                n = N-m
                ind = round(lap.shape[1] - n*(n+1)//2 + m2 + s)
                lap[1, ind] = -coeff1

            if m1 < s and m2 < s:
                coeff2 = -np.sqrt(s*(s+1)-m1*(m1+1))*np.sqrt(s*(s+1)-m2*(m2+1))
                if abs(coeff2) > 1e-10:
                    m = round(m1-m2)
                    if m < 0:
                        continue
                    n = N-m
                    ind = round(lap.shape[1] - n*(n+1)//2 + m2 + s + 1)
                    lap[0, ind] = -coeff2

    if bc:
        lap[1, 0] += 0.5

    return lap


@njit
def dot_direct_(lap, P, W):
    """
    Dot product for direct matrix.

    Parameters
    ----------
    lap: ndarray
        Direct laplacian.
    P: ndarray(shape=(N,N), dtype=complex)
        Input matrix.
    W: ndarray(shape=(N,N), dtype=complex)
        Output matrix.
    """
    N = P.shape[0]

    for m in range(N):
        n = N-m
        start_ind = lap.shape[1]-n*(n+1)//2
        end_ind = start_ind + n
        a = lap[0, start_ind:end_ind]
        b = lap[1, start_ind:end_ind]

        # k = 0
        pk = P[0, m]
        pk_plus = P[1, m+1]
        wk = b[0]*pk + a[1]*pk_plus
        W[0, m] = wk
        if m != 0:
            W[m, 0] = -np.conj(wk)

        # k = 1,...,n-2
        for k in range(1, n-1):
            pk_minus = P[k-1, m+k-1]
            pk = P[k, m+k]
            pk_plus = P[k+1, m+k+1]
            wk = a[k]*pk_minus + b[k]*pk + a[k+1]*pk_plus
            W[k, m+k] = wk
            if m != 0:
                W[k+m, k] = -np.conj(wk)

        # k = n-1
        pk_minus = P[n-2, m+n-2]
        pk = P[n-1, m+n-1]
        wk = a[n-1]*pk_minus + b[n-1]*pk
        W[n-1, m+n-1] = wk
        if m != 0:
            W[n-1+m, n-1] = -np.conj(wk)


@njit(parallel=True)
def solve_direct_(lap, W, P, vtmp, ytmp):
    """
    Highly optimized function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` direct matrix).

    Parameters
    ----------
    lap: ndarray(shape=(2, N*(N+1)/2), dtype=float)
        Direct laplacian.
    W: ndarray(shape=(N, N), dtype=complex)
        Input matrix.
    P: ndarray(shape=(N, N), dtype=complex)
        Output matrix.
    vtmp: ndarray(shape=(N*(N+1)/2,), dtype=float)
        Temporary float memory needed.
    ytmp: ndarray(shape=(N*(N+1)/2,), dtype=complex)
        Temporary complex memory needed.
    """
    N = W.shape[0]

    for m in prange(N):
        n = N-m
        start_ind = lap.shape[1]-n*(n+1)//2
        end_ind = start_ind + n
        a = lap[0, start_ind:end_ind]
        b = lap[1, start_ind:end_ind]
        y = ytmp[start_ind:end_ind]
        v = vtmp[start_ind:end_ind]

        vk = b[0]
        v[0] = vk
        fk = W[0, m]
        yk = fk
        y[0] = yk

        for k in range(1, n):
            lk = a[k]/vk
            fk = W[k, m+k]
            yk = fk - lk*yk
            y[k] = yk
            vk = b[k] - lk*a[k]
            v[k] = vk

        pk = y[n-1]/v[n-1]
        P[n-1, m+n-1] = pk
        if m != 0:
            P[m+n-1, n-1] = -np.conj(pk)

        for k in range(n-2, -1, -1):
            pk = (y[k]-a[k+1]*pk)/v[k]
            P[k, m+k] = pk
            if m != 0:
                P[m+k, k] = -np.conj(pk)

    trP = np.trace(P)/N
    for k in prange(N):
        P[k, k] -= trP


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def laplacian(N, bc=False):
    """
    Return quantized laplacian (as a direct laplacian).

    Parameters
    ----------
    N: int
    bc: bool (optional)
        Whether to include boundary conditions.

    Returns
    -------
    lap : ndarray(shape=(2, N*(N+1)/2), dtype=flaot)
    """
    global _direct_laplacian_cache

    if (N, bc) not in _direct_laplacian_cache:
        lap = compute_direct_laplacian(N, bc=bc)
        _direct_laplacian_cache[(N, bc)] = lap

    return _direct_laplacian_cache[(N, bc)]


def laplace(P):
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
    lap = laplacian(N)
    W = np.zeros_like(P)

    # Apply dot product
    dot_direct_(lap, P, W)

    return W


def solve_poisson(W):
    """
    Return stream matrix `P` for `W`.

    Parameters
    ----------
    W: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    P: ndarray(shape=(N, N), dtype=complex)
    """
    N = W.shape[0]
    lap = laplacian(N, bc=True)
    vtmp = np.zeros(lap.shape[1], dtype=np.float64)
    ytmp = np.zeros(lap.shape[1], dtype=np.complex128)
    P = np.zeros_like(W)
    solve_direct_(lap, W, P, vtmp, ytmp)

    return P


def solve_heat(h_times_nu, W0):
    """
    Solve quantized heat equation using backward Euler method.

    Parameters
    ----------
    h_times_nu: float
        Stepsize (in qtime) times viscosity.
    W0: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    Wh: ndarray(shape=(N, N), dtype=complex)
    """
    global _direct_heat_cache

    N = W0.shape[0]

    if (N, h_times_nu) not in _direct_heat_cache:
        # Get direct laplacian
        lap = laplacian(N, bc=False)

        # Get direct operator for backward Euler
        heat = np.array([[0.], [1.]]) - h_times_nu*lap

        # Store in cache
        _direct_heat_cache[(N, h_times_nu)] = heat
    else:
        heat = _direct_heat_cache[(N, h_times_nu)]

    vtmp = np.zeros(heat.shape[1], dtype=np.float64)
    ytmp = np.zeros(heat.shape[1], dtype=np.complex128)
    Wh = np.zeros_like(W0)
    solve_direct_(heat, W0, Wh, vtmp, ytmp)

    return Wh


def solve_viscdamp(h, W0, nu=1e-4, alpha=0.01, force=None, theta=1):
    """
    Solve quantized viscosity and damping equation

        W' - nu * ∆ W + alpha * W = F

    for one time-step using the theta scheme.

    Parameters
    ----------
    h: float
        Time-step.
    W0: ndarray(shape=(N, N), dtype=complex)
        Initial vorticity matrix.
    nu: float
        Viscosity.
    alpha: float
        Damping.
    force: None or ndarray(shape=(N, N), dtype=complex)
        External forcing F applied.
    theta: float
        Weighting in the theta method (Crank-Nicolson for theta=0.5).

    Returns
    -------
    Wh: ndarray(shape=(N, N), dtype=complex)
    """
    global _direct_viscdamp_cache

    N = W0.shape[0]

    if (N, h, nu, alpha) not in _direct_viscdamp_cache:
        # Get direct laplacian
        lap = laplacian(N, bc=False)

        # Get direct operator for theta method
        viscdamp = (1.0+h*alpha*theta)*np.array([[0.], [1.]]) - (h*nu*theta)*lap

        # Store in cache
        _direct_viscdamp_cache[(N, h, nu, alpha)] = viscdamp
    else:
        viscdamp = _direct_viscdamp_cache[(N, h, nu, alpha)]

    vtmp = np.zeros(viscdamp.shape[1], dtype=np.float64)
    ytmp = np.zeros(viscdamp.shape[1], dtype=np.complex128)

    # Prepare right hand side in Crank-Nicolson
    if theta == 1:
        Wrhs = W0.copy()
    else:
        Wrhs = (1.0-alpha*h*(1-theta))*W0
        Wrhs += (nu*h*(1-theta))*laplace(W0)
    if force is not None:
        Wrhs += h*force

    # Solve linear subsystems
    Wh = np.zeros_like(W0)
    solve_direct_(viscdamp, Wrhs, Wh, vtmp, ytmp)

    return Wh
