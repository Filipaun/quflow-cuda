#include <cupy/complex.cuh>
extern "C"{
__global__ void solve_direct_skewh_(int N, double* lap, complex<double>* W, complex<double>* P) {
    /*
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
    */

    // Thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int n = N-m

    int start_ind = N*(N+1)/2-n*(n+1)/2
    int end_ind = start_ind + n
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
    for k in range(N):
    P[k, k] -= trP
}
}