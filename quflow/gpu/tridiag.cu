#include <cupy/complex.cuh>
extern "C"{
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

__global__ void mat2diagh_interleaved(complex<double>* lowdiag_matrix, const complex<double>* dense_matrix,\
                         unsigned int N)
{
    /*
    CUDA kernel for restructuring input skewhermitian matrix into 
    Return lower diagonal format for skew hermitian matrix W.

    Parameters
    ----------
    lowdiag_matrix : complex<double>*: size N * (N//2+1)
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

    int x_size = (int)(N/2) + 1;

    if ((x_tridiag < x_size) && (y_tridiag < N))
    {
        if( y_tridiag > N-x_tridiag-1){
            x_dense = y_tridiag - (N - x_tridiag);
            y_dense = x_dense + N - x_tridiag;
        }   
        else {
            x_dense = y_tridiag;
            y_dense = x_dense + x_tridiag;
        }

            
        lowdiag_matrix[x_tridiag + y_tridiag*x_size] = dense_matrix[x_dense + y_dense*N];
    }
}

__global__ void diagh2mat_interleaved(complex<double>* dense_matrix, const complex<double>* lowdiag_matrix,\
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

    int x_size = (int)(N/2) + 1;

    if ((x_tridiag < x_size) && (y_tridiag < N))
    {
        if( y_tridiag > N-x_tridiag-1){
            x_dense = y_tridiag - (N - x_tridiag);
            y_dense = x_dense + N - x_tridiag;
        }   
        else {
            x_dense = y_tridiag;
            y_dense = x_dense + x_tridiag;
        }
            
        dense_matrix[x_dense + y_dense*N] = lowdiag_matrix[x_tridiag + y_tridiag*x_size];
        dense_matrix[y_dense + x_dense*N] = -conj(lowdiag_matrix[x_tridiag + y_tridiag*x_size]);
    }
}

__global__ void solve_tridiag_skewh(const unsigned int N, const double* lap, const complex<double>* W, complex<double>* P, double* gamma_tmp) {
        /*
        Solves a N tridiagonal systems where inputs are interleaved

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
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int system_count = (N/2+1);
        //int system_count = 2;
        
        if (tid < system_count) {
            
            // Rescaling of the first row
            // c'_1 = c_1/b_1
            gamma_tmp[tid] = lap[tid+system_count*(N)]/lap[tid];
            P[tid] = W[tid]/lap[tid];

            for(unsigned int i = 1; i<N-1; i++) {
                // Elimination and rescaling in ith row

                // c'_i = c_i /(b_i - a_i*c'_i)
                gamma_tmp[tid+system_count*i] = lap[tid+system_count*(N+i)]/(lap[tid+system_count*i]-lap[tid+system_count*(N+i-1)]*gamma_tmp[tid+system_count*(i-1)]);
                
                // d'_i = (d_i-a_i*d'_{i-1})/(b_i-a_i*c'_{i-1})
                P[tid+system_count*i] = (W[tid+system_count*i]-lap[tid+system_count*(N+i-1)]* P[tid+system_count*(i-1)])/(lap[tid+system_count*i]-lap[tid+system_count*(N+i-1)]*gamma_tmp[tid+system_count*(i-1)]);
            }

            // Rescaling and backward substitution of last row

            // x_n = d'_n
            P[tid+system_count*(N-1)] = (W[tid+system_count*(N-1)]-lap[tid+system_count*(N+(N-1)-1)]* P[tid+system_count*((N-1)-1)])/(lap[tid+system_count*(N-1)]-lap[tid+system_count*(N+(N-1)-1)]*gamma_tmp[tid+system_count*((N-1)-1)]);

            for (unsigned int i = N-2; i> 0;i--){
                // Backward substitution for ith row

                // x_i = d'_i - c'_i*x_{i+1}
                P[tid+system_count*i] = P[tid+system_count*i]- gamma_tmp[tid+system_count*i]*P[tid+system_count*(i+1)];
            }
            P[tid] = P[tid] -gamma_tmp[tid]*P[tid+system_count];
        }     
}

__global__ void solve_tridiag_skewh_cached(const unsigned int N, const double* lap, const complex<double>* W, complex<double>* P, double* gamma_tmp) {
    /*
    Solves a N tridiagonal systems where inputs are interleaved

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
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int system_count = (N/2+1);
    double current;
    double last;
    //int system_count = 2;
    
    if (tid < system_count) {
        
        // Rescaling of the first row
        // c'_1 = c_1/b_1
        current = lap[tid+system_count*(N)];
        gamma_tmp[tid] = current/lap[tid];
        P[tid] = W[tid]/lap[tid];

        for(unsigned int i = 1; i<N-1; i++) {
            // Elimination and rescaling in ith row

            // c'_i = c_i /(b_i - a_i*c'_i)
            last = current;
            current = lap[tid+system_count*(N+i)];
            gamma_tmp[tid+system_count*i] = current/(lap[tid+system_count*i]-last*gamma_tmp[tid+system_count*(i-1)]);
            
            // d'_i = (d_i-a_i*d'_{i-1})/(b_i-a_i*c'_{i-1} )
            P[tid+system_count*i] = (W[tid+system_count*i] - last * P[tid+system_count*(i-1)]) / (lap[tid+system_count*i] - last * gamma_tmp[tid+system_count * (i-1)]);
        }

        // Rescaling and backward substitution of last row

        // x_n = d'_n
        // d'_n = (d_n-a_n*d'_{n-1})/(b_n-a_n*c'_{i-1} )
        P[tid+system_count*(N-1)] = (W[tid+system_count*(N-1)]-current* P[tid+system_count*((N-1)-1)])/(lap[tid+system_count*(N-1)]-current*gamma_tmp[tid+system_count*((N-1)-1)]);

        for (unsigned int i = N-2; i> 0;i--){
            // Backward substitution for ith row

            // x_i = d'_i - c'_i*x_{i+1}
            P[tid+system_count*i] = P[tid+system_count*i]- gamma_tmp[tid+system_count*i]*P[tid+system_count*(i+1)];
        }
        P[tid] = P[tid] -gamma_tmp[tid]*P[tid+system_count];
    }     
}

__global__ void solve_tridiag_skewh_lessmemory(const unsigned int N, const double* lap, complex<double>* W, complex<double>* P) {
    /*
    Solves a N tridiagonal systems where inputs are interleaved

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
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int system_count = (N/2+1);
    
    if (tid < system_count) {
        
        // Rescaling of the first row
        // c'_1 = c_1/b_1
        P[tid] = W[tid]/lap[tid];
        W[tid] = lap[tid+system_count*(N)]/lap[tid];

        for(unsigned int i = 1; i<N-1; i++) {
            // Elimination and rescaling in ith row

            // d'_i = (d_i-a_i*d'_{i-1})/(b_i-a_i*c'_{i-1})
            P[tid+system_count*i] = (W[tid+system_count*i]-lap[tid+system_count*(N+i-1)]* P[tid+system_count*(i-1)])/(lap[tid+system_count*i]-lap[tid+system_count*(N+i-1)]*W[tid+system_count*(i-1)].real());

            // c'_i = c_i /(b_i - a_i*c'_i)
            W[tid+system_count*i] = lap[tid+system_count*(N+i)]/(lap[tid+system_count*i]-lap[tid+system_count*(N+i-1)]*W[tid+system_count*(i-1)].real());
        }

        // Rescaling and backward substitution of last row

        // x_n = d'_n
        P[tid+system_count*(N-1)] = (W[tid+system_count*(N-1)]-lap[tid+system_count*(N+(N-1)-1)]* P[tid+system_count*((N-1)-1)])/(lap[tid+system_count*(N-1)]-lap[tid+system_count*(N+(N-1)-1)]*W[tid+system_count*((N-1)-1)].real());

        for (unsigned int i = N-2; i> 0;i--){
            // Backward substitution for ith row

            // x_i = d'_i - c'_i*x_{i+1}
            P[tid+system_count*i] = P[tid+system_count*i]- W[tid+system_count*i]*P[tid+system_count*(i+1)];
        }
        P[tid] = P[tid] -W[tid]*P[tid+system_count];
    }     
}
}