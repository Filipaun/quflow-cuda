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