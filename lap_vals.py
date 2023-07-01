import numpy as np
import matplotlib.pyplot as plt

def kron_delta(i,j):
    if i == j:
        return 1
    else:
        return 0

def lap(m,N):
    s = (N-1)/2
    lap_mat = np.zeros((N-m,N-m))
    for i in range(N-m):
        for j in range(N-m):
            lap_mat[i,j] = 2*kron_delta(i,j)*(s*(2*i+1+m)-i*(i+m)) - kron_delta(i+1,j)*np.sqrt((i+m+1)*(N-1-i-m))*np.sqrt((i+1)*(N-1-i)) - kron_delta(i-1,j)*np.sqrt((i+m)*(N-i-m))*np.sqrt(i*(N-i))

    return lap_mat

def lap2(N,i_1,i_2,j_1,j_2):
    s = (N-1)/2
    #lap_mat = np.zeros((N-m,N-m))
    i_1 -= (s+1)
    i_2 -= (s+1)
    j_1 -= (s+1)
    j_2 -= (s+1)

    lap_val = 0
    if kron_delta(i_1,j_1)*kron_delta(i_2,j_2) == 1:
        print("main diag")
        lap_val += 2*(s*(s+1)-i_1*i_2)
    elif kron_delta(i_1+1,j_1)*kron_delta(i_2+1,j_2) == 1:
        print("lower diag")
        lap_val -= np.sqrt(s*(s+1)-i_1*(i_1+1))*np.sqrt(s*(s+1)-i_2*(i_2+1))
    elif kron_delta(i_1-1,j_1)*kron_delta(i_2-1,j_2) == 1:
        print("upper diag")
        lap_val -= np.sqrt(s*(s+1)-i_1*(i_1-1))*np.sqrt(s*(s+1)-i_2*(i_2-1))

    return lap_val

import quflow as qf
N = 4  # Size of matrices
lmax = 10  # How many spherical harmonics (SH) coefficients to include
#np.random.seed(42)  # For reproducability
omega0 = np.zeros(N**2) # Array with SH coefficients
omega0[0] = 1  
W0 = qf.shr2mat(omega0, N=N)
qf.plot2(omega0)
plt.show()


def shr():
    from scipy.linalg import eigh_tridiagonal

    N = 4
    basis_break_indices = np.hstack((0, (np.arange(N, 0, -1)**2).cumsum()))
    basis = np.zeros(basis_break_indices[-1], dtype=float)

    # Compute direct laplacian
    lap = qf.laplacian.direct.compute_direct_laplacian(N, bc=False)

    for m in range(N):
        bind0 = basis_break_indices[m]
        bind1 = basis_break_indices[m+1]

        # Compute eigen decomposition
        n = N - m
        start_ind = N*(N+1)//2 - n*(n+1)//2
        end_ind = start_ind + n
        v2, w2 = eigh_tridiagonal(lap[1, start_ind:end_ind], lap[0, start_ind+1:end_ind])

        print(v2)
        print(w2)
        # Reverse the order
        w2 = w2[:, ::-1]

        # The eigenvectors are only defined up to sign.
        # Therefore, we must adjust the sign so that it corresponds with
        # the quantization basis of Hoppe (i.e. with the spherical harmonics).
        qf.adjust_basis_orientation_(w2, m)

        basis[bind0:bind1] = w2.ravel()

    #print(basis)

shr()