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
np.set_printoptions(precision=3)
N = 4  # Size of matrices
lmax = 10  # How many spherical harmonics (SH) coefficients to include
#np.random.seed(42)  # For reproducability
omega0 = np.zeros(N**2) # Array with SH coefficients
omega0[0] = 1  
W0 = qf.shr2mat(omega0, N=N)
qf.plot2(omega0)
plt.show()