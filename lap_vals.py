import numpy as np

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

m = 0
my_lap = lap(m,500)
print(my_lap[1,2])
print(my_lap[0,1]+my_lap[1,1]+my_lap[1,2])
print(np.amax(my_lap))
