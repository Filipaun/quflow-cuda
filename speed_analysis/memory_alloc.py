import cupy as cp

N = 1000
while True:
    a = cp.zeros((N,N),dtype="complex128")
    b = cp.arange(0,N**2,1, dtype ="complex128")

#yo = input("Pause")