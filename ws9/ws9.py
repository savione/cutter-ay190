#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS9

import numpy as np
import scipy as sp
from timeit import timeit

# Part A

def load_data():
    As = []
    bs = []
    for i in range(1, 6):
        As.append(np.loadtxt("LSE{}_m.dat".format(i)))
        bs.append(np.loadtxt("LSE{}_bvec.dat".format(i)))
    return As, bs

# Returns true if x is further than 10^-10 from zero, false otherwise.
def notzero(x):
    return abs(x) > 1e-10

if __name__ == "__main__":

    print("\nPART A:\n")
    As, bs = load_data()
    dets = [np.linalg.det(As[i]) for i in range(5)]
    for i in range(5):
        print("A_({ind}) has shape {A_shape}, while b_({ind}) has shape {b_shape}.".format(ind=i+1, A_shape=np.shape(As[i]), b_shape=np.shape(bs[i])))
        if notzero(dets[i]):
            print("A_({ind}) is nonsingular (has nonzero determinant). The LSE can be solved.".format(ind=i+1))
        else:
            print("A_({ind}) is singular (has zero determinant). The LSE cannot be solved.".format(ind=i+1))


# Part B

# Solves the linear system of equations Ax = b through Gaussian
# elimination. Returns x.
def gauss_elim_solve(A, b):
    if not notzero(np.linalg.det(A)):
        raise ValueError("Determinant of A close to zero; may be uninvertible")
    n,m = np.shape(A)
    if n != m:
        raise ValueError("A must be a square matrix")
    if n != len(b):
        raise ValueError("b must have the same length as A")

    b = np.reshape(b, (len(b), 1))
    M = np.concatenate((A, b), axis=1).astype(np.float)
    
    for i in range(n):
        if not notzero(M[i,i]):
            for j in range(n):
                if notzero(M[j,i]):
                    M[i] += M[j]
                    break
        M[i] /= M[i,i]
        for j in range(n):
            if j == i:
                continue
            M[j] -= (M[j,i] / M[i,i]) * M[i]

    return np.reshape(M[:,-1], (n, 1))

# Simple LSE with known solution (solution is x = [2, 3, -1])
def gen_simple_LSE():
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    b = np.reshape(np.array([8, -11, -3]), (3, 1))
    return A, b

if __name__ == "__main__":

    print("\nPART B:\n")

    # Simple 3x3 LSE

    ntrials = 10
    time = timeit("gauss_elim_solve(A, b)", number=ntrials, setup="import numpy as np; from ws9 import gauss_elim_solve, gen_simple_LSE; A,b = gen_simple_LSE()") / ntrials

    A, b = gen_simple_LSE()

    print("Simple LSE:\nA =\n{}\nb =\n{}\nSolution: x =\n{}\nTime elapsed: {:.3f} ms".format(A, b, gauss_elim_solve(A, b), 1000*time))


    # 5 Given LSEs

    ntrials = 10
    exec_string = "gauss_elim_solve(As[{0}], bs[{0}])"
    setup_string = "import numpy as np; from ws9 import gauss_elim_solve, load_data; As,bs = load_data()"
    times = []
    for i in range(5):
        times.append(timeit(exec_string.format(i), number=ntrials,
                        setup=setup_string) / ntrials)

    info = "#{ind}\t{time:.3f} ms"
    print("\nFor the five given LSEs, the times required were:")
    for i in range(5):
        print(info.format(ind=i+1, time=1000*times[i]))

# Part C

if __name__ == "__main__":
    
    print("\nPART C:\n")

    # 5 Given LSEs using NumPy's linear algebra solver

    ntrials = 1000
    exec_string = "np.linalg.solve(As[{0}], bs[{0}])"
    setup_string = "import numpy as np; from ws9 import load_data; As,bs = load_data()"
    times = []
    for i in range(5):
        times.append(timeit(exec_string.format(i), number=ntrials,
                        setup=setup_string) / ntrials)

    info = "#{ind}\t{time:.3f} ms"
    print("For the five given LSEs, the times required were:")
    for i in range(5):
        print(info.format(ind=i+1, time=1000*times[i]))
