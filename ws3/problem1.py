#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS3 Problem 1

import numpy as np

# Integration Methods

# `Integrates the function `f` from `a` to `b` using the midpoint rule with
# `N` grid-points.
def midpoint(f, a, b, N):
    xs = np.linspace(a, b, N)
    dxs = xs[1:] - xs[:-1] # interval lengths
    ms = xs[:-1] + 0.5 * dxs # midpoints
    return np.sum(dxs * f(ms))

# `Integrates the function `f` from `a` to `b` using the trapezoidal rule with
# `N` grid-points.
def trapezoidal(f, a, b, N):
    xs = np.linspace(a, b, N)
    dxs = xs[1:] - xs[:-1] # interval lengths
    fxs = f(xs) # function values at grid points
    return np.sum(0.5 * dxs * (fxs[1:] + fxs[:-1]))
 
# `Integrates the function `f` from `a` to `b` using Simpson's rule with
# `N` grid-points.
def simpsons(f, a, b, N):
    xs = np.linspace(a, b, N)
    dxs = xs[1:] - xs[:-1] # interval lengths
    ms = xs[:-1] + 0.5 * dxs # midpoints
    fxs = f(xs) # function values at grid points
    fms = f(ms) # function values at midpoints
    return np.sum((1.0/6.0) * dxs * (fxs[1:] + 4*fms + fxs[:-1]))

# Helper function for computing relative error
def rel_err(a, b):
    return (a - b) / b


# Part A

a = 0
b = np.pi
pts1 = 100
pts2 = 2 * pts1

def f(x):
    return np.sin(x)

Q_exact = 2 # analytical integral of `f` from 0 to pi
Q_mid1, Q_mid2 = midpoint(f, a, b, pts1), midpoint(f, a, b, pts2)
Q_trap1, Q_trap2 = trapezoidal(f, a, b, pts1), trapezoidal(f, a, b, pts2)
Q_simpsons1, Q_simpsons2 = simpsons(f, a, b, pts1), simpsons(f, a, b, pts2)

err_mid1, err_mid2 = rel_err(Q_mid1, Q_exact), rel_err(Q_mid2, Q_exact)
err_trap1, err_trap2 = rel_err(Q_trap1, Q_exact), rel_err(Q_trap2, Q_exact)
err_simpsons1, err_simpsons2 = rel_err(Q_simpsons1, Q_exact), rel_err(Q_simpsons2, Q_exact)

print("""Part A

Relative errors in approximations at {pts} grid points:
Midpoint \t {mid}
Trapezoidal \t {trap}
Simpson's \t {simpsons}
""".format(pts=pts1, mid=err_mid1, trap=err_trap1, simpsons=err_simpsons1))

print("""
Convergence order:
Midpoint \t {mid} (should be 1)
Trapezoidal \t {trap} (should be 2)
Simpson's \t {simpsons} (should be 4)
""".format(mid=np.log2(err_mid1/err_mid2), trap=np.log2(err_trap1/err_trap2),
           simpsons=np.log2(err_simpsons1/err_simpsons2)))


# Part B

a = 0
b = np.pi
pts1 = 100
pts2 = 2 * pts1

def f(x):
    return x * np.sin(x)

Q_exact = np.pi # analytical integral of `f` from 0 to pi
Q_mid1, Q_mid2 = midpoint(f, a, b, pts1), midpoint(f, a, b, pts2)
Q_trap1, Q_trap2 = trapezoidal(f, a, b, pts1), trapezoidal(f, a, b, pts2)
Q_simpsons1, Q_simpsons2 = simpsons(f, a, b, pts1), simpsons(f, a, b, pts2)

err_mid1, err_mid2 = rel_err(Q_mid1, Q_exact), rel_err(Q_mid2, Q_exact)
err_trap1, err_trap2 = rel_err(Q_trap1, Q_exact), rel_err(Q_trap2, Q_exact)
err_simpsons1, err_simpsons2 = rel_err(Q_simpsons1, Q_exact), rel_err(Q_simpsons2, Q_exact)


print("""Part B

Relative errors in approximations at {pts} grid points:
Midpoint \t {mid}
Trapezoidal \t {trap}
Simpson's \t {simpsons}
""".format(pts=pts1, mid=err_mid1, trap=err_trap1, simpsons=err_simpsons1))

print("""
Convergence order:
Midpoint \t {mid} (should be 1)
Trapezoidal \t {trap} (should be 2)
Simpson's \t {simpsons} (should be 4)
""".format(mid=np.log2(err_mid1/err_mid2), trap=np.log2(err_trap1/err_trap2),
           simpsons=np.log2(err_simpsons1/err_simpsons2)))
