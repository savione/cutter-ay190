#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS2 Problem 5

import numpy as np
import matplotlib.pyplot as pl
import scipy.interpolate as ip

# times of measurement
times = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])

# apparent magnitudes
mags = np.array([0.302, 0.185, 0.106, 0.093, 0.240, 0.579, 0.561, 0.468, 0.302])

# denser array of times
ts = np.linspace(times[0], times[-1], 10000)

# Piecewise cubic Hermite interpolation at `x` of the points given by `xs`, `fs`.
def p_cubic_hermite(x, xs, fs):
    # Find smallest interval containing `x`
    for j,x_ in enumerate(xs[1:]):
        i = j
        if x < x_:
            break
    # compute interpolation
    x1, x2 = xs[i], xs[i+1] # boundaries
    f1, f2 = fs[i], fs[i+1] # values at boundaries
    f1p = (f2 - f1) / (x2 - x1) # derivative of f(x) at x1
    if i == len(xs) - 2:
        f2p = f1p
    else:
        f2p = (fs[i+2] - f2) / (xs[i+2] - x2) # derivative of f(x) at x2    
    z = (x - x1) / (x2 - x1)
    return f1 * psi0(z) + f2 * psi0(1 - z) + f1p * (x2 - x1) * psi1(z) \
        - f2p * (x2 - x1) * psi1(1 - z)

def psi0(z):
    return 2*z*z*z - 3*z*z + 1

def psi1(z):
    return z*z*z - 2*z*z + z

hermite_plot, = pl.plot(ts, [p_cubic_hermite(t, times, mags) for t in ts], 'r', linewidth=4)
data_plot, = pl.plot(times, mags, 'ko', markersize=10)

pl.legend( (data_plot, hermite_plot), ("Measurements", "Piecewise cubic Hermite interpolation"), frameon=False, loc=(0.05, 0.85) )

pl.xlim(-0.1, 1.1)
pl.ylim(0, 0.8)

pl.xlabel("Time [days]")
pl.ylabel("Apparent Magnitude")
pl.title("Cepheid Lightcurve")

pl.savefig("problem5_fig1.pdf")
pl.show()

pl.cla()

tck = ip.splrep(times, mags, s=0)

spline_plot, = pl.plot(ts, ip.splev(ts, tck, der=0), 'r',\
                       linewidth=4)
data_plot, = pl.plot(times, mags, 'ko', markersize=10)

pl.legend( (data_plot, spline_plot), ("Measurements", "Cubic spline interpolation"), frameon=False, loc=(0.05, 0.85) )

pl.xlim(-0.1, 1.1)
pl.ylim(0, 0.8)

pl.xlabel("Time [days]")
pl.ylabel("Apparent Magnitude")
pl.title("Cepheid Lightcurve")

pl.savefig("problem5_fig2.pdf")
pl.show()
