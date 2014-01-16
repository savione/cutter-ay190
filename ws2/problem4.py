#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS2 Problem 4

import numpy as np
import matplotlib.pyplot as pl

# times of measurement
times = np.array([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])

# apparent magnitudes
mags = np.array([0.302, 0.185, 0.106, 0.093, 0.240, 0.579, 0.561, 0.468, 0.302])

# denser array of times
ts = np.linspace(times[0], times[-1], 10000)

# Lagrange interpolation at `x` of the points given by `xs`, `fs`.
def p_lagrange(x, xs, fs):
    return np.sum( fs * np.array([L(x, j, xs) for j,_ in enumerate(xs)]) )

def L(x, j, xs):
    xs_cut = np.concatenate((xs[:j], xs[j+1:])) 
    return np.product( (x - xs_cut) / (xs[j] - xs_cut) )


# Linear interpolation at `x` of the points given by `xs`, `fs`.
def p_linear(x, xs, fs):
    # Find smallest interval containing `x`
    for j,x_ in enumerate(xs[1:]):
        i = j
        if x < x_:
            break
    # compute interpolation
    x1, x2 = xs[i], xs[i+1] # interval boundaries
    f1, f2 = fs[i], fs[i+1] # values at boundaries
    return f1 + (x - x1) * (f2 - f1) / (x2 - x1)


# Quadratic interpolation at `x` of the points given by `xs`, `fs`.
def p_quadratic(x, xs, fs):
    # Find smallest interval containing `x`
    for j,x_ in enumerate(xs[1:]):
        i = j
        if x < x_:
            break
    # if `x` is in the last interval, pretend it is in the
    # second-to-last interval (because of the right boundary issue)
    if i == len(xs) - 2:
        i -= 1
    # computei interpolation
    x1, x2, x3 = xs[i], xs[i+1], xs[i+2] # nodes
    f1, f2, f3 = fs[i], fs[i+1], fs[i+2] # values at nodes
    return f1 * (x - x2)*(x - x3) / ( (x1 - x2)*(x1 - x3) ) \
        + f2 * (x - x1)*(x - x3) / ( (x2 - x1)*(x2 - x3) ) \
        + f3 * (x - x1)*(x - x2) / ( (x3 - x1)*(x3 - x2) )

lagrange_plot, = pl.plot(ts, [p_lagrange(t, times, mags) for t in ts], 'r', linewidth=4)
data_plot, = pl.plot(times, mags, 'ko', markersize=10)

pl.legend( (data_plot, lagrange_plot), ("Measurements", "Lagrange interpolation"), frameon=False, loc=(0.1, 0.8) )

pl.xlim(-0.1, 1.1)
pl.ylim(0, 2.7)

pl.xlabel("Time [days]")
pl.ylabel("Apparent Magnitude")
pl.title("Cepheid Lightcurve")

pl.savefig("problem4_fig1.pdf")
pl.show()

pl.clf()

linear_plot, = pl.plot(ts, [p_linear(t, times, mags) for t in ts], 'r', linewidth=4)
quadratic_plot, = pl.plot(ts, [p_quadratic(t, times, mags) for t in ts], 'b', linewidth=4)
data_plot, = pl.plot(times, mags, 'ko', markersize=10)

pl.legend( (data_plot, lagrange_plot, quadratic_plot), ("Measurements", "Linear interpolation", "Quadratic interpolation"), frameon=False, loc=(0.03, 0.8) )

pl.xlim(-0.1, 1.1)
pl.ylim(0, 0.7)

pl.xlabel("Time [days]")
pl.ylabel("Apparent Magnitude")
pl.title("Cepheid Lightcurve")

pl.savefig("problem4_fig2.pdf")
pl.show()
