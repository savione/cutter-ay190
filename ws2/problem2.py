#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS2 Problem 2

import numpy as np
import matplotlib.pyplot as pl

def f(x):
    return x*x*x - 5*x*x + x

def fprime(x):
    return 3*x*x - 10*x + 1

# Calculate an array of forward differences based on an input array of
# function values `values` and the fixed grid-spacing `dx`.
# The length of the returned array will be one shorter than the array of
# function values because forward-difference cannot be computed at the
# forward boundary. Hence the forward-boundary grid point will be omitted.
def forward_diffs(values, dx):
    coef = 1 / dx
    diffs = (values[1:] - values[:-1]) * coef
    return diffs

# Calculate an array of central differences based on an input array of
# function values `values` and the fixed grid-spacing `dx`.
# The length of the returned array will be two shorter than the array of
# function values because central-difference cannot be computed at the
# two boundaries. Hence the boundary grid points will be omitted.
def central_diffs(values, dx):
    coef = 0.5 / dx
    diffs = (values[2:] - values[:-2]) * coef
    return diffs

# Problem parameters
a = -2
b = 6
h1 = 1
h2 = 0.5 * h1

xs1 = np.arange(a,b,h1)
xs2 = np.arange(a,b,h2)

forward_error1 = forward_diffs(f(xs1), h1) - fprime(xs1)[:-1]

forward_error2 = forward_diffs(f(xs2), h2) - fprime(xs2)[:-1]

central_error1 = central_diffs(f(xs1), h1) - fprime(xs1)[1:-1]

central_error2 = central_diffs(f(xs2), h2) - fprime(xs2)[1:-1]

fe1_plot, = pl.plot(xs1[:-1], forward_error1, 'r--', linewidth=4)
fe2_plot, = pl.plot(xs2[:-1], forward_error2, 'r:', linewidth=4)
ce1_plot, = pl.plot(xs1[1:-1], central_error1, 'b--', linewidth=4)
ce2_plot, = pl.plot(xs2[1:-1], central_error2, 'b:', linewidth=4)

pl.xlabel("x")
pl.ylabel("Difference between Finite Difference and Derivative")
pl.title("Difference between Finite Difference and Derivative")

pl.legend( (fe1_plot, fe2_plot, ce1_plot, ce2_plot),
           ("Forward Difference, h = 1", "Forward Difference, h = 2",
            "Central Difference, h = 1", "Central Difference, h = 2"),
           frameon=False, loc=(0.4, 0.05) )

pl.savefig("problem2_fig1.pdf")

pl.clf()

fe_plot, = pl.plot(xs1[:-1], forward_error1 / forward_error2[:-1:2], 'r', linewidth=4)
ce_plot, = pl.plot(xs1[1:-1], central_error1 / central_error2[:-2:2], 'b', linewidth=4)

#pl.xlim((-2, 6))
pl.ylim((0, 5))

pl.xlabel("x")
pl.ylabel("Factor Increase in Error when Step-Size is Halved")
pl.title("Convergence Factors of Finite Differencing")

pl.legend( (fe_plot, ce_plot),
           ("Forward Difference", "Central Difference"),
           frameon=False, loc=(0.5, 0.05) )

pl.savefig("problem2_fig2.pdf")
