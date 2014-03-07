#!/usr/bin/env python

# Cutter Coryell
# Ay 190 WS14

import sys,math
import numpy as np
import matplotlib.pyplot as mpl
import scipy as sp
import plot_defaults

# Parameters
L = 100
n = L / 0.1

# apply boundary conditions
def apply_bcs(y):
    # FILL IN CODE
    y[0], y[-1] = y[1], y[-2]
    return y


# calculate new values for y based on the old value `yold`
def calc_rhs(yold):
    ys = yold.copy()
    for i,y in enumerate(yold):
        if y > 0 and i > 0:
            ys[i] = y * (1 - (dt / dx) * (y - yold[i-1]))
        if y < 0 and i < n - 1:
            ys[i] = y * (1 - (dt / dx) * (yold[i+1] - y))
    return ys

# Analytic sine wave psi_0
def psi_0(x, L):
    return 1./8 * np.sin(2*np.pi * x / L)

# set up the grid here. Use a decent number of zones;
# perhaps to get a dx of 0.1
x = np.linspace(0, L, n)
# parameters
dx = x[1]-x[0]
y = np.zeros(n)
dt = 0.1

def run():

    #set up initial conditions
    t = 0.0
    y = psi_0(x, L)

    # evolve (and show evolution)
    mpl.ion()
    mpl.figure()
    mpl.plot(x,y,'x-') # numerical data
    mpl.show()

    yold2 = y
    yold = y
    ntmax = 2000    
    errs = []

    for it in range(ntmax):
        t += dt
        # save previous and previous previous data
        yold2 = yold
        yold = y
        # get new data; ideally just call a function
        y = calc_rhs(yold)
        # after update, apply boundary conditions
        apply_bcs(y) 

        print "it = ",it, ", time = ",t
        mpl.clf()
        # plot numerical result
        mpl.plot(x,y,'x-',lw=4)
        mpl.draw()
        if round(t) % 50 == 0:
            mpl.savefig("fig_t-{}.pdf".format(int(round(t))))

    mpl.show()

    mpl.clf()

run()