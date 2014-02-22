#!/usr/bin/env python

# Cutter Coryell
# Ay 190 WS11
# FTCS implementation

import sys,math
import numpy as np
import matplotlib.pyplot as mpl
import scipy as sp
import plot_defaults

# apply boundary conditions
def apply_bcs(y):
    # FILL IN CODE
    y[0], y[-1] = y[1], y[-2]
    return y

# calculate new values for y based on the old value `yold`
def calc_rhs(yold):
    y = yold[1:-1] - (0.5 * v * dt / dx) * (yold[2:] - yold[:-2])
    y = np.append(0, y) # padding with dummy values
    y = np.append(y, 0)
    return y

# Analytic solution to convection equation with a Gaussian initial condition
def analytic(x, t, v=0.1, x0=30.0, sigma=np.sqrt(15.0)):
    return np.exp(-(x - v * t - x0)**2 / (2 * sigma**2))

# set up the grid here. Use a decent number of zones;
# perhaps to get a dx of 0.1
x = np.linspace(0, 100, 1000)
# parameters
dx = x[1]-x[0]
n = len(x)
y = np.zeros(n)
cfl = 1.0
dt = 0.1
v = 0.9 * dx / dt

# for initial data
sigma = np.sqrt(15.0)
x0 = 30.0

def run():

    #set up initial conditions
    t = 0.0
    y = analytic(x, t, sigma=sigma)

    # evolve (and show evolution)
    mpl.ion()
    mpl.figure()
    mpl.plot(x,y,'x-') # numerical data
    mpl.plot(x,analytic(x, t, sigma=sigma),'r-') # analytic data
    mpl.show()

    yold2 = y
    yold = y
    ntmax = 70    
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
        # get analytic result for time t
        yana = analytic(x, t, sigma=sigma)

        # compute error estimage
        err = np.sqrt(np.sum((y - yana)**2) / n)
        errs.append(err)

        print "it = ",it,err
        mpl.clf()
        # plot numerical result
        mpl.plot(x,y,'x-',lw=4)
        # plot analytic results
        mpl.plot(x,yana,'r-',lw=4)
        mpl.draw()

    mpl.show()

    mpl.savefig("fig-ftcs0.pdf")

    mpl.clf()

    return errs

errs1 = run()

# sigma /= 5

# errs2 = run()

# mpl.ioff()

# # set up the figure and control white space
# myfig = mpl.figure(figsize=(10,8))
# myfig.subplots_adjust(left=0.2)
# myfig.subplots_adjust(bottom=0.16)
# myfig.subplots_adjust(top=0.85)
# myfig.subplots_adjust(right=0.85)

# ax = mpl.gca()

# mpl.xscale("log")
# mpl.yscale("log")
# mpl.xlim(1, len(errs1))
# mpl.title("Error Between Numerical\nand Analytic Result", fontsize=30)
# mpl.ylabel("Error (Euclidean Norm)", labelpad=15)
# mpl.xlabel("Iteration", labelpad=15)
# err_plot1, = mpl.plot(errs1,lw=8)
# err_plot2, = mpl.plot(errs2,lw=8)
# mpl.legend((err_plot1, err_plot2),
#            ("$\sigma$ = \sqrt{15}", "$\sigma$ = \sqrt{15}/5"),
#            frameon=False, loc="lower right")
# mpl.savefig("fig-errors2.pdf")
# mpl.show()
