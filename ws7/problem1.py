#!/usr/bin/env python2.7

# Cutter Coryell
# Ay 190
# WS7 Problem 1

import numpy as np
import matplotlib.pyplot as pl
import plot_defaults

# Parameters
N = 10**2

# Approximate pi using an MC experiment with `N` trials. Returns the
# approximation. If extra=True, returns a triple containing the approximation,
# a numpy 2D array containing the test points, and a numpy 2D array containing
# those test points that are interior to the circle.
def MC_pi(N, extra=False):
    np.random.seed(1)
    test_points = np.random.rand(N, 2)
    interior_points = []
    for pt in test_points:
        if pt[0]*pt[0] + pt[1]*pt[1] < 1:
            interior_points.append(pt)
    n = len(interior_points)
    approx = 4.0 * n / N
    if not extra:
        return approx
    else:
        return (approx, test_points, np.reshape(interior_points, (n, 2)))

# set up the figure and control white space
myfig = pl.figure(figsize=(10,10))
myfig.subplots_adjust(left=0.13)
myfig.subplots_adjust(bottom=0.12)
myfig.subplots_adjust(top=0.85)
myfig.subplots_adjust(right=0.95)

# prepare x and y ranges
xmin = 0
xmax = 1
ymin = 0
ymax = 1

# set axis parameters
pl.axis([xmin,xmax,ymin,ymax])
# get axis object
ax = pl.gca()
# set locators of tick marks
xminorLocator = pl.MultipleLocator(0.04)
xmajorLocator = pl.MultipleLocator(0.2)
yminorLocator = pl.MultipleLocator(0.04)
ymajorLocator = pl.MultipleLocator(0.2)
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)

# label the axes
pl.xlabel("$x$",labelpad=15)
pl.ylabel("$y$",labelpad=15)


# run the experiment
(approx, test_points, interior_points) = MC_pi(N, extra=True)
xs = np.linspace(0,1,10000)

pl.title("Monte Carlo Approximation of $\pi$\nwith {} Points: {}".format(N, approx), fontsize=40)
pl.plot(xs, np.sqrt(1-xs*xs))
pl.plot(test_points[:,0], test_points[:,1], 'go')
pl.plot(interior_points[:,0], interior_points[:,1], 'ro')
pl.savefig("fig-problem1.pdf")

# now check convergence

Ns = 4**np.arange(1, 10)
print(Ns)
for N in Ns:
    approx = MC_pi(N)
    print "{N} & {approx} & {err:.3f} \\\\".format(N=N, approx=approx,
                                               err = abs((approx-np.pi)/np.pi))
