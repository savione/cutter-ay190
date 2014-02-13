#!/usr/bin/env python

# Cutter Coryell
# Ay 190 WS10

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
import plot_defaults

# parameters
npoints = 1000

# set up grid
xmin = 0.0
xmax = 1.0
# set up grid
x = np.linspace(xmin, xmax, npoints)
# dx based on x[1] and x[0]
dx = x[1] - x[0]

# boundary values
A = 0 # inner boundary
B = 0.1 # outer boundary

def y_analyt(x):
    return 2.0*x**3 - 2*x**2 + 0.1*x

def calc_rhs(u,xx):
    # rhs routine
    # rhs[0] is rhs for y
    # rhs[1] is rhs for u
    rhs = np.zeros(2)
    rhs[0] = u 
    rhs[1] = 12*xx - 4

    return rhs

def integrate_FE(z,x):
    # forward-Euler integrator
    
    # make an array for all points
    # entry 0 contains y
    # entry 1 contains y'
    yy = np.zeros((npoints,2))

    yy[0,0] = A # boundary value A for y at x=0
    yy[0,1] = z # guessed boundary value for y' at x=0

    for i in range(npoints-1):
        yy[i+1,:] = yy[i,:] + dx*calc_rhs(yy[i,1],x[i])

    return yy

def integrate_RK4(z,x):
    # fourth order Runge-Kutta integrator
    
    # make an array for all points
    # entry 0 contains y
    # entry 1 contains y'
    yy = np.zeros((npoints,2))

    yy[0,0] = A # boundary value A for y at x=0
    yy[0,1] = z # guessed boundary value for y' at x=0

    for i in range(npoints-1):
        k1 = dx * calc_rhs(yy[i,1], x[i])
        k2 = dx * calc_rhs(yy[i,1] + 0.5*k1[1], x[i] + 0.5*dx)
        k3 = dx * calc_rhs(yy[i,1] + 0.5*k2[1], x[i] + 0.5*dx)
        k4 = dx * calc_rhs(yy[i,1] + k3[1], x[i] + dx)
        yy[i+1,:] = yy[i,:] + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return yy

print("{} points".format(npoints))



def compute_BVP_ODE(integrator_name):

    if integrator_name == "FE":
        integrator = integrate_FE
    elif integrator_name == "RK4":
        integrator = integrate_RK4
    else:
        raise ValueError("Invalid integrator name")
    print(integrator_name + ':')

    # get initial guess for derivative
    z0 = -1100000.0
    z1 = -10000000.0
    yy0 = integrator(z0,x)
    yy1 = integrator(z1,x)
    phi0 = yy0[npoints-1,0] - B
    phi1 = yy1[npoints-1,0] - B
    dphidz = (phi1 - phi0) / (z1 - z0) # dphi/dz

    i = 0
    itmax = 100
    err = 1.0e99
    criterion = 1.0e-12

    z0 = z1
    phi0 = phi1
    while (err > criterion and i < itmax):
        z1 = z0 - phi0 / dphidz # secant update
        yy = integrator(z1,x)
        phi1 = yy[npoints-1,0] - B
        dphidz = (phi1 - phi0) / (z1 - z0) # dphi/dz numerical
        err = np.abs(phi1 / B) # your error measure
        z0 = z1
        phi0 = phi1
        i = i+1

        print i, z1, phi1, yy[npoints/2, 0] - y_analyt(x[npoints/2])

    return yy

yy_FE, yy_RK4 = compute_BVP_ODE("FE"), compute_BVP_ODE("RK4")

myfig = pl.figure(figsize=(10,8))
myfig.subplots_adjust(left=0.18)
myfig.subplots_adjust(bottom=0.15)
myfig.subplots_adjust(top=0.9)
myfig.subplots_adjust(right=0.85)

analyt, = pl.plot(x,y_analyt(x),"k-", linewidth=8)
numer_FE, = pl.plot(x,yy_FE[:,0],"r--", linewidth=8)
numer_RK4, = pl.plot(x,yy_RK4[:,0],"c--", linewidth=6)

ax = pl.gca()
ax.set_yticks(np.linspace(-0.2, 0.1, 4))
ax.set_ylabel("$y$", labelpad=0)
ax.set_xlabel("$x$", labelpad=15)
pl.ylim((-0.25, 0.2))
pl.legend((analyt, numer_FE, numer_RK4), ("Analytical Solution",
          "Forward-Euler Solution", "RK4 Solution"),
          frameon=False, loc="upper center")
pl.savefig("fig-{}pts.pdf".format(npoints))