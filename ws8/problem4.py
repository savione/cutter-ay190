#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS8 Problem 4

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
import plot_defaults

# global constants
ggrav = 6.67e-8
msun  = 1.99e33

# EOS parameters
# for white dwarfs:
polyG = 4.0/3.0
polyK = 1.244e15*0.5**polyG


#######################################
# function definitions

# pressure as a function of density
def p2rho(p):
    return (p / polyK) ** (1 / polyG)

def tov_RHS(rad,rho,m):
    
    # RHS function
    
    rhs = np.zeros(2)
    if(rad > 1.0e-10):
        rhs[0] = - ggrav * m * rho / (rad * rad) # dP/dr
        rhs[1] = 4 * np.pi * rho * rad * rad # dM/dr
    else:
        rhs[0] = 0.0
        rhs[1] = 0.0

    return rhs

def tov_integrate_FE(rad,dr,p,rho,m):

    # Forward-Euler Integrator

    new = np.zeros(2)
    old = np.zeros(2)
    old[0] = p
    old[1] = m

    # forward Euler integrator
    new = old + dr * tov_RHS(rad,rho,m)
    
    # assign outputs
    pnew = new[0]
    mnew = new[1]
    
    return (pnew,mnew)

def tov_integrate_RK2(rad,dr,p,rho,m):

    # Second-Order Runge-Kutta Integrator

    new = np.zeros(2)
    old = np.zeros(2)
    old[0] = p
    old[1] = m

    k1 = dr * tov_RHS(rad,rho,m)
    k2 = dr * tov_RHS(rad + 0.5*dr, p2rho(p + 0.5*k1[0]), m + 0.5*k1[1])

    new = old + k2
    
    # assign outputs
    pnew = new[0]
    mnew = new[1]
    
    return (pnew,mnew)

def tov_integrate_RK3(rad,dr,p,rho,m):

    # Third-Order Runge-Kutta Integrator

    new = np.zeros(2)
    old = np.zeros(2)
    old[0] = p
    old[1] = m

    k1 = dr * tov_RHS(rad,rho,m)
    k2 = dr * tov_RHS(rad + 0.5*dr, p2rho(p + 0.5*k1[0]), m + 0.5*k1[1])
    k3 = dr * tov_RHS(rad + dr, p2rho(p - k1[0] + 2*k2[0]), m - k1[1] + 2*k2[1])

    new = old + (k1 + 4*k2 + k3)/6.0
    
    # assign outputs
    pnew = new[0]
    mnew = new[1]
    
    return (pnew,mnew)

def tov_integrate_RK4(rad,dr,p,rho,m):

    # Fourth-Order Runge-Kutta Integrator

    new = np.zeros(2)
    old = np.zeros(2)
    old[0] = p
    old[1] = m

    k1 = dr * tov_RHS(rad,rho,m)
    k2 = dr * tov_RHS(rad + 0.5*dr, p2rho(p + 0.5*k1[0]), m + 0.5*k1[1])
    k3 = dr * tov_RHS(rad + 0.5*dr, p2rho(p + 0.5*k2[0]), m + 0.5*k2[1])
    k4 = dr * tov_RHS(rad + dr, p2rho(p + k3[0]), m + k3[1])

    new = old + (k1 + 2*k2 + 2*k3 + k4)/6.0
    
    # assign outputs
    pnew = new[0]
    mnew = new[1]
    
    return (pnew,mnew)

#######################################

# Returns a tuple containing the radius, pressure, density, and mass arrays.
# The radius is in km whereas the pressure and density are fractions of central
# values and the mass is a fraction of the total mass.
def stellar_structure(integrate, npoints):
    # set up grid
    radmax = 2.0e8 # 2000 km
    radius = np.linspace(0, radmax, npoints)
    dr = radius[1]-radius[0]

    # set up variables
    press = np.zeros(npoints)
    rho   = np.zeros(npoints)
    mass  = np.zeros(npoints)

    # set up central values
    rho[0]   = 1.0e10
    press[0] = polyK * rho[0]**polyG
    mass[0]  = 0.0

    # set up termination criterion
    press_min = 1.0e-10 * press[0]

    nsurf = 0
    for n in range(npoints-1):
    
        (press[n+1],mass[n+1]) = integrate(radius[n], dr, press[n],
                                                   rho[n], mass[n])
        # check for termination criterion
        if(press[n+1] < press_min and nsurf==0):
            nsurf = n

        if(n+1 > nsurf and nsurf > 0):
            press[n+1] = press[nsurf]
            rho[n+1]   = rho[nsurf]
            mass[n+1]  = mass[nsurf]

        # invert the EOS to get density
        rho[n+1] = p2rho(press[n+1])

    return (radius / 10**5, press / press[0], rho / rho[0], mass / mass[nsurf])

r, p, rho, m = stellar_structure(tov_integrate_RK4, 10000)

# set up the figure and control white space
myfig = pl.figure(figsize=(10,10))
myfig.subplots_adjust(left=0.13)
myfig.subplots_adjust(bottom=0.12)
myfig.subplots_adjust(top=0.9)
myfig.subplots_adjust(right=0.85)

ax1 = pl.gca()
ax2 = ax1.twinx()

# label the axes
ax1.set_xlabel("Radius [km]", labelpad=15)
ax1.set_ylabel("Pressure and Density over Central Values",labelpad=15)
ax2.set_ylabel("Enclosed Mass over Total Mass",labelpad=15)

ax1.set_yticks(np.arange(0.2, 1.2, 0.2))
ax2.set_yticks(np.arange(0.2, 1.2, 0.2))

pl.title("Stellar Structure", fontsize=40)
press, = ax1.plot(r, p)
dens, = ax1.plot(r, rho)
mass, = ax2.plot(r, m, 'r')

pl.legend((press, dens, mass), ("Pressure", "Density", "Enclosed Mass"), frameon=False, loc=5)

pl.savefig("fig-problem4.pdf")
