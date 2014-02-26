#!/usr/bin/env python

# Cutter Coryell
# Ay 190 WS12

import numpy as np
import matplotlib.pyplot as pl
import scipy.interpolate as ip
import plot_defaults

### CONSTANTS ###
G = 6.67e-8

### PARAMETERS ###
npoints = 2000
density_homog = 10**5

# load the data
data = np.loadtxt("presupernova.dat").T

radius = data[2]
density = data[4]
for i, r in enumerate(radius):
    if r > 10**9:
        i_max = i
        break
radius = radius[:i_max]
density = density[:i_max]

# set up the figure and control white space
myfig = pl.figure(figsize=(10,8))
myfig.subplots_adjust(left=0.2)
myfig.subplots_adjust(bottom=0.16)
myfig.subplots_adjust(top=0.85)
myfig.subplots_adjust(right=0.85)
ax = pl.gca()

# plot the density
pl.xscale("log")
pl.yscale("log")
pl.xlim(radius[0], radius[-1] + 10**8)
pl.title("Density versus Radius\nin the Pre-Supernova Star", fontsize=30)
pl.ylabel(r"Density $\rho$ [g cm$^{-3}$]", labelpad=0)
pl.xlabel("Radius $r$ [cm]", labelpad=25)
density_plot, = pl.plot(radius, density,lw=8)
pl.savefig("fig-density.pdf")
pl.close()

# interpolate density in presupernova star
rs = np.linspace(1, radius[-1], npoints)
tck = ip.splrep(radius, density, s=0)
rhos_presn = ip.splev(rs, tck, der=0)

# density grid for homogenous star
rhos_homog = density_homog * np.ones(npoints)

# computes the analytic expression for phi for the homogenous star
def phi_analytic(rs):
    return (2.0/3.0) * np.pi * G * density_homog * (rs**2 - 3*rs[-1]**2)

# compute M(r)

# returns the RHS for the 1st order ODE for M(r)
def M_rhs(r, rho):
    return 4 * np.pi * r**2 * rho

# integrates M(r) to M(r+dr) using the Forward Euler method
def M_integrate_FE(r, dr, M, rho):
    return M + dr * M_rhs(r, rho)

# computes M(r) on equidistant grid of radii `rs`
def compute_M(rs, rhos):
    dr = rs[1] - rs[0]
    Ms = np.zeros(npoints)
    for i in range(npoints - 1):
        Ms[i+1] = M_integrate_FE(rs[i], dr, Ms[i], rhos[i])
    return Ms

# compute z(r)

# returns the RHS for the 1st order ODE for z(r)
def z_rhs(r, z, rho):
    return 4 * np.pi * G * rho - 2 * z / r

# integrates z(r) to z(r+dr) using the Forward Euler method
def z_integrate_FE(r, dr, z, rho):
    return z + dr * z_rhs(r, z, rho)

# computes z(r) on equidistant grid of radii `rs`
def compute_z(rs, rhos):
    dr = rs[1] - rs[0]
    zs = np.zeros(npoints)
    for i in range(npoints - 1):
        zs[i+1] = z_integrate_FE(rs[i], dr, zs[i], rhos[i])
    return zs

# compute phi(r)

# integrates phi(r) to phi(r+dr) using the Forward Euler method
def phi_integrate_FE(dr, phi, z):
    return phi + dr * z

# computes phi(r) on equidistant grid of radii `rs`
def compute_phi(rs, rhos):
    dr = rs[1] - rs[0]
    zs = compute_z(rs, rhos)
    Ms = compute_M(rs, rhos)
    phis = np.zeros(npoints)
    for i in range(npoints - 1):
        phis[i+1] = phi_integrate_FE(dr, phis[i], zs[i])
    # correct for outer boundary condition, phi(r_max) = - GM(r_max) / r_max
    phis -= (phis[-1] + G * Ms[-1] / rs[-1])
    return phis

# homogenous star case

# set up the figure and control white space
myfig = pl.figure(figsize=(10,8))
myfig.subplots_adjust(left=0.2)
myfig.subplots_adjust(bottom=0.16)
myfig.subplots_adjust(top=0.85)
myfig.subplots_adjust(right=0.85)
ax = pl.gca()

# plot the potential
pl.xscale("log")
pl.xlim(radius[0], radius[-1] + 10**8)
pl.title("Gravitational Potential\nfor Homogenous Sphere", fontsize=30)
pl.ylabel(r"Potential $\phi$ [$\times 10^{16}$ erg g$^{-1}$]", labelpad=0)
pl.xlabel("Radius $r$ [cm]", labelpad=25)
phi_analyt, = pl.plot(rs, phi_analytic(rs) / 10**16, lw=8)
phi_numer, = pl.plot(rs, compute_phi(rs, rhos_homog) / 10**16, "r--", lw=8)
pl.legend((phi_analyt, phi_numer),
          ("Analytic", "Numerical"),
          frameon=False, loc="upper left")
pl.savefig("fig-potential-homogenous.pdf")
pl.close()

# presupernova star case

# set up the figure and control white space
myfig = pl.figure(figsize=(10,8))
myfig.subplots_adjust(left=0.2)
myfig.subplots_adjust(bottom=0.16)
myfig.subplots_adjust(top=0.85)
myfig.subplots_adjust(right=0.85)
ax = pl.gca()

# plot the potential
pl.xscale("log")
pl.xlim(radius[0], radius[-1] + 10**8)
pl.title("Gravitational Potential\nfor Pre-Supernova Star", fontsize=30)
pl.ylabel(r"Potential $\phi$ [$\times 10^{18}$ erg g$^{-1}$]", labelpad=0)
pl.xlabel("Radius $r$ [cm]", labelpad=25)
pl.plot(rs, compute_phi(rs, rhos_presn) / 10**18, lw=8)
pl.savefig("fig-potential-presupernova.pdf")
pl.close()

# convergence for homogenous case

npoints = 20
rs = np.linspace(1, radius[-1], npoints)
rhos_homog = density_homog * np.ones(npoints)
phi1_numer = compute_phi(rs, rhos_homog)[0]
phi1_analyt = phi_analytic(rs)[0]

npoints = 40
rs = np.linspace(1, radius[-1], npoints)
rhos_homog = density_homog * np.ones(npoints)
phi2_numer = compute_phi(rs, rhos_homog)[0]
phi2_analyt = phi_analytic(rs)[0]

print abs((phi1_numer - phi1_analyt) / (phi2_numer - phi2_analyt))
