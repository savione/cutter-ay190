#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS3 Problem 2

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as pl
import plot_defaults

# Part A

# parameters
max_exponent = 8

number_density_coeff = 1.05495e35 # cm^(-3)

ns = [2**p for p in range(1,max_exponent)]

def f(x):
    return x * x * np.exp(x) / (np.exp(x) + 1)

[xs, ws] = sp.l_roots(ns[-1], 0)

Qs = np.array([np.sum(ws[:n] * f(xs)[:n]) for n in ns])
number_densities = number_density_coeff * Qs

print("\nPart A\n")
print("Number of nodes:\n{}".format(ns))
print("Number density [cm^(-3)]:\n{}".format(number_densities))
print("Change in number density [cm^(-3)]:\n{}".format(number_densities[1:]
                                             - number_densities[:-1]))

# Answer: 1.902*10^35 cm^(-3)

# Part B

# parameters
n = 100 # number of nodes in Legendre Quadrature
dE = 5 # energy bin size (MeV)
max_E = 155 # energy cutoff (MeV)

Es = np.arange(0, 155, dE) # energies
xs = Es / 20.0 # x parameter, energy / temperature (20 MeV)

def x(y, a, b):
    return 0.5 * (y + 1) * (b - a) + a

def f(y, a, b):
    x_ = x(y, a, b)
    return 0.5 * (b - a) * x_ * x_ / (np.exp(x_) + 1)

[ys, ws] = sp.p_roots(n, 0)

Qs = np.array([np.sum(ws * f(ys, xs[i], xs[i+1])) for i in range(len(xs) - 1)])

print("\nPart B\n")
print("Total number density: {}".format(number_density_coeff * np.sum(Qs)))

myfig = pl.figure(figsize=(10,8))
myfig.subplots_adjust(left=0.13)
myfig.subplots_adjust(bottom=0.14)
myfig.subplots_adjust(top=0.90)
myfig.subplots_adjust(right=0.95)
pl.bar(Es[:-1], number_density_coeff * Qs / 10**34, color='c', width=5)
pl.xlim(0, max_E - dE)
pl.xlabel("Energy bin [MeV]")
pl.ylabel(r"Number density [$\times 10^{34}$ cm$^{-3}$]")
pl.title("Number Density versus Energy", fontsize=30)
pl.savefig("problem2b.pdf")
pl.show()
