#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS4 Problem 1

import numpy as np

def bisect_root_finder(f, a_init, b_init, error_threshold, itercount=False):
    if f(a_init) * f(b_init) >= 0:
        raise ValueError("Function does not have opposite signs at the given initial points")
    a, b = a_init, b_init
    iterations = 0
    while True:
        iterations += 1
        c = 0.5 * (a + b)
        f_a, f_b, f_c = f(a), f(b), f(c)
        if f_c == 0 or np.abs((f_c - f_a) / f_a) < error_threshold:
            if itercount:
                return (c, iterations)
            else:
                return c
        if f_a * f_c < 0:
            a, b = a, c
        else: 
            assert(f_c * f_b < 0)
            a, b = c, b

# Algorithmic parameters
error_threshold = 10**(-10)

# Physical parameters
T = 365.25635 # days
a = 1 # AU
e1 = 0.0167
e2 = 0.99999
t1, t2, t3 = 91, 182, 273

# Derived quantities
b1 = a * np.sqrt(1 - e1*e1)
b2 = a * np.sqrt(1 - e2*e2)
omega = 2 * np.pi / T

# Planet coordinates

def x(E, a):
    return a * np.cos(E)

def y(E, b):
    return b * np.sin(E)

# `E` satisfies the equation "anomaly_func(E, t, omega, e) = 0"
def anomaly_func(E, t, omega, e):
    return omega * t + e * np.sin(E) - E

# Part (a)

E1, iters1 = bisect_root_finder(lambda E: anomaly_func(E, t1, omega, e1),
                                 0, 2*np.pi, error_threshold, itercount=True)

E2, iters2 = bisect_root_finder(lambda E: anomaly_func(E, t2, omega, e1),
                                 0, 2*np.pi, error_threshold, itercount=True)

E3, iters3 = bisect_root_finder(lambda E: anomaly_func(E, t3, omega, e1),
                                 0, 2*np.pi, error_threshold, itercount=True)

info = """Day {day}: \tE = {E:.3f}, (x, y) = ({x:.3f}, {y:.3f});
\t\t{iters} iterations required."""

print("\nEccentricity e = {e}, E in (0, 2*pi)\n".format(e=e1))
print(info.format(day=t1, E=E1, x=x(E1, a), y=y(E1, b1), iters=iters1))
print(info.format(day=t2, E=E2, x=x(E2, a), y=y(E2, b1), iters=iters2))
print(info.format(day=t3, E=E3, x=x(E3, a), y=y(E3, b1), iters=iters3))

# Part (b)

E1, iters1 = bisect_root_finder(lambda E: anomaly_func(E, t1, omega, e2),
                                 0, 2*np.pi, error_threshold, itercount=True)

E2, iters2 = bisect_root_finder(lambda E: anomaly_func(E, t2, omega, e2),
                                 0, 2*np.pi, error_threshold, itercount=True)

E3, iters3 = bisect_root_finder(lambda E: anomaly_func(E, t3, omega, e2),
                                 0, 2*np.pi, error_threshold, itercount=True)

info = """Day {day}: \tE = {E:.3f}, (x, y) = ({x:.3f}, {y:.3f});
\t\t{iters} iterations required."""

print("\nEccentricity e = {e}, E in (0, 2*pi)\n".format(e=e2))
print(info.format(day=t1, E=E1, x=x(E1, a), y=y(E1, b2), iters=iters1))
print(info.format(day=t2, E=E2, x=x(E2, a), y=y(E2, b2), iters=iters2))
print(info.format(day=t3, E=E3, x=x(E3, a), y=y(E3, b2), iters=iters3))

# Narrow intervals (true answer should be near pi)

E1, iters1 = bisect_root_finder(lambda E: anomaly_func(E, t1, omega, e2),
                                 0.5*np.pi, 1.5*np.pi, error_threshold,
                                itercount=True)

E2, iters2 = bisect_root_finder(lambda E: anomaly_func(E, t2, omega, e2),
                                 0.5*np.pi, 1.5*np.pi, error_threshold,
                                itercount=True)

E3, iters3 = bisect_root_finder(lambda E: anomaly_func(E, t3, omega, e2),
                                 0.5*np.pi, 1.5*np.pi, error_threshold,
                                itercount=True)

info = """Day {day}: \tE = {E:.3f}, (x, y) = ({x:.3f}, {y:.3f});
\t\t{iters} iterations required."""

print("\nEccentricity e = {e}, E in (pi/2, 3*pi/2)\n".format(e=e2))
print(info.format(day=t1, E=E1, x=x(E1, a), y=y(E1, b2), iters=iters1))
print(info.format(day=t2, E=E2, x=x(E2, a), y=y(E2, b2), iters=iters2))
print(info.format(day=t3, E=E3, x=x(E3, a), y=y(E3, b2), iters=iters3))
