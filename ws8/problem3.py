#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS8 Problem 3

import numpy as np
import scipy as sp

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

# Returns a tuple containing the stellar radius in km the stellar mass in
# solar masses, and the stepsize in cm,  using the integrator `integrate` to
# integrate the ODE.
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


    return (radius[nsurf]/1.0e5, mass[nsurf]/msun, dr)


(r_FE_1, m_FE_1, h_FE_1) = stellar_structure(tov_integrate_FE, 1000)
(_, m_FE_2, h_FE_2) = stellar_structure(tov_integrate_FE, 2000)
(_, m_FE_3, h_FE_3) = stellar_structure(tov_integrate_FE, 4000)
(_, m_RK2_1, h_RK2_1) = stellar_structure(tov_integrate_RK2, 1000)
(_, m_RK2_2, h_RK2_2) = stellar_structure(tov_integrate_RK2, 2000)
(_, m_RK2_3, h_RK2_3) = stellar_structure(tov_integrate_RK2, 4000)
(_, m_RK3_1, h_RK3_1) = stellar_structure(tov_integrate_RK3, 1000)
(_, m_RK3_2, h_RK3_2) = stellar_structure(tov_integrate_RK3, 2000)
(_, m_RK3_3, h_RK3_3) = stellar_structure(tov_integrate_RK3, 4000)
(_, m_RK4_1, h_RK4_1) = stellar_structure(tov_integrate_RK4, 1000)
(_, m_RK4_2, h_RK4_2) = stellar_structure(tov_integrate_RK4, 2000)
(_, m_RK4_3, h_RK4_3) = stellar_structure(tov_integrate_RK4, 4000)

Q_FE_theoretical = (h_FE_3 - h_FE_2) / (h_FE_2 - h_FE_1)
Q_RK2_theoretical = (h_RK2_3**2 - h_RK2_2**2) / (h_RK2_2**2 - h_RK2_1**2)
Q_RK3_theoretical = (h_RK3_3**3 - h_RK3_2**3) / (h_RK3_2**3 - h_RK3_1**3)
Q_RK4_theoretical = (h_RK4_3**4 - h_RK4_2**4) / (h_RK4_2**4 - h_RK4_1**4)

Q_FE_actual = abs(m_FE_3 - m_FE_2) / abs(m_FE_2 - m_FE_1)
Q_RK2_actual = abs(m_RK2_3 - m_RK2_2) / abs(m_RK2_2 - m_RK2_1)
Q_RK3_actual = abs(m_RK3_3 - m_RK3_2) / abs(m_RK3_2 - m_RK3_1)
Q_RK4_actual = abs(m_RK4_3 - m_RK4_2) / abs(m_RK4_2 - m_RK4_1)

print("FE & {:.3f} & {:.3f} \\\\".format(Q_FE_theoretical, Q_FE_actual))
print("RK2 & {:.3f} & {:.3f} \\\\".format(Q_RK2_theoretical, Q_RK2_actual))
print("RK3 & {:.3f} & {:.3f} \\\\".format(Q_RK3_theoretical, Q_RK3_actual))
print("RK4 & {:.3f} & {:.3f} \\\\".format(Q_RK4_theoretical, Q_RK4_actual))

print(r_FE_1, m_FE_1)
