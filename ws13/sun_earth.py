#!/usr/bin/env python

# Cutter Coryell
# Ay 190 WS13
# Simulates the sun-earth orbital system.

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
import mpl_toolkits.mplot3d as mpl3d
import plot_defaults

# global constants
ggrav = 6.67e-8
msun  = 1.99e33
seconds_per_year = 24.*3600*365 # roughly
cm_per_pc = 3.1e18
distance_to_sgrAstar = 8e3 * cm_per_pc

# system parameters
system_name = "sun_earth"
simulation_duration = 2. # in years
initial_data_file = system_name + ".asc"
distance_unit_to_cm = 1.
time_unit_to_s = 1.
mass_unit_to_g = 1.
Nsteps = 1e4

final_data_file = system_name + "_final_positions.asc"

def NbodyRHS(u, mass):
    N = len(mass)
    a = np.zeros((N, 3))
    for j in range(N):
        if j != 0:
          a[:j] += -(mass[j] * (u[:j, :3] - u[j, :3]) 
                    / (np.linalg.norm(u[:j, :3] - u[j, :3], axis=1)**3)[:,None])
        if j != N - 1:
          a[j+1:] += -(mass[j] * (u[j+1:, :3] - u[j, :3]) 
                      / (np.linalg.norm(u[j+1:, :3] - u[j, :3], axis=1)**3)[:,None])

    return np.hstack((u[:, 3:], ggrav * a)) # u[:, 3:] are the velocities

def NbodyFE(u, mass, dt):
  return u + dt * NbodyRHS(u, mass)

def NbodyRK4(u, mass, dt):
    k1 = NbodyRHS(u, mass)
    k2 = NbodyRHS(u + 0.5 * k1, mass)
    k3 = NbodyRHS(u + 0.5 * k2, mass)
    k4 = NbodyRHS(u + k3, mass)
    return u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.

def TotalEnergy(u, mass):
    N = len(mass)
    kin = 0.5 * np.sum(mass * np.linalg.norm(u[:, 3:], axis=1)**2)
    pots = np.zeros(N)
    for j in range(N):
        pots[:j] += (mass[:j] * mass[j] 
                     / np.linalg.norm(u[:j, :3] - u[j, :3], axis=1))
        # if j != N - 1:
        #   pots[j+1:] += (mass[j+1:] * mass[j]
        #                 / np.linalg.norm(u[j+1:, :3] - u[j, :3], axis=1))
    pot = -ggrav * np.sum(pots)
    return kin + pot
    
# main program
pl.ion()

(x,y,z,vx,vy,vz,mass) = np.loadtxt(initial_data_file, unpack=True)


# convert from units in initial data file to cgs
x *= distance_unit_to_cm
y *= distance_unit_to_cm
z *= distance_unit_to_cm
vx *= distance_unit_to_cm / time_unit_to_s
vy *= distance_unit_to_cm / time_unit_to_s
vz *= distance_unit_to_cm / time_unit_to_s
mass *= mass_unit_to_g

xmin = np.amin(x)
xmax = np.amax(x)
ymin = np.amin(y)
ymax = np.amax(y)
zmin = np.amin(z)
zmax = np.amax(z)
rmax = 2.5*max(abs(xmin),abs(xmax),abs(ymin),abs(ymax),abs(zmin),abs(zmax))

def simulate(Nsteps):

  # use a single state vector to siplify the ODE code
  # indices:
  # u[:,0] = x
  # u[:,1] = y
  # u[:,2] = z
  # u[:,3] = vx
  # u[:,4] = vy
  # u[:,5] = vz
  u = np.array((x,y,z,vx,vy,vz)).transpose()

  t0 = 0
  t1 = simulation_duration * seconds_per_year
  dt = (t1-t0)/Nsteps

  times = []
  energies = []

  for it in np.arange(0, Nsteps):
      time = t0 + it * dt
      u = NbodyRK4(u,mass,dt)
      if it % max(1,Nsteps/100) == 0:


        energy = TotalEnergy(u,mass)
        times.append(time)
        energies.append(energy)
        print "it = %d, time = %g years, energy = %g" % \
              (it, time / seconds_per_year,
               TotalEnergy(u,mass))

        pl.clf()
        fig = pl.gcf()
        ax = mpl3d.Axes3D(fig)
        ax.scatter(u[:,0],u[:,1],u[:,2])
        ax.set_xlim((-rmax,rmax))
        ax.set_ylim((-rmax,rmax))
        ax.set_zlim((-rmax,rmax))
        pl.draw()

  # output result
  file_header = "1:x 2:y 3:z 4:vx 5:vy 6:vz"
  np.savetxt(final_data_file, u, header=file_header)

  return (times, energies)

t1, E1 = simulate(1e4)
t2, E2 = simulate(4e4)

pl.ioff()
pl.close()

# set up the figure and control white space
myfig = pl.figure(figsize=(10,8))
myfig.subplots_adjust(left=0.2)
myfig.subplots_adjust(bottom=0.16)
myfig.subplots_adjust(top=0.85)
myfig.subplots_adjust(right=0.85)

ax = pl.gca()

pl.xlim(0.0, simulation_duration)
pl.ylabel(r"Total Energy [$\times 10^{40}$ ergs]", labelpad=10)
pl.xlabel("Time [years]", labelpad=20)
pl1, = pl.plot(np.array(t1) / seconds_per_year, np.array(E1) / 1e40, lw=8)
pl2, = pl.plot(np.array(t2) / seconds_per_year, np.array(E2) / 1e40, lw=8)
pl.legend( (pl1, pl2), ("$10^4$ timesteps", r"$4 \times 10^4$ timesteps"), frameon=False, loc="upper left")
pl.savefig(system_name + "_energy.pdf")
pl.show()
