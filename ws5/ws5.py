#!/usr/bin/env python2.7

# Cutter Coryell
# Ay 190
# WS5

import numpy as np
import matplotlib.pyplot as pl
import plot_defaults

# Part A

# set up the figure and control white space
myfig = pl.figure(figsize=(10,10))
myfig.subplots_adjust(left=0.13)
myfig.subplots_adjust(bottom=0.16)
myfig.subplots_adjust(top=0.97)
myfig.subplots_adjust(right=0.975)

data = np.genfromtxt("modified_m_sigma_table.dat")
log_sigma = np.log10(data[:,1])
log_M = data[:,7]
e_log_M = data[:,-1]

sigma_min = np.min(log_sigma)
sigma_max = np.max(log_sigma)
sigma_width = sigma_max - sigma_min
# prepare x and y ranges
xmin = sigma_min - sigma_width / 10
xmax = sigma_max + sigma_width / 10
ymin = 4.5
ymax = 9.0
# set axis parameters
pl.axis([xmin,xmax,ymin,ymax])
# get axis object
ax = pl.gca()
# set locators of tick marks
xminorLocator = pl.MultipleLocator(0.05)
xmajorLocator = pl.MultipleLocator(0.25)
yminorLocator = pl.MultipleLocator(0.2)
ymajorLocator = pl.MultipleLocator(1)
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)

# label the axes
pl.xlabel("$\log_{10} (\sigma_* / \mathrm{km \ s}^{-1})$",labelpad=15)
pl.ylabel("$\log_{10} (M_{\sc BH} / M_\odot)$",labelpad=15)

pl.plot(log_sigma, log_M, 'o')
pl.savefig("part-a.pdf")


# Part B

# Returns a linear function that is the linear fit to the given data.
def linear_fit(xs, ys, errors=None):
    if errors is None:
        errors = np.ones(len(xs))
    S = np.sum(1 / (errors*errors))
    Sigma_x = np.sum(xs / (errors*errors))
    Sigma_y = np.sum(ys / (errors*errors))
    Sigma_xx = np.sum(xs*xs / (errors*errors))
    Sigma_xy = np.sum(xs*ys / (errors*errors))
    denom = S * Sigma_xx - Sigma_x * Sigma_x
    a1 = (Sigma_y * Sigma_xx - Sigma_x * Sigma_xy) / denom
    a2 = (S * Sigma_xy - Sigma_x * Sigma_y) / denom
    return lambda x: a1 + a2 * x

fitter_noerror = linear_fit(log_sigma, log_M)
xs = np.arange(sigma_min, sigma_max, sigma_width / 1000)
fit_without_error, = pl.plot(xs, fitter_noerror(xs), color=[0.2,0.5,1])
pl.plot(log_sigma, log_M, 'bo')
pl.savefig("part-b.pdf")

# Part C

fitter_witherror = linear_fit(log_sigma, log_M, errors=e_log_M) 

data, = pl.plot(log_sigma, log_M, 'bo')
ax.errorbar(log_sigma, log_M, yerr=e_log_M, color='b',linestyle="None", marker="None")
fit_with_error, = pl.plot(xs, fitter_witherror(xs), 'r--')

pl.legend( (data, fit_without_error, fit_with_error), ("Data", "Linear Fit without Error", "Linear Fit including Error"), frameon=False, loc=(0.02, 0.78), prop={'size':25} )

pl.savefig("part-c.pdf")
