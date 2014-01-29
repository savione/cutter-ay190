#!/usr/bin/env python2.7

# Cutter Coryell
# Ay 190
# WS6

import numpy as np
import matplotlib.pyplot as pl
from dft import dft
from timeit import timeit
import plot_defaults

# PART A: Test of DFT Implementation (implemented in dft.py)

x = np.random.randn(10)
fft_x = np.fft.fft(x)

print("Relative error of dft compared to np.fft for random input vector:")
print(np.sqrt(np.sum(np.abs((dft(x) - fft_x))**2)) / np.sqrt(np.sum(np.abs(fft_x)**2)))

# PARTS B+C: Characterization of Runtimes

# Parameters
ntrials = 10

Ns = range(10, 101)
dft_runtimes = []
fft_runtimes = []
for N in Ns:
    dft_runtimes.append(timeit("dft(x)", number=ntrials, setup="from dft import dft; import numpy as np; x=np.random.randn({})".format(N)) / ntrials)
    fft_runtimes.append(timeit("np.fft.fft(x)", number=ntrials, setup="import numpy as np; x=np.random.randn({})".format(N)) / ntrials)


# set up the figure and control white space
myfig = pl.figure(figsize=(10,8))
myfig.subplots_adjust(left=0.18)
myfig.subplots_adjust(bottom=0.16)
myfig.subplots_adjust(top=0.95)
myfig.subplots_adjust(right=0.94)

"""
# prepare x and y ranges
xmin = 10
xmax = 100
ymin = 0
ymax = 1
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
"""

# label the axes
pl.xlabel("Length of Input Vector, $N$",labelpad=15)
pl.ylabel("Runtime [s]",labelpad=15)

pl.xscale("log")
pl.yscale("log")
dft, = pl.plot(Ns, dft_runtimes, 'o')
fft, = pl.plot(Ns, fft_runtimes, 'o')
pl.legend( (dft, fft), ("Matrix DFT", r"\texttt{numpy} FFT"), loc=(0.05, 0.75), frameon=False )

pl.savefig("fig-b+c.pdf")
