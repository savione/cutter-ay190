#!/usr/bin/env python2.7

# Cutter Coryell
# Ay 190
# WS7 Problem 2

import numpy as np
import matplotlib.pyplot as pl
import plot_defaults

# Parameters
Ns = 4**np.arange(1, 10)

# Determine the smallest number of people for the probability to be greater than 0.5 that two persons in a group have the same birthday.
def birthday(N):
    np.random.seed(1)
    num_people = 2
    while True:
        successes = 0
        for trial in range(N):
            people = np.sort(np.random.randint(365, size=num_people))
            if not np.array_equal(np.unique(people), people):
                successes += 1
        if float(successes) / N > 0.5:
            return num_people
        num_people += 1
            
for N in Ns:
    num_people = birthday(N)
    print "{N} & {people} & {err:0.3f} \\\\".format(N=N, people=num_people,
                                                    err = abs(float(num_people - 23)/22))
    
