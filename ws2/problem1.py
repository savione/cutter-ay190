#!/usr/bin/env python

# Cutter Coryell
# Ay 190
# WS2 Problem 1

import numpy as np

one_third = 1.0/3
thirteen_thirds = np.float32(13.0/3)
four_thirds = np.float32(4.0/3)

closed_forms = []
# set initial conditions for the recurrence relation
recurs_forms = [np.float32(1), np.float32(one_third)] 

# set first two closed-form values; first two recursive-values already done
for n in range(2):
    closed_forms.append(one_third ** n)
    
for n in range(2, 16):
    # calculate remaining closed-form values
    closed_forms.append(one_third ** n)
    # calculate remaining recursive values
    recurs_forms.append(thirteen_thirds * recurs_forms[n-1]
                        - four_thirds * recurs_forms[n-2])

closed_array = np.array(closed_forms)
recurs_array = np.array(recurs_forms)

absolute_errors = recurs_array - closed_array
relative_errors = absolute_errors / closed_array

# print in \LaTeX form
for n in range(16):
    row = [n, closed_array[n], recurs_array[n], absolute_errors[n], relative_errors[n]]
    for i, item in enumerate(row):
        row[i] = str(item)
    print " & ".join(row), '\\\\'
