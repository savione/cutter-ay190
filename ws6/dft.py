#!/usr/bin/env python2.7                                                        

# Cutter Coryell                                                                
# Ay 190                                                                        
# WS6

import numpy as np

# Computes the discrete Fourier transform of the input vector x.
def dft(x):
    N =len(x)
    J = np.reshape(N * range(N), (N, N))
    M =np.exp(-2j * np.pi / N)** (J *np.transpose(J))
    return M.dot(x)
