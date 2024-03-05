# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:24:49 2019

@author: mouras54
"""

## Run %pylab inline in the kernel
from __future__ import division
import deltasigma as ds
import numpy as np
import pylab as pl
import pandas as pd

def ds_quantize(y, n):
    """v = ds_quantize(y,n)
    Quantize y to:
     
    * an odd integer in [-n+1, n-1], if n is even, or
    * an even integer in [-n, n], if n is odd.

    This definition gives the same step height for both mid-rise
    and mid-tread quantizers.
    """
    n = ds.carray(n)
    v = np.zeros(y.shape)
    for qi in range(n.shape[0]): 
        if n[qi] % 2 == 0: # mid-rise quantizer
            v[qi] = 2*np.floor(0.5*y[qi]) + 1
        else: # mid-tread quantizer
            v[qi] = 2*np.floor(0.5*(y[qi] + 1))
        L = n[qi] - 1
        v[qi] = np.sign(v[qi])*np.min((np.abs(v[qi]), L))
    return v


order = 3
OSR = 10
nlev = 8        ## 3-bit quantizer
form = 'CIFF'   ## FF better for highly linear OpA 

vdd = 0.9
delta = 2 #default value
analog_scale = vdd/(delta*(nlev-1))
digital_scale = 1.0/analog_scale

y = np.linspace(-nlev, nlev, 1000)
v = 0.0*y
for i in range(y.shape[0]):
    v[i] = ds_quantize(np.array([y[i]]), nlev)

    
threshold_vec = np.linspace(-nlev+2, nlev-2, nlev-1)
nlev_vec = np.linspace(-nlev+1, nlev-1, nlev)

vn = 0.0*y
for i in range(y.shape[0]):
    if y[i] < (threshold_vec[0]):
        vn[i] = nlev_vec[0]
    elif y[i] >= (threshold_vec[nlev-1-1]):
        vn[i] = nlev_vec[nlev-1]
    else:
        for mid in range(nlev-2-1):
            if y[i] >= threshold_vec[mid]:
                vn[i] =  nlev_vec[mid+1]


pl.figure(figsize=(7,7))
pl.plot(y,v)
pl.plot(y,vn)
ds.figureMagic([-nlev,nlev], None, None, [-nlev, nlev], 1, None, (16, 6), 'Stair transfer')
pl.xlabel('ADC input signal')
pl.ylabel('ADC output signal')
pl.legend(loc=4)
pl.plot()