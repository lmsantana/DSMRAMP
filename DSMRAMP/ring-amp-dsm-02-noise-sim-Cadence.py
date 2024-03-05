# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:20:30 2019

First sketch to simulate the Ring Amp DSM 3rd order loop clocked at 1GHz


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


"""
## Modulator parameters

order is choosen to increase the OTA/RingAmp power tradeoff
OSR is chossen to achieve BW of 100MHz
nlev is chossen to reduce noise floor and increase total SQNR
form is still open to discussion
"""
order = 3
OSR = 10
nlev = 16        ## 3-bit quantizer
form = 'CIFF'   ## FF better for highly linear OpA 

vdd = 0.9
delta = 2 #default value
analog_scale = vdd/(delta*(nlev-1))
digital_scale = 1.0/analog_scale

"""
 NTF synthesis and initial analysis done in another script
 Here we just retrieve the NTF and the ABCD matrix previously found

"""

ntf = ds.synthesizeNTF(order=order, osr=OSR, opt=1, H_inf=6.56)

if form == 'CRFF':
     ## CRFF Matrix
    ABCD = np.array([[ 1.        ,  0.        ,  0.        ,  1.        , -1.        ],
                     [ 1.        ,  1.        , -0.05892598,  0.        ,  0.        ],
                     [ 1.        ,  1.        ,  0.94107402,  0.        ,  0.        ],
                     [ 2.80536862,  1.81496051,  0.74564328,  1.        ,  0.        ]])
    
    ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.6314776 , -1.6314776 ],
                      [ 1.19337803,  1.        , -0.07934089,  0.        ,  0.        ],
                      [ 0.88631426,  0.74269363,  0.94107402,  0.        ,  0.        ],
                      [ 1.71952629,  0.9321977 ,  0.51565859,  1.        ,  0.        ]])
elif form == 'CIFF':
    ## CIFF matrix  
    ABCD = np.array([[ 1.        ,  0.        ,  0.        ,  1.        , -1.        ],
                     [ 1.        ,  1.        , -0.05805791,  0.        ,  0.        ],
                     [ 0.        ,  1.        ,  1.        ,  0.        ,  0.        ],
                     [ 2.86429459,  2.72678094,  0.74465741,  1.        ,  0.        ]])
        
    ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.154, -1.154],
                      [ 1.081     ,  1.        , -0.0701    ,  0.        ,  0.        ],
                      [ 0.        ,  0.827     ,  1.        ,  0.        ,  0.        ],
                      [ 2.48      ,  2.182     ,  0.714     ,  1.        ,  0.        ]])

print('ABCD matrix of the plant =  \n')
print(pd.DataFrame(ABCD, columns=['x1[n]','x2[n]','x3[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','y[n]']))
print('\n'*3)
print('ABCD Scaled matrix of the plant =  \n')
print(pd.DataFrame(ABCDs, columns=['x1[n]','x2[n]','x3[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','y[n]']))
print('\n'*3)

BW = 500e6
vo_stg1_v2 = 103e-6
psd_vo_stg1_v2_Hz = vo_stg1_v2/BW
Cin = 411e-15
ktc_noise = 2*4e-21/Cin

b1 = 1.154
H1 = np.copy(ntf)
H1[1] = np.concatenate((H1[1], np.array([1])))
H1[2] = b1

integrator1 = (np.array([1.0]), np.array([]), b1)

N = 2**15
NBW = 1.5/N
f = np.linspace(0, 0.5, int(N/2. + 1))
#Sqq = (4/3)*ds.evalTF(ntf, np.exp(2j*np.pi*f)) ** 2
Sqq_in = (4/3)*ds.evalTF(H1, np.exp(2j*np.pi*f)) ** 2
Sqq_int = (4/3)*ds.evalTF(integrator1, np.exp(2j*np.pi*f)) ** 2

pl.figure(figsize=(10,4))
#pl.semilogx(f, ds.dbp(Sqq * NBW) + ds.dbv(analog_scale), 'b--', linewidth=2, label='quantization noise TF')
pl.semilogx(f, ds.dbp(Sqq_in * NBW), 'm--', linewidth=2, label='Input->X1 TF')
pl.semilogx(f, ds.dbp(Sqq_int * NBW), 'r--', linewidth=2, label='Input->X1 TF integrator')
ds.figureMagic([0.00005, 0.5], None, None, [-180, 0], 10, None, (16, 6), 'TF spectrum')
pl.legend(loc=2)
pl.show()

inref_integrator1 = ((np.array([]), np.array([1.0]), 1/b1))
inref_H = (H1[1], H1[0], 1/H[2]) 

