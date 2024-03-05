# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:11:22 2019

@author: mouras54
"""

## run '%pylab inline'
## run '%matplotlib inline'
from __future__ import division
import pylab as pl
import deltasigma as ds
import numpy as np
import pandas as pd

# NTF parameters, no OBG (H_inf) specified for now
order = 5
OSR = 32
arch = 'CRFB'

## Nice vizualization of the NTF fucntion
NTF = ds.synthesizeNTF(order, OSR, opt=1)
print('\n\n Original NTF = \n')
print(ds.pretty_lti(NTF))
print('\n')
#pl.figure(figsize=(10,5))
#ds.plotPZ(NTF, showlist= True)
#pl.title('NTF z-plane')
#pl.show()

## Using CRFB as a base structure, later there will be modifications
a,g,b,c = ds.realizeNTF(NTF, form=arch)
ABCD = ds.stuffABCD(a, g, b, c, form=arch)
ntf, stf = ds.calculateTF(ABCD)

#print('ABCD matrix =  \n')
#print(pd.DataFrame(ABCD, columns=['x1[n]','x2[n]','x3[n]','x4[n]','x5[n]','u[n]','v[n]'], \
#                   index=['x1[n+1]','x2[n+1]','x3[n+1]','x4[n+1]','x5[n+1]','y[n]']))
#print('\n'*3)
#print('NTF = \n')
#print(ds.pretty_lti(ntf))
#print('\n'*3)
#print('STF = \n')
#print(ds.pretty_lti(stf)) ## Is unity the correct STF?
#print('\n'*3)

## Correcting the 'b' vector to prevent peaking in the STF, bi=0 for i>0
b_new = np.concatenate((b[0].reshape((1, )), np.zeros((b.shape[0] - 1, ))), axis=0)

ABCDn = ds.stuffABCD(a, g, b_new, c, form=arch)
ntfn, stfn = ds.calculateTF(ABCDn)

#print('New ABCD matrix =  \n')
#print(pd.DataFrame(ABCDn, columns=['x1[n]','x2[n]','x3[n]','x4[n]','x5[n]','u[n]','v[n]'], \
#                   index=['x1[n+1]','x2[n+1]','x3[n+1]','x4[n+1]','x5[n+1]','y[n]']))
#print('\n'*3)
#print('NTF_new = \n')
#print(ds.pretty_lti(ntfn))
#print('\n'*3)
#print('STF_new = \n')
#print(ds.pretty_lti(stfn)) ## Is unity the correct STF?
#print('\n'*3)

## Running simulations for DC input signals to find the maximum of the state signals
u = np.linspace(0, 0.63, 30)
N = 1000
T = np.ones((1,N))

maxima = np.zeros((order, len(u)))
ymax = np.zeros((1,len(u)))
for i in range(len(u)):
    u_sim = u[i]
    v, xn, xmax, y = ds.simulateDSM(u_sim*T, ABCDn)
    maxima[: , i] = np.squeeze(xmax)
    ymax[0,i] = np.max(y)

pl.figure(figsize=(10,5))
for i in range(order):
    pl.semilogy(u, maxima[i,:], 'o-', label = ('State %d' % (i+1)))
pl.semilogy(u, ymax[0,:], 'o-', label='Qin')
pl.grid(True)
pl.xlabel('DC input')
pl.ylabel('Peak Value')
pl.title('State maxima')
pl.legend(loc=4)
pl.show()

## Scaling the ABCD matrix, MSA changes is a byproduct of dyn range scalling
# Change xlim based on the linearity range of the integrators
# Change ylim based on the design condition for the input of the quantizer
ABCDs, umax, _ = ds.scaleABCD(ABCDn, nlev=2, xlim=1, N_sim=10000)

u = np.linspace(0, umax, 30)
N = 1000
T = np.ones((1,N))

maxima = np.zeros((order, len(u)))
ymax = np.zeros((1,len(u)))
for i in range(len(u)):
    u_sim = u[i]
    v, xn, xmax, y = ds.simulateDSM(u_sim*T, ABCDs)
    maxima[: , i] = np.squeeze(xmax)
    ymax[0, i] = np.max(y)

pl.figure(figsize=(10,5))
for i in range(order):
    pl.semilogy(u, maxima[i,:], 'o-', label = ('State %d' % (i+1)))
pl.semilogy(u, ymax[0,:], 'o-', label='Qin')
pl.grid(True)
pl.xlabel('DC input')
pl.ylabel('Peak Value')
pl.title('State maxima of scaled ABCD')
pl.legend(loc=4)
pl.show()

print('New ABCD matrix =  \n')
print(pd.DataFrame(ABCDn, columns=['x1[n]','x2[n]','x3[n]','x4[n]','x5[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','x4[n+1]','x5[n+1]','y[n]']))
print('\n'*3)

print('Scaled ABCD matrix =  \n')
print(pd.DataFrame(ABCDs, columns=['x1[n]','x2[n]','x3[n]','x4[n]','x5[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','x4[n+1]','x5[n+1]','y[n]']))
print('\n'*3)

    