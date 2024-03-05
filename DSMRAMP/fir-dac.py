# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:21:40 2019

@author: mouras54

FIR DAC script for DSM simulation
The same FIR was used to feed both DACs inthe feedback path
Need further work in the filter design phase and the redefinition of the loop filter
coeficients to restore the original designed NTF

*** The partition of FIR paths is more likely to work, the faster filter path
(closer to the quantizer) should be less strict, while the path to the precision path
(usually close to the input) should be more aggresive

"""

# Run %pylab inline in the kernel
from __future__ import division
import deltasigma as ds
import pylab as pl
import scipy.signal as sg #for FIR filter design
import pandas as pd
import numpy as np

## Create the ABCD plant for the modulator
order = 2
OSR = 64
form = 'CIFB'
Hinf = 1.5

ntf = ds.synthesizeNTF(order, OSR, opt=1, H_inf = Hinf)
#ds.plotPZ(ntf, showlist=True)
#pl.show()

a,g,b,c = ds.realizeNTF(ntf, form)
b = np.hstack((np.atleast_1d(b[0]), np.zeros((b.shape[0] - 1, ))))
ABCD = ds.stuffABCD(a,g,b,c, form)

#print('ABCD matrix of the plant =  \n')
#print(pd.DataFrame(ABCD, columns=['x1[n]','x2[n]','u[n]','v[n]'], \
#                   index=['x1[n+1]','x2[n+1]','y[n]']))
#print('\n'*3)

## Filtering matrix for a 4-tap equal coefficients FIR filter

Afir = np.array([[0. , 0., 0., 0.],
                 [1. , 0., 0., 0.],
                 [0. , 1., 0., 0.],
                 [0. , 0., 1., 0.]])
Bfir = np.array([[1.],
                 [0.], 
                 [0.], 
                 [0.]])
Cfir = np.array([0.241, 0.4465, 0.241, 0.035])
Dfir = np.array([0.035])

A, B, C, D = ds.partitionABCD(ABCD)
Bu, Bv = np.hsplit(B,2) 

## New ABCD matrix with filter states xfir 1-> 3

Anew1 = np.hstack((A, Bv @ Cfir.reshape(1,4)))
Anew2 = np.hstack((np.zeros((np.size(Afir, axis=0), np.size(A, axis=1))) , Afir)) #Axis=0 is line, Axis=1 is colummn
Anew = np.vstack((Anew1, Anew2))

#print('\n\nAnew matrix = \n')
#print(pd.DataFrame(Anew, columns = ['x1[n]', 'x2[n]','xf1[n]', 'xf2[n]','xf3[n]','xf4[n]'],
#                   index = ['x1[n+1]','x2[n+1]','xf1[n+1]','xf2[n+1]','xf3[n+1]','xf4[n+1]']))

Bnew1 = np.hstack((Bu , Bv * Dfir))
Bnew2 = np.hstack((np.zeros((np.size(Bfir),np.size(Bu,axis=1))), Bfir))
Bnew = np.vstack((Bnew1, Bnew2))

#print('\n\nBnew matrix = \n')
#print(pd.DataFrame(Bnew, columns = ['u[n]','v[n]'],
#                   index = ['x1[n+1]','x2[n+1]','xf1[n+1]','xf2[n+1]','xf3[n+1]','xf4[n+1]']))

Cnew = np.hstack((C, np.zeros((1,np.size(Afir,axis=1)))))
Dnew = D

ABCDnew = np.vstack((np.hstack((Anew,Bnew)),
                     np.hstack((Cnew,Dnew))))
print('\n\n New ABCD matrix with FIR DAC states = \n')
pd.set_option('display.max_columns', None)
print(pd.DataFrame(ABCDnew, columns = ['x1[n]', 'x2[n]','xf1[n]', 'xf2[n]','xf3[n]','xf4[n]','u[n]','v[n]'],
                            index = ['x1[n+1]','x2[n+1]','xf1[n+1]','xf2[n+1]','xf3[n+1]','xf4[n+1]','y[n]']))

## ***Need to work in the FIR filter coefficients and the loop filter modification
ds.DocumentNTF(ABCDnew,OSR)
