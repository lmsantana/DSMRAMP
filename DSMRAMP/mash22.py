# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:52:15 2019

@author: mouras54
"""

## run '%pylab inline'
## run '%matplotlib inline'
from __future__ import division
import pylab as pl
import deltasigma as ds
import numpy as np
import pandas as pd
from scipy.signal import lti, ss2zpk, lfilter

def zpk_multiply(a, b):
    za, pa, ka = ds._utils._get_zpk(a)
    zb, pb, kb = ds._utils._get_zpk(b)
    pa = pa.tolist() if hasattr(pa, 'tolist') else pa
    pb = pb.tolist() if hasattr(pb, 'tolist') else pb
    za = za.tolist() if hasattr(za, 'tolist') else za
    zb = zb.tolist() if hasattr(zb, 'tolist') else zb
    return ds.cancelPZ((za+zb, pa+pb, ka*kb))

# Description of the MASH network as an ABCD matrix
ABCD = [[1, 0, 0, 0, 1, -1, 0],
        [1, 1, 0, 0, 0, -2, 0],
        [0, 1, 1, 0, 0, 0, -1],
        [0, 0, 1, 1, 0, 0, -2],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]]
ABCD = np.array(ABCD, dtype=np.float_)

nlev = [9,9] # Both loops have a 9-level quantizers

print(pd.DataFrame(ABCD, columns=['x11[n]','x12[n]','x21[n]','x22[n]','u[n]','v1[n]','v2[n]'], \
                   index=['x11[n+1]','x12[n+1]','x21[n+1]','x22[n+1]','y1[n]','y2[n]']))

# All the transfer functions
ntfs, stfs = ds.calculateTF(ABCD, [1., 1.])

print ("\n\n STF_1:\n")
print (ds.pretty_lti(stfs[0]))
print ("\n\n STF_2:\n")
print (ds.pretty_lti(stfs[1]))
print('***STF_2 is not 0 in this case because the input to the second quantizer is Y1 and not Y1-V1\n\n')

## NTF XY: Noise from quantizer Y to the output X
## From quantizer 1 to output 1
print ("NTF_00: From E1 to V1\n")
print (ds.pretty_lti(ntfs[0, 0]))

## From quantizer 2 to output 1
print ("\n\n NTF_01: From E2 to V1 (always 0)\n")
print (ds.pretty_lti(ntfs[0, 1]))

## From quantizer 1 to output 2
print ("\n\n NTF_10: From E1 to V2\n")
print (ds.pretty_lti(ntfs[1, 0]))

## From quantizer 2 to output 2
print('\n\n NTF_11: From E2 to V2\n')
print(ds.pretty_lti(ntfs[1, 1]))

print('\n\n The regular MASH transfer is of the kind: \n')
print('v1 = STF1 * U + NTF_00 * E1 \n')
print('v2 = STF2 * U + NTF_10 * E1 + NTF_11 * E2 \n')
print('V = (STF1 * NTF_10) * U - (STF2 * NTF_00) * U - (NTF_11 * NTF_00) * E2 \n\n')

pl.figure(figsize = (15,5))
pl.subplot(121)
ds.PlotExampleSpectrum(ntfs[0,0], M=31)
ds.figureMagic(name='NTF of a 2nd order DS')

## Equivalente 4th order NTF
ntf_eq = zpk_multiply(ntfs[0,0], ntfs[1,1])
print ("\n\n NTF_EQ:\n")
print (ds.pretty_lti(ntf_eq))
pl.subplot(122)
ds.PlotExampleSpectrum(ntf_eq, M = 31)
ds.figureMagic(name='NTF of the equivalent 4th from MASH 2-2')

##############################################
## Simulation of the MASH network
##############################################

## Filter coefficients for the noise E1 cancelation
filtM1 = [0., 0., 0., 2., -1.]
filtM2 = [1., -2., 1.]

OSR = 64
N = 2**13
M = nlev[0] -1

## Finding frequencies that fit in a bin, although not necessary after windowing
fb = int(np.ceil(N/(2.*OSR)))
f_in = np.floor(2./3. * fb)

t = np.arange(0, N)
u = 0.5 * M * np.cos(2* np.pi / N * f_in * t)

vx, _, xmax, y = ds.simulateDSM(u, ABCD, nlev = nlev)

v1 = vx[0, :]
v2 = vx[1, :]
vf = lfilter(filtM1, [1.], v1) + lfilter(filtM2, [1.], v2)

## Not normalized spectrum
spec1 = np.fft.fft(v1*ds.ds_hann(N))/(M*N/2)
spec2 = np.fft.fft(v2*ds.ds_hann(N))/(M*N/2)
specf = np.fft.fft(vf*ds.ds_hann(N))/(M*N/2)
freq = np.linspace(0, 0.5, int(N/2 +1))

pl.figure(figsize=(15,10))
pl.subplot(221)
pl.semilogx(freq, ds.dbv(spec1[:int(N/2+1)]))
pl.ylabel('dbV')
pl.xlabel('Frequency')
ds.figureMagic(name = 'V1 spectrum')

pl.subplot(222)
pl.semilogx(freq, ds.dbv(spec2[:int(N/2+1)]))
pl.ylabel('dbV')
pl.xlabel('Frequency')
ds.figureMagic(name = 'V2 spectrum')

pl.subplot(212)
pl.semilogx(freq, ds.dbv(specf[:int(N/2+1)]), label='Output spectrum')
pl.semilogx([0,freq[fb]] , [np.min(ds.dbv(specf)),np.min(ds.dbv(specf))], 'o-', label = 'In band window')
pl.ylabel('dbV')
pl.xlabel('Frequency')
pl.legend(loc=4)
ds.figureMagic(name = 'Vf spectrum')

snr = ds.calculateSNR(specf[2:int(fb+1)], f_in-2)
msg = 'SQNR  =  %.1fdB\n @ A = %.1fdBFS & osr = %.0f\n' % \
      (snr, ds.dbv(specf[int(f_in)]), OSR)
pl.text( 1 / OSR, - 25, msg, horizontalalignment='left',
         verticalalignment='center')
pl.show()

## Slice of digital output
pl.figure(figsize=(15,9))
pl.subplot(411)
pl.step(np.arange(100,400),v1[100:400], label='Output of stage 1')
pl.grid(True)
pl.legend()

pl.subplot(412)
pl.step(np.arange(100,400),v2[100:400], label='Output of stage 2')
pl.grid(True)
pl.legend()

pl.subplot(413)
pl.step(np.arange(100,400),vf[100:400], label='Combined output of stage 1 and 2')
pl.grid(True)
pl.legend()

pl.subplot(414)
pl.step(np.arange(100,400),u[100:400], label='Input signal')
pl.grid(True)
pl.legend()