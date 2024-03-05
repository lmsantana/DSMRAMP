# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:31:40 2019

@author: mouras54
"""

## run '%pylab inline'
## run '%matplotlib inline'
from __future__ import division
from IPython.display import Image
import deltasigma as ds
import pylab as pl
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

order = int(8)
osr = int(32)
nlev = 2
f0 = 0.125 ## fraction of fs
Hinf = 1.5 ## Lee's rule
form = 'CRFB'

ntf = ds.synthesizeNTF(order=order, osr=osr, opt=2, H_inf=Hinf, f0=f0)

print('\n Synthesized 8th order NTF with optmized zeros: \n\n')
print(ds.pretty_lti(ntf))
pl.figure(figsize = (20,4))
ds.plotPZ(ntf, showlist=True)
pl.show()


a,g,b,c = ds.realizeNTF(ntf, form)
## At this point you can choose to zero the b vector besides the first element or not
## Change in the b vector only afects the STF and don't affect the NTF
b = np.hstack((np.atleast_1d(b[0]), np.zeros((b.shape[0] - 1, ))))
ABCD = ds.stuffABCD(a,g,b,c, form)

#pd.describe_option('display')
pd.set_option('display.max_columns', None)
print('ABCD matrix =  \n')
print(pd.DataFrame(ABCD, columns=['x1[n]','x2[n]','x3[n]','x4[n]','x5[n]','x6[n]','x7[n]','x8[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','x4[n+1]','x5[n+1]','x6[n+1]','x7[n+1]','x8[n+1]','y[n]']))
print('\n'*3)

### Check for the formats of the spectrum
ds.DocumentNTF(ABCD, osr, f0)
pl.show()

ds.PlotExampleSpectrum(ntf, M=1, osr = osr, f0 = f0)
pl.show()


## SNR prediction and calculation
snr, amp = ds.simulateSNR(ntf, osr, None, f0, nlev)
snr_pred, amp_pred, _, _, _ = ds.predictSNR(ntf, osr, f0=f0)

pl.plot(amp, snr,'o-.g', label='simulated')
pl.plot(amp_pred, snr_pred, '-', label='predicted')
pl.xlabel('Input Level (dBFS)')
pl.ylabel('SQNR (dB)')
peak_snr, peak_amp = ds.peakSNR(snr, amp)
msg = 'peak SQNR = %4.1fdB  \n@ amp = %4.1fdB  ' % (peak_snr, peak_amp)
pl.text(peak_amp-10,peak_snr,msg, horizontalalignment='right', verticalalignment='center');
msg = 'OSR = %d ' % osr
pl.text(-2, 5, msg, horizontalalignment='right');
ds.figureMagic([-100, 0], 10, None, [0, 80], 10, None, [12, 6], 'Time-Domain Simulations')
pl.legend(loc=2)

## Dynamic scalling - xlim = 1.0 by default

## Plotting before scalling
u = np.linspace(0, 0.9, 30)
N = 10000
N0 = 50
test_tone = np.cos(2*np.pi*f0*np.arange(N))
test_tone[:N0] = test_tone[:N0]*(0.5 - 0.5*np.cos(2*np.pi/N0*np.arange(N0)))
maxima = np.zeros((order, len(u)))
ymax = np.zeros((1,len(u)))
for i in range(len(u)):
    u_sim = u[i]
    v, xn, xmax, y = ds.simulateDSM(u_sim*test_tone, ABCD)
    maxima[: , i] = np.squeeze(xmax)
    ymax[0,i] = np.max(y)

pl.figure(figsize=(10,5))
for i in range(order):
    pl.plot(u, maxima[i,:], 'o-', label = ('State %d' % (i+1)))
#pl.plot(u, ymax[0,:], 'o-', label='Qin')
pl.grid(True)
pl.xlabel('f0 input - DC-like for bandpass')
pl.ylabel('Peak Value')
pl.title('State maxima - Original ABCD matrix')
pl.legend(loc=4)
pl.show()

ABCD0 = ABCD.copy()
ABCD, umax, _ = ds.scaleABCD(ABCD0, nlev=nlev, f=f0) ## Default limit for States xi is 1.0

## Plotting after scalling
u = np.linspace(0, 0.9, 30)
N = 10000
N0 = 50
test_tone = np.cos(2*np.pi*f0*np.arange(N))
test_tone[:N0] = test_tone[:N0]*(0.5 - 0.5*np.cos(2*np.pi/N0*np.arange(N0)))
maxima = np.zeros((order, len(u)))
ymax = np.zeros((1,len(u)))
for i in range(len(u)):
    u_sim = u[i]
    v, xn, xmax, y = ds.simulateDSM(u_sim*test_tone, ABCD)
    maxima[: , i] = np.squeeze(xmax)
    ymax[0,i] = np.max(y)

pl.figure(figsize=(10,5))
for i in range(order):
    pl.plot(u, maxima[i,:], 'o-', label = ('State %d' % (i+1)))
#pl.plot(u, ymax[0,:], 'o-', label='Qin')
pl.grid(True)
pl.xlabel('f0 input - DC-like for bandpass')
pl.ylabel('Peak Value')
pl.title('State maxima - Scaled ABCD matrix')
pl.legend(loc=2)
pl.show()
