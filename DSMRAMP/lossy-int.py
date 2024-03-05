# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:52:52 2019

@author: mouras54
"""

## Run %pylab inline in the kernel
from __future__ import division
import deltasigma as ds
import numpy as np
import pylab as pl

## Basic modulators parameters
order = 4
OSR = 128
nlev = 2
form = 'CIFB'

ntf = ds.synthesizeNTF(order, OSR, opt=1)
a,g,b,c = ds.realizeNTF(ntf, form)
ABCD = ds.stuffABCD(a,g,b,c, form)

A,B,C,D = ds.partitionABCD(ABCD)

## Adding the loss in the feedback path of the integrator plants
## 0.99 is equal to a 1% phase loss
Av = 100 #40dB
Cp = 0
Cs = 5
Ci = 20 #normalized caps
Cc = 5
p = (1+(1./Av))/(1 + ((1 + (Cs+Cp)/Ci + (Cp/Cc)*(1+Cs/Ci))/Av))
lossv = np.array([p , p, p, p])
ABCDl = ABCD.copy()
for i in range(lossv.shape[0]):
    ABCDl[i][i] = ABCD[i][i] * lossv[i]

## New NTF with the added loss in the integrators
ntf_los,_ = ds.calculateTF(ABCDl)

ds.DocumentNTF(ntf, OSR)
pl.show()
ds.DocumentNTF(ntf_los, OSR)
pl.show()

snr, amp = ds.simulateSNR(ntf, OSR)
snrl, ampl = ds.simulateSNR(ntf_los, OSR)

pl.plot(amp, snr, 'o-', color='b')
pl.plot(ampl, snrl, 'o-', color='m')
pl.grid(True)
ds.figureMagic(xRange=(-140,10),dx=10, yRange=(-10,140),dy=10, size=(10,5))
pl.xlabel('Input amplitude (dB)')
pl.ylabel('SQNR (dB)')

peak_snr, peak_amp = ds.peakSNR(snr, amp)
msg = 'Peak SNR %.1fdB at amp = %-.1fdB' % (peak_snr, peak_amp)
pl.text(peak_amp - 10, peak_snr, msg, horizontalalignment='right', verticalalignment='bottom')

peak_snrL, peak_ampL = ds.peakSNR(snrl, ampl)
msg = 'Peak SNR (Lossy) %.1fdB at amp = %-.1fdB' % (peak_snrL, peak_ampL)
pl.text(peak_ampL - 10, peak_snrL + 10, msg, horizontalalignment='right', verticalalignment='bottom')