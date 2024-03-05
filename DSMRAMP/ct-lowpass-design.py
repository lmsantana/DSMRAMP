# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:50:26 2019

@author: mouras54
"""

## Run %pylab inline in the kernel
from __future__ import division
import deltasigma as ds
import pylab as pl
import numpy as np
import warnings
from scipy.signal import ss2zpk
warnings.filterwarnings('ignore')

## Modulator parameters
order = 3
OSR = 32
nlev = 2
####### DAC timing structure
## A tdac = [a, b], describes a DAC which rising edge is at time 'a' and falling 
## edge is at time 'b', where a and b are fraction 1 (sampling time)
tdac = [0, 1] ## NRZ DAC
#tdac = [0, 0.5] ## RZ DAC
#ELD = 0.5
#tdac = [[0+ELD,1+ELD], [0+ELD,1+ELD], [0+ELD,1+ELD], [0+ELD,1+ELD]] ## equivalent to add a new vector vx = [v0, v1, v2, v3] to feed the DACs
M = nlev - 1

ntf0 = ds.synthesizeNTF(order, OSR, opt=2)
#ds.plotPZ(ntf0, showlist=True)
#pl.show()
#ds.DocumentNTF(ntf0, OSR)
#pl.show()
#ds.PlotExampleSpectrum(ntf0, M, OSR)
#pl.show()

## Converting DT to CT
ABCDc, tdac2 = ds.realizeNTF_ct(ntf0, form='FB', tdac=tdac)
#print( '\n\n ABCD matrix: \n')
#print( ABCDc)
#print( "\n\n DAC timing (tdac2):\n")
#print( tdac2)
Ac, Bc, Cc, Dc = ds.partitionABCD(ABCDc)
sys_c = []
for i in range(Bc.shape[1]):
    sys_c.append(ss2zpk(Ac,Bc,Cc,Dc, input=i))
    
n_imp = 10
y = -ds.impL1(ntf0, n_imp)
ds.lollipop(np.arange(n_imp + 1), y)
pl.grid()
dt = 1./16
tppulse = np.vstack((np.zeros((1, 2)), tdac2[1:, :])).tolist()
yy = -ds.pulse(sys_c, tppulse, dt, n_imp).squeeze()
t = np.linspace(0, n_imp + dt, 10/dt + 1)
pl.plot(t, yy, 'g', label='continuous-time')
pl.legend()
pl.title('Loop filter pulse/impulse responses (negated)')
print('\n Impulse response match OK!\n')

## Mapping back to discrete time for simulation purposes
sys_d, Gp = ds.mapCtoD(ABCDc, tdac2)
ABCD = np.vstack((
                  np.hstack((sys_d[0], sys_d[1])),
                  np.hstack((sys_d[2], sys_d[3]))
                ))

ntf, G = ds.calculateTF(ABCD)
pl.subplot(121)
ds.DocumentNTF(ntf0, OSR)
pl.show()
ds.DocumentNTF(ntf, OSR)
pl.subplot(122)

L0 = sys_c[0]
f = np.linspace(0, 0.5)
G = ds.evalTFP(L0, ntf, f)
pl.plot(f, ds.dbv(G), 'm', label='STF')
pl.legend(loc=4)
pl.show()

## SNR curves
snrR, ampR = ds.simulateSNR(ABCD, OSR, None, 0., nlev)
pl.plot(ampR, snrR, 'o-')
peak_snrR, peak_ampR = ds.peakSNR(snrR, ampR)
msg = 'Peak SNR %.1fdB at amp = %-.1fdB' % (peak_snrR, peak_ampR)
pl.text(peak_ampR - 10, peak_snrR, msg, horizontalalignment='right', verticalalignment='bottom')
ds.figureMagic([-80, 0], 10, 1, [0, 80], 10, 1, None,'SQNR vs. Input Amplitude')
pl.xlabel('Input Amplitude (dBFS)')
pl.ylabel('SNR (dB)')
pl.title('Continuous-Time Implementation')