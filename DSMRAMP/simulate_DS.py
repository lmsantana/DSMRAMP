# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:27:09 2019

@author: mouras54
"""

## run '%pylab inline' in the console before starting
from __future__ import division
import pylab as pl
import deltasigma as ds
import numpy as np

## Hard console flush
print('\n'*100)

# NTF parameters, no OBG (H_inf) specified for now
order = 3
OSR = 64



#############################################
## Input signal setup and sinewave simulation
#############################################

NTF = ds.synthesizeNTF(order, OSR, opt =1)
N = 2**13
fb = int(np.ceil(N/(2.*OSR)))
fin = np.floor(2./3. * fb)
amp = 0.5
u = amp * np.sin(2*np.pi* fin/N *np.arange(N))
v, xn, xmax, y = ds.simulateDSM(u, NTF)
t = np.arange(int(N/32))
pl.figure(figsize=(20,4))
pl.step(t, u[t], 'r', label='Input')
pl.step(t, v[t], 'g', label='Output')
pl.axis([0, N/32, -1.2, 1.2])
pl.xlabel('Sample')
pl.ylabel('Amplitude (u,v)')
pl.title('Modulator input and output')
pl.legend(loc=1)
pl.show()

######################################
## Spectrum FFT realized and estimated
######################################
f = np.linspace(0, 0.5, int(N/2. + 1))

# Scaling of FFT due to Hann window
spec = np.fft.fft(v * ds.ds_hann(N))/(N/4)
pl.plot(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
ds.figureMagic([0, 0.5], 0.05, None, [-120, 0], 20, None, (16, 6), 'Output Spectrum')
pl.xlabel('Normalized Frequency')
pl.ylabel('dBFS')

snr = ds.calculateSNR(spec[2:fb+1], fin-2)
pl.text(0.05, -10, 'SNR = %4.1fdB @ OSR = %d' % (snr, OSR), verticalalignment='center')
NBW = 1.5/N

Sqq = (4/3)*ds.evalTF(NTF, np.exp(2j*np.pi*f)) ** 2
pl.plot(f, ds.dbp(Sqq * NBW), 'm', linewidth=2, label='Expected PSD')
pl.text(0.49, -90, 'NBW = %4.1E x $f_s$' % NBW, horizontalalignment='right')
pl.legend(loc=4)
pl.show()

########################################
## SNR prediction versus SNR calculation
########################################

snr_pred, amp_pred, a, b, c = ds.predictSNR(NTF, OSR)

# Simulate SNR evaluate K FFTs with the signal freq at fBW/2 = 0.5/(2*OSR) Normalized
snr_sim, amp_sim = ds.simulateSNR(NTF, OSR)

pl.plot(amp_pred, snr_pred, '-', amp_sim, snr_sim, 'og-.')
ds.figureMagic([-100, 0], 10, None, [0, 100], 10, None, (16, 6),'SQNR')
pl.xlabel('Input Level (dBFS)')
pl.ylabel('SQNR (dB)')
pk_snr, pk_amp = ds.peakSNR(snr_sim, amp_sim)
pl.text(-25, 85, 'peak SNR = %4.1fdB\n@ OSR = %d and %1.2f dBFS\n' % (pk_snr, OSR, pk_amp), horizontalalignment='right');

######################
## Multi bit modulator
######################

OSR = 32
M = 16 # levels
## With a milti bit modulator we can be more agressive in the NTF (increasing H_inf)

H_inf = 2
NTF1m16 = ds.synthesizeNTF(order, OSR, opt = 1, H_inf = 2)
NTF2m16 = ds.synthesizeNTF(order, OSR, opt = 1, H_inf = 7)

pl.figure(figsize=(16,20))
N = 2**13

f_inband = int(np.ceil(N/(2*OSR)))
f_in = int(np.floor(2/3 * f_inband))
u = 0.5*M * np.sin(2*np.pi* f_in/N * np.arange(N))

################################
pl.subplot(421)
v, xn, xmax, y = ds.simulateDSM(u, NTF1m16, M+1)
t = np.arange(int(N/32))

pl.step(t, u[t], 'b', label = 'Input')
pl.step(t, v[t], 'r', label = 'Output')
ds.figureMagic([0, 120], 20, None, [-M, M], 2, None, None, 'Input & Output H_inf = 2')
pl.xlabel('Sample number')
pl.ylabel('(u , v) amplitudes')

pl.subplot(423)
v2, xn, xmax, y = ds.simulateDSM(u, NTF2m16, M+1)
t = np.arange(int(N/32))

pl.step(t, u[t], 'b', label = 'Input')
pl.step(t, v2[t], 'm', label = 'Output')
ds.figureMagic([0, 120], 20, None, [-M, M], 2, None, None, 'Input & Output H_inf = 7')
pl.xlabel('Sample number')
pl.ylabel('(u , v) amplitudes')
################################

################################
pl.subplot(222)
snr, amp = ds.simulateSNR(NTF1m16, OSR, None, 0., M+1)
pl.plot(amp, snr, 'or', amp, snr, '--r')
ds.figureMagic([-130, 0], 10, None, [0,140], 20, None, None, 'SQNR curve')
pl.xlabel('Signal amplitude (dBFS)')
pl.ylabel('SQNR (dB)')
pk_snr, pk_amp = ds.peakSNR(snr, amp)
pl.text(-80, pk_snr+5, 'Peak SNR = %2.2f \n @ OSR = %d and amp = %2.2f dBFS \n H_inf = 2' % (pk_snr, OSR, pk_amp), color = 'r')

snr, amp = ds.simulateSNR(NTF2m16, OSR, None, 0., M+1)
pl.plot(amp, snr, 'om', amp, snr, '--m')
pk_snr, pk_amp = ds.peakSNR(snr, amp)
pl.text(-90, pk_snr+5, 'Peak SNR = %2.2f \n @ OSR = %d and amp = %2.2f dBFS \n H_inf = 7' % (pk_snr, OSR, pk_amp), color = 'm')
###############################


#################################
pl.subplot(212)
f = np.linspace(0, 0.5, int(N/2. + 1))
spec = np.fft.fft(v * ds.ds_hann(N)) / (M * N/4)
pl.plot(f, ds.dbv(spec[:int(N/2 + 1)]), 'r')
snr_spec = ds.calculateSNR(spec[2:f_inband+1], f_in -2)
ds.figureMagic([0, 0.5], 0.1, None, [-140, 0], 10, None, None, 'Spectrum')
pl.xlabel('Normalized frequency')
pl.ylabel('dBFS')

spec2 = np.fft.fft(v2 * ds.ds_hann(N)) / (M* N/4)
pl.plot(f, ds.dbv(spec2[:int(N/2 + 1)]), 'm')
snr_spec2 = ds.calculateSNR(spec2[2:f_inband+1], f_in -2)

pl.tight_layout()

