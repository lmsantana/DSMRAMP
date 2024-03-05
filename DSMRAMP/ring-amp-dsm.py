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

## Modulator parameters
"""
order is choosen to increase the OTA/RingAmp power tradeoff
OSR is chossen to achieve BW of 100MHz
nlev is chossen to reduce noise floor and increase total SQNR
form is still open to discussion
"""
order = 3
OSR = 10
nlev = 16        ## 3-bit quantizer
form = 'CIFF'   ## FF better for highly linear OpA 

"""
 NTF synthesis
 Given a small OSR, the literature says that methods different than butterworth
 may give bettter results, however, after some quick test, the regular butterworth was the 
 one which gave the biggest MSA for the same DR as all the others
"""

ntf = ds.clans(order = order, OSR = OSR, Q = nlev-1, opt=1)
#ntf = ds.synthesizeNTF(order=order, osr=OSR, opt=1, H_inf=6.56)
#ntf = ds.synthesizeNTF(order=order, osr=OSR, opt=1, H_inf=1.5)
#ntf = ds.synthesizeChebyshevNTF(order=order, OSR=OSR, opt=1, H_inf=6.56)

#ds.DocumentNTF(ntf, OSR)
#pl.show()
print(ds.pretty_lti(ntf))



## ABCD matrix
a,g,b,c = ds.realizeNTF(ntf, form=form)
ABCD = ds.stuffABCD(a,g,b,c, form=form)

print('ABCD matrix of the plant =  \n')
print(pd.DataFrame(ABCD, columns=['x1[n]','x2[n]','x3[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','y[n]']))
print('\n'*3)

"""
Code snipet used to characterize the different NTFs and decide for the best to move
"""
#snr, amp = ds.simulateSNR(ABCD, osr=OSR, nlev = nlev)
#pl.plot(amp, snr, 'o-', color='b')
#pl.grid(True)
#ds.figureMagic(xRange=(-120,10),dx=10, yRange=(-30,90),dy=10, size=(10,5))
#peak_snr, peak_amp = ds.peakSNR(snr, amp)
#msg = 'Peak SNR %.1fdB at amp = %-.1fdB' % (peak_snr, peak_amp)
#pl.text(peak_amp - 10, peak_snr, msg, horizontalalignment='right', verticalalignment='bottom')
#pl.xlabel('Input amplitude (dB)')
#pl.ylabel('SQNR (dB)')


"""
State signal levels and dynamic scalling
Done to prevent the amplifiers to get out of the linear region, the resultant
matrix is later checked with a time simulation to guarantee that the state values are within
the safety margin
"""
u = np.linspace(0, 3, 30)
N = 1000
T = np.ones((1,N))

maxima = np.zeros((order, len(u)))
ymax = np.zeros((1,len(u)))
for i in range(len(u)):
    u_sim = u[i]
    v, xn, xmax, y = ds.simulateDSM(u_sim*T, ABCD, nlev = nlev)
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

## Scalling
#ABCDs, umax, _ = ds.scaleABCD(ABCD, nlev=nlev, xlim=np.array([nlev*0.8, nlev*1.5, nlev*2.5])*1.0, N_sim=10000)
ABCDs, umax, _ = ds.scaleABCD(ABCD, nlev=nlev, xlim=nlev, N_sim=10000)
a,g,b,c = ds.mapABCD(ABCDs, form)

maxima = np.zeros((order, len(u)))
ymax = np.zeros((1,len(u)))
for i in range(len(u)):
    u_sim = u[i]
    v, xn, xmax, y = ds.simulateDSM(u_sim*T, ABCDs, nlev = nlev)
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


snr, amp = ds.simulateSNR(ABCDs, osr=OSR, nlev = nlev)
pl.plot(amp, snr, 'o-', color='b')
pl.grid(True)
ds.figureMagic(xRange=(-120,10),dx=10, yRange=(-30,90),dy=10, size=(10,5))
peak_snr, peak_amp = ds.peakSNR(snr, amp)
msg = 'Peak SNR %.1fdB at amp = %-.1fdB' % (peak_snr, peak_amp)
pl.text(peak_amp - 10, peak_snr, msg, horizontalalignment='right', verticalalignment='bottom')
pl.xlabel('Input amplitude (dB)')
pl.ylabel('SQNR (dB)')
pl.show()



 ## CRFF matrix
#ABCD = np.array([[ 1.        ,  0.        ,  0.        ,  1.        , -1.        ],
#                 [ 1.        ,  1.        , -0.05892598,  0.        ,  0.        ],
#                 [ 1.        ,  1.        ,  0.94107402,  0.        ,  0.        ],
#                 [ 2.80536862,  1.81496051,  0.74564328,  1.        ,  0.        ]])
#
#ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.6314776 , -1.6314776 ],
#                  [ 1.19337803,  1.        , -0.07934089,  0.        ,  0.        ],
#                  [ 0.88631426,  0.74269363,  0.94107402,  0.        ,  0.        ],
#                  [ 1.71952629,  0.9321977 ,  0.51565859,  1.        ,  0.        ]])

## CIFF matrix
#ABCD = np.array([[ 1.        ,  0.        ,  0.        ,  1.        , -1.        ],
#                 [ 1.        ,  1.        , -0.05805791,  0.        ,  0.        ],
#                 [ 0.        ,  1.        ,  1.        ,  0.        ,  0.        ],
#                 [ 2.86429459,  2.72678094,  0.74465741,  1.        ,  0.        ]])
#    
#ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.64967266, -1.64967266],
#                  [ 1.08160444,  1.        , -0.07017017,  0.        ,  0.        ],
#                  [ 0.        ,  0.82738729,  1.        ,  0.        ,  0.        ],
#                  [ 1.73628057,  1.52821342,  0.50440738,  1.        ,  0.        ]])
#    
    
"""
ABCD Test region
"""

#ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  0.2      ,  -0.2       ],
#                  [ 0.5       ,  1.        , -0.15      ,  0.        ,  0.        ],
#                  [ 2.        ,  0.5       ,  1.        ,  0.        , -1.5       ],
#                  [ 0.        ,  0.        ,  1.        ,  1.        ,  0.        ]])

    

"""
Time simulation of the scaled ABCD for sanity check and for spectrum analysis
Spectrum is plot for checking also
"""
N = 2**13
decim = 128
fb = int(np.ceil(N/(2.*OSR)))
fin = np.floor(1./3. * fb)
amp = ds.undbv(-10) * nlev

noise_enb = 0
## Noise input vector (w/ input referred noise of input SW and DAC) - de la Rosa SC Model
fs = 1
Cin = 0.5e-12
Cdac = 0.5e-12
sigma_dac2_in = 1 * 4e-21 * (Cdac/Cin) / (Cin) 
sigma_sw2_in =  1 * 4e-21 /(Cin * fs)           # 2* for the differential structure
sigma_in = pl.sqrt(sigma_dac2_in + sigma_sw2_in)
u = (amp * np.sin(2*np.pi* fin/N *np.arange(N))) + noise_enb*sigma_in*(np.random.randn(N) - 0.5)*nlev

#u = amp * np.sin(2*np.pi* fin/N *np.arange(N))
v, xn, xmax, y = ds.simulateDSM(u, ABCDs, nlev=nlev)
t = np.arange(int(N/decim))

vdd = 0.9
delta = 2 #default value
analog_scale = vdd/(delta*(nlev-1))


pl.plot(t, u[t] * analog_scale, 'r', label='Input')
pl.step(t, v[t] * analog_scale, 'g', label='DAC Output')
pl.plot(t, y[t] * analog_scale, 'b', label='Quant in')
ds.figureMagic([0, N/decim], 100, None, [-vdd, vdd], 0.1, None, (20, 8),'Modulator input and output; DAC step = 0.9V/7=128mV')
pl.grid(True)
pl.xlabel('Sample')
pl.ylabel('Amplitude (Volt)')
pl.legend(loc=1)
pl.show()

pl.figure(figsize=(20,4))
pl.step(t, xn[0 , t] * analog_scale, 'g', label='State 1')
pl.step(t, xn[1 , t] * analog_scale, 'b', label='State 2')
pl.step(t, xn[2 , t] * analog_scale, 'r', label='State 3')
pl.axis([0, N/decim, -vdd*0.7, vdd*0.7])
pl.grid(True)
pl.xlabel('Sample')
pl.ylabel('States Amplitudes (Volt)')
pl.title('State amplitudes')
pl.legend(loc=1)
pl.show()

pl.figure(figsize=(20,4))
pl.subplot(141)
pl.hist(xn[0,:] * analog_scale, bins=100)
pl.title('State 1')
pl.subplot(142)
pl.hist(xn[1,:] * analog_scale, bins=100)
pl.title('State 2')
pl.subplot(143)
pl.hist(xn[2,:]* analog_scale, bins=100)
pl.title('State 3')
pl.subplot(144)
pl.hist(y* analog_scale, bins=100)
pl.title('Quant In')
pl.show()

## Plotting spectrums
f = np.linspace(0, 0.5, int(N/2. + 1))
# Scaling of FFT due to Hann window
spec = np.fft.fft(v* analog_scale * ds.ds_hann(N))/(N/4)
pl.semilogx(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
ds.figureMagic([0.005, 0.5], None, None, [-140, 0], 10, None, (16, 6), 'Output Spectrum')
pl.xlabel('Normalized Frequency')
pl.ylabel('dBFS')

pl.plot([0.005, 0.5/OSR], [-140, -140], 'k', linewidth=10, label='Signal Band')


snr = ds.calculateSNR(spec[2:fb+1], fin-2)
pl.text(0.05, -10, 'SNR = %4.1fdB @ OSR = %d' % (snr, OSR), verticalalignment='center')
NBW = 1.5/N

Sqq = (4/3)*ds.evalTF(ntf, np.exp(2*1j*np.pi*f)) ** 2
pl.plot(f, ds.dbp(Sqq * NBW) + ds.dbv(analog_scale), 'm--', linewidth=2, label='Expected PSD')
pl.text(0.49, -90, 'NBW = %4.1E x $f_s$' % NBW, horizontalalignment='right')
pl.legend(loc=4)
pl.show()


ds.DocumentNTF(ABCDs, OSR)
pl.show()

print(ds.pretty_lti(ntf))
print('ABCD matrix of the plant =  \n')
print(pd.DataFrame(ABCDs, columns=['x1[n]','x2[n]','x3[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','y[n]']))
print('\n'*3)

