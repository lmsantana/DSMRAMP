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
nlev = 16        ## 4-bit quantizer
form = 'CIFF'   ## FF better for highly linear OpA 

"""
 NTF synthesis
 Given a small OSR, the literature says that methods different than butterworth
 may give bettter results, however, after some quick test, the regular butterworth was the 
 one which gave the biggest MSA for the same DR as all the others
"""

# ntf = ds.clans(order = order, OSR = OSR, Q = nlev-1, opt=1)
ntf = ds.synthesizeNTF(order=order, osr=OSR, opt=1, H_inf=6.56)
#ntf = ds.synthesizeChebyshevNTF(order=order, OSR=OSR, opt=1, H_inf=6.56)

#ds.DocumentNTF(ntf, OSR)
#pl.show()
#print(ds.pretty_lti(ntf))

# ABCD matrix
a,g,b,c = ds.realizeNTF(ntf, form=form)
ABCD = ds.stuffABCD(a,g,b,c, form=form)
ABCDs = np.copy(ABCD)


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
ABCDs, umax, _ = ds.scaleABCD(ABCD, nlev=nlev, xlim=nlev, N_sim=10000)
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

#ABCD = np.array([[ 1.        ,  0.        ,  0.        ,  1.        , -1.        ],
#                 [ 1.        ,  1.        , -0.05892598,  0.        ,  0.        ],
#                 [ 1.        ,  1.        ,  0.94107402,  0.        ,  0.        ],
#                 [ 2.80536862,  1.81496051,  0.74564328,  1.        ,  0.        ]])
#
#ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.6314776 , -1.6314776 ],
#                  [ 1.19337803,  1.        , -0.07934089,  0.        ,  0.        ],
#                  [ 0.88631426,  0.74269363,  0.94107402,  0.        ,  0.        ],
#                  [ 1.71952629,  0.9321977 ,  0.51565859,  1.        ,  0.        ]])

print('ABCD matrix of the plant =  \n')
print(pd.DataFrame(ABCD, columns=['x1[n]','x2[n]','x3[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','y[n]']))
print('\n'*3)

print('ABCD Scaled matrix of the plant =  \n')
print(pd.DataFrame(ABCDs, columns=['x1[n]','x2[n]','x3[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','y[n]']))
print('\n'*3)


snr, amp = ds.simulateSNR(ABCDs, osr=OSR, nlev = nlev, k=13)
snr2, amp2 = ds.simulateSNR(ABCD, osr=OSR, nlev = nlev, k=13)

pl.plot(amp, snr, 'o--', color='b', linewidth=3, label='Scaled')
pl.plot(amp2, snr2, 'o--', color='k', label='Orignal')

pl.grid(True)
ds.figureMagic(xRange=(-120,10),dx=10, yRange=(-30,100),dy=10, size=(10,5))

peak_snr, peak_amp = ds.peakSNR(snr, amp)
peak_snr2, peak_amp2 = ds.peakSNR(snr2, amp2)

msg = 'Peak SNR Scaled %.1fdB at amp = %-.1fdB' % (peak_snr, peak_amp)
pl.text(peak_amp - 10, peak_snr, msg, horizontalalignment='right', verticalalignment='bottom')

msg = 'Peak SNR Original %.1fdB at amp = %-.1fdB' % (peak_snr2, peak_amp2)
pl.text(peak_amp - 10, peak_snr-10, msg, horizontalalignment='right', verticalalignment='bottom')

pl.xlabel('Input amplitude (dB)')
pl.ylabel('SQNR (dB)')
pl.legend(loc=4)
pl.show()

ds.DocumentNTF(ntf, OSR)




