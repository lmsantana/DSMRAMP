# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## run '%pylab inline' in the console before starting
from __future__ import division
import pylab as pl
import deltasigma as ds
import numpy as np


# NTF parameters, no OBG (H_inf) specified for now
order = 3
OSR = 64

# NTF synthesis, missing parameters H_inf which is "free" and f0 which is DC
# The returned object is an array with complex zeros and poles
NTF = ds.synthesizeNTF(order, OSR, opt=0, f0=0)
#print(NTF)

# $NTF = \frac{(z-1)^3}{(z - (0.7-027j))\times(z - (0.7+027j))\times(z - (0.66))}$

# Pole and zero location
pl.subplot(121)
ds.plotPZ(NTF, markersize = 5)
pl.title('NTF Poles and zeros')

#################################################################
# Main plots of the NTF
################################################################

# Magnitude response plot
# Frequency vector with 100 points resolution between 0-0.75/OSR and 0.75/OSR-0.5(Nyquist normalized)
# Z vector around the unity circle with the resolution of the frequency vector
f = np.concatenate((np.linspace(0, 0.75/OSR, 100), np.linspace(0.75/OSR, 0.5, 100)))
z = np.exp(2j*np.pi*f)
magNTF = ds.dbv(ds.evalTF(NTF, z))
pl.subplot(222)
pl.plot(f, magNTF)
pl.title('NTF Magnitude accross frequencies')
# Formating figure
ds.figureMagic([0, 0.5], 0.05, None, [-100, 10], 10, None, (16, 8))
pl.xlabel('Normalized frequency ($1\\rightarrow f_s)$')
pl.ylabel('dB')
pl.title('NTF Magnitude Response')

# In band NTF response and noise floor
pl.subplot(224)
fstart = 0.001
f_inband = np.linspace(fstart, 1, 200)/(2*OSR)
z_inband = np.exp(2j*np.pi*f_inband)
magNTFin = ds.dbv(ds.evalTF(NTF, z_inband))
pl.semilogx(f_inband, magNTFin)
pl.axis([fstart, 0.5/OSR, -120, 0])
pl.grid(True)
pl.xlabel('Normalized in-band frequency region')
pl.ylabel('dB')
# In band noise RMS value in dB
sigma_NTF = ds.dbv(ds.rmsGain(NTF,0, 0.5/OSR))
pl.text(0.002, sigma_NTF + 5, 'IBN = %5.0f dB' % sigma_NTF )

pl.tight_layout()
pl.show()
#################################################################
# Zero optmization enabled and plot overlap
################################################################

NTF2 = ds.synthesizeNTF(order,OSR, opt=1)

# Pole zero plot
pl.subplot(121)
ds.plotPZ(NTF2, markersize = 9, color='#90EE90')
ds.plotPZ(NTF, markersize = 5)
# No hold is needed in new version os Matplotlib, just a show() before hand to flush the figure
pl.title('Pole-zero plot in Z-plane')

# Magnitude plot
magNTF2 = ds.dbv(ds.evalTF(NTF2, z))
pl.subplot(222)
pl.plot(f, magNTF, label='DC zero location')
pl.plot(f, magNTF2, label='OPT zero location')
ds.figureMagic([0, 0.5], 0.05, None, [-100, 10], 10, None, (16, 8))
pl.xlabel('Normalized frequency ($1\\rightarrow f_s)$')
pl.ylabel('dB')
pl.title('NTF Magnitude Response')
pl.legend(loc=4)
norm_h_inf = ds.infnorm(NTF)
pl.text(0.4, (norm_h_inf[0]) - 10, 'H_infnity = %2.1f' % norm_h_inf[0])

# In band noise plot
pl.subplot(224)
magNTF2in = ds.dbv(ds.evalTF(NTF2, z_inband))
pl.semilogx(f_inband, magNTFin, label='DC zero location')
pl.semilogx(f_inband, magNTF2in, label='OPT zero location')
pl.axis([fstart, 0.5/OSR, -120, 0])
pl.legend(loc=4)
pl.grid(True)
pl.xlabel('Normalized in-band frequency region')
pl.ylabel('dB')

sigma_NTF2 = ds.dbv(ds.rmsGain(NTF2, 0, 0.5/OSR))
pl.text(0.002, sigma_NTF + 5, 'IBN_NTF1 = %5.0f dB' % sigma_NTF )
pl.text(0.003, sigma_NTF2 + 5, 'IBN_NTF2 = %5.0f dB' % sigma_NTF2 )

pl.tight_layout()
pl.show()


pl.tight_layout()