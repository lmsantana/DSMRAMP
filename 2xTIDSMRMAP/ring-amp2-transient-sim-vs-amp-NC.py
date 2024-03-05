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
import scipy as sp


def ds_quantize(y, n):
    """v = ds_quantize(y,n)
    Quantize y to:
     
    * an odd integer in [-n+1, n-1], if n is even, or
    * an even integer in [-n, n], if n is odd.

    This definition gives the same step height for both mid-rise
    and mid-tread quantizers.
    """
    n = ds.carray(n)
    v = np.zeros(y.shape)
    for qi in range(n.shape[0]): 
        if n[qi] % 2 == 0: # mid-rise quantizer
            v[qi] = 2*np.floor(0.5*y[qi]) + 1
        else: # mid-tread quantizer
            v[qi] = 2*np.floor(0.5*(y[qi] + 1))
        L = n[qi] - 1
        v[qi] = np.sign(v[qi])*np.min((np.abs(v[qi]), L))
    return v



"""
## Modulator parameters

order is choosen to increase the OTA/RingAmp power tradeoff
OSR is chossen to achieve BW of 50MHz
nlev is chossen to reduce noise floor and increase total SQNR
form is still open to discussion
"""
order = 3
OSR = 10
nlev = 32        ## 4-bit quantizer
form = 'CIFF'   ## FF better for highly linear OpA 

vdd = 0.9
delta = 2 #default value

###################
## Scales
###################
analog_scale = 1.0

sin2se_dBFS = (nlev-1)/vdd
sin2di_dBFS = (nlev-1)/(2*vdd)
sin2se_dBV = (nlev-1)
sin2di_dBV = (nlev-1)/2

"""
Time simulation of the scaled ABCD for sanity check and for spectrum analysis
Spectrum is plot for checking also
"""
N = 2**13+8
decim = 256
over = 1
fb = int(np.ceil(N/(2.*(over*OSR))))
fin = np.floor(1./7. * fb)
## 5dbV is equivalent to 1.8V-lin
amp = np.array([ds.undbv(5.0),
                ds.undbv(4.5),
                ds.undbv(4.0),
                ds.undbv(3.5),
                ds.undbv(3.0),
                ds.undbv(2.5),
                ds.undbv(2.0),
                ds.undbv(1.5),
                ds.undbv(1.0),
                ds.undbv(0.5),
                ds.undbv(0.0),
                ds.undbv(-1.0),
                ds.undbv(-2.0),
                ds.undbv(-20.0),
                ds.undbv(-50.0),
                ds.undbv(-80.0)])

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
noise_enb = 0.0 # for the sampling input and DAC feedback
noise_amp = 0.0 # for the amplifiers noise ## Noise input vector (w/ input referred noise of input SW and DAC) 
saturation = True
sat_level_stg = 0.9*sin2se_dBV
offset_on = 0.0
offset_calibrated = True
np.random.seed(1)
##########################

Cin = 125e-15*8
sigma2_sw_in = 4 * 4e-21 / Cin

amp_psd = 1e-5 ## V/sqrt(Hz)
sigma2_amp = np.array([(amp_psd)**2 , (amp_psd)**2, (amp_psd)**2])

input_noise = noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(N))*sin2se_dBFS

snr_amp = np.ones(np.size(amp))
maxvalues = np.zeros((np.size(amp),5))   

ABCDs = np.array([[ 1.        ,  0.       ,  0.       ,  2.38     , -2.38],
                  [ 1.081     ,  1.       , -0.0701   ,  0.       ,  0.   ],
                  [ 0.        ,  0.827    ,  1.       ,  0.       ,  0.   ],
                  [ 1.24      ,  1.09     ,  0.35     ,  1.       ,  0.   ]])
    
#ABCDs = np.array([[ 1.        ,  0.       ,  0.       ,  1.0      , -1.0 ],
#                  [ 2.49      ,  1.       , -0.0701   ,  0.       ,  0.   ],
#                  [ 0.        ,  0.827    ,  1.       ,  0.       ,  0.   ],
#                  [ 2.86      ,  1.09     ,  0.35     ,  1.       ,  0.   ]])
    
## ABCD matrix for a CLANS nlev=8 element
#ABCDs = np.array([[ 1.        ,  0.       ,  0.       ,  1.0     , -1.0],
#                  [ 1.        ,  1.       , -0.058    ,  0.       ,  0.   ],
#                  [ 0.        ,  1.       ,  1.       ,  0.       ,  0.   ],
#                  [ 2.83      ,  2.86     ,  1.13     ,  1.       ,  0.   ]])

a,g,b,c = ds.mapABCD(ABCDs, form)

## Simplest model possible for finite DC gain with integrator feedback attenuation
if finite_dc_gain == True:    
    av = ds.undbv(40) # 60db
    a,g,b,c = ds.mapABCD(ABCDs, form)
    
    ## Degradation due to steady state error = \times Aol / (Aol + Acl)
    an = np.divide(av * a, (av + a))
    gn = np.divide(av * g, (av + g))
    bn = np.divide(av * b, (av + b))
    cn = np.divide(av * c, (av + c))
    abcd = ds.stuffABCD(an,gn,bn,cn,form)
    abcd[0,0] = abcd[0,0] * (av / (1 + av))
    abcd[1,1] = abcd[1,1] * (av / (1 + av))
    abcd[2,2] = abcd[2,2] * (av / (1 + av))
    ABCDs = abcd


for amp_index in range(np.size(amp)-7):
    
    ## All architectures subjected to the same input with thermal noise
#    u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBV + input_noise
    u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBFS + input_noise
    
    """
    Hard code implementation of ds.simulateDSM_python function, it is not optmized for time as
    the Cyhton version, but it allows access to internal nodes
    """
    
    ## simulateDSM hard code for noise input and further changes
    nq = 1 # 1 input
    nu = 1 # 1 output
    A = ABCDs[:order, :order]
    B = ABCDs[:order, order:order+nu+nq]
    C = ABCDs[order:order+nq, :order]
    D1 = ABCDs[order:order+nq, order:order+nu] ## assuming no direct feedback from V to Y
    
    u = u.reshape((1,-1))
    x0 = np.zeros((order,), dtype=np.float64)
    v = np.zeros((nq, N), dtype=np.float64)
    vbit = np.zeros((nq, N), dtype=np.float64)
    vbitdac1 = np.zeros((nq, N), dtype=np.float64)
    
    vdac1 = np.zeros((nq, N), dtype=np.float64)
    vdac2 = np.zeros((nq, N), dtype=np.float64)
    vdac2int = np.zeros((nq, N), dtype=np.float64)
    
    y = np.zeros((nq, N), dtype=np.float64)     # to store the quantizer input
    xn = np.zeros((order, N), dtype=np.float64) # to store the state information
    xmax = np.abs(x0) # to keep track of the state maxima
    
    y0 = 0
    yz1 = 0
    vz1 = 0
    
    vdac2int0 = 0
    F = np.array([0 , A[1,0]*B[0,1], 0])
    for i in range(N):
 
        y0 = np.real(np.dot(C, x0) + np.dot(D1, u[:,i])) - (vz1-yz1)
#        y0 = np.real(np.dot(C, x0) + np.dot(D1, u[:,i]))
        
        y[:,i] = y0
        yz1 = y0
        ################
        ## Quantization with offset
        ################
        v[:,i] = ds.ds_quantize(y0,nlev)
        vz1 = v[:,i]
        
        ### Full DAC feedback at input of INT1
        x0 = np.dot(A, x0) + np.dot(B, np.concatenate((u[:,i], v[:,i])))

        
        if saturation == True:
            if np.abs(x0[0]) > sat_level_stg:
                x0[0] = np.sign(x0[0])*sat_level_stg
            if np.abs(x0[1]) > sat_level_stg:
                x0[1] = np.sign(x0[1])*sat_level_stg
            if np.abs(x0[2]) > sat_level_stg:
                x0[2] = np.sign(x0[2])*sat_level_stg
        xn[:, i] = np.real_if_close(x0.T)
        xmax = np.max(np.hstack((np.abs(x0).reshape((-1, 1)), xmax.reshape((-1, 1)))),
                      axis=1, keepdims=True)
    u = u.squeeze()
    v = v.squeeze()
    vdac1 = vdac1.squeeze()
    vdac2 = vdac2.squeeze()
    vdac2int = vdac2int.squeeze()
    xn = xn.squeeze()
    y = y.squeeze()
    
    ##########################
    ## Save max and min values of vairiable
    ##########################
    maxvalues[amp_index,0] = xmax[0]
    maxvalues[amp_index,1] = xmax[1]
    maxvalues[amp_index,2] = xmax[2]
    maxvalues[amp_index,3] = np.max(np.abs(y))
    maxvalues[amp_index,4] = np.max(np.abs(u))
    
    ##########################
    ## Plot some DAC values for sanity check
    ##########################
    if amp_index == 8:
        pl.figure()
        ds.figureMagic([1, 100], None, None, [-31, 31], 2, None, (14, 8), 'DAC plot')
        pl.step(np.linspace(1,100,num=99), v[1:100], label = 'Complete DOUT')
        pl.step(np.linspace(1,100,num=99), vdac1[1:100], label = 'DOUT[4:2]')
        pl.step(np.linspace(1,100,num=99), vdac2[1:100], label = 'DOUT[1:0]')
        pl.step(np.linspace(1,100,num=99), vdac2int[1:100], label = 'INT DOUT[1:0]')
        pl.legend()
        pl.plot()
    
    f = np.linspace(0, 0.5, int(N/2. + 1))
    spec = np.fft.fft(v* analog_scale * ds.ds_hann(N))/(N/4)
    snr_amp[amp_index] = ds.calculateSNR(spec[2:fb+1], fin-2)
    print(amp_index)
    if amp_index == 8:
        pl.figure()
        pl.semilogx(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-130, 20], 10, None, (16, 6), 'Output Spectrum')
        pl.plot([0.0005, 0.5/(over*OSR)], [-130, -130], 'k', linewidth=10, label='Signal Band')
        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr_amp[amp_index], OSR), verticalalignment='center')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
        pl.figure()
        pl.plot(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum')
        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr_amp[amp_index], OSR), verticalalignment='center')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
        vdec = sp.signal.decimate(v,OSR, ftype='fir')
        N2 = np.size(vdec)
        f2 = np.linspace(0, 0.5, int(N2/2. + 1))
        spec2 = np.fft.fft(vdec* analog_scale * ds.ds_hann(N2))/(N2/4)
        pl.figure()
        pl.plot(f2, ds.dbv(spec2[:int(N2/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum')
        pl.plot([0.0005, 0.5], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(spec2[1:int(N2/2)],fin,nsig=2), 1.0), verticalalignment='center')
        pl.xlabel('Normalized Decimated Frequency')
        pl.ylabel('dBFS')
        pl.plot()

print('\n'*3)
print('Maximum absolute values of state variables =  \n')
print(pd.DataFrame(maxvalues, columns=['x1','x2','x3','y','u'], \
                   index=['5','4.5','4','3.5','3','2.5','2','1.5','1','0.5','0','-1','-2','-20','-50','-80']))
print('\n'*3)

pl.figure(figsize=(10,4))
pl.plot(ds.dbv(amp),snr_amp, label='No dac mismatch')
ds.figureMagic([-80, 5], 5, None, [-10,100], 5, None, (10, 6), 'OUT Dynamic Range')
pl.text(-20, 75,'Peak SNR = %2.1f dB @ %2.1f dbV' %(snr_amp.max(), ds.dbv(amp[np.argmax(snr_amp)])))
pl.xlabel('Amplitude (dBVp-p differential)')
pl.ylabel('SNR (dB)')
pl.legend(loc=4)
pl.plot()


ntf,_ = ds.calculateTF(ABCDs)

ds.DocumentNTF(ntf,OSR)
f1, f2 = ds.ds_f1f2(OSR, 0, 0)
NG0 = ds.dbv(ds.rmsGain(ntf, f1, f2))
NG0_lin = ds.undbv(NG0)
NG0_lin_squared = NG0_lin**2

NTF_gain = ds.rmsGain(ntf, 0, 0.5)**2


## Calculation for an extra NC zero
z,p,k = ntf[0], ntf[1], ntf[2]
z2 = np.append(z, 1)
ntf2 = tuple([z2, p, k])
ds.DocumentNTF(ntf2,1*OSR)

## Calculation for NC-TI extra zero
z3 = np.append(z2, 1)
ntf3 = tuple([z3, p, k])
ds.DocumentNTF(ntf3,1*OSR)
#pl.plot()
#pl.figure()
#f = ds.ds_freq(OSR, 0, 0)
#z = np.exp(2j * np.pi * f)
#H = ds.dbv(ds.evalTF(ntf2, z))
#
#pl.figure()
#pl.plot(f, H, 'b')
#pl.plot([0.0, 0.5/OSR], [-120, -120], 'k' )
#pl.grid()
#pl.plot()
#pl.figure()
#pl.semilogx(f, H, 'b')
#pl.semilogx([0.0, 0.5/OSR], [-120, -120], 'k' )
#pl.grid()
#pl.plot()
#
#print(ds.pretty_lti(ntf))
#print(ds.pretty_lti(ntf2))

#pl.figure(figsize=(10,4))
#pl.step(np.concatenate((np.array([-nlev]),threshold_vec, np.array([nlev]))), np.concatenate((nlev_vec, np.array([nlev-1]))), where = 'post', label='No offset')
#for i in range(nlev-1):
#    pl.step(np.concatenate((np.array([-nlev]),threshold_vec+offset[i,:], np.array([nlev]))), np.concatenate((nlev_vec, np.array([nlev-1]))), where = 'post', label='With Offset')
#ds.figureMagic([-nlev,nlev], None, None, [-nlev, nlev], 1, None, (16, 10), 'Stair transfer')
#pl.xlabel('ADC input signal')
#pl.ylabel('ADC output signal')
#pl.legend(loc=4)
#pl.plot()
#
#pl.figure(figsize=(10,4))
#pl.step(np.concatenate((np.array([-nlev]),threshold_vec, np.array([nlev])))/(sin2se_dBV*flash_scale), np.concatenate((nlev_vec, np.array([nlev-1]))), where = 'post', label='No offset')
#for i in range(nlev-1):
#    pl.step(np.concatenate((np.array([-nlev]),threshold_vec+offset[i,:], np.array([nlev])))/(sin2se_dBV*flash_scale), np.concatenate((nlev_vec, np.array([nlev-1]))), where = 'post', label='With Offset')
#ds.figureMagic([-nlev/(sin2se_dBV*flash_scale),nlev/(sin2se_dBV*flash_scale)], None, None, [-nlev, nlev], 1, None, (16, 10), 'Stair transfer')
#pl.xlabel('ADC input signal scaled (V)')
#pl.ylabel('ADC output signal')
#pl.legend(loc=4)
#pl.plot()
