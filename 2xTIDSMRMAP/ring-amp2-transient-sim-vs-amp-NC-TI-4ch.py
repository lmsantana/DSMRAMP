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
N = 2**13
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
finite_dc_gain = False
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


for amp_index in range(np.size(amp)):
    
    ## All architectures subjected to the same input with thermal noise
#    u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBV + input_noise
    u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBFS + input_noise
    
    # Downsample the input sampling to the Nyquist frequency after decimantion of OSR
#    for i in range(N):
#        if i%10 != 0:
#            u[i] = u[i - i%10]
#            
#    
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
    v = np.zeros((nq, N), dtype=np.float64)
    x01 = np.zeros((order,), dtype=np.float64)
    v1 = np.zeros((nq, N), dtype=np.float64)
    y1 = np.zeros((nq, N), dtype=np.float64)     # to store the quantizer input
    xn1 = np.zeros((order, N), dtype=np.float64) # to store the state information
    xmax1 = np.abs(x01) # to keep track of the state maxima
    
    x02 = np.zeros((order,), dtype=np.float64)
    v2 = np.zeros((nq, N), dtype=np.float64)
    y2 = np.zeros((nq, N), dtype=np.float64)     # to store the quantizer input
    xn2 = np.zeros((order, N), dtype=np.float64) # to store the state information
    xmax2 = np.abs(x02) # to keep track of the state maxima
    
    x03 = np.zeros((order,), dtype=np.float64)
    v3 = np.zeros((nq, N), dtype=np.float64)
    y3 = np.zeros((nq, N), dtype=np.float64)     # to store the quantizer input
    xn3 = np.zeros((order, N), dtype=np.float64) # to store the state information
    xmax3 = np.abs(x03) # to keep track of the state maxima
    
    x04 = np.zeros((order,), dtype=np.float64)
    v4 = np.zeros((nq, N), dtype=np.float64)
    y4 = np.zeros((nq, N), dtype=np.float64)     # to store the quantizer input
    xn4 = np.zeros((order, N), dtype=np.float64) # to store the state information
    xmax4 = np.abs(x04) # to keep track of the state maxima
    
    y01 = 0
    yz1 = 0
    vz1 = 0
    
    y02 = 0
    yz2 = 0
    vz2 = 0
    
    y03 = 0
    yz3 = 0
    vz3 = 0
    
    y04 = 0
    yz4 = 0
    vz4 = 0

#   Coupling matrixes required
#   1st order (1 - z^-1)
#   [0     0     0     -z^-1
#    -z^-1 0     0     0
#    0     -z^-1 0     0
#    0     0     -z^-1 0]
   
#   2nd order (1 - 2z^-1 + z^2)
#   [0      0      z^2     -2z^-1
#    -2z^-1 0      0       z^2
#    z^2    -2z^-1 0       0
#    0      z^2    -2z^-1  0]
    
#   3nd order (1 - 3z^-1 +3z^-2 -z^3)
#   [0        -z^3   3z^2     -3z^-1
#    -3z^-1   0      -z^3     3z^2
#    3z^2    -3z^-1  0        -z^3
#    -z^3     3z^2   -3z^-1   0]
    
#   4th order (1 - 4z^-1 +6z^-2 -4z^3+1z^4)
#   [ 1+z^4    -4z^3   6z^2    -4z^-1
#    -4z^-1    1+z^4  -4z^3     6z^2
#     6z^2    -4z^-1   1+z^4   -4z^3
#    -4z^3     6z^2   -4z^-1   1+z^4]
   


    for i in range(N):
        
        if i%4 == 0:
            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) -(vz4-yz4)
#            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) -2*(vz4-yz4)+(vz3-yz3)
#            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) -3*(vz4-yz4)+3*(vz3-yz3)-(vz2-yz2)
#            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) -4*(vz4-yz4)+6*(vz3-yz3)-4*(vz2-yz2)+(vz1-yz1)
            y1[:,i] = y01
            yz1 = y01

            v1[:,i] = ds.ds_quantize(y01,nlev)
            vz1 = v1[:,i]
            v[:,i] = v1[:,i]
            
            ### Full DAC feedback at input of INT1
            x01 = np.dot(A, x01) + np.dot(B, np.concatenate((u[:,i], v1[:,i])))
    
            if saturation == True:
                if np.abs(x01[0]) > sat_level_stg:
                    x01[0] = np.sign(x01[0])*sat_level_stg
                if np.abs(x01[1]) > sat_level_stg:
                    x01[1] = np.sign(x01[1])*sat_level_stg
                if np.abs(x01[2]) > sat_level_stg:
                    x01[2] = np.sign(x01[2])*sat_level_stg
            xn1[:, i] = np.real_if_close(x01.T)
            xmax1 = np.max(np.hstack((np.abs(x01).reshape((-1, 1)), xmax1.reshape((-1, 1)))),
                          axis=1, keepdims=True)
            
        if i%4 == 1:
            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) -(vz1-yz1)
#            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) -2*(vz1-yz1)+(vz4-yz4)
#            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) -3*(vz1-yz1)+3*(vz4-yz4)-(vz3-yz3)
#            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) -4*(vz1-yz1)+6*(vz4-yz4)-4*(vz3-yz3)+(vz2-yz2)
            y2[:,i] = y02
            yz2 = y02

            v2[:,i] = ds.ds_quantize(y02,nlev)
            vz2 = v2[:,i]
            v[:,i] = v2[:,i]
            
            ### Full DAC feedback at input of INT1
            x02 = np.dot(A, x02) + np.dot(B, np.concatenate((u[:,i], v2[:,i])))
    
            if saturation == True:
                if np.abs(x02[0]) > sat_level_stg:
                    x02[0] = np.sign(x02[0])*sat_level_stg
                if np.abs(x02[1]) > sat_level_stg:
                    x02[1] = np.sign(x02[1])*sat_level_stg
                if np.abs(x02[2]) > sat_level_stg:
                    x02[2] = np.sign(x02[2])*sat_level_stg
            xn2[:, i] = np.real_if_close(x02.T)
            xmax2 = np.max(np.hstack((np.abs(x02).reshape((-1, 1)), xmax2.reshape((-1, 1)))),
                          axis=1, keepdims=True)
            
        if i%4 == 2:
            y03 = np.real(np.dot(C, x03) + np.dot(D1, u[:,i])) -(vz2-yz2)
#            y03 = np.real(np.dot(C, x03) + np.dot(D1, u[:,i])) -2*(vz2-yz2)+(vz1-yz1)
#            y03 = np.real(np.dot(C, x03) + np.dot(D1, u[:,i])) -3*(vz2-yz2)+3*(vz1-yz1)-(vz4-yz4)
#            y03 = np.real(np.dot(C, x03) + np.dot(D1, u[:,i])) -4*(vz2-yz2)+6*(vz1-yz1)-4*(vz4-yz4)+(vz3-yz3)
            y3[:,i] = y03
            yz3 = y03

            v3[:,i] = ds.ds_quantize(y03,nlev)
            vz3 = v3[:,i]
            v[:,i] = v3[:,i]
            
            ### Full DAC feedback at input of INT1
            x03 = np.dot(A, x03) + np.dot(B, np.concatenate((u[:,i], v3[:,i])))
    
            if saturation == True:
                if np.abs(x03[0]) > sat_level_stg:
                    x03[0] = np.sign(x03[0])*sat_level_stg
                if np.abs(x03[1]) > sat_level_stg:
                    x03[1] = np.sign(x03[1])*sat_level_stg
                if np.abs(x03[2]) > sat_level_stg:
                    x03[2] = np.sign(x03[2])*sat_level_stg
            xn3[:, i] = np.real_if_close(x03.T)
            xmax3 = np.max(np.hstack((np.abs(x03).reshape((-1, 1)), xmax3.reshape((-1, 1)))),
                          axis=1, keepdims=True)
            
        if i%4 == 3:
            y04 = np.real(np.dot(C, x04) + np.dot(D1, u[:,i])) -(vz3-yz3)
#            y04 = np.real(np.dot(C, x04) + np.dot(D1, u[:,i])) -2*(vz3-yz3)+(vz2-yz2)
#            y04 = np.real(np.dot(C, x04) + np.dot(D1, u[:,i])) -3*(vz3-yz3)+3*(vz2-yz2)-(vz1-yz1)
#            y04 = np.real(np.dot(C, x04) + np.dot(D1, u[:,i])) -4*(vz3-yz3)+6*(vz2-yz2)-4*(vz1-yz1)+(vz4-yz4)
            y4[:,i] = y04
            yz4 = y04

            v4[:,i] = ds.ds_quantize(y04,nlev)
            vz4 = v4[:,i]
            v[:,i] = v4[:,i]
            
            ### Full DAC feedback at input of INT1
            x04 = np.dot(A, x04) + np.dot(B, np.concatenate((u[:,i], v4[:,i])))
    
            if saturation == True:
                if np.abs(x04[0]) > sat_level_stg:
                    x04[0] = np.sign(x04[0])*sat_level_stg
                if np.abs(x04[1]) > sat_level_stg:
                    x04[1] = np.sign(x04[1])*sat_level_stg
                if np.abs(x04[2]) > sat_level_stg:
                    x04[2] = np.sign(x04[2])*sat_level_stg
            xn4[:, i] = np.real_if_close(x04.T)
            xmax4 = np.max(np.hstack((np.abs(x04).reshape((-1, 1)), xmax4.reshape((-1, 1)))),
                          axis=1, keepdims=True)
    u = u.squeeze()
    v1 = v1.squeeze()
    v2 = v2.squeeze()
    v3 = v3.squeeze()
    v4 = v4.squeeze()
    v = v.squeeze()
    xn1 = xn1.squeeze()
    y1 = y1.squeeze()
    y2 = y2.squeeze()
    y3 = y3.squeeze()
    y4 = y4.squeeze()
    
    ##########################
    ## Save max and min values of vairiable
    ##########################
    maxvalues[amp_index,0] = xmax1[0]
    maxvalues[amp_index,1] = xmax1[1]
    maxvalues[amp_index,2] = xmax1[2]
    maxvalues[amp_index,3] = np.max(np.abs(y1))
    maxvalues[amp_index,4] = np.max(np.abs(u))
    
#    ##########################
#    ## Plot some DAC values for sanity check
#    ##########################
#    if amp_index == 8:
#        pl.figure()
#        ds.figureMagic([1, 100], None, None, [-31, 31], 2, None, (14, 8), 'DAC plot')
#        pl.step(np.linspace(1,100,num=99), v[1:100], label = 'Complete DOUT')
#        pl.step(np.linspace(1,100,num=99), vdac1[1:100], label = 'DOUT[4:2]')
#        pl.step(np.linspace(1,100,num=99), vdac2[1:100], label = 'DOUT[1:0]')
#        pl.step(np.linspace(1,100,num=99), vdac2int[1:100], label = 'INT DOUT[1:0]')
#        pl.legend()
#        pl.plot()
    
    f = np.linspace(0, 0.5, int(N/2. + 1))
    spec = np.fft.fft(v* analog_scale * ds.ds_hann(N))/(N/4)
    snr_amp[amp_index] = ds.calculateSNR(spec[2:fb+1], fin-2)
    print(amp_index)
    if amp_index == 12:
        pl.figure()
        pl.semilogx(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum')
        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
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


#ntf,_ = ds.calculateTF(ABCDs)
#
#ds.DocumentNTF(ntf,OSR)
#f1, f2 = ds.ds_f1f2(OSR, 0, 0)
#NG0 = ds.dbv(ds.rmsGain(ntf, f1, f2))
#NG0_lin = ds.undbv(NG0)
#NG0_lin_squared = NG0_lin**2
#
#NTF_gain = ds.rmsGain(ntf, 0, 0.5)**2


## Calculation for an extra self NC zero
#z,p,k = ntf[0], ntf[1], ntf[2]
#z2 = np.append(z, 1)
#ntf2 = tuple([z2, p, k])
#ds.DocumentNTF(ntf2,1*OSR)

## Calculation for NC-TI extra zero
## Calculation for a polyphase extrapolation by z^-2 self NC and 2z^-1 crossed NC
#z,p,k = ntf[0], ntf[1], ntf[2]
#num,den = sp.signal.zpk2tf(z,p,k)
#new_num = np.array([num[0] , 0.0, num[1], 0.0, num[2], 0.0, num[3]])
#new_den = np.array([den[0] , 0.0, den[1], 0.0, den[2], 0.0, den[3]])
#z3,p3,k3 = sp.signal.tf2zpk(new_num, new_den)
#z3 = np.append(z3, 0.99)
#z3 = np.append(z3, 0.99)
#ntf3 = tuple([z3, p3, k3])

### Calculation for a polyphase extrapolation by z^-1 self NC and z^-1 crossed NC
#z,p,k = ntf2[0], ntf2[1], ntf2[2]
#num,den = sp.signal.zpk2tf(z,p,k)
#new_num = np.array([num[0] , 0.0, num[1], 0.0, num[2], 0.0, num[3], 0.0, num[4]])
#new_den = np.array([den[0] , 0.0, den[1], 0.0, den[2], 0.0, den[3]])
#z3,p3,k3 = sp.signal.tf2zpk(new_num, new_den)
#z3 = np.append(z3, 0.99)
#ntf3 = tuple([z3, p3, k3])

## common part of the code
#ds.DocumentNTF(ntf3,1*OSR)
#
#freq = 1e9
#
#f = ds.ds_freq(OSR, 0, 0)
#z = np.exp(2j * np.pi * f)
#ntf = ntf2
#H = ds.dbv(ds.evalTF(ntf, z))
#
#pl.figure(figsize=(10,4))
#pl.plot(f*freq, H, 'b')
#pl.plot([0.0, 0.5*freq/OSR], [ds.dbv(ds.rmsGain(ntf, 0, 1/(2*OSR))), ds.dbv(ds.rmsGain(ntf, 0, 1/(2*OSR)))], 'k' )
#msg = 'QN attenuation by NTF = %2.f' %(ds.dbv(ds.rmsGain(ntf, 0, 1/(2*OSR))))
#pl.text(freq/(2*OSR), ds.dbv(ds.rmsGain(ntf, 0, 1/(2*OSR))), msg)
#pl.xlabel('frequency - linear [Hz]')
#pl.ylabel('|NTF| [dB]')
#pl.grid()
#pl.plot()
#pl.figure(figsize=(10,4))
#pl.semilogx(f*freq, H, 'b')
#pl.semilogx([0.0, 0.5*freq/OSR], [ds.dbv(ds.rmsGain(ntf, 0, 1/(2*OSR))), ds.dbv(ds.rmsGain(ntf, 0, 1/(2*OSR)))], 'k' )
#msg = 'QN attenuation by NTF = %2.f' %(ds.dbv(ds.rmsGain(ntf, 0, 1/(2*OSR))))
#pl.text(freq/(2*OSR), ds.dbv(ds.rmsGain(ntf, 0, 1/(2*OSR))), msg)
#pl.xlabel('frequency - log [Hz]')
#pl.ylabel('|NTF| [dB]')
#pl.grid()
#pl.plot()
#
#print('\n\n')
#print(ds.pretty_lti(ntf))
#print('\n\n')
#print(ds.pretty_lti(ntf2))
#print('\n\n')
#print(ds.pretty_lti(ntf3))
#print('\n\n')

