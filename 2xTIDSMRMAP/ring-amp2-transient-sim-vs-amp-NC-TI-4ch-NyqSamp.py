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
OSR = 12
nlev = 32        ## 4-bit quantizer
form = 'CIFF'   ## FF better for highly linear OpA 

vdd = 0.9
delta = 2 #default value

nch = 4
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
N = 2**14+8
#N = 2**13+4
decim = 256
over = 1
fb = int(np.ceil(N/(2.*(over*OSR))))
fin = np.floor(6./7. * fb)
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
noise_enb = 1.0 # for the sampling input and DAC feedback
noise_amp = 1.0 # for the amplifiers noise ## Noise input vector (w/ input referred noise of input SW and DAC) 
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
snr_amp_ti = np.ones(np.size(amp))
maxvalues = np.zeros((np.size(amp),5))   

ABCDs = np.array([[ 1.        ,  0.       ,  0.       ,  2.38     , -2.38],
                  [ 1.081     ,  1.       , -0.0701   ,  0.       ,  0.   ],
                  [ 0.        ,  0.827    ,  1.       ,  0.       ,  0.   ],
                  [ 1.24      ,  1.09     ,  0.35     ,  1.       ,  0.   ]])

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
#for amp_index in range(np.size(amp)):
    
    ## All architectures subjected to the same input with thermal noise
#    u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBFS + input_noise
    u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBFS
    u1 = np.copy(u)
    u2 = np.copy(u)
    u3 = np.copy(u)
    u4 = np.copy(u)
    ##Working part   
    for i in range(N):
        if i%(nch*OSR) != 0:
            u1[i] = u[i - i%(nch*OSR)] + noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(1)[0])*sin2se_dBFS
    for i in range(-OSR,N-OSR):
        if i%(nch*OSR) != 0:
            u2[i+OSR] = u[i - i%(nch*OSR) +OSR] + noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(1)[0])*sin2se_dBFS
    for i in range(-2*OSR,N-2*OSR):
        if i%(nch*OSR) != 0:
            u3[i+2*OSR] = u[i - i%(nch*OSR)+2*OSR] + noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(1)[0])*sin2se_dBFS
    for i in range(-3*OSR,N-3*OSR):
        if i%(nch*OSR) != 0:
            u4[i+3*OSR] = u[i - i%(nch*OSR) +3*OSR] + noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(1)[0])*sin2se_dBFS
  
#    pl.figure(figsize=(10,8));plot(u1[1:80]);plot(u2[1:80]);plot(u3[1:80]);plot(u4[1:80]);plot(u[1:80])
            
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
    u1 = u1.reshape((1,-1))
    u2 = u2.reshape((1,-1))
    u3 = u3.reshape((1,-1))
    u4 = u4.reshape((1,-1))
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
   
    v1con = np.zeros((nq, int(N/nch)), dtype=np.float64)
    v2con = np.zeros((nq, int(N/nch)), dtype=np.float64)
    v3con = np.zeros((nq, int(N/nch)), dtype=np.float64)
    v4con = np.zeros((nq, int(N/nch)), dtype=np.float64)

    for i in range(N):
        
        if i%4 == 0:
            y01 = np.real(np.dot(C, x01) + np.dot(D1, u1[:,i]))
#            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) -(vz4-yz4)
#            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) -2*(vz4-yz4)+(vz3-yz3)
#            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) -3*(vz4-yz4)+3*(vz3-yz3)-(vz2-yz2)
#            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) -4*(vz4-yz4)+6*(vz3-yz3)-4*(vz2-yz2)+(vz1-yz1)
            y1[:,i] = y01
            yz1 = y01

            v1[:,i] = ds.ds_quantize(y01,nlev)
            vz1 = v1[:,i]
            v[:,i] = v1[:,i]
            v1con[:,int(i/nch)] = v1[:,i]
            
            ### Full DAC feedback at input of INT1
            x01 = np.dot(A, x01) + np.dot(B, np.concatenate((u1[:,i], v1[:,i])))
    
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
            y02 = np.real(np.dot(C, x02) + np.dot(D1, u2[:,i]))
#            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) -(vz1-yz1)
#            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) -2*(vz1-yz1)+(vz4-yz4)
#            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) -3*(vz1-yz1)+3*(vz4-yz4)-(vz3-yz3)
#            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) -4*(vz1-yz1)+6*(vz4-yz4)-4*(vz3-yz3)+(vz2-yz2)
            y2[:,i] = y02
            yz2 = y02

            v2[:,i] = ds.ds_quantize(y02,nlev)
            vz2 = v2[:,i]
            v[:,i] = v2[:,i]
            v2con[:,int((i-1)/nch)] = v2[:,i]
            
            ### Full DAC feedback at input of INT1
            x02 = np.dot(A, x02) + np.dot(B, np.concatenate((u2[:,i], v2[:,i])))
    
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
            y03 = np.real(np.dot(C, x03) + np.dot(D1, u3[:,i]))
#            y03 = np.real(np.dot(C, x03) + np.dot(D1, u[:,i])) -(vz2-yz2)
#            y03 = np.real(np.dot(C, x03) + np.dot(D1, u[:,i])) -2*(vz2-yz2)+(vz1-yz1)
#            y03 = np.real(np.dot(C, x03) + np.dot(D1, u[:,i])) -3*(vz2-yz2)+3*(vz1-yz1)-(vz4-yz4)
#            y03 = np.real(np.dot(C, x03) + np.dot(D1, u[:,i])) -4*(vz2-yz2)+6*(vz1-yz1)-4*(vz4-yz4)+(vz3-yz3)
            y3[:,i] = y03
            yz3 = y03

            v3[:,i] = ds.ds_quantize(y03,nlev)
            vz3 = v3[:,i]
            v[:,i] = v3[:,i]
            v3con[:,int((i-2)/nch)] = v3[:,i]
            
            ### Full DAC feedback at input of INT1
            x03 = np.dot(A, x03) + np.dot(B, np.concatenate((u3[:,i], v3[:,i])))
    
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
            y04 = np.real(np.dot(C, x04) + np.dot(D1, u4[:,i]))
#            y04 = np.real(np.dot(C, x04) + np.dot(D1, u[:,i])) -(vz3-yz3)
#            y04 = np.real(np.dot(C, x04) + np.dot(D1, u[:,i])) -2*(vz3-yz3)+(vz2-yz2)
#            y04 = np.real(np.dot(C, x04) + np.dot(D1, u[:,i])) -3*(vz3-yz3)+3*(vz2-yz2)-(vz1-yz1)
#            y04 = np.real(np.dot(C, x04) + np.dot(D1, u[:,i])) -4*(vz3-yz3)+6*(vz2-yz2)-4*(vz1-yz1)+(vz4-yz4)
            y4[:,i] = y04
            yz4 = y04

            v4[:,i] = ds.ds_quantize(y04,nlev)
            vz4 = v4[:,i]
            v[:,i] = v4[:,i]
            v4con[:,int((i-3)/nch)] = v4[:,i]
            
            ### Full DAC feedback at input of INT1
            x04 = np.dot(A, x04) + np.dot(B, np.concatenate((u4[:,i], v4[:,i])))
    
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
    u1 = u1.squeeze()
    u2 = u2.squeeze()
    u3 = u3.squeeze()
    u4 = u4.squeeze()
    
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
    
    v1con = v1con.squeeze()
    v2con = v2con.squeeze()
    v3con = v3con.squeeze()
    v4con = v4con.squeeze()
    
    ##########################
    ## Save max and min values of vairiable
    ##########################
    maxvalues[amp_index,0] = xmax1[0]
    maxvalues[amp_index,1] = xmax1[1]
    maxvalues[amp_index,2] = xmax1[2]
    maxvalues[amp_index,3] = np.max(np.abs(y1))
    maxvalues[amp_index,4] = np.max(np.abs(u))
    
    f = np.linspace(0, 0.5, int(N/2. + 1))
    spec = np.fft.fft(v* analog_scale * ds.ds_hann(N))/(N/4)
    snr_amp[amp_index] = ds.calculateSNR(spec[2:fb+1], fin-2)
    

    bf, af = sp.signal.firwin((4*10*OSR)+1, 1. / OSR, window='hamming'), 1.
    v1fil = sp.signal.filtfilt(bf, af, v1con, axis=-1)
    v2fil = sp.signal.filtfilt(bf, af, v2con, axis=-1)
    v3fil = sp.signal.filtfilt(bf, af, v3con, axis=-1)
    v4fil = sp.signal.filtfilt(bf, af, v4con, axis=-1)
    v1dec = v1fil[slice(0,None,OSR)]
    v2dec = v2fil[slice(3,None,OSR)]
    v3dec = v3fil[slice(6,None,OSR)]
    v4dec = v4fil[slice(9,None,OSR)]
    
    vinter = np.zeros(int(N/OSR))
    for k in range(np.size(vinter)):
        if k%nch==0:
            vinter[k] = v1dec[int(k/nch)]
        elif k%nch==1:
            vinter[k] = v2dec[int((k-1)/nch)]
        elif k%nch==2:
            vinter[k] = v3dec[int((k-2)/nch)]
        elif k%nch==3:
            vinter[k] = v4dec[int((k-3)/nch)]
    N2 = np.size(vinter)
    f2 = np.linspace(0, 0.5, int(N2/2. + 1))* nch* 1e9*N2/N
    spec2 = np.fft.fft(vinter* analog_scale * ds.ds_hann(N2))/(N2/4)
    snr_amp_ti[amp_index] = ds.calculateSNR(spec2[1:int(N2/2)],fin,2)
    
    print(amp_index)
    if amp_index == 9:
        f = np.linspace(0, 0.5, int(N/(nch*2) + 1))
        spec = np.fft.fft(v1con* analog_scale * ds.ds_hann(int(N/nch)))/(N/(4*nch))
        pl.figure()
        pl.semilogx(f, ds.dbv(spec[:int(N/(nch*2)) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum Log Channel 1')
        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
        pl.figure()
        pl.plot(f, ds.dbv(spec[:int(N/(nch*2)) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum Lin Channel 1')
        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
        pl.figure()
        pl.plot(f2, ds.dbv(spec2[:int(N2/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, f2[np.size(f2)-1]], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum: Low Pass FIR +Decimation/10 -> Out MUX')
        pl.plot([0.0005, 0.5], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.text(0.1, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(spec2[1:int(N2/2)],fin,nsig=2), 1.0), verticalalignment='center')
        pl.xlabel('Frequency [Hz]')
        pl.ylabel('dBFS')
        pl.plot()

print('\n'*3)
print('Maximum absolute values of state variables =  \n')
print(pd.DataFrame(maxvalues, columns=['x1','x2','x3','y','u'], \
                   index=['5','4.5','4','3.5','3','2.5','2','1.5','1','0.5','0','-1','-2','-20','-50','-80']))
print('\n'*3)

pl.figure(figsize=(10,4))
pl.plot(ds.dbv(amp),snr_amp_ti, label='No dac mismatch')
ds.figureMagic([-80, 5], 5, None, [-10,100], 5, None, (10, 6), 'OUT Dynamic Range')
pl.text(-20, 75,'Peak SNR = %2.1f dB @ %2.1f dbV' %(snr_amp_ti.max(), ds.dbv(amp[np.argmax(snr_amp_ti)])))
pl.xlabel('Amplitude (dBVp-p differential)')
pl.ylabel('SNR (dB)')
pl.legend(loc=4)
pl.plot()


