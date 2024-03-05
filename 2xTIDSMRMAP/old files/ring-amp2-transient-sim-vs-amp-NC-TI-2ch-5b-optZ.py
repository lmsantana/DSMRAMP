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
nlev = 32        ## 5-bit quantizer
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
saturation = False
sat_sum = False
sat_level_stg = 0.9*sin2se_dBV
offset_on = 0.0
offset_calibrated = True
np.random.seed(1)
##########################

Cin = 125e-15*4
#sigma2_sw_in = 4 * 4e-21 / Cin  ## Value used previously
sigma2_sw_in = 2 * 4e-21 / Cin  ## corrected values after last Verilog-A sim

amp_psd = 1e-5 ## V/sqrt(Hz)
sigma2_amp = np.array([(amp_psd)**2 , (amp_psd)**2, (amp_psd)**2])

input_noise = noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(N))*sin2se_dBFS

snr_amp = np.ones(np.size(amp))
maxvalues = np.zeros((np.size(amp),5))   


ABCDs = np.array([[ 1.        ,  0.       ,  0.       ,  2.38     , -2.38 ],
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
    
    ## All architectures subjected to the same input with thermal noise
#    u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBFS + input_noise
    u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBFS
    for i in range(N):
        if i%2 != 0:
            u[i] = u[i] + noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(1)[0])*sin2se_dBFS
        else:
            u[i] = u[i] + noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(1)[0])*sin2se_dBFS


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
    
    y01 = 0
    yz1 = 0
    vz1 = 0
    vz1p = 0
    sum1 = 0
    
    y02 = 0
    yz2 = 0
    vz2 = 0
    vz2p = 0
    sum2 = 0
    
    v1con = np.zeros((nq, int(N/2)), dtype=np.float64)
    v2con = np.zeros((nq, int(N/2)), dtype=np.float64)
    opt_g1 = 1.0
#    opt_g2 = 0.95 #80.5dB at 1.5V 
    opt_g2 = 0.945
    
    for i in range(N):
        
        if i%2 == 0:
            
            ## No delay ideal time model with output saturation
#            if sat_sum == True:
#                y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i]))  - 2*(-yz2)
#                if np.abs(y01) > sat_level_stg:
#                    y01 = np.sign(y01)*sat_level_stg
#                y01 = y01 -2*vz2 + (vz1-yz1)
#            elif sat_sum == False:
#            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) + opt_g1*(vz1-yz1)  - opt_g2*2*(vz2-yz2)
##            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) - (vz2-yz2)   
##            y01 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i]))
                
            ## Delayed model (between summation and quantization)
            sum1 = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) + opt_g1*(-sum1 + vz1 +opt_g2*2*vz2p) + opt_g2*(2*sum2 -opt_g2*4*vz1) 
            y01 = sum1 - opt_g2*2*vz2
            
            y1[:,i] = y01
            yz1 = y01

            vz1p = vz1
            
            v1[:,i] = ds.ds_quantize(y01,nlev)
            vz1 = v1[:,i]
            v[:,i] = v1[:,i]
            v1con[:,int(i/2)] = v1[:,i]
            
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
            
        if i%2 == 1:
#            if sat_sum == True:
#                y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i]))  - 2*(-yz1)
#                if np.abs(y02) > sat_level_stg:
#                    y02 = np.sign(y02)*sat_level_stg
#                y02 = y02 -2*vz1 + (vz2-yz2)
#            elif sat_sum == False:
#            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) + opt_g1*(vz2-yz2) - opt_g2*2*(vz1-yz1)
##            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) - (vz1-yz1)
##            y02 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i]))
            
            sum2 = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) + opt_g1*(-sum2 + vz2 +opt_g2*2*vz1p) + opt_g2*(2*sum1 -opt_g2*4*vz2) 
            y02 = sum2 - opt_g2*2*vz1

            
            y2[:,i] = y02
            yz2 = y02

            vz2p = vz2
            
            v2[:,i] = ds.ds_quantize(y02,nlev)
            vz2 = v2[:,i]
            v[:,i] = v2[:,i]
            v2con[:,int((i-1)/2)] = v2[:,i]
            
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
    u = u.squeeze()
    v1 = v1.squeeze()
    v2 = v2.squeeze()
    v = v.squeeze()
    xn1 = xn1.squeeze()
    y1 = y1.squeeze()
    y2 = y2.squeeze()
    v1con = v1con.squeeze()
    v2con = v2con.squeeze()
    
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
    print(amp_index)
    if amp_index == 10:
        pl.figure()
        pl.semilogx(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum TI')
        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr_amp[amp_index], OSR), verticalalignment='center')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
        pl.figure()
        pl.plot(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum TI')
        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr_amp[amp_index], OSR), verticalalignment='center')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
#        fch1 = np.linspace(0, 0.5, int(N/2.))
#        specch1 = np.fft.fft(v1* analog_scale * ds.ds_hann(int(N)))/(N/4)
#        pl.figure()
#        pl.semilogx(f, ds.dbv(specch1[:int(N/2.) +1]), 'b', label='Simulation spectrum')
#        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum Channel 1 RN vector')
#        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
#        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(specch1[1:fb+1],fin,nsig=2), OSR), verticalalignment='center')
#        pl.xlabel('Normalized Frequency')
#        pl.ylabel('dBFS')
#        pl.plot()
#        
#        pl.figure()
#        pl.plot(f, ds.dbv(specch1[:int(N/2.) +1]), 'b', label='Simulation spectrum')
#        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum Channel 1 RN vector')
#        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
#        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(specch1[1:fb+1],fin,nsig=2), OSR), verticalalignment='center')
#        pl.xlabel('Normalized Frequency')
#        pl.ylabel('dBFS')
#        pl.plot()

        vdec = sp.signal.decimate(v,OSR, ftype='fir')
        N2 = np.size(vdec)
        f3 = np.linspace(0, 0.5, int(N2/2. + 1))
        spec3 = np.fft.fft(vdec* analog_scale * ds.ds_hann(N2))/(N2/4)
        pl.figure()
        pl.plot(f3, ds.dbv(spec3[:int(N2/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum: Interpolation -> Decimation/10 FIR')
        pl.plot([0.0005, 0.5], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(spec3[1:int(N2/2)],fin,nsig=2), 1.0), verticalalignment='center')
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

