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
import sys


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
nlev = 2**6        
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
N = 2**10
decim = 256
over = 1
fb = int(np.ceil(N/(2.*(over*OSR))))
fin = np.floor(1./7. * fb)
## 5dbV is equivalent to 1.8V-lin
#amp = np.array([ds.undbv(10.0),
#                ds.undbv(9.0),
#                ds.undbv(8.0),
#                ds.undbv(7.0),
#                ds.undbv(6.0),
#                ds.undbv(5.0),
#                ds.undbv(4.0),
#                ds.undbv(3.0),
#                ds.undbv(2.0),
#                ds.undbv(1.0),
#                ds.undbv(0.0),
#                ds.undbv(-1.0),
#                ds.undbv(-2.0),
#                ds.undbv(-20.0),
#                ds.undbv(-50.0)])

amp = np.array([ds.undbv(6.0),
                ds.undbv(5.0),
                ds.undbv(4.0),
                ds.undbv(3.0),
                ds.undbv(2.0),
                ds.undbv(1.0),
                ds.undbv(0.0),
                ds.undbv(-20.0),
                ds.undbv(-50.0)])

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
noise_enb = 0.0 # for the sampling input and DAC feedback
noise_amp = 0.0 # for the amplifiers noise ## Noise input vector (w/ input referred noise of input SW and DAC) 
saturation = False
offset_on = 0.0
offset_calibrated = True

comp_noise_factor = 0.0625*0.5

np.random.seed(1)
##########################

Cin = 50e-15
sigma2_sw_in = 4 * 4e-21 / Cin


amp_psd = 1e-5 ## V/sqrt(Hz)
sigma2_amp = np.array([(amp_psd)**2 , (amp_psd)**2, (amp_psd)**2])

input_noise = noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(N))*sin2se_dBFS

snr_amp = np.ones(np.size(amp))
maxvalues = np.zeros((np.size(amp),5))   



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
    yz1p = 0
    
    y02 = 0
    yz2 = 0
    vz2 = 0
    yz2p = 0
    
    vdac1 = 0
    vdac1p = 0
    vdac2 = 0
    vdac2p = 0
    
    v1con = np.zeros((nq, int(N/2)), dtype=np.float64)
    v2con = np.zeros((nq, int(N/2)), dtype=np.float64)
    opt_g1 = 1.0
    opt_g2 = 1.0
    
    vb51 = 0.0
    vb41 = 0.0
    vb31 = 0.0
    vb21 = 0.0
    vb21 = 0.0
    vb11 = 0.0
    vb01 = 0.0
    vb11r = 0.0
    vb01r1 = 0.0
    vb01r2 = 0.0
    vbm11 = 0.0
    vbm21 = 0.0
    vbm31 = 0.0
    vbm41 = 0.0
    vbm51 = 0.0
    vbm61 = 0.0
    eq1 = 0.0
    
    vb52 = 0.0
    vb42 = 0.0
    vb32 = 0.0
    vb22 = 0.0
    vb22 = 0.0
    vb12 = 0.0
    vb02 = 0.0
    vb12r = 0.0
    vb02r1 = 0.0
    vb02r2 = 0.0
    vbm12 = 0.0
    vbm22 = 0.0
    vbm32 = 0.0
    vbm42 = 0.0
    vbm52 = 0.0
    vbm62 = 0.0
    eq2 = 0.0
    
    att_shr = 1
    
    for i in range(N):
        
        if i%2 == 0:
            y01int =  u[:,i]  ## mid conversion coupling signal
            y1[:,i] = y01int
            yz1 = y01int
            
#            y01int = y01int - opt_g1*(vbm11+vbm21+vbm31+vbm41) +opt_g2*2*(vbm12+vbm22+vbm32+vbm42)
            y01int = y01int - opt_g1*(vbm11+vbm21+vbm31+vbm41+vbm51) 
#            y01int = y01int - eq1 + 2*eq2
#            y01int = y01int - eq1 
            
            vb51 = ds_quantize(y01int,2)*32.0
            y01int = y01int-vb51
            vb41 = ds_quantize(y01int,2)*16.0
            y01int = y01int-vb41
            vb31 = ds_quantize(y01int,2)*8.0
            y01int = y01int-vb31
            vb21 = ds_quantize(y01int,2)*4.0
            y01int = y01int-vb21
            vb11 = ds_quantize(y01int,2)*2.0
            y01int = y01int-vb11
            y01int = y01int +opt_g2*2*(vbm12+vbm22+vbm32+vbm42+vbm52)
            vb11r = ds_quantize(y01int,2)*2.0
            y01int = y01int-vb11r
            vb01 = ds_quantize(y01int,2)*1.0
                               
            # eq1 digital approximation
            eq1 = y01int-vb01
            y01int = y01int-vb01
            vbm11 = ds_quantize(y01int,2)*0.5
            y01int = y01int-vbm11
            vbm21 = ds_quantize(y01int,2)*0.25
            y01int = y01int-vbm21
            vbm31 = ds_quantize(y01int,2)*0.125
            y01int = y01int-vbm31
            vbm41 = ds_quantize(y01int,2)*0.0625
            y01int = y01int - vbm41
            vbm51 = ds_quantize(y01int,2)*0.03125
                                
            vdac1 = (vb51+vb41+vb31+vb21+vb11+vb01+vb11r)
            
#            if vdac1 != vdac1x:
#                print(vb51,vb41,vb31,vb21,vb11,vdac1x,vdac1)
#                sys.exit('fault calculation')

            v1[:,i] = vdac1
            v[:,i] = v1[:,i]
            v1con[:,int(i/2)] = v1[:,i]
            
            ### Full DAC feedback at input of INT1

            
        if i%2 == 1:
            y02int = u[:,i]  ## mid conversion coupling signal
            y2[:,i] = y02int
            yz2 = y02int
            
#            y02int = y02int - opt_g1*(vbm12+vbm22+vbm32+vbm42)+opt_g2*2*(vbm11+vbm21+vbm31+vbm41)
            y02int = y02int - opt_g1*(vbm12+vbm22+vbm32+vbm42+vbm52)
#            y02int = y02int - eq2 + 2*eq1
#            y02int = y02int - eq2 
            
            vb52 = ds_quantize(y02int,2)*32.0
            y02int = y02int-vb52
            vb42 = ds_quantize(y02int,2)*16.0
            y02int = y02int-vb42
            vb32 = ds_quantize(y02int,2)*8.0
            y02int = y02int-vb32
            vb22 = ds_quantize(y02int,2)*4.0
            y02int = y02int-vb22
            vb12 = ds_quantize(y02int,2)*2.0
            y02int = y02int-vb12
            y02int = y02int +opt_g2*2*(vbm11+vbm21+vbm31+vbm41+vbm51)
            vb12r = ds_quantize(y02int,2)*2.0
            y02int = y02int-vb12r
            vb02 = ds_quantize(y02int,2)*1.0
                               
            # eq1 digital approximation
            eq2 = y02int-vb02
            y02int = y02int-vb02
            vbm12 = ds_quantize(y02int,2)*0.5
            y02int = y02int-vbm12
            vbm22 = ds_quantize(y02int,2)*0.25
            y02int = y02int-vbm22
            vbm32 = ds_quantize(y02int,2)*0.125
            y02int = y02int-vbm32
            vbm42 = ds_quantize(y02int,2)*0.0625
            y02int = y02int - vbm42
            vbm52 = ds_quantize(y02int,2)*0.03125
            
            vdac2 = (vb52+vb42+vb32+vb22+vb12+vb02+vb12r)
            

            
            
            v2[:,i] = vdac2
            v[:,i] = v2[:,i]
            v2con[:,int((i-1)/2)] = v2[:,i]
            
            ### Full DAC feedback at input of INT1

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
    if amp_index == 5:
        pl.figure()
        pl.semilogx(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-100, 40], 10, None, (16, 6), 'Output Spectrum TI log scale')
#        pl.plot([0.0005, 0.5/(over*OSR)], [-70, -70], 'k', linewidth=10, label='Signal Band')
#        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr_amp[amp_index], OSR), verticalalignment='center')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
#        a = v1[2::2]
#        f_ch1 = np.linspace(0, 0.5, int(N/2./2 + 1))
#        spec_ch1 = np.fft.fft(a* analog_scale * ds.ds_hann(N/2 -1))/(N/4/2)
#        pl.figure()
#        pl.semilogx(f_ch1, ds.dbv(spec_ch1[:int(N/2./2) +1]), 'b', label='Simulation spectrum')
#        ds.figureMagic([0.0005, 0.5], None, None, [-70, 30], 10, None, (16, 6), 'Output Spectrum TI')
#        pl.plot([0.0005, 0.5/(over*OSR)], [-70, -70], 'k', linewidth=10, label='Signal Band')
##        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(spec_ch1[2:fb+1],fin-2), OSR), verticalalignment='center')
#        pl.xlabel('Normalized Frequency')
#        pl.ylabel('dBFS')
#        pl.plot()
        
        pl.figure()
        pl.plot(f*2, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 1.0], None, None, [-100, 40], 10, None, (16, 6), 'Output Spectrum 2xTI linear scale')
#        pl.plot([0.0005, 0.5/(over*OSR)], [-70, -70], 'k', linewidth=10, label='Signal Band')
#        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr_amp[amp_index], OSR), verticalalignment='center')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
#        pl.figure()
#        pl.plot(f_ch1, ds.dbv(spec_ch1[:int(N/2./2) +1]), 'b', label='Simulation spectrum')
#        ds.figureMagic([0.0005, 0.5], None, None, [-70, 40], 10, None, (8, 6), 'Output Spectrum Single Channel')
##        pl.plot([0.0005, 0.5/(over*OSR)], [-70, -70], 'k', linewidth=10, label='Signal Band')
##        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr_amp[amp_index], OSR), verticalalignment='center')
#        pl.xlabel('Normalized Frequency')
#        pl.ylabel('dBFS')
#        pl.plot()
        
#        pl.figure(figsize=(16,6))
#        pl.plot(v[100:200])
#        pl.plot(v1[np.arange(100,300,2)])
#        pl.plot(v2[np.arange(101,300,2)])
        

#        vdec = sp.signal.decimate(v,OSR, ftype='fir')
#        N2 = np.size(vdec)
#        f3 = np.linspace(0, 0.5, int(N2/2. + 1))
#        spec3 = np.fft.fft(vdec* analog_scale * ds.ds_hann(N2))/(N2/4)
#        pl.figure()
#        pl.plot(f3, ds.dbv(spec3[:int(N2/2.) +1]), 'b', label='Simulation spectrum')
#        ds.figureMagic([0.0005, 0.5], None, None, [-100, 30], 10, None, (16, 6), 'Output Spectrum: Interpolation -> Decimation/10 FIR')
#        pl.plot([0.0005, 0.5], [-100, -100], 'k', linewidth=10, label='Signal Band')
#        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(spec3[1:int(N2/2)],fin,nsig=2), 1.0), verticalalignment='center')
#        pl.xlabel('Normalized Decimated Frequency')
#        pl.ylabel('dBFS')
#        pl.plot()       

#print('\n'*3)
#print('Maximum absolute values of state variables =  \n')
#print(pd.DataFrame(maxvalues, columns=['x1','x2','x3','y','u'], \
#                   index=['5','4.5','4','3.5','3','2.5','2','1.5','1','0.5','0','-1','-2','-20','-50','-80']))
#print('\n'*3)

pl.figure(figsize=(10,4))
pl.plot(ds.dbv(amp),snr_amp, label='SQNR')
ds.figureMagic([-50, 10], 5, None, [-10,80], 5, None, (10, 6), 'Dynamic Range')
pl.text(-40, 75,'Peak SNR = %2.1f dB @ %2.1f dbV' %(snr_amp.max(), ds.dbv(amp[np.argmax(snr_amp)])))
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

