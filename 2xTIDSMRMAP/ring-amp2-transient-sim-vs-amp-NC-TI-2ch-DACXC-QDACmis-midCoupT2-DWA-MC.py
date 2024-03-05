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
nlev = 2**6        ## 5-bit quantizer
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
N = 2**11+8
decim = 256
over = 1
fb = int(np.ceil(N/(2.*(over*OSR))))
fin = np.floor(1./7. * fb)
## 5dbV is equivalent to 1.8V-lin
amp = np.array([ds.undbv(4.0),
                ds.undbv(3.0),
                ds.undbv(2.0),
                ds.undbv(1.0)])

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
noise_enb = 0.0 # for the sampling input and DAC feedback
noise_amp = 0.0 # for the amplifiers noise ## Noise input vector (w/ input referred noise of input SW and DAC) 
saturation = False
sat_sum = False
sat_level_stg = 0.9*sin2se_dBV
np.random.seed(1)
qdac_mismatch_enable = 1.0

qdac_unit_sigma = 1/(pl.sqrt(1.36)*100) #Matching for Cu = 1.3fF


##########################
mc_points = 1000
sndr_mc_dacmismatch = np.zeros((mc_points,1))
msa_mc_dacmismatch = np.zeros((mc_points,1))
for mc_index in range(mc_points):
    
    qdac1_unit = 1+qdac_mismatch_enable*qdac_unit_sigma*np.random.randn(65)
    qdac2_unit = 1+qdac_mismatch_enable*qdac_unit_sigma*np.random.randn(65)
    
    qdac_factor1 = qdac_mismatch_enable*(qdac_unit_sigma)*(np.random.randn(7))/np.array([pl.sqrt(32),pl.sqrt(16),pl.sqrt(8),pl.sqrt(4),pl.sqrt(2),pl.sqrt(2),pl.sqrt(1)])
    qdacsub_factor1 = qdac_mismatch_enable*(qdac_unit_sigma)*(np.random.randn(4))/np.array([pl.sqrt(4),pl.sqrt(2),pl.sqrt(1),pl.sqrt(0.5)])
    qdac_factor2 = qdac_mismatch_enable*(qdac_unit_sigma)*(np.random.randn(7))/np.array([pl.sqrt(32),pl.sqrt(16),pl.sqrt(8),pl.sqrt(4),pl.sqrt(2),pl.sqrt(2),pl.sqrt(1)])
    qdacsub_factor2 = qdac_mismatch_enable*(qdac_unit_sigma)*(np.random.randn(4))/np.array([pl.sqrt(4),pl.sqrt(2),pl.sqrt(1),pl.sqrt(0.5)])

    
    qdac_11 = qdac_mismatch_enable*(qdac_unit_sigma)*(np.random.randn(4))/np.array([pl.sqrt(8),pl.sqrt(4),pl.sqrt(2),pl.sqrt(1)])
    qdac_22 = qdac_mismatch_enable*(qdac_unit_sigma)*(np.random.randn(4))/np.array([pl.sqrt(8),pl.sqrt(4),pl.sqrt(2),pl.sqrt(1)])
    
    qdac_12 = qdac_mismatch_enable*(qdac_unit_sigma)*(np.random.randn(4))/np.array([pl.sqrt(16),pl.sqrt(8),pl.sqrt(4),pl.sqrt(2)])
    qdac_21 = qdac_mismatch_enable*(qdac_unit_sigma)*(np.random.randn(4))/np.array([pl.sqrt(16),pl.sqrt(8),pl.sqrt(4),pl.sqrt(2)])
    
    Cin = 125e-15*4
    sigma2_sw_in = 4 * 4e-21 / Cin
    
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
        opt_g2 = 0.96
        att_shr = 0.98
        
        vb51 = np.array([0.0])
        vb41 = np.array([0.0])
        vb31 = np.array([0.0])
        vb21 = np.array([0.0])
        vb21 = np.array([0.0])
        vb11 = np.array([0.0])
        vb11r = np.array([0.0])
        vb01 = np.array([0.0])
        vbm11 = np.array([0.0])
        vbm21 = np.array([0.0])
        vbm31 = np.array([0.0])
        vbm41 = np.array([0.0])
        
        vb52 = np.array([0.0])
        vb42 = np.array([0.0])
        vb32 = np.array([0.0])
        vb22 = np.array([0.0])
        vb22 = np.array([0.0])
        vb12 = np.array([0.0])
        vb12r = np.array([0.0])
        vb02 = np.array([0.0])
        vbm12 = np.array([0.0])
        vbm22 = np.array([0.0])
        vbm32 = np.array([0.0])
        vbm42 = np.array([0.0])
        
        flip1 = 0
        flip2 = 0
        
        for i in range(N):
            
            if i%2 == 0:
                y01int = np.real(np.dot(C, x01) + np.dot(D1, u[:,i])) - opt_g2*(round(vbm11[0]/0.5)*(1+qdac_11[0])*0.5 + round(vbm21[0]/0.25)*(1+qdac_11[1])*0.25 + round(vbm31[0]/0.125)*(1+qdac_11[2])*0.125 + round(vbm41[0]/0.0625)*(1+qdac_11[3])*0.0625)
                y1[:,i] = y01int
                yz1 = y01int
                
#                if vb51>0 and flip1 == 0:
#                    flip1 = 1
#                elif vb51>0 and flip1 == 1:
#                    flip1 = 0
                
                if flip1 == 0:
                    flip1 = 1
                elif flip1 == 1:
                    flip1 = 0
                
                if flip1 == 1:
                    qdac_factor1[0] = sum(qdac1_unit[range(0,32)])
                    qdac_factor1[1] = sum(qdac1_unit[range(32,48)])
                    qdac_factor1[2] = sum(qdac1_unit[range(48,56)])
                    qdac_factor1[3] = sum(qdac1_unit[range(56,60)])
                    qdac_factor1[4] = sum(qdac1_unit[range(60,62)])
                    qdac_factor1[5] = sum(qdac1_unit[range(62,64)])
                    qdac_factor1[6] = (qdac1_unit[64])
                if flip1 == 0:
                    qdac_factor1[0] = sum(qdac1_unit[range(33,65)])
                    qdac_factor1[1] = sum(qdac1_unit[range(17,33)])
                    qdac_factor1[2] = sum(qdac1_unit[range(9,17)])
                    qdac_factor1[3] = sum(qdac1_unit[range(5,9)])
                    qdac_factor1[4] = sum(qdac1_unit[range(3,5)])
                    qdac_factor1[5] = sum(qdac1_unit[range(1,3)])
                    qdac_factor1[6] = (qdac1_unit[0])
                    
                
                vb51 = ds_quantize(y01int,2)*qdac_factor1[0]
                y01int = y01int-vb51
                vb41 = ds_quantize(y01int,2)*qdac_factor1[1]
                y01int = y01int-vb41
                vb31 = ds_quantize(y01int,2)*qdac_factor1[2]
                y01int = y01int-vb31
                vb21 = ds_quantize(y01int,2)*qdac_factor1[3]
                y01int = y01int-vb21
                vb11 = ds_quantize(y01int,2)*qdac_factor1[4]
                y01int = y01int-vb11
                y01int = y01int + opt_g2*(round(vbm12[0]/0.5)*(1+qdac_21[0]) + round(vbm22[0]/0.25)*(1+qdac_21[1])*0.5 + round(vbm32[0]/0.125)*(1+qdac_21[2])*0.25 + round(vbm42[0]/0.0625)*(1+qdac_21[3])*0.125)
                vb11r = ds_quantize(y01int,2)*qdac_factor1[5]
                y01int = y01int-vb11r
                vb01 = ds_quantize(y01int,2)*qdac_factor1[6]
                                   
                # eq1 digital approximation
                y01int = y01int-vb01
                y01int = y01int*att_shr
                vbm11 = ds_quantize(y01int,2)*0.5*(1+qdacsub_factor1[0])
                y01int = y01int-vbm11
                vbm21 = ds_quantize(y01int,2)*0.25*(1+qdacsub_factor1[1])
                y01int = y01int-vbm21
                vbm31 = ds_quantize(y01int,2)*0.125*(1+qdacsub_factor1[2])
                y01int = y01int-vbm31
                vbm41 = ds_quantize(y01int,2)*0.0625
                
    
                
#                vdac1 = vb51+vb41+vb31+vb21+vb11+vb01                    
                vdac1 = round(vb51[0]/32)*32+round(vb41[0]/16)*16+round(vb31[0]/8)*8+round(vb21[0]/4)*4+round(vb11[0]/2)*2+round(vb11r[0]/2)*2+round(vb01[0]/1)*1
                vdac1p = round(vb51[0]+vb41[0]+vb31[0]+vb21[0]+vb11[0]+vb01[0])
                vdac1x = ds.ds_quantize(y01int,nlev)
                
                
    #            if vdac1 != vdac1x:
    #                print(vb51,vb41,vb31,vb21,vb11,vb01,vdac1,vdac1x)
    #                sys.exit('fault calculation')
    
                v1[:,i] = vdac1
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
                y02int = np.real(np.dot(C, x02) + np.dot(D1, u[:,i])) - opt_g2*(round(vbm12[0]/0.5)*(1+qdac_22[0])*0.5 + round(vbm22[0]/0.25)*(1+qdac_22[1])*0.25 + round(vbm32[0]/0.125)*(1+qdac_22[2])*0.125 + round(vbm42[0]/0.0625)*(1+qdac_22[3])*0.0625)
    
                y2[:,i] = y02int
                yz2 = y02int
                
#                if vb52>0 and flip2 == 0:
#                    flip2 = 1
#                elif vb52>0 and flip2 == 1:
#                    flip2 = 0
                    
                if  flip2 == 0:
                    flip2 = 1
                elif flip2 == 1:
                    flip2 = 0
                
                if flip1 == 1:
                    qdac_factor2[0] = sum(qdac2_unit[range(0,32)])
                    qdac_factor2[1] = sum(qdac2_unit[range(32,48)])
                    qdac_factor2[2] = sum(qdac2_unit[range(48,56)])
                    qdac_factor2[3] = sum(qdac2_unit[range(56,60)])
                    qdac_factor2[4] = sum(qdac2_unit[range(60,62)])
                    qdac_factor2[5] = sum(qdac2_unit[range(62,64)])
                    qdac_factor2[6] = (qdac2_unit[64])
                if flip1 == 0:
                    qdac_factor2[0] = sum(qdac2_unit[range(33,65)])
                    qdac_factor2[1] = sum(qdac2_unit[range(17,33)])
                    qdac_factor2[2] = sum(qdac2_unit[range(9,17)])
                    qdac_factor2[3] = sum(qdac2_unit[range(5,9)])
                    qdac_factor2[4] = sum(qdac2_unit[range(3,5)])
                    qdac_factor2[5] = sum(qdac2_unit[range(1,3)])
                    qdac_factor2[6] = (qdac2_unit[0])
                
                vb52 = ds_quantize(y02int,2)*qdac_factor2[0]
                y02int = y02int-vb52
                vb42 = ds_quantize(y02int,2)*qdac_factor2[1]
                y02int = y02int-vb42
                vb32 = ds_quantize(y02int,2)*qdac_factor2[2]
                y02int = y02int-vb32
                vb22 = ds_quantize(y02int,2)*qdac_factor2[3]
                y02int = y02int-vb22
                vb12 = ds_quantize(y02int,2)*qdac_factor2[4]
                y02int = y02int-vb12
                y02int = y02int + opt_g2*(round(vbm11[0]/0.5)*(1+qdac_12[0]) + round(vbm21[0]/0.25)*(1+qdac_12[1])*0.5 + round(vbm31[0]/0.125)*(1+qdac_12[2])*0.25 + round(vbm41[0]/0.0625)*(1+qdac_12[3])*0.125)
                vb12r = ds_quantize(y02int,2)*qdac_factor1[5]
                y02int = y02int-vb12r
                vb02 = ds_quantize(y02int,2)*qdac_factor1[6]
                                   
                # eq1 digital approximation
                y02int = y02int-vb02
                y02int = y02int*att_shr
                vbm12 = ds_quantize(y02int,2)*0.5*(1+qdacsub_factor2[0])
                y02int = y02int-vbm12
                vbm22 = ds_quantize(y02int,2)*0.25*(1+qdacsub_factor2[1])
                y02int = y02int-vbm22
                vbm32 = ds_quantize(y02int,2)*0.125*(1+qdacsub_factor2[2])
                y02int = y02int-vbm32
                vbm42 = ds_quantize(y02int,2)*0.0625
                
    
                
#                vdac1 = vb51+vb41+vb31+vb21+vb11+vb01                    
                vdac2 = round(vb52[0]/32)*32+round(vb42[0]/16)*16+round(vb32[0]/8)*8+round(vb22[0]/4)*4+round(vb12[0]/2)*2+round(vb12r[0]/2)*2+round(vb02[0]/1)*1
                
                
                v2[:,i] = vdac2
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
#        print(amp_index)
        
    
#    pl.figure(figsize=(10,4))
#    pl.plot(ds.dbv(amp),snr_amp, label='No dac mismatch')
#    ds.figureMagic([-80, 5], 5, None, [-10,100], 5, None, (10, 6), 'OUT Dynamic Range')
#    pl.text(-20, 75,'Peak SNR = %2.1f dB @ %2.1f dbV' %(snr_amp.max(), ds.dbv(amp[np.argmax(snr_amp)])))
#    pl.xlabel('Amplitude (dBVp-p differential)')
#    pl.ylabel('SNR (dB)')
#    pl.legend(loc=4)
#    pl.plot()
    
    print(mc_index)
    
    sndr_mc_dacmismatch[mc_index] = snr_amp.max()
    msa_mc_dacmismatch[mc_index] = ds.dbv(amp[np.argmax(snr_amp)])

pl.figure(figsize=(10,4))    
pl.hist(sndr_mc_dacmismatch)
pl.figure(figsize=(10,4))
pl.scatter(sndr_mc_dacmismatch,msa_mc_dacmismatch)

sndr_sort = np.sort(sndr_mc_dacmismatch,axis=0)
pl.figure(figsize=(10,4))
pl.plot(sndr_sort)
pl.plot(np.array([0, mc_points]), np.array([pl.mean(sndr_mc_dacmismatch),pl.mean(sndr_mc_dacmismatch)]))
pl.text(mc_points*0.01, pl.mean(sndr_mc_dacmismatch)+1,'Average SQNR for %d points = %2.1f dB' %(mc_points,pl.mean(sndr_mc_dacmismatch)))
pl.scatter(mc_points*0.01,sndr_sort[int(mc_points*0.01)])
pl.text(mc_points*0.05, sndr_sort[int(mc_points*0.01)],'Peak SQNR at 99%% confidence = %2.1f dB' %(sndr_sort[int(mc_points*0.01)]))
ds.figureMagic([0, mc_points], mc_points/10, None, [75,95], 2, None, (10, 6), 'Monte Carlo')
pl.xlabel('Sorted Occurence (#)')
pl.ylabel('Peak SQNR (dB)')
pl.legend(loc=4)
pl.show()


##################################################################################
#ntf,_ = ds.calculateTF(ABCDs)
#
#ds.DocumentNTF(ntf,OSR)
#f1, f2 = ds.ds_f1f2(OSR, 0, 0)
#NG0 = ds.dbv(ds.rmsGain(ntf, f1, f2))
#NG0_lin = ds.undbv(NG0)
#NG0_lin_squared = NG0_lin**2
#
#NTF_gain = ds.rmsGain(ntf, 0, 0.5)**2
#
#
## Calculation for an extra self NC zero
#z,p,k = ntf[0], ntf[1], ntf[2]
#z2 = np.append(z, 1)
#ntf2 = tuple([z2, p, k])
#ds.DocumentNTF(ntf2,1*OSR)
#
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
#
#### Calculation for a polyphase extrapolation by z^-1 self NC and z^-1 crossed NC
##z,p,k = ntf2[0], ntf2[1], ntf2[2]
##num,den = sp.signal.zpk2tf(z,p,k)
##new_num = np.array([num[0] , 0.0, num[1], 0.0, num[2], 0.0, num[3], 0.0, num[4]])
##new_den = np.array([den[0] , 0.0, den[1], 0.0, den[2], 0.0, den[3]])
##z3,p3,k3 = sp.signal.tf2zpk(new_num, new_den)
##z3 = np.append(z3, 0.99)
##ntf3 = tuple([z3, p3, k3])
#
## common part of the code
#

#freq = 2e9
#
#f = ds.ds_freq(OSR, 0, 0)
#z = np.exp(2j * np.pi * f)
#ntf = ntf3
#H = ds.dbv(ds.evalTF(ntf, z))
#ds.DocumentNTF(ntf,1*OSR)
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

    