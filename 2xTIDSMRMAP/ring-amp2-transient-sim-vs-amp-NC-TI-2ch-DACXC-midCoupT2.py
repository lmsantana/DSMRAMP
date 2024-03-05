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
import scipy.io as spio
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
N = 2**14
decim = 256
over = 1
fb = int(np.ceil(N/(2.*(over*OSR))))
fin = np.floor(1./2. * fb)
# 5dbV is equivalent to 1.8V-lin
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
# amp = np.array([ds.undbv(5.0),
#                 ds.undbv(4.5),
#                 ds.undbv(4.0),
#                 ds.undbv(3.5),
#                 ds.undbv(3.0),
#                 ds.undbv(2.5),
#                 ds.undbv(2.0),
#                 ds.undbv(1.5),
#                 ds.undbv(1.0),
#                 ds.undbv(0.5),
#                 ds.undbv(0.0)])

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = False
DCgain = 40
noise_enb = 1.0 # for the sampling input and DAC feedback
noise_amp = 1.0 # for the amplifiers noise ## Noise input vector (w/ input referred noise of input SW and DAC) 

noise_comp = 0.0 
sigma_ncomp = 100/100 * 1 *1/6.6 ## percentage with respect to LSB in schematic
saturation = True
sat_level_stg = 1.0*sin2se_dBV
inter_gain = 1

nl_fact_p = 12.2/100.0/(sin2se_dBV**3)

np.random.seed(1)
##########################

Cin = (22e-15)*(65)
sigma2_sw_in = 1* 5 * 1.3806e-23 * 298 / Cin

amp_psd = 1e-5 ## V/sqrt(Hz)
sigma2_amp = np.array([(amp_psd)**2 , (amp_psd)**2, (amp_psd)**2])

snr_amp = np.ones(np.size(amp))
maxvalues = np.zeros((np.size(amp),5))   

form = 'CIFF'   ## FF better for highly linear OpA 
# ABCDs = np.array([[ 1.        ,  0.       ,  0.       ,  4.63     , -4.63 ],
#                   [ 1.826     ,  1.       , -0.089    ,  0.       ,  0.   ],
#                   [ 0.        ,  0.813    ,  1.       ,  0.       ,  0.   ],
#                   [ 0.676     ,  0.418    ,  0.182    ,  1.       ,  0.   ]])
    
ABCDs = np.array([[ 1.        ,  0.       ,  0.       ,  4.63     , -4.63],
                  [ 0.913     ,  1.       , -0.089    ,  0.       ,  0.   ],
                  [ 0.        ,  0.813    ,  1.       ,  0.       ,  0.   ],
                  [ 0.676     ,  0.835    ,  0.365    ,  1.       ,  0.   ]])
    

a,g,b,c = ds.mapABCD(ABCDs, form)
ABCDs2 = np.copy(ABCDs)

## Simplest model possible for finite DC gain with integrator feedback attenuation
if finite_dc_gain == True:    
    av = ds.undbv(DCgain) # 60db
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
    ABCDs2 = abcd

ntf1,stf1 = ds.calculateTF(ABCDs)
# ds.DocumentNTF(ntf1,OSR)
data_ch1 = np.zeros((np.size(amp), 7, int(N/2)))
data_ch2 = np.zeros((np.size(amp), 7, int(N/2)))

for amp_index in range(np.size(amp)):
    
    ## All architectures subjected to the same input with thermal noise
    input_noise = noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(N))*sin2se_dBFS 
    # u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBFS + input_noise
    
    delta_ch1_p = np.pi/180 * (10) *0
    delta_ch1_n = np.pi/180 * (-7) *0
    delta_ch2_p = np.pi/180 * (-10) *0
    delta_ch2_n = np.pi/180 * (9) *0
    delta_ti = np.pi/180 * 0.01 *0
    u_ch1_p = amp[amp_index] * (np.sin(2 * np.pi * fin * (np.arange(N)/N) + delta_ch1_p)) * sin2di_dBFS / 2
    u_ch1_n = amp[amp_index] * (np.sin(2 * np.pi * fin * (np.arange(N)/N) + delta_ch1_n + np.pi)) * sin2di_dBFS / 2
    
    u_ch2_p = amp[amp_index] * (np.sin(2 * np.pi * fin * (np.arange(N)/N) + delta_ch2_p + delta_ti)) * sin2di_dBFS / 2
    u_ch2_n = amp[amp_index] * (np.sin(2 * np.pi * fin * (np.arange(N)/N) + delta_ch2_n + delta_ti + np.pi)) * sin2di_dBFS / 2
    
    u_ch1 = u_ch1_p - u_ch1_n
    u_ch2 = u_ch2_p - u_ch2_n
    
    u = np.zeros(N)
    u[0::2] = u_ch1[0::2]
    u[1::2] = u_ch2[1::2]
    u += input_noise


    """
    Hard code implementation of ds.simulateDSM_python function, it is not optmized for time as
    the Cyhton version, but it allows access to internal nodes
    """
    
    ## simulateDSM hard code for noise input and further changes
    nq = 1 # 1 input
    nu = 1 # 1 output
    A1 = ABCDs[:order, :order]
    B1 = ABCDs[:order, order:order+nu+nq]
    C1 = ABCDs[order:order+nq, :order]
    D11 = ABCDs[order:order+nq, order:order+nu] ## assuming no direct feedback from V to Y
    
    A2 = ABCDs2[:order, :order]
    B2 = ABCDs2[:order, order:order+nu+nq]
    C2 = ABCDs2[order:order+nq, :order]
    D12 = ABCDs2[order:order+nq, order:order+nu] ## assuming no direct feedback from V to Y
    
    u = u.reshape((1,-1))
    v = np.zeros((nq, N), dtype=np.float64)
    v_noLSB = np.zeros((nq, N), dtype=np.float64)
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
    vdac2 = 0
    
    v1con = np.zeros((nq, int(N/2)), dtype=np.float64)
    v2con = np.zeros((nq, int(N/2)), dtype=np.float64)
    opt_g2 = 1.0
    att_shr = 1.0
    y_att = 1.0
    sum_att = 1.0
    dac_att = (1 - np.array([1/100/32, 2/100/16, 3/100/8, 4/100/4, 5/100/2, 6/100/2, 7/100])*0)  
    
    vb51 = 0.0
    vb41 = 0.0
    vb31 = 0.0
    vb21 = 0.0
    vb21 = 0.0
    vb11 = 0.0
    vb01 = 0.0
    vb11r = 0.0
    vbm11 = 0.0
    vbm21 = 0.0
    vbm31 = 0.0
    vbm41 = 0.0
    
    vb52 = 0.0
    vb42 = 0.0
    vb32 = 0.0
    vb22 = 0.0
    vb22 = 0.0
    vb12 = 0.0
    vb02 = 0.0
    vb12r = 0.0
    vbm12 = 0.0
    vbm22 = 0.0
    vbm32 = 0.0
    vbm42 = 0.0
    lsb_comp_offset = 0.0
    
    for i in range(N):
        
        if i%2 == 0:
            y01int = np.real(np.dot(C1, x01+ nl_fact_p*(x01**3))*sum_att + np.dot(D11, u[:,i]+ nl_fact_p*u[:,i]**3)) - opt_g2*(vbm11+vbm21+vbm31+vbm41)  ## mid conversion coupling signal
            y01int = y01int*y_att
            y1[:,i] = y01int
            yz1 = y01int
            
            vb51 = ds_quantize(y01int + noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*32.0
            y01int = y01int-vb51*dac_att[0]
            vb41 = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*16.0
            y01int = y01int-vb41*dac_att[1]
            vb31 = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*8.0
            y01int = y01int-vb31*dac_att[2]
            vb21 = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*4.0
            y01int = y01int-vb21*dac_att[3]
            
            vb11 = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*2.0
            y01int = y01int-vb11*dac_att[4]
            y01int = y01int +opt_g2*2*(vbm12+vbm22+vbm32+vbm42) 
            vb11r = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*2.0
            y01int = y01int-vb11r*dac_att[5]
            vb01 = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0] + lsb_comp_offset,2)*1.0
                               
            # eq1 digital approximation
            y01int = y01int-vb01*dac_att[6]
            y01int = y01int*att_shr
            vbm11 = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0]/inter_gain,2)*0.5
            y01int = y01int-vbm11
            vbm21 = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0]/inter_gain,2)*0.25
            y01int = y01int-vbm21
            vbm31 = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0]/inter_gain,2)*0.125
            y01int = y01int-vbm31
            vbm41 = ds_quantize(y01int+ noise_comp*sigma_ncomp*np.random.randn(1)[0]/inter_gain,2)*0.0625
                                
            vdac1 = (vb51+vb41+vb31+vb21+vb11+vb01+vb11r)
            vdac1_noLSB = (vb51+vb41+vb31+vb21+vb11+vb11r-1)
            vdacsub1 = vbm11+vbm21+vbm31+vbm41
            
            data_ch1[amp_index,0, int(i/2)] = (vb51+32)/64
            data_ch1[amp_index,1, int(i/2)] = (vb41+16)/32
            data_ch1[amp_index,2, int(i/2)] = (vb31+8)/16
            data_ch1[amp_index,3, int(i/2)] = (vb21+4)/8
            data_ch1[amp_index,4, int(i/2)] = (vb11+2)/4
            data_ch1[amp_index,5, int(i/2)] = (vb11r+2)/4
            data_ch1[amp_index,6, int(i/2)] = (vb01+1)/2
            
            
#            vdac1 = ds.ds_quantize(y01,nlev)
            
#            if vdac1 != vdac1x:
#                print(vb51,vb41,vb31,vb21,vb11,vdac1x,vdac1)
#                sys.exit('fault calculation')

            v1[:,i] = vdac1
            v[:,i] = v1[:,i]
            v_noLSB[:,i] = vdac1_noLSB
            v1con[:,int(i/2)] = v1[:,i]
            # v1con[:,int(i/2)] = vdacsub1
            
            ### Full DAC feedback at input of INT1
            x01 = np.dot(A1, x01 + nl_fact_p*(x01**3)) + np.dot(B1, np.concatenate((u[:,i]+ noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(1)[0])*sin2se_dBFS , vdac1)))
    
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
            y02int = np.real(np.dot(C2, x02+ nl_fact_p*(x02**3))*sum_att + np.dot(D12, u[:,i]+ nl_fact_p*u[:,i]**3)) - opt_g2*(vbm12+vbm22+vbm32+vbm42)  ## mid conversion coupling signal
            y02int = y02int*y_att
            y2[:,i] = y02int
            yz2 = y02int
            
            
            vb52 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*32.0
            y02int = y02int-vb52*dac_att[0]
            vb42 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*16.0
            y02int = y02int-vb42*dac_att[1]
            vb32 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*8.0
            y02int = y02int-vb32*dac_att[2]
            vb22 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*4.0
            y02int = y02int-vb22*dac_att[3]
            vb12 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*2.0
            y02int = y02int-vb12*dac_att[4]
            y02int = y02int +opt_g2*2*(vbm11+vbm21+vbm31+vbm41) 
            vb12r = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0],2)*2.0
            y02int = y02int-vb12r*dac_att[5]
            vb02 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0] + lsb_comp_offset,2)*1.0
                               
            # eq1 digital approximation
            y02int = y02int-vb02*dac_att[6]
            y02int = y02int*att_shr
            vbm12 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0]/inter_gain,2)*0.5
            y02int = y02int-vbm12
            vbm22 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0]/inter_gain,2)*0.25
            y02int = y02int-vbm22
            vbm32 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0]/inter_gain,2)*0.125
            y02int = y02int-vbm32
            vbm42 = ds_quantize(y02int+ noise_comp*sigma_ncomp*np.random.randn(1)[0]/inter_gain,2)*0.0625

            
            vdac2 = (vb52+vb42+vb32+vb22+vb12+vb02+vb12r)
            vdac2_noLSB = (vb52+vb42+vb32+vb22+vb12+vb12r-1)
            vdacsub2 = vbm12+vbm22+vbm32+vbm42
            
            data_ch2[amp_index,0, int((i-1)/2)] = (vb52+32)/64
            data_ch2[amp_index,1, int((i-1)/2)] = (vb42+16)/32
            data_ch2[amp_index,2, int((i-1)/2)] = (vb32+8)/16
            data_ch2[amp_index,3, int((i-1)/2)] = (vb22+4)/8
            data_ch2[amp_index,4, int((i-1)/2)] = (vb12+2)/4
            data_ch2[amp_index,5, int((i-1)/2)] = (vb12r+2)/4
            data_ch2[amp_index,6, int((i-1)/2)] = (vb02+1)/2
            
#            if vdac2 != vdac2x:
#                print(vb52,vb42,vb32,vb22,vb12,vb02,vdac2x,vdac2)
#                sys.exit('fault calculation')
            
            
            v2[:,i] = vdac2
            v[:,i] = v2[:,i]
            v_noLSB[:,i] = vdac2_noLSB
            v2con[:,int((i-1)/2)] = v2[:,i]
            # v2con[:,int((i-1)/2)] = vdacsub2
            
            ### Full DAC feedback at input of INT1
            x02 = np.dot(A2, x02 + nl_fact_p*(x02**3)) + np.dot(B2, np.concatenate((u[:,i]+ noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(1)[0])*sin2se_dBFS , vdac2)))
    
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
    v_noLSB = v_noLSB.squeeze()
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
    spec = np.fft.fft(v* analog_scale * ds.ds_hann(N) / 2/nlev)/(N/4)
    snr_amp[amp_index] = ds.calculateSNR(spec[2:fb+1], fin-2)
    
    print(amp_index)
    if amp_index == 2:
        pl.figure()
        pl.semilogx(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 40], 10, None, (16, 6), 'Output Spectrum TI')
        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr_amp[amp_index], OSR), verticalalignment='center')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
        pl.figure()
        pl.plot(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        ds.figureMagic([0.0005, 0.5], None, None, [-150, 40], 10, None, (16, 6), 'Output Spectrum TI')
        pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr_amp[amp_index], OSR), verticalalignment='center')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.plot()
        
        # spec_noLSB = np.fft.fft(v_noLSB* analog_scale * ds.ds_hann(N))/(N/4)
        
        # pl.figure()
        # pl.semilogx(f, ds.dbv(spec_noLSB[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        # ds.figureMagic([0.0005, 0.5], None, None, [-150, 40], 10, None, (16, 6), 'Output Spectrum TI')
        # pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        # pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(spec_noLSB[2:fb+1], fin-2), OSR), verticalalignment='center')
        # pl.xlabel('Normalized Frequency')
        # pl.ylabel('dBFS')
        # pl.plot()
        
        # pl.figure()
        # pl.plot(f, ds.dbv(spec_noLSB[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        # ds.figureMagic([0.0005, 0.5], None, None, [-150, 40], 10, None, (16, 6), 'Output Spectrum TI')
        # pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        # pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(spec_noLSB[2:fb+1], fin-2), OSR), verticalalignment='center')
        # pl.xlabel('Normalized Frequency')
        # pl.ylabel('dBFS')
        # pl.plot()
        
        # delta_v1 = np.zeros(np.size(v1con))
        # delta_v2 = np.zeros(np.size(v2con))
        # for v_index in range(np.size(v1con)-1):
        #     delta_v1[v_index] = v1con[v_index+1] - v1con[v_index]
        #     delta_v2[v_index] = v2con[v_index+1] - v2con[v_index]
            
        # delta_v = np.zeros(np.size(v))
        # for v_index in range(np.size(v)-1):
        #     delta_v[v_index] = v[v_index+1] - v[v_index]
            
        # pl.figure()
        # pl.hist(delta_v1)
        # pl.figure()
        # pl.hist(delta_v2)
        # pl.figure()
        # pl.hist(delta_v)
            
        
        
        # spec1 = np.fft.fft(v1con* analog_scale * ds.ds_hann(int(N/2)))/(int(N/8))
        # spec2 = np.fft.fft(v2con* analog_scale * ds.ds_hann(int(N/2)))/(int(N/8))
        
        # pl.figure()
        # pl.plot(f[0:1028], ds.dbv(spec1[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        # ds.figureMagic([0.0005, 0.5], None, None, [-150, 40], 10, None, (16, 6), 'Sub SAR Channel 1')
        # pl.xlabel('Normalized Frequency')
        # pl.ylabel('dBFS')
        # pl.plot()
        
        # pl.figure()
        # pl.plot(f[0:1028], ds.dbv(spec2[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        # ds.figureMagic([0.0005, 0.5], None, None, [-150, 40], 10, None, (16, 6), 'Sub SAR Channel 2')
        # pl.xlabel('Normalized Frequency')
        # pl.ylabel('dBFS')
        # pl.plot()
        # pl.figure()
        # pl.hist(xn1[0,::2],range=(-64,64))
        # pl.plot()
        # pl.figure()
        # pl.hist(xn1[1,::2],range=(-64,64))
        # pl.plot()
        # pl.figure()
        # pl.hist(xn1[2,::2],range=(-64,64))
        # pl.plot()
        
        # fch1 = np.linspace(0, 0.5, int(N/2.))
        # specch1 = np.fft.fft(v1* analog_scale * ds.ds_hann(int(N)))/(N/4)
        # pl.figure()
        # pl.semilogx(f, ds.dbv(specch1[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        # ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum Channel 1 RN vector')
        # pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        # pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(specch1[1:fb+1],fin,nsig=2), OSR), verticalalignment='center')
        # pl.xlabel('Normalized Frequency')
        # pl.ylabel('dBFS')
        # pl.plot()
        
        # pl.figure()
        # pl.plot(f, ds.dbv(specch1[:int(N/2.) +1]), 'b', label='Simulation spectrum')
        # ds.figureMagic([0.0005, 0.5], None, None, [-150, 30], 10, None, (16, 6), 'Output Spectrum Channel 1 RN vector')
        # pl.plot([0.0005, 0.5/(over*OSR)], [-150, -150], 'k', linewidth=10, label='Signal Band')
        # pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(specch1[1:fb+1],fin,nsig=2), OSR), verticalalignment='center')
        # pl.xlabel('Normalized Frequency')
        # pl.ylabel('dBFS')
        # pl.plot()

#        vdec = sp.signal.decimate(v,OSR, ftype='iir')
#        N2 = np.size(vdec)
#        f3 = np.linspace(0, 0.5, int(N2/2. + 1))
#        spec3 = np.fft.fft(vdec* analog_scale * ds.ds_hann(N2))/(N2/4)
#        pl.figure()
#        pl.plot(f3, ds.dbv(spec3[:int(N2/2.) +1]), 'b', label='Simulation spectrum')
#        ds.figureMagic([0.0005, 0.5], None, None, [-150, 40], 10, None, (16, 6), 'Output Spectrum: Interpolation -> Decimation/10 FIR')
#        pl.plot([0.0005, 0.5], [-150, -150], 'k', linewidth=10, label='Signal Band')
#        pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (ds.calculateSNR(spec3[1:int(N2/2)],fin,nsig=3), 1.0), verticalalignment='center')
#        pl.xlabel('Normalized Decimated Frequency')
#        pl.ylabel('dBFS')
#        pl.plot()       

#print('\n'*3)
#print('Maximum absolute values of state variables =  \n')
#print(pd.DataFrame(maxvalues, columns=['x1','x2','x3','y','u'], \
#                   index=['5','4.5','4','3.5','3','2.5','2','1.5','1','0.5','0','-1','-2','-20','-50','-80']))
#print('\n'*3)

pl.figure(figsize=(10,4))
pl.plot(ds.dbv(amp),snr_amp, label='No dac mismatch')
ds.figureMagic([-80, 5], 5, None, [-10,100], 5, None, (10, 6), 'OUT Dynamic Range')
pl.text(-40, 75,'Peak SNR = %2.1f dB @ %2.1f dbV' %(snr_amp.max(), ds.dbv(amp[np.argmax(snr_amp)])-6))
pl.text(-40, 70,'Peak ENOB = %2.1f dB @ %2.1f dbV' %((snr_amp.max()-1.76)/6.02, ds.dbv(amp[np.argmax(snr_amp)])-6))
pl.xlabel('Amplitude (dBVpeak)')
pl.ylabel('SNR (dB)')
pl.legend(loc=4)
pl.plot()

result_dict = {'amp_dB': ds.dbv(amp), 
               'data_ch1': data_ch1,
               'data_ch2': data_ch2}
spio.savemat('data_out_BWov2_2to14ppt.mat', result_dict)


