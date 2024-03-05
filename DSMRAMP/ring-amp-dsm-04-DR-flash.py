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
OSR is chossen to achieve BW of 100MHz
nlev is chossen to reduce noise floor and increase total SQNR
form is still open to discussion
"""
order = 3
OSR = 10
nlev = 16        ## 3-bit quantizer
form = 'CIFF'   ## FF better for highly linear OpA 

vdd = 0.9
delta = 2 #default value

###################
## Scales
###################
analog_scale = vdd/(delta*(nlev-1))
digital_scale = 1.0/analog_scale

sin2se_dBFS = (nlev-1)/vdd
sin2di_dBFS = (nlev-1)/(2*vdd)
sin2se_dBV = (nlev-1)
sin2di_dBV = (nlev-1)/2

"""
Time simulation of the scaled ABCD for sanity check and for spectrum analysis
Spectrum is plot for checking also
"""
N = 2**15
decim = 256
over = 1
fb = int(np.ceil(N/(2.*(over*OSR))))
fin = np.floor(1./11. * fb)
## 5dbV is equivalent to 1.8V-lin
amp = np.array([ds.undbv(5.0),
                ds.undbv(4.0),
                ds.undbv(3.0),
                ds.undbv(2.0),
                ds.undbv(1.0),
                ds.undbv(0.5),
                ds.undbv(-0.0),
                ds.undbv(-1.0),
                ds.undbv(-2.0),
                ds.undbv(-3.0),
                ds.undbv(-4.0),
                ds.undbv(-5),
                ds.undbv(-6),
                ds.undbv(-7),
                ds.undbv(-8),
                ds.undbv(-9),
                ds.undbv(-10),
                ds.undbv(-40),
                ds.undbv(-80)])

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
DCgain = 40.0
noise_enb = 0.0 # for the sampling input and DAC feedback
noise_amp = 0.0 # for the amplifiers noise ## Noise input vector (w/ input referred noise of input SW and DAC) - de la Rosa SC Model
coef_mismatch = False
saturation = True
sat_level_stg = 0.75*sin2se_dBV
offset_on = 0.0
offset_calibrated = False
##########################

Cin = 500e-15/2
#Cin = 160e-15
sigma2_sw_in = 4 * 4e-21 / Cin

amp_psd = 1e-5 ## V/sqrt(Hz)
sigma2_amp = np.array([(amp_psd)**2 , (amp_psd)**2, (amp_psd)**2])

input_noise = noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(N))*sin2se_dBV
snr_amp = np.ones(np.size(amp))
snr_mis_worst = np.ones(np.size(amp))
snr_DWA_worst = np.ones(np.size(amp))

#######################
## Threshold vector and offset
#######################
np.random.seed(2)
flash_scale = 6.4 
LSB_V = 1/(flash_scale*sin2di_dBV)
if nlev == 8:
    sigma_off = 9e-3
elif nlev == 16:
    sigma_off = 9e-3
else:
    sigma_off = 9e-3
if offset_calibrated == True:
    sigma_off = sigma_off - 7e-3
    
sigma_off_scaled = sigma_off*(1/LSB_V) ## Due to calculations being single ended
offset =  sigma_off_scaled*np.random.randn(nlev-1, nlev-1)*offset_on
threshold_vec = np.linspace(-nlev+2, nlev-2, nlev-1)
nlev_vec = np.linspace(-nlev+1, nlev-1, nlev)

if form == 'CRFF':
 ## CRFF Matrix
    ABCD = np.array([[ 1.        ,  0.        ,  0.        ,  1.        , -1.        ],
                     [ 1.        ,  1.        , -0.05892598,  0.        ,  0.        ],
                     [ 1.        ,  1.        ,  0.94107402,  0.        ,  0.        ],
                     [ 2.80536862,  1.81496051,  0.74564328,  1.        ,  0.        ]])

    ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.6314776 , -1.6314776 ],
                      [ 1.19337803,  1.        , -0.07934089,  0.        ,  0.        ],
                      [ 0.88631426,  0.74269363,  0.94107402,  0.        ,  0.        ],
                      [ 1.71952629,  0.9321977 ,  0.51565859,  1.        ,  0.        ]])
elif form == 'CIFF':
## CIFF matrix  
    ABCD = np.array([[ 1.        ,  0.        ,  0.        ,  1.        , -1.        ],
                     [ 1.        ,  1.        , -0.05805791,  0.        ,  0.        ],
                     [ 0.        ,  1.        ,  1.        ,  0.        ,  0.        ],
                     [ 2.86429459,  2.72678094,  0.74465741,  1.        ,  0.        ]])
    
    ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.154     , -1.154],
                      [ 1.081     ,  1.        , -0.0701    ,  0.        ,  0.   ],
                      [ 0.        ,  0.827     ,  1.        ,  0.        ,  0.   ],
                      [ 2.48      ,  2.182     ,  0.714     ,  1.        ,  0.   ]])

if coef_mismatch == True:
    a,g,b,c = ds.mapABCD(ABCDs, form)
    
    ## Capacitances in fF
    Cin1 = 511
    Cin1 = Cin1*(1 + (0.01/pl.sqrt(Cin1/(nlev-1)))*np.sum(np.random.randn(nlev-1)))
    Cint1 = 411
    Cint1 = Cint1*(1 + (0.01/pl.sqrt(Cint1))*np.random.randn(1)[0])
    
    Csample1 = 250
    Csample1 = Csample1*(1 + (0.01/pl.sqrt(Csample1))*np.random.randn(1)[0])
    Cin2 = 250
    Cin2 = Cin2*(1 + (0.01/pl.sqrt(Cin2))*np.random.randn(1)[0])

    Csampleg = 16.7
    Csampleg = Csampleg*(1 + (0.01/pl.sqrt(Csampleg))*np.random.randn(1)[0])
    Cing = 16.7
    Cing = Cing*(1 + (0.01/pl.sqrt(Cing))*np.random.randn(1)[0])
    
    Cint2 = 119.25
    Cint2 = Cint2*(1 + (0.01/pl.sqrt(Cint2))*np.random.randn(1)[0])
    
    Csample2 = 150
    Csample2 = Csample2*(1 + (0.01/pl.sqrt(Csample2))*np.random.randn(1)[0])
    Cin3 = 150
    Cin3 = Cin3*(1 + (0.01/pl.sqrt(Cin3))*np.random.randn(1)[0])
    
    Cint3 = 88.9
    Cint3 = Cint3*(1 + (0.01/pl.sqrt(Cint3))*np.random.randn(1)[0])
    
    Cyinput = 40.1
    Cyinput = Cyinput*(1 + (0.01/pl.sqrt(Cyinput))*np.random.randn(1)[0])
    Cyint1 = 98.86
    Cyint1 = Cyint1*(1 + (0.01/pl.sqrt(Cyint1))*np.random.randn(1)[0])
    Cyint2 = 87.37
    Cyint2 = Cyint2*(1 + (0.01/pl.sqrt(Cyint2))*np.random.randn(1)[0])
    Cyint3 = 28.27
    Cyint3 = Cyint3*(1 + (0.01/pl.sqrt(Cyint3))*np.random.randn(1)[0])
    
    ref_scale = 6.4
    b[0] = Cin1/Cint1
    b[3] = Cyinput/(Cyinput+Cyint1+Cyint2+Cyint3)*ref_scale
    c[0] = Cin1/Cint1
    c[1] = (Cin2/(Cin2+Csample1))*(Cin2/Cint2)
    c[2] = (Cin3/(Cin3+Csample2))*(Cin3/Cint3)
    a[0] = Cyint1/(Cyinput+Cyint1+Cyint2+Cyint3)*ref_scale
    a[1] = Cyint2/(Cyinput+Cyint1+Cyint2+Cyint3)*ref_scale
    a[2] = Cyint3/(Cyinput+Cyint1+Cyint2+Cyint3)*ref_scale
    
    ABCDs = ds.stuffABCD(a,g,b,c, form)


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


for amp_index in range(np.size(amp)):
    
    ## All architectures subjected to the same input with thermal noise
    u = (amp[amp_index] * np.sin(2*np.pi* fin/N *np.arange(N)))*sin2di_dBV + input_noise

    
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
    y = np.zeros((nq, N), dtype=np.float64)     # to store the quantizer input
    xn = np.zeros((order, N), dtype=np.float64) # to store the state information
    xmax = np.abs(x0) # to keep track of the state maxima
    
    point = 0

    for i in range(N):
        y0 = np.real(np.dot(C, x0) + np.dot(D1, u[:,i]))
        y[:,i] = y0
        
        ################
        ## Quantization with offset
        ################
        if y0[0] < (threshold_vec[0] + offset[0, point]):
            v[:,i] = nlev_vec[0]
        elif y0[0] >= (threshold_vec[nlev-1-1] + offset[nlev-1-1, point]):
            v[:,i] = nlev_vec[nlev-1]
        else:
            for mid in range(nlev-2):
                if y0[0] >= threshold_vec[mid]+offset[mid,point]:
                    v[:,i] =  nlev_vec[mid+1]
                    
        ## Update of DWA scheme shuffling pointer
        nelements = int((nlev-1 + v[:,i])/2)
        if point+nelements >= nlev-1:
            point = nelements-(nlev-1-point)
        else:
            if point+nelements == nlev-1:
                point = 0
            else:
                point = point+nelements
#                
#        ## Random single increment
#        point = int(point + 1 + np.floor(1+ np.random.randn(1)[0]))
#        if point >= nlev-1:
#            point = 0
        
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
    xn = xn.squeeze()
    y = y.squeeze()
    
    
    f = np.linspace(0, 0.5, int(N/2. + 1))
    spec = np.fft.fft(v* analog_scale * ds.ds_hann(N))/(N/4)
    snr_amp[amp_index] = ds.calculateSNR(spec[2:fb+1], fin-2)
    print(amp_index)

    
    ######################
    ## Input mismatch vector
    ######################
    ndac = nlev - 1
    ## Thermometer code vector
    sv0 = -np.ones((ndac, np.size(v)))
    for i in range(np.size(v)):
        sv0[0:int((ndac +v[i])/2) , i] = 1
    
    ## DWA vector
    svDWA = -np.ones((ndac, np.size(v)))
    p = 0
    for i in range(np.size(v)):
        nelements = int((ndac+v[i])/2)
        if p+nelements >= ndac :
            svDWA[p:ndac, i] = 1
            svDWA[0:nelements-(ndac-p), i] = 1
            p = nelements-(ndac-p)
        else:
            svDWA[p:p+nelements, i] = 1
            if p+nelements == ndac:
                p = 0
            else:
                p = p+nelements
    
    dmismatch = 0.01/np.sqrt(Cin*1e15/(nlev-1)) ## Regular MOM sigma
    
    snr_mis = np.ones(100)
    snr_DWA = np.ones(100)
    for i in range(np.size(snr_mis)):
        ue = 1 + dmismatch*np.random.randn(ndac, 1)
        
        dv0 = np.dot(ue.T, sv0)
        dv0 = dv0.squeeze()    
        
        dvDWA = np.dot(ue.T, svDWA)
        dvDWA = dvDWA.squeeze()
        
        specm = np.fft.fft(dv0*analog_scale * ds.ds_hann(N)) / (N/4)
        snr_mis[i] = ds.calculateSNR(specm[2:fb+1], fin-2)
        
        specDWA = np.fft.fft(dvDWA*analog_scale * ds.ds_hann(N)) / (N/4)
        snr_DWA[i] = ds.calculateSNR(specDWA[2:fb+1], fin-2)
        
    snr_mis_worst[amp_index] = np.sort(snr_mis)[0]
    snr_DWA_worst[amp_index] = np.sort(snr_DWA)[0]
    
    if amp_index == 4:
        pl.semilogx(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Offset spectrum')
        pl.semilogx(f, ds.dbv(specm[:int(N/2.) +1]), 'r--',linewidth=1, label='Plus mismatch')
        pl.semilogx(f, ds.dbv(specDWA[:int(N/2.) +1]), 'g--',linewidth=1, label='Plus mismatch w/ DWA')
        pl.plot([0.0005, 0.5/(over*OSR)], [-130, -130], 'k', linewidth=10, label='Signal Band')
        ds.figureMagic([0.0005, 0.5], None, None, [-140, 0], 10, None, (16, 6), 'Output Spectrum')
        pl.xlabel('Normalized Frequency')
        pl.ylabel('dBFS')
        pl.legend(loc=1)
        pl.plot()


pl.figure(figsize=(10,4))
pl.plot(ds.dbv(amp),snr_amp, label='No dac mismatch')
pl.plot(ds.dbv(amp),snr_mis_worst, label='2.5sigma mismatch no shaping')
pl.plot(ds.dbv(amp),snr_DWA_worst, label='2.5sigma mismatch DWA')
ds.figureMagic([-80, 5], 5, None, [-10, 80], 5, None, (10, 6), 'OUT Dynamic Range')
pl.text(-50, 75,'Peak SNR = %2.1f dB @ %2.1f dbV' %(snr_amp.max(), ds.dbv(amp[np.argmax(snr_amp)])))
pl.text(-50, 70,'Peak SNR-mis = %2.1f dB @ %2.1f dbV' %(snr_mis_worst.max(), ds.dbv(amp[np.argmax(snr_mis_worst)])))
pl.text(-50, 65,'Peak SNR-DWA = %2.1f dB @ %2.1f dbV' %(snr_DWA_worst.max(), ds.dbv(amp[np.argmax(snr_DWA_worst)])))
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
