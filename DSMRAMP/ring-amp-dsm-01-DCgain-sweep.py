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

"""
Time simulation of the scaled ABCD for sanity check and for spectrum analysis
Spectrum is plot for checking also
"""
pl.rcParams["font.family"] = "Times New Roman"
pl.rcParams["font.weight"] = "bold"
pl.rcParams["font.size"] = 20
pl.rcParams["legend.fontsize"] = 16
pl.figure(figsize=(10,4))
#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
DCgain = np.linspace(0,100,20)
##########################
OSR = np.array([10, 20, 40, 80, 160, 240])
DCgain_complete = np.zeros((np.size(DCgain),np.size(OSR)))
idx_osr = 0
for idx_osr in range(np.size(OSR)):
    
    #ntf = ds.clans(order = order, OSR = OSR, Q = nlev, opt=1)
    ntf = ds.synthesizeNTF(order=order, osr=OSR[idx_osr], opt=1, H_inf=6.56)
    
    # ABCD matrix
    a,g,b,c = ds.realizeNTF(ntf, form=form)
    ABCD = ds.stuffABCD(a,g,b,c, form=form)
    ABCDs = np.copy(ABCD)
    print(ABCDs)
    
    att_rms = np.ones(np.size(DCgain))
    idx_dcgain = 0
    for idx_dcgain in range(np.size(DCgain)):   
        ABCDs = np.copy(ABCD)
    #    if form == 'CIFB':
    #        ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  3.29918276, -3.29918276],
    #                          [ 0.33310912,  1.        , -0.04278519,  3.28963322, -3.28963322],
    #                          [ 0.        ,  1.35696266,  1.        ,  4.68902782, -4.68902782],
    #                          [ 0.        ,  0.        ,  0.61085042,  1.        ,  0.        ]])
    #    elif form == 'CIFF':
    #    ## CIFF matrix      
    #        ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.154     , -1.154],
    #                          [ 1.081     ,  1.        , -0.0701    ,  0.        ,  0.   ],
    #                          [ 0.        ,  0.827     ,  1.        ,  0.        ,  0.   ],
    #                          [ 2.48      ,  2.182     ,  0.714     ,  1.        ,  0.   ]])
        
        
        ## Simplest model possible for finite DC gain with integrator feedback attenuation
        if finite_dc_gain == True:    
            av = ds.undbv(DCgain[idx_dcgain]) # 60db
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
        
        
        
        ntf,_ = ds.calculateTF(ABCDs)
        
    #    ds.DocumentNTF(ntf,OSR)
        f1, f2 = ds.ds_f1f2(OSR[idx_osr], 0, 0)
        NG0 = ds.dbv(ds.rmsGain(ntf, f1, f2))
        NG0_lin = ds.undbv(NG0)
        NG0_lin_squared = NG0_lin**2
        
        att_rms[idx_dcgain] = NG0
        

    pl.plot(DCgain,att_rms, label='OSR='+str(OSR[idx_osr]))
    ds.figureMagic([0, 100], 10, None, [-160,0], 20, None, (10, 6))
    DCgain_complete[:,idx_osr] = att_rms
    #pl.text(-20, 75,'Peak SNR = %2.1f dB @ %2.1f dbV' %(snr_amp.max(), ds.dbv(amp[np.argmax(snr_amp)])))
pl.xlabel('DC gain (dB)')
pl.ylabel('RMS attenuation (dB)')
pl.legend(loc=4)
pl.title('Noise shaping accross DC gain and optimum OSR')
pl.plot()
