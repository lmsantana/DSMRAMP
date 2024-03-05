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
analog_scale = vdd/(delta*(nlev-1))
digital_scale = 1.0/analog_scale

"""
 NTF synthesis and initial analysis done in another script
 Here we just retrieve the NTF and the ABCD matrix previously found

"""

ntf = ds.synthesizeNTF(order=order, osr=OSR, opt=1, H_inf=6.56)

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

#print('ABCD matrix of the plant =  \n')
#print(pd.DataFrame(ABCD, columns=['x1[n]','x2[n]','x3[n]','u[n]','v[n]'], \
#                   index=['x1[n+1]','x2[n+1]','x3[n+1]','y[n]']))
#print('\n'*3)



"""
Time simulation of the scaled ABCD for sanity check and for spectrum analysis
Spectrum is plot for checking also
"""
N = 2**15
decim = 256
over = 1
fb = int(np.ceil(N/(2.*(over*OSR))))
fin = np.floor(1./11. * fb)
amp = ds.undbv(-5.0) * nlev 

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
noise_enb = 1 # for the sampling input and DAC feedback
noise_amp = 0 # for the amplifiers noise ## Noise input vector (w/ input referred noise of input SW and DAC) - de la Rosa SC Model
saturation = True
percentage_of_nlev = 1.0
coef_mismatch = True
##########################

noise_in_type = 2
Cin = 125e-15

sigma2_dac_in = 0
sigma2_sw_in = 4 * 4e-21 / Cin

amp_psd = 1e-5 ## V/sqrt(Hz)
gm = 1e-3
bw = 50e8
amp_psd2 = 4 * 4e-21 *2/3 /gm  ## V^2 / Hz
sigma2_amp_gm = amp_psd2*bw
sigma2_amp = np.array([(amp_psd)**2 , (amp_psd)**2, (amp_psd)**2])

## All architectures subjected to the same input with thermal noise
u = (amp * np.sin(2*np.pi* fin/N *np.arange(N))) + noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(N) - 0.5)*nlev

snr_mc = np.ones(10)
for mc_i in range(np.size(snr_mc)):
       
    ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.154*2     , -1.154*2],
                      [ 1.081     ,  1.        , -0.0701    ,  0.        ,  0.   ],
                      [ 0.        ,  0.827     ,  1.        ,  0.        ,  0.   ],
                      [ 2.48/2      ,  2.182/2     ,  0.714/2     ,  1.        ,  0.   ]])    
    
    """
    Hard code implementation of ds.simulateDSM_python function, it is not optmized for time as
    the Cyhton version, but it allows access to internal nodes
    """
    
    if coef_mismatch == True:
        a,g,b,c = ds.mapABCD(ABCDs, form)
        
        ## Capacitances in fF
        Cin1 = 125
        Cin1 = Cin1*(1 + (0.01/pl.sqrt(Cin1/(nlev-1)))*np.sum(np.random.randn(nlev-1)))
        Cint1 = 54.15
        Cint1 = Cint1*(1 + (0.01/pl.sqrt(Cint1))*np.random.randn(1)[0])
        
        Csample1 = 62.5
        Csample1 = Csample1*(1 + (0.01/pl.sqrt(Csample1))*np.random.randn(1)[0])
        Cin2 = 62.5
        Cin2 = Cin2*(1 + (0.01/pl.sqrt(Cin2))*np.random.randn(1)[0])
    
        Csampleg = 4.05
        Csampleg = Csampleg*(1 + (0.01/pl.sqrt(Csampleg))*np.random.randn(1)[0])
        Cing = 4.05
        Cing = Cing*(1 + (0.01/pl.sqrt(Cing))*np.random.randn(1)[0])
        
        Cint2 = 28.9
        Cint2 = Cint2*(1 + (0.01/pl.sqrt(Cint2))*np.random.randn(1)[0])
        
        Csample2 = 31.25
        Csample2 = Csample2*(1 + (0.01/pl.sqrt(Csample2))*np.random.randn(1)[0])
        Cin3 = 31.25
        Cin3 = Cin3*(1 + (0.01/pl.sqrt(Cin3))*np.random.randn(1)[0])
        
        Cint3 = 17.85
        Cint3 = Cint3*(1 + (0.01/pl.sqrt(Cint3))*np.random.randn(1)[0])
        
        Cyinput = 50
        Cyinput = Cyinput*(1 + (0.01/pl.sqrt(Cyinput))*np.random.randn(1)[0])
        Cyint1 = 62
        Cyint1 = Cyint1*(1 + (0.01/pl.sqrt(Cyint1))*np.random.randn(1)[0])
        Cyint2 = 54.5
        Cyint2 = Cyint2*(1 + (0.01/pl.sqrt(Cyint2))*np.random.randn(1)[0])
        Cyint3 = 17.8
        Cyint3 = Cyint3*(1 + (0.01/pl.sqrt(Cyint3))*np.random.randn(1)[0])
        
        ref_scale = 4.0
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
    
    ## fast simulateDSM without DAC noise and amp noise
    v, xn, xmax, y = ds.simulateDSM(u, ABCDs, nlev=nlev)
    
    
#    ## simulateDSM hard code for noise input and further changes
#    nq = 1 # 1 input
#    nu = 1 # 1 output
#    A = ABCDs[:order, :order]
#    B = ABCDs[:order, order:order+nu+nq]
#    C = ABCDs[order:order+nq, :order]
#    D1 = ABCDs[order:order+nq, order:order+nu] ## assuming no direct feedback from V to Y
#    
#    u = u.reshape((1,-1))
#    x0 = 0.0*np.ones((order,), dtype=np.float64)
#    v = np.empty((nq, N), dtype=np.float64)
#    y = np.empty((nq, N), dtype=np.float64)     # to store the quantizer input
#    xn = np.empty((order, N), dtype=np.float64) # to store the state information
#    xmax = np.abs(x0) # to keep track of the state maxima
#    
#    
#    
#    for i in range(N):
#    
#        y0 = np.real(np.dot(C, x0) + np.dot(D1, u[:,i]))
#        y[:,i] = y0
#        
#        v[:,i] = ds_quantize(y0, nlev) + noise_enb*pl.sqrt(sigma2_dac_in)*(np.random.randn(1) - 0.5)*nlev
#        x0 = np.dot(A, x0) + np.dot(B, np.concatenate((u[:,i], v[:,i]))) + noise_amp*pl.sqrt(sigma2_amp)*(np.random.randn(1) - 0.5)*nlev
#        
#        ## Hard saturation model for noise shaping prediction
#        if saturation == True:
#            if np.abs(x0[0]) > percentage_of_nlev*nlev:
#                x0[0] = np.sign(x0[0])*percentage_of_nlev*nlev
#            if np.abs(x0[1]) > percentage_of_nlev*nlev:
#                x0[1] = np.sign(x0[1])*percentage_of_nlev*nlev
#            if np.abs(x0[2]) > percentage_of_nlev*nlev:
#                x0[2] = np.sign(x0[2])*percentage_of_nlev
#        
#        xn[:, i] = np.real_if_close(x0.T)
#        xmax = np.max(np.hstack((np.abs(x0).reshape((-1, 1)), xmax.reshape((-1, 1)))),
#                      axis=1, keepdims=True)
#    
#    u = u.squeeze()
#    v = v.squeeze()
#    xn = xn.squeeze()
#    y = y.squeeze()
    
    
    f = np.linspace(0, 0.5, int(N/2. + 1))
    spec = np.fft.fft(v* analog_scale * ds.ds_hann(N))/(N/4)
    snr_mc[mc_i] = ds.calculateSNR(spec[2:fb+1], fin-2)
    print(mc_i)
    print('\n')

pl.hist(snr_mc,20)
pl.xlabel('SNR [dB]')
pl.ylabel('absolute frequency [#]')
pl.plot()