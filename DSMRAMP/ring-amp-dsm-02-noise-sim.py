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
nlev = 8        ## 3-bit quantizer
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
        
    ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.154, -1.154],
                      [ 1.081     ,  1.        , -0.0701    ,  0.        ,  0.        ],
                      [ 0.        ,  0.827     ,  1.        ,  0.        ,  0.        ],
                      [ 2.48      ,  2.182     ,  0.714     ,  1.        ,  0.        ]])

print('ABCD matrix of the plant =  \n')
print(pd.DataFrame(ABCD, columns=['x1[n]','x2[n]','x3[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','y[n]']))
print('\n'*3)
print('ABCD Scaled matrix of the plant =  \n')
print(pd.DataFrame(ABCDs, columns=['x1[n]','x2[n]','x3[n]','u[n]','v[n]'], \
                   index=['x1[n+1]','x2[n+1]','x3[n+1]','y[n]']))
print('\n'*3)


"""
Time simulation of the scaled ABCD for sanity check and for spectrum analysis
Spectrum is plot for checking also
"""
N = 2**15
decim = 256
over = 1
fb = int(np.ceil(N/(2.*(over*OSR))))
fin = np.floor(1./11. * fb)
amp = ds.undbv(-4.5) * nlev

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
noise_enb = 1 # for the sampling input and DAC feedback
noise_amp = 1 # for the amplifiers noise ## Noise input vector (w/ input referred noise of input SW and DAC) - de la Rosa SC Model
saturation = True
percentage_of_nlev = 1.0
##########################

noise_in_type = 2
Cin = 500e-15
if noise_in_type == 1: ## Split Input and DAC CAP structure
    Cdac = Cin
    sigma2_dac_in = 2 * 4 * 4e-21 * 0.38 / Cdac 
    sigma2_sw_in =  2 * 4 * 4e-21 * 0.38 / Cin           # 4* because of the switches
elif noise_in_type == 2:    ## Single input CAP structure
    sigma2_dac_in = 0
    sigma2_sw_in = 4 * 4e-21 / Cin          # 0.38 factor from the integrated sinc^2

amp_psd = 1e-5 ## V/sqrt(Hz)
sigma2_amp = np.array([(amp_psd)**2 , (amp_psd)**2, (amp_psd)**2])

u = (amp * np.sin(2*np.pi* fin/N *np.arange(N))) + noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(N) - 0.5)*nlev


"""
Hard code implementation of ds.simulateDSM_python function, it is not optmized for time as
the Cyhton version, but it allows access to internal nodes
"""

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

## simulateDSM hard code for noise input and further changes
nq = 1 # 1 input
nu = 1 # 1 output
A = ABCDs[:order, :order]
B = ABCDs[:order, order:order+nu+nq]
C = ABCDs[order:order+nq, :order]
D1 = ABCDs[order:order+nq, order:order+nu] ## assuming no direct feedback from V to Y

u = u.reshape((1,-1))
x0 = 0.0*np.ones((order,), dtype=np.float64)
v = np.empty((nq, N), dtype=np.float64)
y = np.empty((nq, N), dtype=np.float64)     # to store the quantizer input
xn = np.empty((order, N), dtype=np.float64) # to store the state information
xmax = np.abs(x0) # to keep track of the state maxima



for i in range(N):

    y0 = np.real(np.dot(C, x0) + np.dot(D1, u[:,i]))
    y[:,i] = y0
    
    v[:,i] = ds_quantize(y0, nlev) + noise_enb*pl.sqrt(sigma2_dac_in)*(np.random.randn(1) - 0.5)*nlev
    x0 = np.dot(A, x0) + np.dot(B, np.concatenate((u[:,i], v[:,i]))) + noise_amp*pl.sqrt(sigma2_amp)*(np.random.randn(1) - 0.5)*nlev
    
    ## Hard saturation model for noise shaping prediction
    if saturation == True:
        if np.abs(x0[0]) > percentage_of_nlev*nlev:
            x0[0] = np.sign(x0[0])*percentage_of_nlev*nlev
        if np.abs(x0[1]) > percentage_of_nlev*nlev:
            x0[1] = np.sign(x0[1])*percentage_of_nlev*nlev
        if np.abs(x0[2]) > percentage_of_nlev*nlev:
            x0[2] = np.sign(x0[2])*percentage_of_nlev
    
    xn[:, i] = np.real_if_close(x0.T)
    xmax = np.max(np.hstack((np.abs(x0).reshape((-1, 1)), xmax.reshape((-1, 1)))),
                  axis=1, keepdims=True)

u = u.squeeze()
v = v.squeeze()
xn = xn.squeeze()
y = y.squeeze()


############################
## Mismatch vector
###########################
ndac = nlev - 1
sv0 = -np.ones((ndac, np.size(v)))
for i in range(np.size(v)):
    sv0[0:int((ndac +v[i])/2) , i] = 1


dmismatch = 0.01/np.sqrt(Cin*1e15/(nlev-1)) ## Regular MOM sigma
#dmismatch = 0.005/np.sqrt(Cin*1e15/(nlev-1)) ## MEX MOM sigma

ue = 1 + dmismatch*np.random.randn(ndac, 1)
dv0 = np.dot(ue.T, sv0)
dv0 = dv0.squeeze()

specm = np.fft.fft(dv0*analog_scale * ds.ds_hann(N)) / (N/4)


"""
Time simulation for spectrum analysis and histogram of the state signals to statitcs
"""
t = np.arange(int(N/decim))


pl.plot(t, u[t] * analog_scale, 'r', label='Input')
pl.step(t, v[t] * analog_scale, 'g', label='DAC Output')
pl.plot(t, y[t] * analog_scale, 'b', label='Quant in')
ds.figureMagic([0, N/decim], 100, None, [-vdd, vdd], 0.1, None, (20, 8),'Modulator input and output; DAC step = 0.9V/7=128mV')
pl.grid(True)
pl.xlabel('Sample')
pl.ylabel('Amplitude (Volt)')
pl.legend(loc=1)
pl.show()

pl.figure(figsize=(20,4))
pl.step(t, xn[0 , t] * analog_scale, 'g', label='State 1')
pl.step(t, xn[1 , t] * analog_scale, 'b', label='State 2')
pl.step(t, xn[2 , t] * analog_scale, 'r', label='State 3')
pl.axis([0, N/decim, -vdd*0.7, vdd*0.7])
pl.grid(True)
pl.xlabel('Sample')
pl.ylabel('States Amplitudes (Volt)')
pl.title('State amplitudes')
pl.legend(loc=1)
pl.show()

pl.figure(figsize=(20,4))
pl.subplot(141)
pl.hist(xn[0,:] * analog_scale, bins=100, range=(-1,1), density=True)
pl.ylabel('Percentual (%)')
pl.title('State 1')

pl.subplot(142)
pl.hist(xn[1,:] * analog_scale, bins=100, range=(-1,1), density=True)
pl.ylabel('Percentual (%)')
pl.title('State 2')

pl.subplot(143)
pl.hist(xn[2,:]* analog_scale, bins=100, range=(-1,1), density=True)
pl.ylabel('Percentual (%)')
pl.title('State 3')

pl.subplot(144)
pl.hist(y* analog_scale, bins=100, density=True)
pl.ylabel('Percentual (%)')
pl.title('Quant In')
pl.show()

pl.figure(figsize=(20,4))
pl.subplot(131)
pl.hist(np.diff(xn[0,:]) * analog_scale, bins=100, range=(-1,1), density=True)
pl.ylabel('Percentual (%)')
pl.title('Step size State 1')

pl.subplot(132)
pl.hist(np.diff(xn[1,:]) * analog_scale, bins=100, range=(-1,1), density=True)
pl.ylabel('Percentual (%)')
pl.title('Step size State 2')

pl.subplot(133)
pl.hist(np.diff(xn[2,:])* analog_scale, bins=100, range=(-1,1), density=True)
pl.ylabel('Percentual (%)')
pl.title('Step size State 3')
pl.show()


pl.figure(figsize=(10,4))
## Plotting spectrums
f = np.linspace(0, 0.5, int(N/2. + 1))
# Scaling of FFT due to Hann window
spec = np.fft.fft(v* analog_scale * ds.ds_hann(N))/(N/4)
pl.semilogx(f, ds.dbv(spec[:int(N/2.) +1]), 'b', label='Simulation spectrum')
#pl.plot(f[:fb], ds.dbv(spec[1:fb +1]), 'b', label='Simulation spectrum')
#pl.grid(True)
ds.figureMagic([0.0005, 0.5], None, None, [-130, 0], 10, None, (16, 6), 'Output Spectrum')
pl.xlabel('Normalized Frequency')
pl.ylabel('dBFS')

### Mismatch
pl.semilogx(f, ds.dbv(specm[:int(N/2.) +1]), '--r', label='Simulated Specturm with mismatch')
snrm = ds.calculateSNR(specm[2:fb+1], fin-2)
pl.text(0.05, -10, 'SNR = %4.1fdB @ OSR = %d with mismatch' % (snrm, OSR), verticalalignment='center')


## Plot of inband range for reference
pl.plot([0.0005, 0.5/(over*OSR)], [-130, -130], 'k', linewidth=10, label='Signal Band')

snr = ds.calculateSNR(spec[2:fb+1], fin-2)
pl.text(0.05, -5, 'SNR = %4.1fdB @ OSR = %d' % (snr, OSR), verticalalignment='center')

NBW = 1.5/N
Sqq = (4/3)*ds.evalTF(ntf, np.exp(2j*np.pi*f)) ** 2
pl.plot(f, ds.dbp(Sqq * NBW) + ds.dbv(analog_scale), 'm--', linewidth=2, label='Expected PSD')
pl.text(0.49, -90, 'NBW = %4.1E x $f_s$' % NBW, horizontalalignment='right')

pl.legend(loc=2)
pl.show()

ds.DocumentNTF(ABCD, OSR)
ds.DocumentNTF(ABCDs, OSR)
