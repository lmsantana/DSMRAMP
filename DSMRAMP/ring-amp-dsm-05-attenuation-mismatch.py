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
fin = np.floor(1./7. * fb)
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
                ds.undbv(-4.0)])

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
noise_enb = 1 # for the sampling input and DAC feedback
noise_amp = 1 # for the amplifiers noise ## Noise input vector (w/ input referred noise of input SW and DAC) - de la Rosa SC Model
coef_mismatch = False
saturation = True
sat_level_stg = 0.75*sin2se_dBV
offset_on = 1.0
offset_calibrated = False
##########################

Cin = 500e-15/4.0
#Cin = 160e-15
sigma2_sw_in = 4 * 4e-21 / Cin

amp_psd = 1e-5 ## V/sqrt(Hz)
sigma2_amp = np.array([(amp_psd)**2 , (amp_psd)**2, (amp_psd)**2])

input_noise = noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(N))*sin2di_dBV
snr_amp = np.ones(np.size(amp))
snr_mis_worst = np.ones(np.size(amp))
snr_DWA_worst = np.ones(np.size(amp))
    
re_scale = 2.0
ABCDs = np.array([[ 1.                 ,  0.                 ,  0.                 ,  1.154              , -1.154],
                  [ 1.081              ,  1.                 , -0.0701             ,  0.                 ,  0.   ],
                  [ 0.                 ,  0.827              ,  1.                 ,  0.                 ,  0.   ],
                  [ 2.48               ,  2.182              ,  0.714              ,  1.                 ,  0.   ]])

new_flash_scale = 1.0 + ABCDs[3,0]+ ABCDs[3,1]+ ABCDs[3,2]
print("\n New attenuation factor: %2.3f \n\n" %new_flash_scale)

a,g,b,c = ds.mapABCD(ABCDs, form)
a = a/re_scale
c[0] = c[0]*re_scale
b[0] = b[0]*re_scale

Cin1 = Cin*1e15
Cint1 = Cin1/b[0]
Csample1 = Cin1/2

Cin2 = Csample1
Cint2 = (Cin2*Cin2/(Cin2+Csample1))/c[1]
Cg = Cint2*(2*g[0])
Csampleg = Cg
Csample2 = Cin2/2

Cin3 = Csample2
Cint3 = (Cin3*Cin3/(Cin3+Csample2))/c[2]

flash_scale_vec = np.ones(10000)
for scale_i in range(np.size(flash_scale_vec)):
    Cflash = 1*(nlev-1)/1.0 *(1 + (0.01)*np.random.randn(1)[0])
    Cyinput = 50*(1 + (0.01)*np.random.randn(1)[0])
    Cyint1 = 50*a[0]*(1 + (0.01)*np.random.randn(1)[0])
    Cyint2 = 50*a[1]*(1 + (0.01)*np.random.randn(1)[0])
    Cyint3 = 50*a[2]*(1 + (0.01)*np.random.randn(1)[0])
    
    flash_scale_vec[scale_i] = 1/(Cyinput/(Cyinput+Cyint1+Cyint2+Cyint3 + Cflash))

pl.hist(flash_scale_vec,100)
pl.plot()

Cflash = 1*(nlev-1)/1.0
Cyinput = 50
Cyint1 = Cyinput*a[0]
Cyint2 = Cyinput*a[1]
Cyint3 = Cyinput*a[2]
flash_scale = 1/(Cyinput/(Cyinput+Cyint1+Cyint2+Cyint3 + Cflash))

flash_scale = 3.99 
flash_scale_worst = np.sort(flash_scale_vec)[0]
b[3] = b[3]*flash_scale/flash_scale_worst
a[0] = a[0]*flash_scale/flash_scale_worst
a[1] = a[1]*flash_scale/flash_scale_worst
a[2] = a[2]*flash_scale/flash_scale_worst

ABCDs = ds.stuffABCD(a,g,b,c, form)

print("Cin1: %3.3f fF \n" % Cin1)
print("Cint1: %3.3f fF \n" % Cint1)
print("Cin2: %3.3f fF \n" % Cin2)
print("Cint2: %3.3f fF \n" % Cint2)
print("Cin3: %3.3f fF \n" % Cin3)
print("Cint3: %3.3f fF \n" % Cint3)
print("Cg: %3.3f fF \n" % Cg)
print("Cyinput: %3.3f fF \n" % Cyinput)
print("Cyint1: %3.3f fF \n" % Cyint1)
print("Cyint2: %3.3f fF \n" % Cyint2)
print("Cyint3: %3.3f fF \n" % Cyint3)
print("Cflash: %3.3f fF \n" % Cflash)

print("\nAttenuation factor: %3.3f \n" % flash_scale)
#######################
## Threshold vector and offset
#######################
np.random.seed(2)

LSB_V = 1/(flash_scale * sin2di_dBFS)
if nlev == 8:
    sigma_off = 9e-3
elif nlev == 16:
    sigma_off = 9e-3
else:
    sigma_off = 9e-3
if offset_calibrated == True:
    sigma_off = sigma_off - 7e-3
    
print("LSB: %2.2f mV" %(LSB_V*1e3))
sigma_off_scaled = sigma_off*(1/LSB_V) ## Due to calculations being single ended
offset =  sigma_off_scaled*np.random.randn(nlev-1, nlev-1)*offset_on
threshold_vec = np.linspace(-nlev+2, nlev-2, nlev-1)
nlev_vec = np.linspace(-nlev+1, nlev-1, nlev)

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
    u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBV + input_noise
    
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
ds.figureMagic([-5, 5], 1, None, [30, 80], 5, None, (10, 6), 'OUT Dynamic Range')
pl.text(-4, 75,'Peak SNR = %2.1f dB @ %2.1f dbV' %(snr_amp.max(), ds.dbv(amp[np.argmax(snr_amp)])))
pl.text(-4, 65,'Peak SNR-mis = %2.1f dB @ %2.1f dbV' %(snr_mis_worst.max(), ds.dbv(amp[np.argmax(snr_mis_worst)])))
pl.text(-4, 70,'Peak SNR-DWA = %2.1f dB @ %2.1f dbV' %(snr_DWA_worst.max(), ds.dbv(amp[np.argmax(snr_DWA_worst)])))
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
