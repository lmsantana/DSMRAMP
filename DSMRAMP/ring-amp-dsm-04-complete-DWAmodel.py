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
analog_scale = 1.0

sin2se_dBFS = (nlev-1)/vdd
sin2di_dBFS = (nlev-1)/(2*vdd)
sin2se_dBV = (nlev-1)
sin2di_dBV = (nlev-1)/2

"""
Time simulation of the scaled ABCD for sanity check and for spectrum analysis
Spectrum is plot for checking also
"""
N = 2**13
over = 1
fb = int(np.ceil(N/(2.*(over*OSR))))
fin = np.floor(1./7. * fb)
## 5dbV is equivalent to 1.8Vpp = Full Scale
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
                ds.undbv(0.0)])

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
noise_enb = 1.0 # for the sampling input and DAC feedback
saturation = True
sat_level_stg = 0.9*sin2se_dBV
offset_on = 1.0
offset_calibrated = True

coef_mismatch = True

#dmismatch = 0.01
dmismatch = (1/np.sqrt(8)/100) # Mismacth for 8fF from TSMC datasheet

drop_vdd_on = False
drop_sigma = 5e-3

dwa_off = False
print('DWA ON = %s \n' %(not dwa_off))
##########################

#########################
# Plot config
#########################
pl.rcParams['font.family'] = 'Times New Roman'
pl.rcParams['font.size'] = 24
pl.rcParams['font.weight'] = 'bold'
pl.rcParams['figure.titleweight'] = 'bold'
pl.rcParams['axes.titleweight'] = 'bold'
pl.rcParams['axes.labelweight'] = 'bold'
pl.rcParams['lines.linewidth'] = 2
pl.rcParams['lines.markersize'] = 10
##########################

Cin = 125e-15
sigma2_sw_in = 4 * 4e-21 / Cin

snr_amp = np.ones(np.size(amp))

mc_points = 1000
snr_amp_mc = np.zeros(mc_points)
snr_1dBV5_mc = np.zeros(mc_points)

snr_all_mc = np.zeros((np.size(amp), mc_points))

pl.figure(figsize=(10,4))
for mc_index in range(np.size(snr_amp_mc)):   
    input_noise = noise_enb*pl.sqrt(sigma2_sw_in)*(np.random.randn(N))*sin2se_dBFS
    
    ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.154*2     , -1.154*2],
                      [ 1.081     ,  1.        , -0.0701    ,  0.        ,  0.   ],
                      [ 0.        ,  0.827     ,  1.        ,  0.        ,  0.   ],
                      [ 2.48/2      ,  2.182/2     ,  0.714/2     ,  1.        ,  0.   ]]) 
    
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
        
        
        Cflash = 1*(nlev-1)
      
        flash_scale = 1/(Cyinput/(Cyinput+Cyint1+Cyint2+Cyint3 + Cflash))
    
        b[0] = Cin1/Cint1
        b[3] = Cyinput/(Cyinput+Cyint1+Cyint2+Cyint3+ Cflash)*flash_scale
        c[0] = Cin1/Cint1
        c[1] = (Cin2/(Cin2+Csample1))*(Cin2/Cint2)
        c[2] = (Cin3/(Cin3+Csample2))*(Cin3/Cint3)
        a[0] = Cyint1/(Cyinput+Cyint1+Cyint2+Cyint3+ Cflash)*flash_scale
        a[1] = Cyint2/(Cyinput+Cyint1+Cyint2+Cyint3+ Cflash)*flash_scale
        a[2] = Cyint3/(Cyinput+Cyint1+Cyint2+Cyint3+ Cflash)*flash_scale
        
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
    
    
#    print("Cin1: %3.3f fF \n" % Cin1)
#    print("Cint1: %3.3f fF \n" % Cint1)
#    print("Cin2: %3.3f fF \n" % Cin2)
#    print("Cint2: %3.3f fF \n" % Cint2)
#    print("Cin3: %3.3f fF \n" % Cin3)
#    print("Cint3: %3.3f fF \n" % Cint3)
#    print("Cg: %3.3f fF \n" % Cg)
#    print("Cyinput: %3.3f fF \n" % Cyinput)
#    print("Cyint1: %3.3f fF \n" % Cyint1)
#    print("Cyint2: %3.3f fF \n" % Cyint2)
#    print("Cyint3: %3.3f fF \n" % Cyint3)
#    print("Cflash: %3.3f fF \n" % Cflash)
#    
#    print("\nAttenuation factor: %3.3f \n" % flash_scale)
    #######################
    ## Threshold vector and offset
    #######################
    #flash_scale = 3.99 
    LSB_V = 1/(flash_scale * sin2di_dBFS)
    # LSB_V = 1/(flash_scale * sin2se_dBFS)
    sigma_off = 9e-3
    if offset_calibrated == True:
        sigma_off = 2e-3
        
#    print("LSB: %2.2f mV" %(LSB_V*1e3))
    sigma_off_scaled = sigma_off*(1/LSB_V) ## Due to calculations being single ended
    ## uncalibrated offset are normal distributed
    if offset_calibrated == False:
        offset =  sigma_off_scaled*np.random.randn(nlev-1, nlev-1)*offset_on
            
    ## calibrated offsets are uniform distribution
    elif offset_calibrated == True:
        offset =  sigma_off_scaled*(np.random.rand(nlev-1, nlev-1)-0.5)*offset_on
    
    ## Rotation of the baseline offsets due to DWA
    for offset_rol in range(1,(nlev-1)):
            offset[offset_rol,:] = np.roll(offset[0,:],offset_rol)
    
    threshold_vec = np.linspace(-nlev+2, nlev-2, nlev-1)
    
    if drop_vdd_on == False:
        drop_sigma = 0
    
    vdd_pointer=900e-3 - drop_sigma*(np.random.rand(nlev-1))
    att_pointer = vdd_pointer / 900e-3
    
    
    nlev_vec = np.linspace(-nlev+1, nlev-1, nlev)
    
#    sigma_dyn_off = 75e-3
#    dyn_offset = sigma_dyn_off*sin2di_dBFS*np.random.randn(nlev-1, nlev-1)*dyn_offset_on
    
    ## Simplest model possible for finite DC gain with integrator feedback attenuation
    if finite_dc_gain == True:    
        av = ds.undbv(33) # 60db
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
        u = (amp[amp_index] * (np.sin(2*np.pi* fin/N *np.arange(N))))*sin2di_dBFS + input_noise
        
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
        v_coef = np.ones(15)*B[0,1]/15*(1+dmismatch*np.random.randn(15))
        v_coef_conc = np.concatenate((v_coef,v_coef))
        
        for i in range(N):
            y0 = np.real(np.dot(C, x0) + np.dot(D1, u[:,i]))*att_pointer[point]
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
            
#            if i%2000 == 0:
#                v[:,i] = nlev_vec[np.random.randint(0,nlev)]
#                print("Bin output ERROR!")
            ## Update of DWA scheme shuffling pointer
            ndac = nlev - 1
            nelements = int((ndac + v[:,i])/2)
            p_coef = np.sum(v_coef_conc[point           : point+nelements])
            n_coef = np.sum(v_coef_conc[point+nelements : point+nelements + (ndac-nelements)])        
            
#            # DWA rotation in the right direction
#            if point+nelements >= nlev-1:
#                point = nelements-(nlev-1-point)
#            else:
#                if point+nelements == nlev-1:
#                    point = 0
#                else:
#                    point = point+nelements
                    
            # DWA rotation as it was implemented
            if point-nelements >=0:
                point = point - nelements
            else:
                if point-nelements == -1:
                    point = 0
                else:
                    point = (nlev-1) + (point-nelements)
            
            if dwa_off == True:
                point = 0
            
            BxV_input_sum = (n_coef*(-15) + p_coef*15)/B[0,1]
            in_vector = np.concatenate((u[:,i], np.array([BxV_input_sum])))
            x0 = np.dot(A, x0) + np.dot(B, in_vector)
            
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
#        print(amp_index)
        snr_all_mc[amp_index, mc_index] = ds.calculateSNR(spec[2:fb+1], fin-2)
        
    snr_amp_mc[mc_index] = snr_amp.max()
    snr_1dBV5_mc[mc_index] = snr_amp[7]
#    print('MC = %d \n' %(mc_index))
    # pl.plot(ds.dbv(amp)-5,snr_amp, label='Mismacth with DWA')
    print(mc_index)


pl.rcParams['lines.linewidth'] = 2
pl.rcParams['lines.markersize'] = 10
pl.rcParams['boxplot.flierprops.linewidth'] = 4
pl.rcParams['boxplot.flierprops.markersize'] = 10
pl.rcParams['boxplot.flierprops.markeredgewidth'] = 4
pl.rcParams['boxplot.boxprops.linewidth'] = 4
pl.rcParams['boxplot.boxprops.color'] = 'black'
pl.rcParams['boxplot.medianprops.linewidth'] = 2
pl.rcParams['boxplot.whiskerprops.linewidth'] = 4
pl.rcParams['boxplot.capprops.linewidth'] = 2
pl.rcParams['boxplot.capprops.color'] = 'black'
pl.rcParams['boxplot.whiskerprops.color'] = 'black'
pl.rcParams['boxplot.medianprops.color'] = 'black'
pl.figure(figsize=(10,10))
for i in range(mc_points):
    pl.plot(ds.dbv(amp)-5, snr_all_mc[:, i])

amp_cadence = np.concatenate((np.ones(17)*1.58, np.ones(8)*2.92))-5
sndr_cadence = np.array((63,67.4,66.8,65.2,67.2,67,67.1,67.6,67.9,67.2,66.9,67.2,66.1,66.6,64.8,66.4,66.7,66.1,70.3,52.2,62.3,67.2,64.5,63.5,68.8))

amp_cadence_1_5 = (np.ones(17)*1.58)-5
sndr_cadence_1_5 = np.array((63,67.4,66.8,65.2,67.2,67,67.1,67.6,67.9,67.2,66.9,67.2,66.1,66.6,64.8,66.4,66.7))

amp_cadence_2_5 = (np.ones(8)*2.92)-5
sndr_cadence_2_9 = np.array((66.1,70.3,52.2,62.3,67.2,64.5,63.5,68.8))

# pl.boxplot(sndr_cadence_1_5, manage_ticks=False, showmeans=False, positions=[1.58-5], patch_artist=True)
# pl.boxplot(sndr_cadence_2_9, manage_ticks=False, showmeans=False, positions=[2.92-5], patch_artist=True)

pl.grid(True)
ax = pl.gca()
ax.set_autoscale_on(False)
ax.set_aspect('auto', 'box')

xRange = [-5, 0]
dx = 1
yRange = [52,72]
dy = 2

ax.set_xlim(xRange)
ax.set_ylim(yRange)
x1, x2 = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(x1, x2, dx))
y1, y2 = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(y1, y2, dy))
pl.title('SNDR versus Input Voltage')
pl.xlabel('Input Voltage (dBFS)')
pl.ylabel('SNDR (dB)')
pl.plot()


# Histogram plot
pl.figure(figsize=(10,5))
pl.hist(snr_amp_mc,20)
pl.title('Peak SNDR histogram')
pl.xlabel('SNDR (dB)')
pl.ylabel('Absolute frequency (#)')
pl.plot()

pl.rcParams['lines.markersize'] = 15
pl.rcParams['lines.linewidth'] = 5
## Sorted plot
sndr_sort = np.sort(snr_amp_mc,axis=0)
pl.figure(figsize=(10,10))
pl.plot(np.arange(1,mc_points+1), sndr_sort)
pl.plot(np.array([0, mc_points+1]), np.array([pl.mean(snr_amp_mc),pl.mean(snr_amp_mc)]))
pl.text(mc_points*0.01+1, pl.mean(snr_amp_mc)+0.5,'Average SNDR for %d points = %2.1f dB' %(mc_points,pl.mean(snr_amp_mc)))
pl.scatter(mc_points*0.01+1,sndr_sort[int(mc_points*0.01)])
pl.text(mc_points*0.05, sndr_sort[int(mc_points*0.01)],'Peak SNDR at 99%% confidence = %2.1f dB' %(sndr_sort[int(mc_points*0.01)]))

pl.grid(True)
ax = pl.gca()
ax.set_autoscale_on(False)
ax.set_aspect('auto', 'box')

xRange = [0, mc_points]
dx = mc_points/10
yRange = [60,75]
dy = 1

ax.set_xlim(xRange)
ax.set_ylim(yRange)
x1, x2 = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(x1, x2, dx))
y1, y2 = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(y1, y2, dy))

pl.title('Sorted Monte Carlo plot of peak SNDR', )
pl.xlabel('Sorted Occurence (#)')
pl.ylabel('Peak SNDR (dB)')
# pl.legend(loc=4)
pl.show()

