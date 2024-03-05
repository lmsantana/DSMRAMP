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


"""
## Modulator parameters

order is choosen to increase the OTA/RingAmp power tradeoff
OSR is chossen to achieve BW of 100MHz
nlev is chossen to reduce noise floor and increase total SQNR
form is still open to discussion
"""
order = 3
OSR = 10
nlev = 15        ## 3-bit quantizer
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

#########################
# Non ideal parameters setting
#########################
finite_dc_gain = True
dc_gaindB = 40
dc_gainlin = ds.undbv(dc_gaindB)
##########################

re_scale = 2.0
ABCDs = np.array([[ 1.*dc_gainlin/(1+dc_gainlin)   ,  0.                            ,  0.                           ,  1.154*re_scale     , -1.154*re_scale],
                  [ 1.081                          ,  1.*dc_gainlin/(1+dc_gainlin)  , -0.0701                       ,  0.                 ,  0.   ],
                  [ 0.                             ,  0.827                         ,  1.*dc_gainlin/(1+dc_gainlin) ,  0.                 ,  0.   ],
                  [ 2.48/re_scale                  ,  2.182/re_scale                ,  0.714/re_scale               ,  1.                 ,  0.   ]])



ntf,_ = ds.calculateTF(ABCDs)

ds.DocumentNTF(ntf,OSR)
f1, f2 = ds.ds_f1f2(OSR, 0, 0)
NG0 = ds.dbv(ds.rmsGain(ntf, f1, f2))
NG0_lin = ds.undbv(NG0)
NG0_lin_squared = NG0_lin**2

NTF_gain = ds.rmsGain(ntf, 0, 0.5)**2

