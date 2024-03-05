# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:44:51 2019

@author: mouras54
"""

from __future__ import division
import deltasigma as ds
import numpy as np
import pylab as pl
import pandas as pd

order = 3
OSR = 10
nlev = 8        ## 3-bit quantizer
form = 'CIFF'   ## FF better for highly linear OpA 
scaled = True

vdd = 0.9
delta = 2 #default value
analog_scale = vdd/(delta*(nlev-1))
digital_scale = 1.0/analog_scale

if form == 'CIFF':
    if scaled == True:
        ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.64967266, -1.64967266],
                          [ 1.08160444,  1.        , -0.07017017,  0.        ,  0.        ],
                          [ 0.        ,  0.82738729,  1.        ,  0.        ,  0.        ],
                          [ 1.73628057,  1.52821342,  0.50440738,  1.        ,  0.        ]])
    else:
        ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.        , -1.        ],
                         [ 1.        ,  1.        , -0.05805791,  0.        ,  0.        ],
                         [ 0.        ,  1.        ,  1.        ,  0.        ,  0.        ],
                         [ 2.86429459,  2.72678094,  0.74465741,  1.        ,  0.        ]])

elif form == 'CRFF':
    if scaled == True:
        ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.6314776 , -1.6314776 ],
                          [ 1.19337803,  1.        , -0.07934089,  0.        ,  0.        ],
                          [ 0.88631426,  0.74269363,  0.94107402,  0.        ,  0.        ],
                          [ 1.71952629,  0.9321977 ,  0.51565859,  1.        ,  0.        ]])
    else:
        ABCDs = np.array([[ 1.        ,  0.        ,  0.        ,  1.        , -1.        ],
                         [ 1.        ,  1.        , -0.05892598,  0.        ,  0.        ],
                         [ 1.        ,  1.        ,  0.94107402,  0.        ,  0.        ],
                         [ 2.80536862,  1.81496051,  0.74564328,  1.        ,  0.        ]])
    
else:
    print("Inavlid form -- exit")
    exit()

ntot = 100
ntfs, _ = ds.calculateTF(ABCDs)
impulse_response_ideal = ds.impL1(ntfs, ntot)

finite_dc_gain = True
av = ds.undbv(40) # 60db
if finite_dc_gain == True:    
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
    ABCDs_real = abcd
    
ntfs_real, _ = ds.calculateTF(ABCDs_real)
impulse_response_real = ds.impL1(ntfs_real, ntot)


t = np.linspace(0, ntot, ntot+1)


pl.figure(figsize=(15,10))
ds.lollipop(t, impulse_response_ideal, color = 'r')
ds.lollipop(t, impulse_response_real, color = 'b')

#df = pd.read_csv('data/CIFF_impulse_resp_OTAgm60dB_100MHz.csv')
#df = pd.read_csv('data/CIFF_impulse_resp_OTAgm40dB_100MHz.csv')
#df = pd.read_csv('data/CIFF_impulse_resp_OTAgm40dB_100MHz_w_restoration.csv')
df = pd.read_csv('data/CIFF_split_dac_impulse_resp_OTAgm40dB_100MHz_w_restoration.csv')
#df = pd.read_csv('data/CIFF_split_dac_impulse_resp_OTAgm40dB_100MHz.csv')

## Ideal DAC scalling for a 250uV input at the DAC
#pl.plot(t[0:99], df.values[0:99,1]/1.0, color = 'k', linewidth = 2.0)

## Split DAC scalling for a 1bit input at the DAC
pl.plot(t[0:99], df.values[0:99,1]/(900e-3/7)/2, color = 'k', linewidth = 2.0)
pl.grid(True)
pl.show()

#abcd_1st = np.array([[1.0*av/(1+av), 1, -1],
#                     [1.0, 0.0, 0.0]])
#ntf1st, _ = ds.calculateTF(abcd_1st)
#
#print(ds.pretty_lti(ntf1st))
#impulse_1st = ds.impL1(ntf1st, ntot)
#
#ds.lollipop(t, impulse_1st)
#pl.grid(True)
#pl.show()

