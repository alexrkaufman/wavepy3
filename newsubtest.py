# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:55:00 2016

@author: jpbos
"""

# file to test improved subharmonic method

import numpy as np
import wavepy as wp

from math import gamma, pi

import matplotlib as plt

test = wp.wavepy()

test.r0scrn = 0.01
test.N = 1024
test.SideLen = 1
test.L0 = 100;
test.l0 = 0.001;
test.dx = test.SideLen/test.N;
nsub = 9;

dq = 1/test.SideLen
na = test.alpha/6.0
        
Bnum = gamma(na/2.0)
Bdenom = (2**(2-na)) * pi * na * gamma(-na/2)
Bfac = (2*pi)**(2-na) * (Bnum/Bdenom)
        
# c1 Striblings Consistency parameter. Evaluates to 6.88 in Kolmogorov turb.
cone = (2* (8/(na-2) * gamma(2/(na-2)))**((na-2)/2))        
        
#Anisotropy factors
b = test.aniso
c=1
f0 = 1/test.L0
lof_phz = np.zeros((test.N,test.N))

a = test.N/2

temp_m = np.linspace(-0.5,0.5,test.N)

m_indices, n_indices = np.meshgrid(temp_m, -1*np.transpose(temp_m))

temp_mp = np.linspace(-2.5,2.5,6)

m_prime_indices,n_prime_indices = np.meshgrid(temp_mp,-1*np.transpose(temp_mp));


for Np in range(1,nsub+1):
    
    temp_phz = np.zeros((test.N,test.N))
    #Subharmonic frequency
    dqp = dq/(3.0**Np)
    #Set samples
    
    f_x = 3**(-Np)*m_prime_indices*dq;
    f_y = 3**(-Np)*n_prime_indices*dq;
    
    f = np.sqrt((f_x**2)/(b**2) + (f_y**2)/(c**2))
    #Sample PSD
    PSD_fi = cone*Bfac*((b*c)**(-na/2))*(test.r0scrn)**(2-na)*(f**2 + f0**2)**(-na/2)
    PSD_fi[3,3] = 0;
    
    #Generate normal circ complex RV
    w = np.random.randn(6,6) + 1j*np.random.randn(6,6)
    #Covariances
    cv = w * np.sqrt(PSD_fi)*dqp
    #Sum over subharmonic components
    temp_shape = np.shape(cv)
    for n in range(0, temp_shape[0]):
        for m in range(0,temp_shape[1]):
            
            indexMap =  ( m_prime_indices[n][m]*m_indices + 
            n_prime_indices[n][m]*n_indices )
            
            
            
            temp_phz = temp_phz + cv[m][n] * np.exp(1j*2*pi*(3**(-Np))*indexMap)

    
    #Accumulate components to phase screen
    lof_phz = lof_phz + temp_phz

lof_phz = np.real(lof_phz) - np.mean(np.real(lof_phz))
 
