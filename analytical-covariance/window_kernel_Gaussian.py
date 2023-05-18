# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 09:53:38 2021

@author: Yanxiang Lai
"""

import numpy as np
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt, amin, amax, mean, dot, power, conj
import sys
import time
from configobj import ConfigObj
import time
from numba import njit
import itertools as itt

# @njit()
#     The last dim corresponds to multipoles: [L0xL0,L2xL2,L4xL4,L2xL0,L4xL0,L4xL2]
def WinFun(Nbin):
    """
    Returns an array with [7,15,6] dimensions. 
    The first dim corresponds to the k-bin of k2 
    (only 3 bins on each side of diagonal are included as the Gaussian covariance drops quickly away from diagonal)

    The second dim corresponds to elements to be multiplied by various power spectrum multipoles
    to obtain the final covariance (see function 'Wij' below)

    The last dim corresponds to multipoles: [L0xL0,L2xL2,L4xL4,L2xL0,L4xL0,L4xL2]

    Parameters
    ----------
    Nbin : int.
        The total number of k bins.
    icut_x : float
        The index of maximum kx.
    icut_y : float
        The index of maximum ky.
    icut_z : float
        The index of maximum kz.
    Bin_ModeNum : 1d numpy array of int.
        This array contains the number of k-modes inside each k-bin.
    Bin_kmodes : 4d numpy array of float. 
        This array contains kx, ky, kz and the length of k vectors for each bin.
    kfun_x : float.
        The fundamental modes in the x direction.
    kfun_y : float
        The fundamental modes in the y direction.
    kfun_z : float
        The fundamental modes in the z direction.
    kBinWidth : float
        The width of the k-bin.
    all the W functions: 3d numpy array of complex.
        They are the 22 different window functions which are used to calculate the Gaussian part of the covariance matrix.
    I22 : float
        The sum of W22. This is a normalization factor.

    Returns
    -------
    avgWij: numpy array.
        This is the array requires to calculate the Gaussian part of the covariance matrix. 

    """
    
    #Initialise the output arrays.
    avgWij=np.zeros((2*3+1,15,6)); avgW00=np.zeros((2*3+1,15),dtype=np.complex128);
    avgW22=avgW00.copy(); avgW44=avgW00.copy(); avgW20=avgW00.copy(); avgW40=avgW00.copy(); avgW42=avgW00.copy()
    
    #this is the number of grid cells in each direction. 
    size_x = 2*icut_x + 1
    size_y = 2*icut_y + 1
    size_z = 2*icut_z + 1
    
    [ix,iy,iz,k2xh,k2yh,k2zh]=np.zeros((6, size_x, size_y, size_z))
    # [ix,iy,iz,k2xh,k2yh,k2zh]=np.zeros((6,2*icut+1,2*icut+1,2*icut+1))
    
    #These arrays will store all possible different k-bins. 
    for i in range(size_x):
        ix[i,:,:] += i - icut_x
    
    for i in range(size_y):
        iy[:,i,:] += i - icut_y
        
    for i in range(size_z):
        iz[:,:,i] += i - icut_z
    
    #If the number of kmodes being sampled is less than the number of kmodes in the bin, we will use the Monte-Carlo integration to calculate the analytical covariance matrix.
    #Otherwise, we will smapled each kmode in the bin once. 
    if (kmodes_sampled<Bin_ModeNum[Nbin]):
        norm=kmodes_sampled
        sampled=(np.random.rand(kmodes_sampled)*Bin_ModeNum[Nbin]).astype(np.int32)
    else:
        norm=Bin_ModeNum[Nbin]
        sampled=np.arange(Bin_ModeNum[Nbin],dtype=np.int32)
    
    # Randomly select a mode in the k1 bin
    for n in sampled:
        #First, we need to calculate the unit vector of k1 and k2. 
        [ik1x,ik1y,ik1z,rk1]=Bin_kmodes[Nbin][n]
        if (rk1 <= 10e-10): k1xh=0; k1yh=0; k1zh=0
        else: k1xh=ik1x*kfun_x/rk1; k1yh=ik1y*kfun_y/rk1; k1zh=ik1z*kfun_z/rk1
        
        # print(k1xh, k1yh,k1zh)
            
    # Build a 3D array of modes around the selected mode   
        k2xh=ik1x-ix; k2yh=ik1y-iy; k2zh=ik1z-iz;
        rk2 = np.sqrt((k2xh*kfun_x)**2 + (k2yh*kfun_y)**2 + (k2zh*kfun_z)**2)
        
        sort=(rk2/kBinWidth).astype(np.int32)-Nbin # to decide later which shell the k2 mode belongs to
        ind=(rk2 <= 10e-10);
        if (ind.any()>0): rk2[ind]=1e10
        
        k2xh= k2xh*kfun_x/rk2; k2yh=k2yh*kfun_y/rk2; k2zh=k2zh*kfun_z/rk2;
        #k2 hat arrays built
        
    # Now calculating window multipole kernels by taking dot products of cartesian FFTs with k1-hat, k2-hat arrays
    # W corresponds to W22(k) and Wc corresponds to conjugate of W22(k)
    # L(i) refers to multipoles
        
        W_L0 = W
        Wc_L0 = conj(W)
        
        xx=Wxx*k1xh**2+Wyy*k1yh**2+Wzz*k1zh**2+2.*Wxy*k1xh*k1yh+2.*Wyz*k1yh*k1zh+2.*Wxz*k1zh*k1xh
        
        W_k1L2=1.5*xx-0.5*W
        W_k2L2=1.5*(Wxx*k2xh**2+Wyy*k2yh**2+Wzz*k2zh**2 \
        +2.*Wxy*k2xh*k2yh+2.*Wyz*k2yh*k2zh+2.*Wxz*k2zh*k2xh)-0.5*W
        Wc_k1L2=conj(W_k1L2)
        Wc_k2L2=conj(W_k2L2)
        
        W_k1L4=35./8.*(Wxxxx*k1xh**4 +Wyyyy*k1yh**4+Wzzzz*k1zh**4 \
     +4.*Wxxxy*k1xh**3*k1yh +4.*Wxxxz*k1xh**3*k1zh +4.*Wxyyy*k1yh**3*k1xh \
     +4.*Wyyyz*k1yh**3*k1zh +4.*Wxzzz*k1zh**3*k1xh +4.*Wyzzz*k1zh**3*k1yh \
     +6.*Wxxyy*k1xh**2*k1yh**2+6.*Wxxzz*k1xh**2*k1zh**2+6.*Wyyzz*k1yh**2*k1zh**2 \
     +12.*Wxxyz*k1xh**2*k1yh*k1zh+12.*Wxyyz*k1yh**2*k1xh*k1zh +12.*Wxyzz*k1zh**2*k1xh*k1yh) \
     -5./2.*W_k1L2 -7./8.*W_L0
        Wc_k1L4=conj(W_k1L4)
        
        k1k2=Wxxxx*(k1xh*k2xh)**2+Wyyyy*(k1yh*k2yh)**2+Wzzzz*(k1zh*k2zh)**2 \
            +Wxxxy*(k1xh*k1yh*k2xh**2+k1xh**2*k2xh*k2yh)*2\
            +Wxxxz*(k1xh*k1zh*k2xh**2+k1xh**2*k2xh*k2zh)*2\
            +Wyyyz*(k1yh*k1zh*k2yh**2+k1yh**2*k2yh*k2zh)*2\
            +Wyzzz*(k1zh*k1yh*k2zh**2+k1zh**2*k2zh*k2yh)*2\
            +Wxyyy*(k1yh*k1xh*k2yh**2+k1yh**2*k2yh*k2xh)*2\
            +Wxzzz*(k1zh*k1xh*k2zh**2+k1zh**2*k2zh*k2xh)*2\
            +Wxxyy*(k1xh**2*k2yh**2+k1yh**2*k2xh**2+4.*k1xh*k1yh*k2xh*k2yh)\
            +Wxxzz*(k1xh**2*k2zh**2+k1zh**2*k2xh**2+4.*k1xh*k1zh*k2xh*k2zh)\
            +Wyyzz*(k1yh**2*k2zh**2+k1zh**2*k2yh**2+4.*k1yh*k1zh*k2yh*k2zh)\
            +Wxyyz*(k1xh*k1zh*k2yh**2+k1yh**2*k2xh*k2zh+2.*k1yh*k2yh*(k1zh*k2xh+k1xh*k2zh))*2\
            +Wxxyz*(k1yh*k1zh*k2xh**2+k1xh**2*k2yh*k2zh+2.*k1xh*k2xh*(k1zh*k2yh+k1yh*k2zh))*2\
            +Wxyzz*(k1yh*k1xh*k2zh**2+k1zh**2*k2yh*k2xh+2.*k1zh*k2zh*(k1xh*k2yh+k1yh*k2xh))*2
        
        W_k2L4=35./8.*(Wxxxx*k2xh**4 +Wyyyy*k2yh**4+Wzzzz*k2zh**4 \
     +4.*Wxxxy*k2xh**3*k2yh +4.*Wxxxz*k2xh**3*k2zh +4.*Wxyyy*k2yh**3*k2xh \
     +4.*Wyyyz*k2yh**3*k2zh +4.*Wxzzz*k2zh**3*k2xh +4.*Wyzzz*k2zh**3*k2yh \
     +6.*Wxxyy*k2xh**2*k2yh**2+6.*Wxxzz*k2xh**2*k2zh**2+6.*Wyyzz*k2yh**2*k2zh**2 \
     +12.*Wxxyz*k2xh**2*k2yh*k2zh+12.*Wxyyz*k2yh**2*k2xh*k2zh +12.*Wxyzz*k2zh**2*k2xh*k2yh) \
     -5./2.*W_k2L2 -7./8.*W_L0
        Wc_k2L4=conj(W_k2L4)
        
        W_k1L2_k2L2= 9./4.*k1k2 -3./4.*xx -1./2.*W_k2L2
        W_k1L2_k2L4=2/7.*W_k1L2+20/77.*W_k1L4 #approximate as 6th order FFTs not simulated
        W_k1L4_k2L2=W_k1L2_k2L4 #approximate
        W_k1L4_k2L4=1/9.*W_L0+100/693.*W_k1L2+162/1001.*W_k1L4
        Wc_k1L2_k2L2= conj(W_k1L2_k2L2)
        Wc_k1L2_k2L4=conj(W_k1L2_k2L4); Wc_k1L4_k2L2=Wc_k1L2_k2L4
        Wc_k1L4_k2L4=conj(W_k1L4_k2L4)
        
        k1k2W12=W12xxxx*(k1xh*k2xh)**2+W12yyyy*(k1yh*k2yh)**2+W12zzzz*(k1zh*k2zh)**2 \
            +W12xxxy*(k1xh*k1yh*k2xh**2+k1xh**2*k2xh*k2yh)*2\
            +W12xxxz*(k1xh*k1zh*k2xh**2+k1xh**2*k2xh*k2zh)*2\
            +W12yyyz*(k1yh*k1zh*k2yh**2+k1yh**2*k2yh*k2zh)*2\
            +W12yzzz*(k1zh*k1yh*k2zh**2+k1zh**2*k2zh*k2yh)*2\
            +W12xyyy*(k1yh*k1xh*k2yh**2+k1yh**2*k2yh*k2xh)*2\
            +W12xzzz*(k1zh*k1xh*k2zh**2+k1zh**2*k2zh*k2xh)*2\
            +W12xxyy*(k1xh**2*k2yh**2+k1yh**2*k2xh**2+4.*k1xh*k1yh*k2xh*k2yh)\
            +W12xxzz*(k1xh**2*k2zh**2+k1zh**2*k2xh**2+4.*k1xh*k1zh*k2xh*k2zh)\
            +W12yyzz*(k1yh**2*k2zh**2+k1zh**2*k2yh**2+4.*k1yh*k1zh*k2yh*k2zh)\
            +W12xyyz*(k1xh*k1zh*k2yh**2+k1yh**2*k2xh*k2zh+2.*k1yh*k2yh*(k1zh*k2xh+k1xh*k2zh))*2\
            +W12xxyz*(k1yh*k1zh*k2xh**2+k1xh**2*k2yh*k2zh+2.*k1xh*k2xh*(k1zh*k2yh+k1yh*k2zh))*2\
            +W12xyzz*(k1yh*k1xh*k2zh**2+k1zh**2*k2yh*k2xh+2.*k1zh*k2zh*(k1xh*k2yh+k1yh*k2xh))*2
        
        xxW12=W12xx*k1xh**2+W12yy*k1yh**2+W12zz*k1zh**2+2.*W12xy*k1xh*k1yh+2.*W12yz*k1yh*k1zh+2.*W12xz*k1zh*k1xh
    
        W12_L0 = W12
        W12_k1L2=1.5*xxW12-0.5*W12
        W12_k1L4=35./8.*(W12xxxx*k1xh**4 +W12yyyy*k1yh**4+W12zzzz*k1zh**4 \
     +4.*W12xxxy*k1xh**3*k1yh +4.*W12xxxz*k1xh**3*k1zh +4.*W12xyyy*k1yh**3*k1xh \
     +6.*W12xxyy*k1xh**2*k1yh**2+6.*W12xxzz*k1xh**2*k1zh**2+6.*W12yyzz*k1yh**2*k1zh**2 \
     +12.*W12xxyz*k1xh**2*k1yh*k1zh+12.*W12xyyz*k1yh**2*k1xh*k1zh +12.*W12xyzz*k1zh**2*k1xh*k1yh) \
     -5./2.*W12_k1L2 -7./8.*W12_L0
        W12_k1L4_k2L2=2/7.*W12_k1L2+20/77.*W12_k1L4
        W12_k1L4_k2L4=1/9.*W12_L0+100/693.*W12_k1L2+162/1001.*W12_k1L4
        W12_k2L2=1.5*(W12xx*k2xh**2+W12yy*k2yh**2+W12zz*k2zh**2\
        +2.*W12xy*k2xh*k2yh+2.*W12yz*k2yh*k2zh+2.*W12xz*k2zh*k2xh)-0.5*W12
        W12_k2L4=35./8.*(W12xxxx*k2xh**4 +W12yyyy*k2yh**4+W12zzzz*k2zh**4 \
     +4.*W12xxxy*k2xh**3*k2yh +4.*W12xxxz*k2xh**3*k2zh +4.*W12xyyy*k2yh**3*k2xh \
     +4.*W12yyyz*k2yh**3*k2zh +4.*W12xzzz*k2zh**3*k2xh +4.*W12yzzz*k2zh**3*k2yh \
     +6.*W12xxyy*k2xh**2*k2yh**2+6.*W12xxzz*k2xh**2*k2zh**2+6.*W12yyzz*k2yh**2*k2zh**2 \
     +12.*W12xxyz*k2xh**2*k2yh*k2zh+12.*W12xyyz*k2yh**2*k2xh*k2zh +12.*W12xyzz*k2zh**2*k2xh*k2yh) \
     -5./2.*W12_k2L2 -7./8.*W12_L0
        
        W12_k1L2_k2L2= 9./4.*k1k2W12 -3./4.*xxW12 -1./2.*W12_k2L2
        
        W_k1L2_Sumk2L22=1/5.*W_k1L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L2_k2L4
        W_k1L2_Sumk2L24=2/7.*W_k1L2_k2L2+20/77.*W_k1L2_k2L4
        W_k1L4_Sumk2L22=1/5.*W_k1L4+2/7.*W_k1L4_k2L2+18/35.*W_k1L4_k2L4
        W_k1L4_Sumk2L24=2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4
        W_k1L4_Sumk2L44=1/9.*W_k1L4+100/693.*W_k1L4_k2L2+162/1001.*W_k1L4_k2L4
        
        C00exp = [Wc_L0*W_L0,Wc_L0*W_k2L2,Wc_L0*W_k2L4,\
                Wc_k1L2*W_L0,Wc_k1L2*W_k2L2,Wc_k1L2*W_k2L4,\
                Wc_k1L4*W_L0,Wc_k1L4*W_k2L2,Wc_k1L4*W_k2L4]
        
        C00exp += [2.*W_L0*W12_L0,W_k1L2*W12_L0,W_k1L4*W12_L0,\
                W_k2L2*W12_L0,W_k2L4*W12_L0,conj(W12_L0)*W12_L0]
        
        C22exp = [Wc_k2L2*W_k1L2 + Wc_L0*W_k1L2_k2L2,\
                Wc_k2L2*W_k1L2_k2L2 + Wc_L0*W_k1L2_Sumk2L22,\
                Wc_k2L2*W_k1L2_k2L4 + Wc_L0*W_k1L2_Sumk2L24,\
                Wc_k1L2_k2L2*W_k1L2 + Wc_k1L2*W_k1L2_k2L2,\
                Wc_k1L2_k2L2*W_k1L2_k2L2 + Wc_k1L2*W_k1L2_Sumk2L22,\
                Wc_k1L2_k2L2*W_k1L2_k2L4 + Wc_k1L2*W_k1L2_Sumk2L24,\
                Wc_k1L4_k2L2*W_k1L2 + Wc_k1L4*W_k1L2_k2L2,\
                Wc_k1L4_k2L2*W_k1L2_k2L2 + Wc_k1L4*W_k1L2_Sumk2L22,\
                Wc_k1L4_k2L2*W_k1L2_k2L4 + Wc_k1L4*W_k1L2_Sumk2L24]
        
        C22exp += [W_k1L2*W12_k2L2 + W_k2L2*W12_k1L2\
                   +W_k1L2_k2L2*W12_L0+W_L0*W12_k1L2_k2L2,\
                 0.5*((1/5.*W_L0+2/7.*W_k1L2+18/35.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L2\
+(1/5.*W_k2L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L2_k2L2),\
    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L2\
+(2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L2_k2L2),\
0.5*(W_k1L2_k2L2*W12_k2L2+(1/5.*W_L0+2/7.*W_k2L2+18/35.*W_k2L4)*W12_k1L2\
+(1/5.*W_k1L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L2_k2L4)*W12_L0 + W_k2L2*W12_k1L2_k2L2),\
0.5*(W_k1L2_k2L4*W12_k2L2+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L2\
+(2/7.*W_k1L2_k2L2+20/77.*W_k1L2_k2L4)*W12_L0 + W_k2L4*W12_k1L2_k2L2),\
                 conj(W12_k1L2_k2L2)*W12_L0+conj(W12_k1L2)*W12_k2L2]
        
        C44exp = [Wc_k2L4*W_k1L4 + Wc_L0*W_k1L4_k2L4,\
                Wc_k2L4*W_k1L4_k2L2 + Wc_L0*W_k1L4_Sumk2L24,\
                Wc_k2L4*W_k1L4_k2L4 + Wc_L0*W_k1L4_Sumk2L44,\
                Wc_k1L2_k2L4*W_k1L4 + Wc_k1L2*W_k1L4_k2L4,\
                Wc_k1L2_k2L4*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L24,\
                Wc_k1L2_k2L4*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L44,\
                Wc_k1L4_k2L4*W_k1L4 + Wc_k1L4*W_k1L4_k2L4,\
                Wc_k1L4_k2L4*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L24,\
                Wc_k1L4_k2L4*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L44]
        
        C44exp += [W_k1L4*W12_k2L4 + W_k2L4*W12_k1L4\
                   +W_k1L4_k2L4*W12_L0+W_L0*W12_k1L4_k2L4,\
                 0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L4 + W_k1L2_k2L4*W12_k1L4\
+(2/7.*W_k1L2_k2L4+20/77.*W_k1L4_k2L4)*W12_L0 + W_k1L2*W12_k1L4_k2L4),\
0.5*((1/9.*W_L0+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_k2L4 + W_k1L4_k2L4*W12_k1L4\
+(1/9.*W_k2L4+100/693.*W_k1L2_k2L4+162/1001.*W_k1L4_k2L4)*W12_L0 + W_k1L4*W12_k1L4_k2L4),\
0.5*(W_k1L4_k2L2*W12_k2L4+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L4\
+(2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4)*W12_L0 + W_k2L2*W12_k1L4_k2L4),\
0.5*(W_k1L4_k2L4*W12_k2L4+(1/9.*W_L0+100/693.*W_k2L2+162/1001.*W_k2L4)*W12_k1L4\
+(1/9.*W_k1L4+100/693.*W_k1L4_k2L2+162/1001.*W_k1L4_k2L4)*W12_L0 + W_k2L4*W12_k1L4_k2L4),\
                 conj(W12_k1L4_k2L4)*W12_L0+conj(W12_k1L4)*W12_k2L4] #1/(nbar)^2
        
        C20exp = [Wc_L0*W_k1L2,Wc_L0*W_k1L2_k2L2,Wc_L0*W_k1L2_k2L4,\
                Wc_k1L2*W_k1L2,Wc_k1L2*W_k1L2_k2L2,Wc_k1L2*W_k1L2_k2L4,\
                Wc_k1L4*W_k1L2,Wc_k1L4*W_k1L2_k2L2,Wc_k1L4*W_k1L2_k2L4]
        
        C20exp += [W_k1L2*W12_L0 + W*W12_k1L2,\
                 0.5*((1/5.*W+2/7.*W_k1L2+18/35.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L2),\
                0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L2),\
                0.5*(W_k1L2_k2L2*W12_L0 + W_k2L2*W12_k1L2),\
                 0.5*(W_k1L2_k2L4*W12_L0 + W_k2L4*W12_k1L2),\
                 np.conj(W12_k1L2)*W12_L0]
        
        C40exp = [Wc_L0*W_k1L4,Wc_L0*W_k1L4_k2L2,Wc_L0*W_k1L4_k2L4,\
                Wc_k1L2*W_k1L4,Wc_k1L2*W_k1L4_k2L2,Wc_k1L2*W_k1L4_k2L4,\
                Wc_k1L4*W_k1L4,Wc_k1L4*W_k1L4_k2L2,Wc_k1L4*W_k1L4_k2L4]
        
        C40exp += [W_k1L4*W12_L0 + W*W12_k1L4,\
                 0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L4),\
                0.5*((1/9.*W+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L4),\
                0.5*(W_k1L4_k2L2*W12_L0 + W_k2L2*W12_k1L4),\
                 0.5*(W_k1L4_k2L4*W12_L0 + W_k2L4*W12_k1L4),\
                 np.conj(W12_k1L4)*W12_L0]
        
        C42exp = [Wc_k2L2*W_k1L4 + Wc_L0*W_k1L4_k2L2,\
                Wc_k2L2*W_k1L4_k2L2 + Wc_L0*W_k1L4_Sumk2L22,\
                Wc_k2L2*W_k1L4_k2L4 + Wc_L0*W_k1L4_Sumk2L24,\
                Wc_k1L2_k2L2*W_k1L4 + Wc_k1L2*W_k1L4_k2L2,\
                Wc_k1L2_k2L2*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L22,\
                Wc_k1L2_k2L2*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L24,\
                Wc_k1L4_k2L2*W_k1L4 + Wc_k1L4*W_k1L4_k2L2,\
                Wc_k1L4_k2L2*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L22,\
                Wc_k1L4_k2L2*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L24]
        
        C42exp += [W_k1L4*W12_k2L2 + W_k2L2*W12_k1L4+\
                   W_k1L4_k2L2*W12_L0+W*W12_k1L4_k2L2,\
                 0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L4\
    +(2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L4_k2L2),\
    0.5*((1/9.*W+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L4\
+(1/9.*W_k2L2+100/693.*W_k1L2_k2L2+162/1001.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L4_k2L2),\
0.5*(W_k1L4_k2L2*W12_k2L2+(1/5.*W+2/7.*W_k2L2+18/35.*W_k2L4)*W12_k1L4\
+(1/5.*W_k1L4+2/7.*W_k1L4_k2L2+18/35.*W_k1L4_k2L4)*W12_L0 + W_k2L2*W12_k1L4_k2L2),\
0.5*(W_k1L4_k2L4*W12_k2L2+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L4\
+(2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4)*W12_L0 + W_k2L4*W12_k1L4_k2L2),\
                 conj(W12_k1L4_k2L2)*W12_L0+conj(W12_k1L4)*W12_k2L2] #1/(nbar)^2
        
        for i in range(-3,4):
            ind=(sort==i);
            for j in range(15):
                avgW00[i+3,j]+=np.sum(C00exp[j][ind])
                avgW22[i+3,j]+=np.sum(C22exp[j][ind])
                avgW44[i+3,j]+=np.sum(C44exp[j][ind])
                avgW20[i+3,j]+=np.sum(C20exp[j][ind])
                avgW40[i+3,j]+=np.sum(C40exp[j][ind])
                avgW42[i+3,j]+=np.sum(C42exp[j][ind])
            
    for i in range(0,2*3+1):
        if(i+Nbin-3>=nBins or i+Nbin-3<0): 
            avgW00[i]*=0; avgW22[i]*=0; avgW44[i]*=0;
            avgW20[i]*=0; avgW40[i]*=0; avgW42[i]*=0; continue
        avgW00[i]=avgW00[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
        avgW22[i]=avgW22[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
        avgW44[i]=avgW44[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
        avgW20[i]=avgW20[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
        avgW40[i]=avgW40[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
        avgW42[i]=avgW42[i]/(norm*Bin_ModeNum[Nbin+i-3]*I22**2)
        
    avgWij[:,:,0]=2.*np.real(avgW00); avgWij[:,:,1]=25.*np.real(avgW22); avgWij[:,:,2]=81.*np.real(avgW44);
    avgWij[:,:,3]=5.*2.*np.real(avgW20); avgWij[:,:,4]=9.*2.*np.real(avgW40); avgWij[:,:,5]=45.*np.real(avgW42);
    return(avgWij)
        
def fft(temp, W22_flag):
    """
    As the window falls steeply with k, only low-k regions are needed for the calculation.
    Therefore cutting out the high-k modes in the FFTs using the icut parameter. This will also reduce the memory requires to run the code. 

    Parameters
    ----------
    temp : 3d cpmplex numpy array
        The input window function.
    W22_flag : int or bool
        For W_22 window function, return the original form. For W12, return the complex conjugate. 

    Returns
    -------
    temp: numpy array.
        W_22 or the complex conjugate of W12. 

    """
    
    
    
    NX, NY, NZ = np.shape(temp)
    
    ia_x = np.int32((NX-1)/2)
    ia_y = np.int32((NY-1)/2)
    ia_z = np.int32((NZ-1)/2)
    
    # print(temp[ia_x][ia_y][ia_z])
    
    temp=temp[ia_x-icut_x:ia_x+icut_x+1,ia_y-icut_y:ia_y+icut_y+1,ia_z-icut_z:ia_z+icut_z+1]
    
    if(W22_flag): return(temp)
    else: return(conj(temp))

def input_variable(input_file):
    pardict = ConfigObj(input_file)
    pardict['omega_m'] = np.float32(pardict['omega_m'])
    pardict['hem_num'] = np.int32(pardict['hem_num'])
    pardict['red_num'] = np.int32(pardict['red_num'])
    pardict['k_max'] = np.float32(pardict['k_max'])
    pardict['FKP'] = np.float32(pardict['FKP'])
    pardict['icut_kmode'] = np.float32(pardict['icut_kmode'])
    pardict['k_bin'] = np.int32(pardict['k_bin'])
    pardict['alpha'] = np.float32(pardict['alpha'])
    pardict['ln_10_10_As'] = np.float32(pardict['ln_10_10_As'])
    pardict['h'] = np.float32(pardict['h'])
    pardict['omega_b'] = np.float32(pardict['omega_b'])
    pardict['omega_cdm'] = np.float32(pardict['omega_cdm'])
    pardict['L_x'] = np.array(pardict['L_x'], dtype=np.float32).reshape(pardict['hem_num']*pardict['red_num'])
    pardict['L_y'] = np.array(pardict['L_y'], dtype=np.float32).reshape(pardict['hem_num']*pardict['red_num'])
    pardict['L_z'] = np.array(pardict['L_z'], dtype=np.float32).reshape(pardict['hem_num']*pardict['red_num'])
    pardict['N_x'] = np.array(pardict['N_x'], dtype=np.int32).reshape(pardict['hem_num']*pardict['red_num'])
    pardict['N_y'] = np.array(pardict['N_y'], dtype=np.int32).reshape(pardict['hem_num']*pardict['red_num'])
    pardict['N_z'] = np.array(pardict['N_z'], dtype=np.int32).reshape(pardict['hem_num']*pardict['red_num'])
    pardict['red_min'] = np.array(pardict['red_min'], dtype=np.float32).reshape(pardict['red_num'])
    pardict['red_max'] = np.array(pardict['red_max'], dtype=np.float32).reshape(pardict['red_num'])
    pardict['bias'] = np.array(pardict['bias'], dtype=np.float32).reshape(len(pardict['bias']))
    pardict['z_eff'] = np.array(pardict['z_eff'], dtype=np.float32).reshape(pardict['red_num'])
    pardict['weights_length'] = np.int32(pardict['weights_length'])
    
    return pardict

#The path to the configuration file. 
configfile = sys.argv[1]

#The job number determines which patch of the sky we will calculate the window kernel. 
job_num = int(sys.argv[2])

#Read in all the input parameters. 
pardict = input_variable(configfile)

nBins = pardict['k_bin']
red_num = pardict['red_num']
hem_num = pardict['hem_num']
icut_kmode = pardict['icut_kmode']
kBinWidth = pardict['k_max']/nBins
survey_name = pardict['name']

remainder, k1_bin = divmod(job_num, nBins)
hemisphere, red_bin = divmod(remainder, red_num)

if (hemisphere == 0):
    dire_1 = 'NGC' + '_' + str(red_bin)
else:
    dire_1 = 'SGC' + '_' + str(red_bin)
    
print(k1_bin, dire_1, kBinWidth, survey_name, icut_kmode)

#The length of the simulation box for each patch of the SDSS survey. Please change these accordingly if you are using a different survey. 
Lx_all = np.reshape(pardict['L_x'], (hem_num, red_num))
Ly_all = np.reshape(pardict['L_y'], (hem_num, red_num))
Lz_all = np.reshape(pardict['L_z'], (hem_num, red_num))

# Lx_all = np.array([[1350.0, 1500.0, 1800.0], [1000.0, 850.0, 1000.0]])
# Ly_all = np.array([[2450.0, 2850.0, 3400.0], [1900.0, 2250.0, 2600.0]])
# Lz_all = np.array([[1400.0, 1600.0, 1900.0], [1100.0, 1300.0, 1500.0]])

Lbox_x = Lx_all[hemisphere][red_bin]
Lbox_y = Ly_all[hemisphere][red_bin]
Lbox_z = Lz_all[hemisphere][red_bin]

#The fundamental kmodes in the x, y or z direction. 
kfun_x = 2.0*np.pi/Lbox_x
kfun_y = 2.0*np.pi/Lbox_y
kfun_z = 2.0*np.pi/Lbox_z

# As the window falls steeply with k, only low-k regions are needed for the calculation.
# Therefore cutting out the high-k modes in the FFTs using the icut parameter
icut_x = np.int32(np.ceil(icut_kmode/kfun_x))
icut_y = np.int32(np.ceil(icut_kmode/kfun_y))
icut_z = np.int32(np.ceil(icut_kmode/kfun_z))

#The path to the file contains the normalization factors. 
file = '/data/s4479813/Normalization_' + dire_1 + '_' + survey_name +'.npy'
I22, I12, I11, I10, I24, I14, I34, I44, I32, I12oI22 = np.load(file)

# file = 'Normalization_' + dire_1 + '_' + survey_name +'.npy'
# I22, I12, I11, I10, I24, I14, I34, I44, I32, I12oI22 = np.load(file)

label_all = []
for w in ['W22', 'W12']:
    label_all.append(w)
    
    for (i,i_label),(j,j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
        label_all.append(w + i_label + j_label)
        
    for (i,i_label),(j,j_label),(k,k_label),(l,l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
        label_all.append(w + i_label + j_label + k_label + l_label)

FFT_all = []
for i in range(len(label_all)):
    label = label_all[i]
    output_name = '/data/s4479813/FFT/FFT_grid_'+ label + '_' + dire_1 + '_' + survey_name +'W22.npy'
    if (i < 21.5):
        flag = 1
    else:
        flag = 0
    FFT_all.append(fft(np.load(output_name), W22_flag=flag))
    
[W, Wxx, Wxy, Wxz, Wyy, Wyz, Wzz, Wxxxx, Wxxxy, Wxxxz, Wxxyy, Wxxyz, Wxxzz, Wxyyy, Wxyyz, Wxyzz,\
 Wxzzz, Wyyyy, Wyyyz, Wyyzz, Wyzzz, Wzzzz, W12, W12xx, W12xy, W12xz, W12yy, W12yz, W12zz, W12xxxx,\
 W12xxxy, W12xxxz, W12xxyy, W12xxyz, W12xxzz, W12xyyy, W12xyyz, W12xyzz, W12xzzz, W12yyyy, W12yyyz,\
 W12yyzz, W12yzzz, W12zzzz] = FFT_all 

print("Finish loading in all FFT files.")

#The number of k-modes in each direction. 
length_k_x = np.int32(kBinWidth*nBins/kfun_x)+1
length_k_y = np.int32(kBinWidth*nBins/kfun_y)+1
length_k_z = np.int32(kBinWidth*nBins/kfun_z)+1

#ix, iy, iz contains the index of all possible k-modes in each direction. 
[ix, iy, iz] = np.zeros((3, 2*length_k_x+1, 2*length_k_y+1, 2*length_k_z+1))
#Bin_kmodes contain kx, ky, kz and the length of wavevectors in each bin.
Bin_kmodes = []
#Bin_ModeNum contains the number of kmodes in each kbin. 
Bin_ModeNum = np.zeros(nBins, dtype=np.int32)

for i in range(nBins):
    Bin_kmodes.append([])

for i in range(2*length_k_x+1):
    ix[i,:,:] += i - length_k_x

for j in range(2*length_k_y+1):
    iy[:,j,:] += j - length_k_y

for k in range(2*length_k_z+1):
    iz[:,:,k] += k - length_k_z

#The length of the wavevectors. 
rk = np.sqrt((ix*kfun_x)**2 + (iy*kfun_y)**2 + (iz*kfun_z)**2)
# print(np.min(rk), np.max(rk))

#This determines which bin the wavevectors belong to. 
sort=(rk/kBinWidth).astype(np.int32)
# print(np.max(sort), np.min(sort))

for i in range(nBins):
    ind=(sort==i)
    Bin_ModeNum[i] = len(ix[ind])
    # print(Bin_ModeNum[i])
    Bin_kmodes[i] = np.hstack((ix[ind].reshape(-1, 1), iy[ind].reshape(-1, 1), iz[ind].reshape(-1, 1), rk[ind].reshape(-1, 1)))
    
print(I22, length_k_x, length_k_y, length_k_z, Bin_ModeNum[k1_bin], icut_x, icut_y, icut_z, np.shape(W))

# Lm2 = np.int32(kBinWidth*nBins/kfun)+1
# [ix,iy,iz] = np.zeros((3,2*Lm2+1,2*Lm2+1,2*Lm2+1));
# #Number of k-vectors in x, y and z direction. 
# Bin_kmodes=[]; Bin_ModeNum=np.zeros(nBins,dtype=int)

# for i in range(nBins): Bin_kmodes.append([])
# for i in range(len(ix)):
#     ix[i,:,:]+=i-Lm2; iy[:,i,:]+=i-Lm2; iz[:,:,i]+=i-Lm2

# rk=np.sqrt(ix**2+iy**2+iz**2)
# sort=(rk*kfun/kBinWidth).astype(int)

# for i in range(0,nBins):
#     ind=(sort==i); Bin_ModeNum[i]=len(ix[ind]);
#     #This finds the number of kmodes that with bin i.\
#     Bin_kmodes[i]=np.hstack((ix[ind].reshape(-1,1),iy[ind].reshape(-1,1),iz[ind].reshape(-1,1),rk[ind].reshape(-1,1)))
#     #This finds all the kmodes that are within a certain bin. 

# #Index of the k1-bin for which the window kernels are calculated
# k1_bin=40


kmodes_sampled=30000 # Number of k-modes sampled in the k1 shell. This is the number of smapled kmodes we used to calculate the Gaussian covaraince matrix. You can adjust this
#value based on your demands for accuray. 

start_time = time.time()
Wij = WinFun(k1_bin)
end_time = time.time()
print (time.time() - start_time)

print(np.shape(Wij), np.min(Wij), np.max(Wij))

# The window kernel is being saved to this file. 
name = '/data/s4479813/window/window_kernel_' + dire_1 + '_bin_' + str(k1_bin) + '_' + survey_name +'Wij.npy'
np.save(name, Wij)