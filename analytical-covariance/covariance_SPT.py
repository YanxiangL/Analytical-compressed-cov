# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:00:21 2022

@author: s4479813
"""

import scipy, time, sys
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import log, log10, cos, sin, pi, cosh, sinh , sqrt, amin, amax, mean, dot, power, conj
from scipy.misc import derivative
import seaborn as sns; sns.set_theme(style='white')
from configobj import ConfigObj
import camb
from scipy.optimize import minimize

# sys.path.append("../")
from detail import T0

def power_spectrum(Omega_bh2=0.0221, Omega_cdmh2=0.11662, Omega_k=0.0, H0= 68.00, DE_EoS=-1.0, 
             scalar_amplitude = 2.105e-9, max_pk_redshift=10.0, scalar_index = 0.9667):
    
    my_cosmology = camb.set_params(ombh2 = Omega_bh2, omch2 = Omega_cdmh2, omk = Omega_k, H0=H0, w=DE_EoS, 
                               As=scalar_amplitude, ns=scalar_index, lmax=2500, tau = 0.066, mnu = 0.06, 
                               neutrino_hierarchy = 'degenerate')
    
    # Here we will also need to tell CAMB what redshifts we might want it to compute the matter power spectrum at
    # We don't have to do this for the Cls as the CMB is only at one redshift.
    my_cosmology = my_cosmology.set_matter_power(redshifts=np.concatenate([np.logspace(1.0, -2.0, 10),[0.0]]), nonlinear=False) 
    
    my_cosmology.WantCls = True               # We want the CMB power spectra
    my_cosmology.WantCMB = True               # We want the temperature and polarization power spectra
    my_cosmology.WantTransfer = True          # We want the low redshift matter power spectrum
    my_cosmology.WantDerivedParameters = True # We also want to compute some derived parameters
    my_cosmology.want_zstar = True #The redshift of the recombination
    
    first_run = camb.get_results(my_cosmology)
    
    #Get the matter power spectrum at CMB.
    TT, EE, BB, TE = np.split(first_run.get_cmb_power_spectra(spectra=['total'], CMB_unit='muK')['total'],4,axis=1)
    #Get the linear matter power spectrum.
    Pk_interpolator = first_run.get_matter_power_interpolator(nonlinear=False)
    
    #Get the growth rate of structure at redshift zero. 
    fsigma8 = first_run.get_fsigma8()
    
    #Get the sigma8 factor at redshift zero. 
    sigma8_0 = first_run.get_sigma8_0()
    
    return [my_cosmology, first_run, TT, EE, BB, TE, Pk_interpolator, fsigma8, sigma8_0]

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
    try:
        pardict['bias'] = np.array(pardict['bias'], dtype=np.float32).reshape(pardict['hem_num']*pardict['red_num'], 10)
    except:
        pardict['bias'] = np.array(pardict['bias'], dtype=np.float32).reshape(len(pardict['bias']))
   
    pardict['z_eff'] = np.array(pardict['z_eff'], dtype=np.float32).reshape(pardict['red_num'])
    pardict['weights_length'] = np.int32(pardict['weights_length'])
    
    return pardict

# For generating individual elements of the covariance matrix
# see Survey_window_kernels.ipynb for further details where the same function is used

def Cij(kt,Wij):
    temp=np.zeros((7,6));
    for i in range(-3,4):
        if(kt+i<0 or kt+i>=kbins):
            temp[i+3]=0.
            continue
        temp[i+3]=Wij[i+3,0]*Pfit[0][kt]*Pfit[0][kt+i]+\
        Wij[i+3,1]*Pfit[0][kt]*Pfit[2][kt+i]+\
        Wij[i+3,2]*Pfit[0][kt]*Pfit[4][kt+i]+\
        Wij[i+3,3]*Pfit[2][kt]*Pfit[0][kt+i]+\
        Wij[i+3,4]*Pfit[2][kt]*Pfit[2][kt+i]+\
        Wij[i+3,5]*Pfit[2][kt]*Pfit[4][kt+i]+\
        Wij[i+3,6]*Pfit[4][kt]*Pfit[0][kt+i]+\
        Wij[i+3,7]*Pfit[4][kt]*Pfit[2][kt+i]+\
        Wij[i+3,8]*Pfit[4][kt]*Pfit[4][kt+i]+\
        1.01*(Wij[i+3,9]*(Pfit[0][kt]+Pfit[0][kt+i])/2.+\
        Wij[i+3,10]*Pfit[2][kt]+Wij[i+3,11]*Pfit[4][kt]+\
        Wij[i+3,12]*Pfit[2][kt+i]+Wij[i+3,13]*Pfit[4][kt+i])+\
        1.01**2*Wij[i+3,14]
    return(temp)

# Returns the full (Monopole+Quadrupole) Gaussian covariance matrix
def CovMatGauss():
    covMat=np.zeros((2*kbins,2*kbins))
    for i in range(kbins):
        temp=Cij(i,Wij[i])
        C00=temp[:,0]; C22=temp[:,1]; C20=temp[:,3];
        for j in range(-3,4):
            if(i+j>=0 and i+j<kbins):
                covMat[i,i+j]=C00[j+3]
                covMat[kbins+i,kbins+i+j]=C22[j+3]
                covMat[kbins+i,i+j]=C20[j+3]
    covMat[:kbins,kbins:kbins*2]=np.transpose(covMat[kbins:kbins*2,:kbins])
    covMat=(covMat+np.transpose(covMat))/2.
    return(covMat)

# Growth factor and growth rate for LCDM case
def Dz(z,Om0):
    return(scipy.special.hyp2f1(1/3., 1, 11/6., (1-1/Om0)/(1+z)**3)
                                /scipy.special.hyp2f1(1/3., 1, 11/6., 1-1/Om0)/(1+z))

def fgrowth(z,Om0):
    return(1. + 6*(Om0-1)*scipy.special.hyp2f1(4/3., 2, 17/6., (1-1/Om0)/(1+z)**3)
                  /( 11*Om0*(1+z)**3*scipy.special.hyp2f1(1/3., 1, 11/6., (1-1/Om0)/(1+z)**3) ))

def trispIntegrand(u12,k1,k2):
    return( (8*i44*(Plin(k1)**2*T0.e44o44_1(u12,k1,k2) + Plin(k2)**2*T0.e44o44_1(u12,k2,k1))
            +16*i44*Plin(k1)*Plin(k2)*T0.e44o44_2(u12,k1,k2)
             +8*i34*(Plin(k1)*T0.e34o44_2(u12,k1,k2)+Plin(k2)*T0.e34o44_2(u12,k2,k1))
            +2*i24*T0.e24o44(u12,k1,k2))
            *Plin(np.sqrt(k1**2+k2**2+2*k1*k2*u12)) )/10**5

# Returns the tree-level trispectrum as a function of multipoles and k1, k2

def trisp(l1,l2,k1,k2):
    T0.l1=l1; T0.l2=l2
    
    expr = i44*(Plin(k1)**2*Plin(k2)*T0.ez3(k1,k2) + Plin(k2)**2*Plin(k1)*T0.ez3(k2,k1))\
            +8*i34*Plin(k1)*Plin(k2)*T0.e34o44_1(k1,k2)
    
    # expr = 8*i34*Plin(k1)*Plin(k2)*T0.e34o44_1(k1,k2)

    temp = (10**5*quad(trispIntegrand, -1, 1,args=(k1,k2), epsabs=0.0, epsrel=1e-7, limit=1000)[0]/2. + expr)/i22**2
    return(temp)

def Z12Kernel(l,mu,dlnpk):
    if(l==0):
        exp=(7*b1**2*be*(70 + 42*be + (-35*(-3 + dlnpk) + 3*be*(28 + 13*be - 14*dlnpk - 5*be*dlnpk))*mu**2) + 
            b1*(35*(47 - 7*dlnpk) + be*(798 + 153*be - 98*dlnpk - 21*be*dlnpk + 
            4*(84 + be*(48 - 21*dlnpk) - 49*dlnpk)*mu**2)) + 
            98*(5*b2*(3 + be) + 4*g2*(-5 + be*(-2 + mu**2))))/(735.*b1**2)
    elif(l==2):
        exp=(14*b1**2*be*(14 + 12*be + (2*be*(12 + 7*be) - (1 + be)*(7 + 5*be)*dlnpk)*mu**2) + 
            b1*(4*be*(69 + 19*be) - 7*be*(2 + be)*dlnpk + 
            (24*be*(11 + 6*be) - 7*(21 + be*(22 + 9*be))*dlnpk)*mu**2 + 7*(-8 + 7*dlnpk + 24*mu**2)) + 
            28*(7*b2*be + g2*(-7 - 13*be + (21 + 11*be)*mu**2)))/(147.*b1**2)
    elif(l==4):
        exp=(8*be*(b1*(-132 + 77*dlnpk + be*(23 + 154*b1 + 14*dlnpk)) - 154*g2 + 
            (b1*(396 - 231*dlnpk + be*(272 + 308*b1 + 343*b1*be - 7*(17 + b1*(22 + 15*be))*dlnpk)) + 
            462*g2)*mu**2))/(2695.*b1**2)
    return(exp)

# Legendre polynomials
def lp(l,mu):
    if (l==0): exp=1
    if (l==2): exp=((3*mu**2 - 1)/2.)
    if (l==4): exp=((35*mu**4 - 30*mu**2 + 3)/8.)
    return(exp)

# For transforming the linear array used in the next cell to a matrix
def MatrixForm(a):
    b=np.zeros((3,3))
    if len(a)==6:
        b[0,0]=a[0]; b[1,0]=b[0,1]=a[1]; 
        b[2,0]=b[0,2]=a[2]; b[1,1]=a[3];
        b[2,1]=b[1,2]=a[4]; b[2,2]=a[5];
    if len(a)==9:
        b[0,0]=a[0]; b[0,1]=a[1]; b[0,2]=a[2]; 
        b[1,0]=a[3]; b[1,1]=a[4]; b[1,2]=a[5];
        b[2,0]=a[6]; b[2,1]=a[7]; b[2,2]=a[8];
    return(b)

# Calculating multipoles of the Z12 kernel
def Z12Multipoles(i,l,dlnpk):
    return(quad(lambda mu: lp(i,mu)*Z12Kernel(l,mu,dlnpk), -1, 1)[0])

def covaSSC(l1,l2):
    covaBC=np.zeros((len(k),len(k)))
    for i in range(3):
        for j in range(3):
            covaBC+=1/4.*sigma22Sq[i,j]*np.outer(Plin(k)*Z12Multipoles(2*i,l1,dlnPk),Plin(k)*Z12Multipoles(2*j,l2,dlnPk))
            sigma10Sq[i,j]=1/4.*sigma10Sq[i,j]*quad(lambda mu: lp(2*i,mu)*(1 + be*mu**2), -1, 1)[0]\
            *quad(lambda mu: lp(2*j,mu)*(1 + be*mu**2), -1, 1)[0]

    covaLA=-rsd[l2]*np.outer(Plin(k)*(covaLAterm[int(l1/2)]+i32/i22/i10*rsd[l1]*Plin(k)*b2/b1**2+2/i10*rsd[l1]),Plin(k))\
           -rsd[l1]*np.outer(Plin(k),Plin(k)*(covaLAterm[int(l2/2)]+i32/i22/i10*rsd[l2]*Plin(k)*b2/b1**2+2/i10*rsd[l2]))\
           +(np.sum(sigma10Sq)+1/i10)*rsd[l1]*rsd[l2]*np.outer(Plin(k),Plin(k))

    return(covaBC+covaLA)

# def Z3EFT(params, l1, l2, k1, k2):
    
#     b3 = b3EFT
    
#     c5, c6, c7, c8, c9 = params
    
#     if ((l1 == 2) and (l2 == 2)):
#         if (k1 != k2):
#             return ((4*k1*k2*(-3150*(9*b1 - 2*b3)*be**2*(33 + 17*be)*k1**14 - 
#             525*(99*b1*(3 + be)*(63 + be*(78 + 23*be)) - 2*b3*(2079 + be*(3267 + be*(1419 + 151*be))))*k1**12*k2**2 + 
#             210*(9*b1*(10395 + be*(16335 + 7106*be + 974*be**2)) - b3*(31185 + be*(49005 + be*(22297 + 3213*be))))*k1**10*k2**4 - 
#             4*k1**8*(7920*b1*(1617 + be*(2541 + be*(2887 + 923*be))) - 1617*(1667*b3 - 128*(45*c5 - 86*c6 - 90*c8)) + 
#                be*(-(b3*(4235847 + 2*be*(2375769 + 753551*be))) + 
#                   1408*(45*(231 + be*(699 + 271*be))*c5 - (19866 + be*(44679 + 16691*be))*c6 + 
#                      45*(98*be*(7 + 3*be)*c7 - 2*(231 + 2*be*(521 + 209*be))*c8 - 147*be*(7 + 3*be)*c9))) + 
#                4928*(735 + be*(1155 + be*(849 + 221*be)))*(b3 - c6)*k1**2)*k2**6 - 
#             2*k1**6*(315*b1*(82929 + be*(130317 + 94886*be + 24394*be**2)) + 
#                90*b3*(-41503 + be*(-65219 + 2*be*(-9053 + 213*be))) + 896*(2541 + be*(3993 + be*(3179 + 879*be)))*(b3 - c6)*k1**2)
#               *k2**8 + 105*(495*b1*(3 + be)*(63 + be*(78 + 23*be)) - 2*b3*(31185 + be*(49005 + be*(22297 + 3213*be))))*k1**4*
#              k2**10 + 1050*(27*b1*be**2*(33 + 17*be) + b3*(2079 + be*(3267 + be*(1419 + 151*be))))*k1**2*k2**12 + 
#             6300*b3*be**2*(33 + 17*be)*k2**14) - 1575*(k1 - k2)**3*(k1 + k2)**3*((9*b1 - 2*b3)*k1**2 + 2*b3*k2**2)*
#           (2*be**2*(33 + 17*be)*k1**8 + (693 + be*(1089 + 5*be*(143 + 35*be)))*k1**6*k2**2 + 
#             2*(231 + be*(363 + be*(407 + 131*be)))*k1**4*k2**4 + (693 + be*(1089 + 5*be*(143 + 35*be)))*k1**2*k2**6 + 
#             2*be**2*(33 + 17*be)*k2**8)*(np.log((k1 - k2)**2) - 2*np.log(k1 + k2)))/(9.779616e7*b1**3*k1**9*k2**7))
#         else:
#             return ((88*(-135*b1*(441 + be*(693 + 25*be*(29 + 9*be))) + 294*(23*b3 - 90*c5 + 172*c6 + 180*c8) + 
#             2*be*(b3*(5313 + be*(5787 + 1823*be)) - 90*(231 + be*(699 + 271*be))*c5 + 924*(43*c6 + 45*c8) + 
#                2*be*((44679 + 16691*be)*c6 + 45*(-98*(7 + 3*be)*c7 + 4*(521 + 209*be)*c8 + 147*(7 + 3*be)*c9)))) - 
#          224*(5313 + be*(8349 + be*(6259 + 1655*be)))*(b3 - c6)*k2**2)/(1.528065e6*b1**3))
            
#     elif ((l1 == 0) and (l2 == 0)):
#         if (k1 != k2):
#             return ((4*k1*k2*(-225*(9*b1 - 2*b3)*be**2*(7 + 3*be)*k1**10 - 
#                 75*(2*b3*(-315 + be*(-315 - 70*be + 6*be**2)) + 9*b1*(315 + be*(315 + be*(91 + 3*be))))*k1**8*k2**2 - 
#                 6*k1**6*(120*b1*(2625 + be*(2625 + be*(1211 + 219*be))) - 
#                    175*(379*b3 - 6120*c5 + 7916*c6 + 180*(-42*c7 + 110*c8 + 63*c9)) + 
#                    be*(-(b3*(66325 + be*(30597 + 5533*be))) + 360*(2975 + be*(1281 + 209*be))*c5 - 
#                       700*(1979*c6 + 45*(-42*c7 + 110*c8 + 63*c9)) - 
#                       4*be*((150507 + 24923*be)*c6 + 15*(-490*(19 + 3*be)*c7 + (24682 + 3978*be)*c8 + 735*(19 + 3*be)*c9))) + 
#                    560*(35 + be*(35 + be*(21 + 5*be)))*(b3 - c6)*k1**2)*k2**4 - 
#                 k1**4*(6930*b3*(5 + be)*(5 + be*(4 + be)) + 45*b1*(9975 + be*(9975 + be*(5887 + 1383*be))) + 
#                    64*(735 + be*(735 + be*(413 + 93*be)))*(b3 - c6)*k1**2)*k2**6 + 
#                 75*(27*b1*be**2*(7 + 3*be) + 2*b3*(315 + be*(315 + 70*be - 6*be**2)))*k1**2*k2**8 + 450*b3*be**2*(7 + 3*be)*k2**10) - 
#              225*(k1 - k2)**3*(k1 + k2)**3*((9*b1 - 2*b3)*k1**2 + 2*b3*k2**2)*
#               (be**2*(7 + 3*be)*k1**4 + (105 + be*(105 + be*(49 + 9*be)))*k1**2*k2**2 + be**2*(7 + 3*be)*k2**4)*
#               (np.log((k1 - k2)**2) - 2*np.log(k1 + k2)))/(3.969e6*b1**3*k1**7*k2**5))
#         else:
#             return ((6*(-45*b1*(4725 + be*(4725 + be*(2219 + 411*be))) + 350*(76*b3 - 1530*c5 + 1979*c6 + 45*(-42*c7 + 110*c8 + 63*c9)) + 
#             2*be*(4*b3*(3325 + be*(1547 + 283*be)) - 90*(2975 + be*(1281 + 209*be))*c5 + 
#                175*(1979*c6 + 45*(-42*c7 + 110*c8 + 63*c9)) + 
#                be*((150507 + 24923*be)*c6 + 15*(-490*(19 + 3*be)*c7 + (24682 + 3978*be)*c8 + 735*(19 + 3*be)*c9)))) - 
#          16*(5145 + be*(5145 + be*(3031 + 711*be)))*(b3 - c6)*k2**2)/(496125.*b1**3))
            
#     elif ((l1 == 0) and (l2 == 2)):
#          if (k1 != k2):
#              return ((be*(4*k1*k2*(-1575*(9*b1 - 2*b3)*be**2*k1**14 - 
#              1050*(9*b1*(63 + be*(60 + 11*be)) - b3*(126 + be*(120 + 19*be)))*k1**12*k2**2 + 
#               105*(9*b1*(420 + be*(640 + 97*be)) - 2*b3*(1050 + be*(1240 + 207*be)))*k1**10*k2**4 - 
#               6*k1**8*(1920*b1*(882 + be*(665 + 183*be)) - 196*(1819*b3 - 8*(2790*c5 - 3757*c6 + 3150*c7 - 8730*c8 - 4725*c9)) + 
#                  be*(-3*b3*(89600 + 24449*be) + 32*be*(27630*c5 - 37369*c6 + 45*(686*c7 - 1914*c8 - 1029*c9)) + 
#                     33600*(90*c5 - 123*c6 + 98*c7 - 278*c8 - 147*c9)) + 896*(245 + be*(210 + 53*be))*(b3 - c6)*k1**2)*k2**6 - 
#               k1**6*(30*b3*(6468 + 1049*be**2) + 315*b1*(15372 + be*(13120 + 3267*be)) + 
#                  7168*(63 + be*(53 + 14*be))*(b3 - c6)*k1**2)*k2**8 + 
#               210*(45*b1*(63 + be*(60 + 11*be)) - b3*(1050 + be*(1240 + 207*be)))*k1**4*k2**10 + 
#               525*(27*b1*be**2 + b3*(252 + 240*be + 38*be**2))*k1**2*k2**12 + 3150*b3*be**2*k2**14) - 
#            1575*(k1 - k2)**3*(k1 + k2)**3*((9*b1 - 2*b3)*k1**2 + 2*b3*k2**2)*
#             (be**2*k1**8 + 2*(21 + 5*be*(4 + be))*k1**6*k2**2 + 2*(42 + be*(32 + 9*be))*k1**4*k2**4 + 
#               2*(21 + 5*be*(4 + be))*k1**2*k2**6 + be**2*k2**8)*(np.log((k1 - k2)**2) - 2*np.log(k1 + k2))))/(2.22264e7*b1**3*k1**9*k2**7))
#          else:
#              return ((4*be*(-135*b1*(3381 + 5*be*(518 + 141*be)) + 294*(188*b3 - 2790*c5 + 3757*c6 - 3150*c7 + 8730*c8 + 4725*c9) + 
#            6*be*(4*b3*(1750 + 479*be) - 90*(1050 + 307*be)*c5 + 37369*be*c6 + 1050*(123*c6 - 98*c7 + 278*c8 + 147*c9) + 
#               45*be*(-686*c7 + 1914*c8 + 1029*c9)) - 56*(987 + be*(842 + 215*be))*(b3 - c6)*k2**2))/(694575.*b1**3))
         
# def Z3SPT(params, l1, l2, k1, k2):
    
#     g2x,g21,b3 = params
    
#     if(l1==0 and l2==0):
#         if(k1!=k2):
#             return(-(4*k1*k2*(1155*b1*be**4*k1**14 + 
#                   70*be**2*(b1*(2310 + be*(1584 + 253*be + 9*b1*(198 + be*(132 + 25*be)))) + 
#                     693*(7 + 3*be)*g21)*k1**12*k2**2 + 
#                   7*(b1*(242550 + 1540*be*(180 + 77*be) + 33*be**3*(760 + 111*be) + 
#                         60*b1*be*(3465 + 2*be*(2772 + be*(1683 + 2*be*(297 + 50*be))))) - 
#                     2310*(-315 + be*(-315 - 70*be + 6*be**2))*g21)*k1**10*k2**4 - 
#                   2*(21*b1**2*be*(92400 + be*(120120 + be*(62766 + be*(18964 + 2675*be)))) + 
#                     11*b1*(183750 + be*(191100 + be*(88200 + be*(22092 + 2845*be)))) + 
#                     231*(980*b3*(3 + be)*(15 + be*(10 + 3*be)) + 
#                         144*b2*(2975 + be*(2975 + be*(1281 + 209*be))) + 
#                         1617*(5 + be)*(5 + be*(4 + be))*g21 - 224*(525 + be*(525 + be*(203 + 27*be)))*g2x
#                         ))*k1**8*k2**6 + (b1*(6387150 + 
#                         be*(1848*b1**2*be*(35 + 3*be*(14 + 5*be))**2 + 
#                           11*(882000 + be*(727356 + be*(297024 + 50797*be))) + 
#                           12*b1*(256025 + be*(970200 + be*(1086393 + 501886*be + 89370*be**2))))) - 
#                     747054*(5 + be)*(5 + be*(4 + be))*g21)*k1**6*k2**8 - 
#                   14*(b1*(34650 + be*(69300 + 
#                           be*(51590 + 33*be*(340 + 11*be) + 
#                               15*b1*(5544 + be*(6732 + be*(2024 + 133*be)))))) + 
#                     1155*(-315 + be*(-315 - 70*be + 6*be**2))*g21)*k1**4*k2**10 - 
#                   35*be**2*(b1*(3696 + be*(3960 + 913*be + 12*b1*(297 + 2*be*(231 + 65*be)))) - 
#                     1386*(7 + 3*be)*g21)*k1**2*k2**12 - 210*b1*be**4*(11 + 15*b1*be)*k2**14) + 
#               105*(k1 - k2)**3*(k1 + k2)**3*(11*b1*be**4*k1**10 + 
#                   2*be**2*(b1*(770 + 66*(8 + 9*b1)*be + 99*(1 + 4*b1)*be**2 + 75*b1*be**3) + 
#                     231*(7 + 3*be)*g21)*k1**8*k2**2 + 
#                   2*(b1*(8085 + 2*be*(4620 + 3465*b1 + 11*be*(273 + 17*be*(6 + be)) + 
#                           6*b1*be*(924 + be*(693 + 286*be + 50*be**2)))) + 
#                     693*(35 + be*(35 + 2*be*(7 + be)))*g21)*k1**6*k2**4 + 
#                   2*(2*b1*(1155 + be*(2310 + be*
#                             (2541 + 22*be*(57 + 11*be) + 6*b1*(462 + be*(693 + be*(374 + 75*be)))))) - 
#                     693*(35 + be*(35 + 2*be*(7 + be)))*g21)*k1**4*k2**6 + 
#                   be**2*(b1*(1232 + 3*be*(440 + 121*be + 4*b1*(99 + 2*be*(77 + 25*be)))) - 
#                     462*(7 + 3*be)*g21)*k1**2*k2**8 + 2*b1*be**4*(11 + 15*b1*be)*k2**10)*
#                 (np.log((k1 - k2)**2) - 2*np.log(k1 + k2)))/(4.07484e7*b1**3*k1**9*k2**7))
#         else:
#             return(-(42*b1**3*be**2*(35 + 3*be*(14 + 5*be))**2 + 
#               12*b1**2*be*(1225 + 3*be*(4900 + be*(6566 + 3276*be + 615*be**2))) + 
#               2*b1*(40425 + be*(73500 + be*(70462 + 9*be*(3500 + 633*be)))) - 
#               42*(36*b2*(2975 + be*(2975 + be*(1281 + 209*be))) + 
#                   7*(35*b3*(3 + be)*(15 + be*(10 + 3*be)) + 
#                     4*(525 + be*(525 + be*(203 + 27*be)))*(g21 - 2*g2x))))/(231525.*b1**3))
            
                    
#     if(l1==0 and l2==2):
#         if(k1!=k2):
#             return(-(be*(4*k1*k2*(105*be**2*(2*b1*(715 + be*(221 + 45*b1*(13 + 3*be))) + 3003*g21)*k1**14 + 
#                     35*(2*b1*(63063 + be*(74217 + 377*be*(77 + 13*be) + 
#                             9*b1*(7722 + be*(11154 + be*(4979 + 905*be))))) + 
#                       3003*(126 + be*(120 + 19*be))*g21)*k1**12*k2**2 + 
#                     7*(26*b1*(4620 + be*(3080 + 9*be*(825 + 179*be))) + 
#                       30*b1**2*(72072 + be*(95238 + be*(52338 + be*(23829 + 4547*be)))) - 
#                       3003*(1050 + be*(1240 + 207*be))*g21)*k1**10*k2**4 - 
#                     13*(b1*(918456 + 4*be*(209286 + be*(121132 + 20645*be))) + 
#                       6*b1**2*(517440 + be*(809886 + be*(499422 + be*(191267 + 30145*be)))) + 
#                       33*(1152*b2*(1519 + be*(1050 + 307*be)) + 
#                           7*(1568*b3*(15 + be*(10 + 3*be)) + (6468 + 1049*be**2)*g21 - 
#                             256*(147 + be*(70 + 27*be))*g2x)))*k1**8*k2**6 + 
#                     (96096*b1**3*be*(1 + be)*(7 + 5*be)*(35 + 3*be*(14 + 5*be)) + 
#                       26*b1*(3172554 + be*(3883110 + be*(1842643 + 346949*be))) + 
#                       6*b1**2*(5325320 + be*(23621598 + be*(30019990 + be*(15187159 + 2925165*be)))) - 
#                       3003*(6468 + 1049*be**2)*g21)*k1**6*k2**8 - 
#                     7*(2*b1*(615615 + be*(545545 + 1677*(33 - 7*be)*be + 
#                             15*b1*(66924 + be*(77220 + be*(17875 + 53*be))))) + 
#                       3003*(1050 + be*(1240 + 207*be))*g21)*k1**4*k2**10 - 
#                     35*(2*b1*(18018 + be*(45474 + 
#                             be*(37895 + 7813*be + 3*b1*(15444 + be*(18603 + 4585*be))))) - 
#                       3003*(126 + be*(120 + 19*be))*g21)*k1**2*k2**12 - 
#                     105*be**2*(2*b1*(572 + be*(442 + 15*b1*(39 + 41*be))) - 3003*g21)*k2**14) + 
#                 105*(k1 - k2)**3*(k1 + k2)**3*
#                   (be**2*(2*b1*(715 + be*(221 + 45*b1*(13 + 3*be))) + 3003*g21)*k1**10 + 
#                     (26*b1*(1617 + be*(1903 + 9*be*(99 + 19*be))) + 
#                       6*b1**2*be*(7722 + be*(11154 + be*(5499 + 1025*be))) + 
#                       3003*(42 + be*(40 + 9*be))*g21)*k1**8*k2**2 + 
#                     2*(2*b1*(6006*(5 + 6*b1) + 429*(80 + 183*b1)*be + 715*(25 + 99*b1)*be**2 + 
#                           13*(263 + 2559*b1)*be**3 + 6225*b1*be**4) + 3003*(21 + 4*be*(3 + be))*g21)*k1**6*
#                     k2**4 + 2*(b1*(57057 + 13*be*(5907 + be*(3091 + 641*be)) + 
#                           12*b1*be*(5577 + be*(9867 + be*(5863 + 1275*be)))) - 
#                       3003*(21 + 4*be*(3 + be))*g21)*k1**4*k2**6 + 
#                     (2*b1*(6006 + be*(15158 + 39*be*(363 + 97*be) + 
#                             3*b1*be*(5148 + be*(6721 + 2075*be)))) - 3003*(42 + be*(40 + 9*be))*g21)*
#                     k1**2*k2**8 + be**2*(2*b1*(572 + be*(442 + 15*b1*(39 + 41*be))) - 3003*g21)*k2**10)*
#                   (np.log((k1 - k2)**2) - 2*np.log(k1 + k2))))/(2.1189168e8*b1**3*k1**9*k2**7))
    
#         else:
#              return((4*be*(-21*b1**3*be*(1 + be)*(7 + 5*be)*(35 + 3*be*(14 + 5*be)) - 
#                 2*b1*(7203 + be*(9289 + 9*be*(505 + 99*be))) - 
#                 6*b1**2*(245 + be*(3255 + be*(5089 + 57*be*(49 + 10*be)))) + 
#                 3*(36*b2*(1519 + be*(1050 + 307*be)) + 
#                    7*(49*b3*(15 + be*(10 + 3*be)) + 4*(147 + be*(70 + 27*be))*(g21 - 2*g2x)))))/
#             (46305.*b1**3))
    
#     if(l1==2 and l2==2):
#         if(k1!=k2):
#             return(-(4*k1*k2*(14175*b1*be**4*k1**18 + 
#                   210*be**2*(b1*(8580 + be*(7072 + 1358*be + 9*b1*(780 + be*(590 + 117*be)))) + 
#                     546*(33 + 17*be)*g21)*k1**16*k2**2 + 
#                   70*(b1*(189189 + 2*be*(169884 + be*(115401 + be*(39572 + 6943*be)) + 
#                           9*b1*(11583 + be*(24453 + be*(19539 + be*(8326 + 1531*be)))))) + 
#                     273*(2079 + be*(3267 + be*(1419 + 151*be)))*g21)*k1**14*k2**4 + 
#                   2*(b1*(-11351340 - 540540*(33 + 25*b1)*be - 110110*(64 + 135*b1)*be**2 - 
#                         266994*(2 + 5*b1)*be**3 + 9*(8957 + 248360*b1)*be**4 + 720048*b1*be**5) - 
#                     1911*(31185 + be*(49005 + be*(22297 + 3213*be)))*g21)*k1**12*k2**6 - 
#                   4*(3*b1**2*be*(-1789788 + be*(1183182 + be*(4574856 + be*(2502259 + 463655*be)))) + 
#                     b1*(-2816814 + be*(-2510508 + be*(4578288 + be*(3456648 + 666035*be)))) + 
#                     39*(8448*b2*(147 + be*(231 + be*(699 + 271*be))) + 
#                         7*(9856*b3*be**2*(7 + 3*be) + 
#                           3*(-41503 + be*(-65219 + 2*be*(-9053 + 213*be)))*g21 - 
#                           1408*(-147 + be*(-231 + be*(-13 + 23*be)))*g2x)))*k1**10*k2**8 + 
#                   2*(b1*(38804766 + be*(74522448 + 384384*b1**2*be*(1 + be)**2*(7 + 5*be)**2 + 
#                           be*(92317368 + be*(53941056 + 11332691*be)) + 
#                           6*b1*(9971962 + be*(27333306 + be*(31994248 + be*(17984861 + 3763132*be)))))) - 
#                     1638*(-41503 + be*(-65219 + 2*be*(-9053 + 213*be)))*g21)*k1**8*k2**10 + 
#                   2*(b1*(-2837835 + 2*be**2*(1494493 + 3*be*(283920 + 87719*be)) + 
#                         42*b1*be*(-250965 + be*(-225225 + be*(31785 + be*(64520 + 23917*be))))) - 
#                     1911*(31185 + be*(49005 + be*(22297 + 3213*be)))*g21)*k1**6*k2**12 - 
#                   14*(b1*(270270 + be*(849420 + 
#                           be*(969540 + 35*be*(12428 + 2395*be) + 
#                               6*b1*(173745 + be*(293085 + 2*be*(76605 + 14966*be)))))) - 
#                     1365*(2079 + be*(3267 + be*(1419 + 151*be)))*g21)*k1**4*k2**14 - 
#                   105*be**2*(b1*(13728 + be*(17680 + 5027*be + 12*b1*(1170 + be*(2115 + 719*be)))) - 
#                     1092*(33 + 17*be)*g21)*k1**2*k2**16 - 5670*b1*be**4*(5 + 7*b1*be)*k2**18) + 
#               105*(k1 - k2)**3*(k1 + k2)**3*(135*b1*be**4*k1**14 + 
#                   2*be**2*(b1*(8580 + be*(7072 + 1538*be + 9*b1*(780 + be*(590 + 117*be)))) + 
#                     546*(33 + 17*be)*g21)*k1**12*k2**2 + 
#                   (b1*(126126 + 5148*(44 + 27*b1)*be + 572*(349 + 513*b1)*be**2 + 
#                         156*(580 + 1743*b1)*be**3 + 9*(1907 + 14248*b1)*be**4 + 23988*b1*be**5) + 
#                     546*(693 + be*(1089 + be*(649 + 141*be)))*g21)*k1**10*k2**4 + 
#                   2*(b1*(60060 + 18876*(7 + 3*b1)*be + 2574*(70 + 97*b1)*be**2 + 
#                         78*(1282 + 4287*b1)*be**3 + (20299 + 180582*b1)*be**4 + 36525*b1*be**5) + 
#                     819*(-77 + be*(-121 + be*(33 + 29*be)))*g21)*k1**8*k2**6 + 
#                   (b1*(150150 + be*(302016 + be*(355212 + be*(209040 + 46543*be)) + 
#                           12*b1*(16731 + be*(45903 + be*(55731 + be*(33323 + 7500*be)))))) - 
#                     1638*(-77 + be*(-121 + be*(33 + 29*be)))*g21)*k1**6*k2**8 + 
#                   2*(b1*(18018 + be*(56628 + be*
#                             (82940 + 6*be*(8762 + 2159*be) + 
#                               3*b1*(23166 + be*(45318 + be*(31708 + 8135*be)))))) - 
#                     273*(693 + be*(1089 + be*(649 + 141*be)))*g21)*k1**4*k2**10 + 
#                   be**2*(b1*(13728 + be*(17680 + 5747*be + 12*b1*(1170 + be*(2115 + 803*be)))) - 
#                     1092*(33 + 17*be)*g21)*k1**2*k2**12 + 54*b1*be**4*(5 + 7*b1*be)*k2**14)*
#                 (np.log((k1 - k2)**2) - 2*np.log(k1 + k2)))/(1.69513344e8*b1**3*k1**11*k2**9))
            
#         else:
#             return((8*(-21*b1**3*be**2*(1 + be)**2*(7 + 5*be)**2 - 
#                 6*b1**2*be*(490 + be*(1281 + be*(1498 + be*(873 + 190*be)))) - 
#                 b1*(1911 + be*(3696 + be*(4402 + be*(2608 + 567*be)))) + 
#                 3*(98*b3*be**2*(7 + 3*be) + 12*b2*(147 + be*(231 + be*(699 + 271*be))) + 
#                   7*(-147 + be*(-231 + be*(-13 + 23*be)))*(g21 - 2*g2x))))/(9261.*b1**3))

# def min_func(params):
    
#     """
#     This calculates the difference between the Z3 kernel from the SPT model and the EFT model.

#     Parameters
#     ----------
#     params : 1d numpy array.
#         This array should contain the value for b3, g21 and g2x in the SPT model.
#     l1 : int
#         The first multipole.
#     l2 : int
#         The second multipole.

#     Returns
#     -------
#     total : float
#         The difference between Z3 for the SPT and the EFT model.

#     """
    
#     c5, c6, c7, c8, c9, g2x,g21,b3 = params
    
#     #This will generate all possible combination of kmodes.
#     test_k1 = np.repeat(k, kbins)
#     test_k2 = np.tile(k, kbins)
    
#     #The UV subtraction is only important when k2 >> k1, so we select the modes where k1 << k2, so UV subtraction should be negligible. 
#     index_useful = np.where((test_k1 <= 0.075) & (test_k2 <= 0.075))[0]
    
#     k1 = test_k1[index_useful]
#     k2 = test_k2[index_useful]
    
#     total = 0.0
    
#     params_EFT = c5, c6, c7, c8, c9
#     params_SPT = g2x, g21, b3
    
#     for i in range(len(k1)):
        
#         Z3SPT_00 = Z3SPT(params_SPT, 0, 0, k1[i], k2[i])
#         Z3EFT_00 = Z3EFT(params_EFT, 0, 0, k1[i], k2[i])
#         Z3EFT_02 = Z3EFT(params_EFT, 0, 2, k1[i], k2[i])
#         Z3SPT_02 = Z3SPT(params_SPT, 0, 2, k1[i], k2[i])
#         Z3SPT_22 = Z3SPT(params_SPT, 2, 2, k1[i], k2[i])
#         Z3EFT_22 = Z3EFT(params_EFT, 2, 2, k1[i], k2[i])
        
        
#         # total += np.abs((Z3SPT_00 - Z3EFT_00)/Z3EFT_00) 
#         # + np.abs((Z3SPT_02 - Z3EFT_02)/Z3EFT_02) + np.abs((Z3SPT_22 - Z3EFT_22)/Z3EFT_22)
        
#         total += np.abs((Z3SPT_00 - Z3EFT_00)) 
#         # + np.abs((Z3SPT_02 - Z3EFT_02)) + np.abs((Z3SPT_22 - Z3EFT_22))
        
#     return total

# def find_best_fit():
#     """
#     This function will use the scipy minimization scheme to calculate the best-fit of the unkown SPT parameters. 

#     Parameters
#     ----------
#     end_point : int or float
#         The maximum ratio between k2 and k1 given the data.
#     Num : int
#         Total numbers of points to generate the sample.

#     Returns
#     -------
#     spline_00 : Interpolation object.
#         The interpolator for T0 of C00.
#     spline_02 : Interpolation object.
#         The interpolator for T0 of C02.
#     spline_22 : Interpolation object.
#         The interpolator for T0 of C22.

#     """
#     #We will use the scipy minimmization scheme to calculate the best-fit SPT parameters. The Powell method is the only method that is robust enough. If you require different
#     #precision, you can also chnage the tolerance through options. 
#     start = time.time()
#     result_00 = minimize(min_func, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), method = 'Powell', options={'xtol': 1e-6, 'ftol': 1e-6})
#     # b3_00, g21_00, g2x_00 = result_00.x
#     end = time.time()
#     print('It takes ' + str(end - start) + ' seconds to find the best-fit parameters for l1 = 0, l2 = 0')
#     print('The minimization is successful: '+str(result_00.success)+'. The best-fit are ' + str(result_00.x))
    
#     return result_00

if __name__ == "__main__":
    #The path to the input file
    configfile = sys.argv[1]
    #This determine which patch of the survey we will calculate the covariance matrix for
    job_num = int(sys.argv[2])
    #The skycut parameter from the montepython routine. 
    guess = int(sys.argv[3])
    
    option = int(sys.argv[4])
    
    if guess == 1:
        keyword = '_guess_'
    else:
        keyword = '_best_'
    
    pardict = input_variable(configfile)
    k_num = pardict['k_bin']
    kmax = pardict['k_max']
    red_num = pardict['red_num']
    hem_num = pardict['hem_num']
    kbinwidth = kmax/k_num
    survey_name = pardict['name']
    
    # if option == 0:
    #     pardict['option'] = 'Anal'
    # elif option == 1:
    #     pardict['option'] = 'Anal_comp'
    # elif option == 2:
    #     pardict['option'] = 'Anal_comp_taylor'
    # else:
    #     raise Exception('Incorrect option in the config file.')
        
    if option == 0:
        pardict['option'] = 'Anal'
    elif option == 1:
        pardict['option'] = 'Anal_comp'
    elif option == 2:
        pardict['option'] = 'Anal_comp_taylor'    
    elif option == 3:
        pardict['option'] = 'Anal_guess'
    elif option == 4:
        pardict['option'] = 'Anal_comp_guess'
    else:
        raise Exception('Incorrect option in the config file.')
        
    #The BOSS survey has 3 different redshift bins on each hemisphere. You may need to change this if you are using a different survey.
    hemisphere, red_bin = divmod(job_num, red_num)
    
    #The effective redshift of each redshift bin. Please change this accordingly if you are using a different survey. 
    redshift_all = pardict['z_eff']
    redshift = redshift_all[red_bin]
    
    if (hemisphere == 0):
        dire_1 = 'NGC' + '_' + str(red_bin)
        dire_2 = 'NGC'
    else:
        dire_1 = 'SGC' + '_' + str(red_bin)
        dire_2 = 'SGC'
    
    
    if (option != 0) and (option != 3) and guess == 1:
        if option == 1 or option == 2:
            print('Copy the analytical covariance matrix with fiducial parameters.')
                
            input_keyword = 'Anal'
        elif option == 4:
            print('Copy the analytical covariance matrix with guess parameters.')
           
            input_keyword = 'Anal_guess'
            
        input_1 = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_SPT_UV_sub_' + dire_1 + '_' + survey_name + keyword + input_keyword + '.npy'
        
        input_G = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_Gaussian' + dire_1 + '_' + survey_name + keyword + input_keyword + '.npy'
        input_SSC = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_SSC' + dire_1 + '_' + survey_name + keyword + input_keyword + '.npy'
        input_T0_1 = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_T0_SPT_UV_sub' + dire_1 + '_' + survey_name + keyword + input_keyword + '.npy'
        
        covaAnl = np.load(input_1)
        covaG = np.load(input_G)
        covaSSCmult = np.load(input_SSC)
        covaT0_EFT_newmult = np.load(input_T0_1)
        
        output_1 = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_SPT_UV_sub_' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        
        output_G = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_Gaussian' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        output_SSC = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_SSC' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        output_T0_1 = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_T0_SPT_UV_sub' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        
        
        np.save(output_1, covaAnl)
        np.save(output_G, covaG)
        np.save(output_SSC, covaSSCmult)
        np.save(output_T0_1, covaT0_EFT_newmult)
        
    else:
            
        if guess == 0:
            # bestfits = np.load('./resource/bestfit_marg_' + pardict['option'] + '.npy')
            bestfits = np.load(str('/data/s4479813/pybird-desi/model_PS/bestfit_marg_' + pardict['option'] + '.npy'))
            log_10_10_As = bestfits[0]
            h = bestfits[1]
            Omega_cdmh2 = bestfits[2]
            Omega_bh2 = bestfits[3]
            
            mock_index = np.int16(pardict['mock_index'])
            if (red_num == 1 and hem_num == 1):
                index = mock_index
            else:
                index = mock_index[job_num]
                
            b1 = bestfits[4 + index*10]
            
        else:
            if option < 2.5:
                log_10_10_As, h, Omega_bh2, Omega_cdmh2 = np.float64(pardict['fiducial'])
            else:
                Omega_cdmh2 = pardict['omega_cdm']
                Omega_bh2 = pardict['omega_b']
                h = pardict['h']
                log_10_10_As = pardict['ln_10_10_As']
            
            try: 
                b1 = pardict['bias'][0][0]
            except:
                b1 = pardict['bias'][0]
                
        print(dire_1, log_10_10_As, h, Omega_cdmh2, Omega_bh2, b1)
            
        # print(log_10_10_As, h, Omega_cdmh2, Omega_bh2, b1)
        
        #Read in the Gaussian window kernels. 
        Wij = np.zeros((k_num, 7, 15, 6))
        for i in range(k_num):
            name = '/data/s4479813/window/window_kernel_' + dire_1 + '_bin_' + str(i) + '_' + survey_name +'Wij.npy'
            
            # name = './resource/window_kernel_' + dire_1 + '_bin_' + str(i) + '_' + survey_name +'Wij.npy'
            Wij[i] = np.load(name)
            # print(np.shape(Wij[i]))
        
        #Here, we will assume the same kmin, kmax for mono-, quadru- and hexadeca-pole. 
        nbins = k_num
        k_min = kbinwidth/2.0
        k_max = kmax - kbinwidth/2.0
        k = np.linspace(k_min, k_max, nbins)
        kbins = len(k)
        print(k)
        kmode_file = str('/data/s4479813/pybird-desi/model_PS/kmode_' + dire_1 + '_' + survey_name + '.npy')
        # kmode_file = str('./resource/kmode_' + dire_1 + '_' + survey_name + '.npy')
        
        PS_file = str("/data/s4479813/pybird-desi/model_PS/PS_MP_" + dire_1 + '_' + survey_name + keyword + "model_" + pardict['option'] + ".npy")
        # PS_file = str("./resource/PS_MP_" + dire_1 + '_' + survey_name + keyword + "model_" + pardict['option'] + ".npy")
        
        kmode = np.load(kmode_file)
        PS_data = np.load(PS_file)
        k_length = np.int32(len(kmode)/3)
        model_all = []
        for i in range(3):
            kmode_pole = kmode[i*k_length:(i+1)*k_length]
            PS_data_pole = PS_data[i*k_length:(i+1)*k_length]
            model = InterpolatedUnivariateSpline(kmode_pole, PS_data_pole)
            model_all.append(model(k))    
            
        Pfit = [0, 0, 0, 0, 0]
        Pfit[0] = model_all[0]
        Pfit[2] = model_all[1]
        Pfit[4] = model_all[2]
        
        print(np.shape(Pfit[0]), np.shape(Pfit[2]), np.shape(Pfit[4]))
        
        
        # Calculate the fiducial matter density. 
        Om = (Omega_bh2 + Omega_cdmh2)/h**2
        z = redshift
        
        # try:
        #     bias_params = pardict['bias'][job_num]
            
        #     b1 = bias_params[0]
        #     #beta = f/b1, zero for real space. This is the redshift space distortion parameter. 
            
        #     # b1EFT = b1
        #     b2EFT = bias_params[1]
        #     b3EFT = bias_params[2]
        #     b4EFT = bias_params[3]
            
        # except:
        #     bias_params = pardict['bias']
            
        #     b1 = bias_params[0]
        #     #beta = f/b1, zero for real space. This is the redshift space distortion parameter. 
            
        #     # b1EFT = b1
        #     b2EFT = bias_params[1]
        #     b3EFT = bias_params[2]
        #     b4EFT = bias_params[3]
        
        # print(bias_params)
        
        be = fgrowth(z, Om)/b1
        
        # g2 = -2/7*(b1 - 1)
        # b2 = 0.0
        # g3 = 11/63*(b1 - 1);
        # # b2 = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3
         
        # g2x = -2/7*b2;
        # g21 = -22/147*(b1 - 1);
        # # b3 = -1.028 + 7.646*b1 - 6.227*b1**2 + 0.912*b1**3
        # b3 = -1.028 + 7.646*b1 - 6.227*b1**2 + 0.912*b1**3 + 4*g2x - 4/3*g3 - 8/3*g21 - 32/21*g2
        
        
        # g2 = 2.0/7.0*(b2EFT-b1)
        g2 = -2.0/7.0*(b1 - 1.0)
        # b2 = 2.0*(b4EFT+7.0/2.0*g2)
        # b2 = 0.0
        b2 = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 + 4/3*g2
        # b2 = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3
        # b2 = 0.0
        # bg3 = 1.0/6.0*((b3EFT-b1)/2.0-15.0*g2)
        # bg3 = (b3EFT-b1)/12.0 - 2.5*g2
        
        # # test = find_best_fit()
        
        # # c5, c6, c7, c8, c9, g2x,g21,b3 = test.x
        
        # g2x = 0.0
        g2x = -2/7.*b2;
        g21 = -22/147.*(b1 - 1);
        # g21 = (b3EFT - b1 - 30.0*g2)/12.0
        g3 = 11/63.*(b1 - 1);
        b3 = -1.028 + 7.646*b1 - 6.227*b1**2 + 0.912*b1**3 + 4*g2x - 4/3.*g3 - 8/3.*g21 - 32/21.*g2
        # b3 = 0.0
        
        # b3 = -1.028 + 7.646*b1 - 6.227*b1**2 + 0.912*b1**3 + 4*g2x + 796.0/1323.0*(b1-1)
        # b3 = -3.0*b2
        # b3 = 0.0
        # b3 = -1.028 + 7.646*b1 - 6.227*b1**2 + 0.912*b1**3
        
        # g2x = 0.0
        # g21 = 0.0
        # b3 = 0.0
        # g3 = 0.0
    
        T0.InitParameters([b1,be,g2,b2,0.0,g2x,g21,b3])
        
        # T0.InitParameters([b1,be,g2,b2,bg3])
        
        # print(g2, b2, bg3)
        
        scalar_amplitude = np.exp(log_10_10_As)/10**10
        my_cosmology, run, TT, EE, BB, TE, Pk_interpolator, fsigma8, sigma8_0 = power_spectrum(Omega_bh2 = Omega_bh2, Omega_cdmh2= Omega_cdmh2, H0 = 100*h,
        scalar_amplitude = scalar_amplitude)
        k_mode = np.logspace(-5, np.log10(kmax), 10000)
        #Find the power spectrum at redshift zero. 
        pdata = Pk_interpolator.P(0.0, k_mode)
        
        #Interpolate the power spectrum at the effective redshift. 
        Plin=InterpolatedUnivariateSpline(k_mode, Dz(redshift, Om)**2*b1**2*pdata)
        
        # Derivatives of the linear power spectrum
        dlnPk=derivative(Plin,k,dx=1e-4)*k/Plin(k)
        
        alpha = pardict['alpha'] 
        #The random catalogue I am using is 50 times bigger than the actual catalogue. alpha = Ng/Nr where Nr is the number
        #of galaxies in the random catalogue. 
        
        #Read in the pre-computed normalization factors.
        file = '/data/s4479813/Normalization_' + dire_1 + '_' + survey_name +'.npy'
        # file = './resource/Normalization_' + dire_1 + '_' + survey_name +'.npy'
        i22, i12, i11, i10, i24, i14, i34, i44, i32, i12oi22 = np.load(file)
        
        #The normalization needs to be corrected for the number of galaxies in the data catalogue. 
        
        i22 = i22*alpha
        i12 = i12*alpha
        i11 = i11*alpha
        i10 = i10*alpha
        i24 = i24*alpha
        i14 = i14*alpha
        i34 = i34*alpha
        i44 = i44*alpha
        i32 = i32*alpha
    
        _=np.load('/data/s4479813/FFT/SSC_window_kernel' + dire_1+ '_' + survey_name + '.npy')
        
        # _=np.load('./resource/SSC_window_kernel' + dire_1+ '_' + survey_name + '.npy')
        
        delete_index = np.where(np.isnan(_))
        if (len(delete_index[1]) != 0):
            _ = np.delete(_, np.unique(delete_index[1]), axis = 1)
        
        kwin = _[0]; powW22 = _[1:7]; powW10 = _[7:13]; powW22x10 = _[13:]
        
        [temp,temp2]=np.zeros((2,6)); temp3 = np.zeros(9)
        
        print("Finish reading in all pre-computed data.")
        
        # raise Exception('Test complete.')
        
        #This part calculates equation (65), (68) and (78) in https://arxiv.org/pdf/1910.02914.pdf .
        
        start = time.time()
        for i in range(9):
            Pwin=InterpolatedUnivariateSpline(kwin, powW22x10[i])
            temp3[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1], epsabs = 0.0, epsrel=1.0e-6, limit=500)[0]
    
            if(i<6):
                Pwin=InterpolatedUnivariateSpline(kwin, powW22[i])
                temp[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1], epsabs = 0.0, epsrel=1.0e-6, limit=500)[0]
                Pwin=InterpolatedUnivariateSpline(kwin, powW10[i])
                temp2[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1], epsabs = 0.0, epsrel=1.0e-6, limit=500)[0]
            else:
                continue
        
        end = time.time()
        print("It takes " + str(end - start) + " seconds to finish the integration.")
        
        sigma22Sq = MatrixForm(temp); sigma10Sq = MatrixForm(temp2); sigma22x10 = MatrixForm(temp3)
        
        # Kaiser terms
        rsd=np.zeros(5)
        rsd[0]=1 + (2*be)/3 + be**2/5
        rsd[2]=(4*be)/3 + (4*be**2)/7
        rsd[4]=(8*be**2)/35
        
        Z12Multipoles = np.vectorize(Z12Multipoles)
    
        # Terms used in the LA calculation
        covaLAterm=np.zeros((3,len(k)))
        for l in range(3):
            for i in range(3):
                for j in range(3):
                    covaLAterm[l]+=1/4.*sigma22x10[i,j]*Z12Multipoles(2*i,2*l,dlnPk)\
                    *quad(lambda mu: lp(2*j,mu)*(1 + be*mu**2), -1, 1)[0]
                    
        print("Finish computing the local average covariance matrix.")
        
        start = time.time()
        covaSSCmult=np.zeros((2*kbins,2*kbins))
        covaSSCmult[:kbins,:kbins]=covaSSC(0,0)
        covaSSCmult[kbins:,kbins:]=covaSSC(2,2)
        covaSSCmult[:kbins,kbins:]=covaSSC(0,2); 
        covaSSCmult[kbins:,:kbins]=np.transpose(covaSSCmult[:kbins,kbins:])
        end = time.time()
        print("It takes " + str(end - start) + " seconds to finish computing the super survey covariance matrix")
        
        start = time.time()
        covaG = CovMatGauss()
        end = time.time()
        print("It takes " + str(end-start) + " seconds to compute the Gaussian part of the covariance matrix.")
        
        # Constructing multipole covariance
        # Warning: the trispectrum takes a while to run
            
        #Start calculating the EFT covariance matrix with the UV subtraction. 
        covaT0_EFT_newmult=np.zeros((2*kbins,2*kbins))
        trisp = np.vectorize(trisp)
        
        start = time.time()
        for i in range(len(k)):
            covaT0_EFT_newmult[i,:kbins]=trisp(0,0,k[i],k)
            covaT0_EFT_newmult[i,kbins:]=trisp(0,2,k[i],k)
            covaT0_EFT_newmult[kbins+i,kbins:]=trisp(2,2,k[i],k)
        
        covaT0_EFT_newmult[kbins:,:kbins]=np.transpose(covaT0_EFT_newmult[:kbins,kbins:])
        
        end = time.time()
        print("It takes " + str(end - start) + " seconds to finish the trispectrum T0_EFT_new calculation.")
        
        covaNG=covaT0_EFT_newmult+covaSSCmult
        covaAnl=covaG+covaNG
        
        output_1 = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_SPT_UV_sub_' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        
        output_G = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_Gaussian' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        output_SSC = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_SSC' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        output_T0_1 = '/data/s4479813/pybird-desi/BOSS_DR12_FullShape/ACM_T0_SPT_UV_sub' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        
        # output_1 = './ACM_SPT_UV_sub_' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        
        # output_G = './ACM_Gaussian' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        # output_SSC = './ACM_SSC' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        # output_T0_1 = './ACM_T0_SPT_UV_sub' + dire_1 + '_' + survey_name + keyword + pardict['option'] + '.npy'
        
        np.save(output_1, covaAnl)
        np.save(output_G, covaG)
        np.save(output_SSC, covaSSCmult)
        np.save(output_T0_1, covaT0_EFT_newmult)