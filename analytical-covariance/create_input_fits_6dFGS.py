# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:26:46 2022

@author: s4479813
"""

import numpy as np
import pandas as pd
import sys
from configobj import ConfigObj
from astropy.io import fits
from astropy.table import Table
import fitsio
from nbodykit.source.catalog import FITSCatalog, CSVCatalog
from nbodykit.lab import * 

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

if __name__ == "__main__":
    configfile = sys.argv[1]
    pardict = input_variable(configfile)
    
    random_all = []
    for i in range(60):
        random_all.append(np.loadtxt('ran_'+str(i)+'_FKP_P10000.dat.orig'))
        print(i)
    
    
    random = np.vstack(random_all)

    # xmock = random[:,0]

    # ymock = random[:, 1]

    # zmock = random[:,2]

    w_FKP = random[:, 3]

    xdata = random[:, 0]

    ydata = random[:, 1]

    zdata = random[:, 2]

    P0 = 10000

    P0 = 10000.0

    nbar = (1.0/w_FKP - 1.0)/P0

    alpha = 75117.0/len(random)
    
    weights = np.ones(len(random))
        
    print(alpha)
        
    FKP_weight = pardict['FKP']
    
    FKP = w_FKP
    
    pos = np.array([xdata, ydata, zdata]).T
    
    data = Table([pos, nbar, weights, FKP], names=('pos', 'NZ', 'WEIGHT', 'WEIGHT_FKP'))
    data.write('6dFGS_temfile.fits', format='fits')
    
    randoms = FITSCatalog('6dFGS_temfile.fits')
    
    #The fiducial matter density. 
    omega_m = pardict['omega_m']
    
    cosmo = cosmology.Cosmology(h=1.0).match(Omega0_m=omega_m)
    
    randoms['RA'], randoms['Dec'], randoms['z'] = transform.CartesianToSky(randoms['pos'], cosmo)
    
    RA = np.array(randoms['RA'])
    Dec = np.array(randoms['Dec'])
    z = np.array(randoms['z'])
    
    print(np.max(z))
    print(len(np.where((z >= 0.01) & (z <= 0.20))[0]))
    
    output_name = '6dFGS_random.fits'
    
    data_final = Table([RA, Dec, z, nbar, weights, FKP, np.array(randoms['pos'])], names=('RA', 'DEC', 'Z', 'NZ', 'WEIGHT', 'WEIGHT_FKP', 'pos'))
    data_final.write(output_name, format='fits')
    
    
    
    
    