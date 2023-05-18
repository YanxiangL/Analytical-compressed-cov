# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:43:18 2022

@author: s4479813
"""

import numpy as np
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt, amin, amax, mean, dot, conj
import dask.array as da
import itertools as itt
import fitsio
from nbodykit.source.catalog import FITSCatalog, CSVCatalog
from nbodykit.lab import *
from configobj import ConfigObj
import sys

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

def fft(temp):
    """
    This function will cut out all the wavevectors beyond the specified kmax. This is because these wavevectors won't be used when calculating the analytical covariance 
    matrices and without cutting them out, the output file is very big. This means it would otherwise requires a lot of space to store and lots of memory when calculating 
    the analyical covariance matrix. This function will also shift the zero wavevector to the center of the array, the negative vector at the left and the postive vectors
    at the right.

    Parameters
    ----------
    temp : 3d numpy array.
        The window function after the Fourier transform.

    Returns
    -------
    temp2 : 3d numpy array.
        The window function after the cutting and shifting.

    """
    
    #This finds the center of the array. 
    
    ia_x = int(NX//2-1)
    ib_x = int(NX//2+1)
    ia_y = int(NY//2-1)
    ib_y = int(NY//2+1)
    ia_z = int(NZ//2-1)
    ib_z = int(NZ//2+1)
    
    #This will shift the window function corresponds to the negative wavevector to the left, postive to the right and the zero mode at the center. 
    temp2=np.zeros((NX,NY,NZ),dtype='<c8')
    temp2[ia_x:NX,ia_y:NY,ia_z:NZ]=temp[0:ib_x,0:ib_y,0:ib_z]; temp2[0:ia_x,ia_y:NY,ia_z:NZ]=temp[ib_x:NX,0:ib_y,0:ib_z]
    temp2[ia_x:NX,0:ia_y,ia_z:NZ]=temp[0:ib_x,ib_y:NY,0:ib_z]; temp2[ia_x:NX,ia_y:NY,0:ia_z]=temp[0:ib_x,0:ib_y,ib_z:NZ]
    temp2[0:ia_x,0:ia_y,ia_z:NZ]=temp[ib_x:NX,ib_y:NY,0:ib_z]; temp2[0:ia_x,ia_y:NY,0:ia_z]=temp[ib_x:NX,0:ib_y,ib_z:NZ]
    temp2[ia_x:NX,0:ia_y,0:ia_z]=temp[0:ib_x,ib_y:NY,ib_z:NZ]; temp2[0:ia_x,0:ia_y,0:ia_z]=temp[ib_x:NX,ib_y:NY,ib_z:NZ]
    
    temp2=temp2[ia_x-icut_x:ia_x+icut_x+1,ia_y-icut_y:ia_y+icut_y+1,ia_z-icut_z:ia_z+icut_z+1]
    
    return temp2

if __name__ == "__main__":
    configfile = sys.argv[1]
    #This is the job number which determines which FFT to calculate. 
    job_num = int(sys.argv[2])
    
    pardict = input_variable(configfile)
    red_num = pardict['red_num']
    hem_num = pardict['hem_num']
    
    #The power_num determines which window function to calculate. 
    remainder, power_num = divmod(job_num, 22)
    #The hemisphere here coresponds to the NGC/SGC hemisphere of BOSS and red_bin is the redshift bin in each hemisphere. 3 here indicates there 
    #are 3 different patches in each hemisphere in BOSS survey. 
    hemisphere, red_bin = divmod(remainder, red_num)
    
    red_min = pardict['red_min'][red_bin]
    red_max = pardict['red_max'][red_bin]
    kmax_cut = pardict['k_max']
    
    #The file path to the random file. 
    if (red_num*hem_num == 1):
        name = pardict['fitsfile']
    else:
        name = pardict['fitsfile'][hemisphere]
    
    survey_name = pardict['name']
    print(name, power_num, red_bin, red_min, red_max, kmax_cut)
    
    if (hemisphere == 0):
        dire_1 = 'NGC' + '_' + str(red_bin)
    else:
        dire_1 = 'SGC' + '_' + str(red_bin)
        
    #The fiducial matter density. 
    omega_m = pardict['omega_m']
    
    #The length of the simulation box in each direction for 6 different patches. You need to change these accordingly if you are using a different survey. 
    Lx_all = np.reshape(pardict['L_x'], (hem_num, red_num))
    Ly_all = np.reshape(pardict['L_y'], (hem_num, red_num))
    Lz_all = np.reshape(pardict['L_z'], (hem_num, red_num))
    
    # Lx_all = np.array([[1350.0, 1500.0, 1800.0], [1000.0, 850.0, 1000.0]])
    # Ly_all = np.array([[2450.0, 2850.0, 3400.0], [1900.0, 2250.0, 2600.0]])
    # Lz_all = np.array([[1400.0, 1600.0, 1900.0], [1100.0, 1300.0, 1500.0]])
    
    #The number of grid cells in each direction for 6 different patches. You also need to change this when you are using a different survey.  
    NX_all = np.reshape(pardict['N_x'], (hem_num, red_num))
    NY_all = np.reshape(pardict['N_y'], (hem_num, red_num))
    NZ_all = np.reshape(pardict['N_z'], (hem_num, red_num))
    
    # NX_all = np.array([[250, 290, 340], [190, 160, 190]], dtype=np.int16)
    # NY_all = np.array([[460, 540, 650], [360, 430, 500]], dtype=np.int16)
    # NZ_all = np.array([[260, 300, 360], [210, 240, 280]], dtype=np.int16)
    
    Lx = Lx_all[hemisphere][red_bin]
    Ly = Ly_all[hemisphere][red_bin]
    Lz = Lz_all[hemisphere][red_bin]
    
    NX = NX_all[hemisphere][red_bin]
    NY = NY_all[hemisphere][red_bin]
    NZ = NZ_all[hemisphere][red_bin]
    
    # NX = 16
    # NY = 16
    # NZ = 16
    
    #The distance is measured in Mpc/h, so we set h = 1.0. 
    cosmo = cosmology.Cosmology(h=1.0).match(Omega0_m=omega_m)
    # Nmesh = np.max([NX, NY, NZ]) # FFT size
    Nmesh = np.array([NX, NY, NZ]) # FFT size
    BoxSize = np.array([Lx, Ly, Lz]) #Box size, should encompass all the galaxies
    print(Nmesh, BoxSize)
    
    #The fundamental wavevectors in the x, y and z direction. 
    kfun_x = 2.0*np.pi/Lx
    kfun_y = 2.0*np.pi/Ly
    kfun_z = 2.0*np.pi/Lz
    
    #Determines the edge of the useful grid cells. All cells beyond these limits are being cut off. 
    icut_x = np.int32(np.ceil(kmax_cut/kfun_x))
    icut_y = np.int32(np.ceil(kmax_cut/kfun_y))
    icut_z = np.int32(np.ceil(kmax_cut/kfun_z))
    
    # Download the BOSS random catalog from https://data.sdss.org/sas/dr12/boss/lss/
    # RA, Dec, z, w_fkp, nbar are given by columns 0,1,2,3,4 in the fits files
    # Reference: https://nbodykit.readthedocs.io/en/latest/catalogs/reading.html?highlight=fits#FITS-Data
    
    randoms = FITSCatalog(name)
    randoms = randoms[(randoms['Z'] > red_min) * (randoms['Z'] < red_max)]
    print("Finish redshift cuts.")
    print(len(randoms['Z']))
    
    randoms['W12'] = randoms['WEIGHT_FKP']**2 
    randoms['W22'] = (randoms['WEIGHT_FKP']**2) * randoms['NZ']
    
    # Calculating Iij
    print("Finish reading in fits file.")
    
    if (power_num == 0):
        I22 = da.sum(randoms['NZ'] * randoms['WEIGHT_FKP']**2)
        I10 = da.sum(randoms['WEIGHT'])
        I12 = da.sum(randoms['WEIGHT_FKP']**2)
        I11 = da.sum(randoms['WEIGHT_FKP'])
        I24 = da.sum(randoms['NZ'] * randoms['WEIGHT_FKP']**4)
        I14 = da.sum(randoms['WEIGHT_FKP']**4)
        I34 = da.sum(randoms['NZ']**2 * randoms['WEIGHT_FKP']**4)
        I44 = da.sum(randoms['NZ']**3 * randoms['WEIGHT_FKP']**4)
        I32 = da.sum(randoms['NZ']**2 * randoms['WEIGHT_FKP']**2)
    
        output = np.array([I22.compute(), I12.compute(), I11.compute(), I10.compute(), I24.compute(), I14.compute(), I34.compute(), 
                            I44.compute(), I32.compute(), I12.compute()/I22.compute()])
        file = '/data/s4479813/Normalization_' + dire_1 + '_' + survey_name +'.npy'
        
        np.save(file, output)
        
        print(output)
    
    # randoms['OriginalPosition'] = transform.SkyToCartesian(
    #     randoms['RA'], randoms['DEC'], randoms['Z'], degrees=False, cosmo=cosmo)
    
    if pardict['name'] == '6dFGS':
        randoms['OriginalPosition'] = randoms['pos']
    else:
        randoms['OriginalPosition'] = transform.SkyToCartesian(
            randoms['RA'], randoms['DEC'], randoms['Z'], degrees=True, cosmo=cosmo)
    
    
    x = randoms['OriginalPosition'].T
    
    #This small factor is added in to avoid the floating point error. 
    AFE = 10e-10
    
    #This calculates the size of grid cells in each direction. 
    size_x = Lx/NX
    size_y = Ly/NY
    size_z = Lz/NZ
    
    x_new = np.array(x)
    x_min = np.min(x_new[0])
    y_min = np.min(x_new[1])
    z_min = np.min(x_new[2])
    print(x_min, y_min, z_min)
    
    grid_x_start = x_min - size_x - AFE
    grid_y_start = y_min - size_y - AFE
    grid_z_start = z_min - size_z - AFE
    
    print(grid_x_start, grid_y_start, grid_z_start)
    # # Shifting the points such that the survey center is in the center of the box
    # randoms['Position'] = randoms['OriginalPosition'] + da.array([BoxSize[0]/2.0, BoxSize[1]/2.0, BoxSize[2]/2.0])
    randoms['Position'] = randoms['OriginalPosition'] + da.array([-grid_x_start, -grid_y_start, -grid_z_start])
    
    pos = np.array(randoms['Position'].T)
    print(np.min(pos[0]), np.min(pos[1]), np.min(pos[2]), np.max(pos[0]), np.max(pos[1]), np.max(pos[2]))
    
    # # randoms['Position'] = randoms['OriginalPosition']
    # # Types of fourth-order FFTs that will be generated below
    # # w='W22'
    # # label=[]
    # # for (i,i_label),(j,j_label),(k,k_label),(l,l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
    # #         label.append(w + i_label + j_label + k_label + l_label)
    
    # # print(', '.join(label))
    
    label_all = []
    power_all = []
    for w in ['W22', 'W12']:
        label_all.append(w)
        power_all.append([])
        
        for (i,i_label),(j,j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
            label_all.append(w + i_label + j_label)
            power_all.append([i, j])
            
        for (i,i_label),(j,j_label),(k,k_label),(l,l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
            label_all.append(w + i_label + j_label + k_label + l_label)
            power_all.append([i, j, k, l])
    
    label_new = [label_all[power_num], label_all[power_num + 22]]
    power = power_all[power_num]
    
    print(label_new, power, len(power))
    
    count = 0
    for w in ['W22', 'W12']:
        
        label = label_new[count]
        if len(power) == 0:
            Wij = np.fft.fftn(randoms.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, value=w, resampler='tsc', interlaced=True, compensated=True).paint())
            print((da.sum(randoms[w]).compute()), np.real(Wij[0,0,0]))
            Wij *= (da.sum(randoms[w]).compute())/np.real(Wij[0,0,0]) #Fixing normalization, e.g., zero mode should be I22 for 'W22'
            
        elif len(power) == 2:
            i = power[0]
            j = power[1]
            randoms[label] = randoms[w] * x[i]*x[j] /(x[0]**2 + x[1]**2 + x[2]**2)
            Wij = np.fft.fftn(randoms.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, value=label, resampler='tsc', interlaced=True, compensated=True).paint())
            print((da.sum(randoms[label]).compute()), np.real(Wij[0,0,0]))
            Wij *= (da.sum(randoms[label]).compute())/np.real(Wij[0,0,0])
            print(power_num, i, j)
    
        else:
            i = power[0]
            j = power[1]
            k = power[2]
            l = power[3]
            
            randoms[label] = randoms[w] * x[i]*x[j]*x[k]*x[l] /(x[0]**2 + x[1]**2 + x[2]**2)**2
            Wij = np.fft.fftn(randoms.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, value=label, resampler='tsc', interlaced=True, compensated=True).paint())
            print((da.sum(randoms[label]).compute()), np.real(Wij[0,0,0]))
            Wij *= (da.sum(randoms[label]).compute())/np.real(Wij[0,0,0])
            
            print(power_num, i, j, k, l)
            
        
        Wij = fft(Wij)
        output_name = '/data/s4479813/FFT/FFT_grid_'+ label + '_' + dire_1 + '_' + survey_name +'W22.npy'
        # output_name = 'FFT_grid_'+ label + '_' + dire_1 + '_' + survey_name +'W22.npy'
        
        np.save(output_name, Wij)
        count += 1