# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:06:10 2022

@author: s4479813
"""

import numpy as np
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt, amin, amax, mean, dot, conj
import dask.array as da
import itertools as itt
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
# Calculating window power spectrum from FFT array
def PowerCalc(arr):
    _=np.zeros(nBins,dtype='<c8')
    for i in range(nBins):
        ind=(sort==i)
        _[i]=np.average(arr[ind])
    return(np.real(_))

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
    
    epsilon_cut = float(sys.argv[3])
    
    pardict = input_variable(configfile)
    red_num = pardict['red_num']
    hem_num = pardict['hem_num']
    
    #The hemisphere here coresponds to the NGC/SGC hemisphere of BOSS and red_bin is the redshift bin in each hemisphere. 3 here indicates there 
    #are 3 different patches in each hemisphere in BOSS survey. 
    hemisphere, red_bin = divmod(job_num, red_num)
    
    red_min = pardict['red_min'][red_bin]
    red_max = pardict['red_max'][red_bin]
    kmax_cut = pardict['k_max']
    
    #The file path to the random file. 
    if (red_num*hem_num == 1):
        name = pardict['fitsfile']
    else:
        name = pardict['fitsfile'][hemisphere]
    
    survey_name = pardict['name']
    print(name, job_num, red_bin, red_min, red_max, kmax_cut)
    
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
    
    NX = NX_all[hemisphere][red_bin]
    NY = NY_all[hemisphere][red_bin]
    NZ = NZ_all[hemisphere][red_bin]
    
    #Choose a larger size length to sample smaller kmodes because SSC mainly affect small kmodes. 
    Lx = Lx_all[hemisphere][red_bin]*2.0
    Ly = Ly_all[hemisphere][red_bin]*2.0
    Lz = Lz_all[hemisphere][red_bin]*2.0
    
    #The fundamental wavevectors in the x, y and z direction. 
    kfun_x = 2.0*np.pi/Lx
    kfun_y = 2.0*np.pi/Ly
    kfun_z = 2.0*np.pi/Lz
    
    #Determines the edge of the useful grid cells. All cells beyond these limits are being cut off. 
    icut_x = np.int32(np.ceil(epsilon_cut/kfun_x))
    icut_y = np.int32(np.ceil(epsilon_cut/kfun_y))
    icut_z = np.int32(np.ceil(epsilon_cut/kfun_z))
    
    print(Lx, Ly, Lz, icut_x, icut_y, icut_z)
    
    #The distance is measured in Mpc/h, so we set h = 1.0. 
    cosmo = cosmology.Cosmology(h=1.0).match(Omega0_m=omega_m)
    Nmesh = np.int32(np.array([NX, NY, NZ])) # FFT size
    BoxSize = np.array([Lx, Ly, Lz]) #Box size, should encompass all the galaxies
    
    # Download the BOSS random catalog from https://data.sdss.org/sas/dr12/boss/lss/
    # RA, Dec, z, w_fkp, nbar are given by columns 0,1,2,3,4 in the fits files
    # Reference: https://nbodykit.readthedocs.io/en/latest/catalogs/reading.html?highlight=fits#FITS-Data
    randoms = FITSCatalog(name)
    randoms = randoms[(randoms['Z'] > red_min) * (randoms['Z'] < red_max)]
    randoms['W22'] = (randoms['WEIGHT_FKP']**2) * randoms['NZ']
    try:
        randoms['W10'] = randoms['WEIGHT']
    except:
        randoms['W10'] = np.ones(len(randoms['W22']))
    
    print("Finish reading random files.")
    print(Nmesh, BoxSize)
    
    if pardict['name'] == '6dFGS':
        randoms['OriginalPosition'] = transform.SkyToCartesian(
            randoms['RA'], randoms['DEC'], randoms['Z'], degrees=False, cosmo=cosmo)
    else:
        randoms['OriginalPosition'] = transform.SkyToCartesian(
            randoms['RA'], randoms['DEC'], randoms['Z'], degrees=True, cosmo=cosmo)
        
    r = randoms['OriginalPosition'].T    
    
    #This small factor is added in to avoid the floating point error. 
    AFE = 10e-10
    
    #This calculates the size of grid cells in each direction. 
    size_x = Lx/NX
    size_y = Ly/NY
    size_z = Lz/NZ
    
    x_new = np.array(r)
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
    
    # # Shifting the points such that the survey center is in the center of the box
    # randoms['Position'] = randoms['OriginalPosition'] + da.array([BoxSize[0]/2.0, BoxSize[1]/2.0, BoxSize[2]/2.0])
    
    num_ffts = lambda n: int((n+1)*(n+2)/2) # Number of FFTs of nth order
    
    export=np.zeros((2*(1+num_ffts(2)+num_ffts(4)),2*icut_x+1,2*icut_y+1,2*icut_z+1),dtype='complex128')

    ind=0
    
    for w in ['W22', 'W10']:
        print(f'Computing FFTs of {w}')
        
        print('Computing 0th order FFTs')
        Wij = np.fft.fftn(randoms.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, value=w, resampler='tsc', interlaced=True, compensated=True).paint())
        Wij *= (da.sum(randoms[w]).compute())/np.real(Wij[0,0,0]) #Fixing normalization, e.g., zero mode should be I22 for 'W22'
        export[ind]=fft(Wij); ind+=1
        
        print('Computing 2nd order FFTs')
        for (i,i_label),(j,j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
            label = w + i_label + j_label
            randoms[label] = randoms[w] * r[i]*r[j] /(r[0]**2 + r[1]**2 + r[2]**2)
            Wij = np.fft.fftn(randoms.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, value=label, resampler='tsc', interlaced=True, compensated=True).paint())
            Wij *= (da.sum(randoms[label]).compute())/np.real(Wij[0,0,0])
            export[ind]=fft(Wij); ind+=1
    
        print('Computing 4th order FFTs')
        for (i,i_label),(j,j_label),(k,k_label),(l,l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
            label = w + i_label + j_label + k_label + l_label
            randoms[label] = randoms[w] * r[i]*r[j]*r[k]*r[l] /(r[0]**2 + r[1]**2 + r[2]**2)**2
            Wij = np.fft.fftn(randoms.to_mesh(Nmesh=Nmesh, BoxSize=BoxSize, value=label, resampler='tsc', interlaced=True, compensated=True).paint())
            Wij *= (da.sum(randoms[label]).compute())/np.real(Wij[0,0,0])
            export[ind]=fft(Wij); ind+=1
            
    
    # # For shifting the zero-frequency component to the center of the FFT array
    # for i in range(len(export)):
    #     export[i]=np.fft.fftshift(export[i])
        
    # Recording the k-modes in different shells
    # Bin_kmodes contains [kx,ky,kz,radius] values of all the modes in the bin
    
    # [kx,ky,kz] = np.zeros((3,Nmesh,Nmesh,Nmesh));
    
    # for i in range(len(kx)):
    #     kx[i,:,:]+=i-Nmesh/2; ky[:,i,:]+=i-Nmesh/2; kz[:,:,i]+=i-Nmesh/2
    
    # rk = np.sqrt((kx*kfun_x)**2 + (ky*kfun_y)**2 + (kz*kfun_z)**2)
    # binwidth = (kfun_x + kfun_y + kfun_z)/3.0
    # sort = (rk/binwidth).astype(int)
    # rk[nBins,nBins,nBins]=1e10;
    # kx = kx*kfun_x/rk
    # ky = ky*kfun_y/rk
    # kz = kz*kfun_z/rk
    # rk[nBins,nBins,nBins]=0
    
    [kx,ky,kz] = np.zeros((3,2*icut_x+1,2*icut_y+1,2*icut_z+1));
    
    # for i in range(len(kx)):
    #     kx[i,:,:]+=i-Nmesh/2; ky[:,i,:]+=i-Nmesh/2; kz[:,:,i]+=i-Nmesh/2
    for i in range(2*icut_x+1):
        kx[i, :, :] += i - icut_x
        
    for i in range(2*icut_y+1):
        ky[:, i, :] += i - icut_y
        
    for i in range(2*icut_z+1):
        kz[:, :, i] += i - icut_z
    
    rk = np.sqrt((kx*kfun_x)**2 + (ky*kfun_y)**2 + (kz*kfun_z)**2)
    binwidth = (kfun_x + kfun_y + kfun_z)/3.0
    sort = (rk/binwidth).astype(int)
    rk[icut_x,icut_y,icut_z]=1e10;
    kx = kx*kfun_x/rk
    ky = ky*kfun_y/rk
    kz = kz*kfun_z/rk
    rk[icut_x,icut_y,icut_z]=0
    
    print(np.min(sort), np.max(sort))
    
    # rk=np.sqrt(kx**2+ky**2+kz**2)
    # sort=(rk).astype(int)
    
    # rk[nBins,nBins,nBins]=1e10; kx/=rk; ky/=rk; kz/=rk; rk[nBins,nBins,nBins]=0 #rk being zero at the center causes issues so fixed that
    
    # Reading the FFT files for W22 (referred to as W hereafter for brevity) and W10

    [W, Wxx, Wxy, Wxz, Wyy, Wyz, Wzz, Wxxxx, Wxxxy, Wxxxz, Wxxyy, Wxxyz, Wxxzz, Wxyyy, Wxyyz, Wxyzz,\
     Wxzzz, Wyyyy, Wyyyz, Wyyzz, Wyzzz, Wzzzz, W10, W10xx, W10xy, W10xz, W10yy, W10yz, W10zz, W10xxxx,\
     W10xxxy, W10xxxz, W10xxyy, W10xxyz, W10xxzz, W10xyyy, W10xyyz, W10xyzz, W10xzzz, W10yyyy, W10yyyz,\
     W10yyzz, W10yzzz, W10zzzz] = export
    
    W_L0 = W
            
    W_L2=1.5*(Wxx*kx**2+Wyy*ky**2+Wzz*kz**2+2.*Wxy*kx*ky+2.*Wyz*ky*kz+2.*Wxz*kz*kx)-0.5*W
            
    W_L4=35./8.*(Wxxxx*kx**4 +Wyyyy*ky**4+Wzzzz*kz**4 \
         +4.*Wxxxy*kx**3*ky +4.*Wxxxz*kx**3*kz +4.*Wxyyy*ky**3*kx \
         +4.*Wyyyz*ky**3*kz +4.*Wxzzz*kz**3*kx +4.*Wyzzz*kz**3*ky \
         +6.*Wxxyy*kx**2*ky**2+6.*Wxxzz*kx**2*kz**2+6.*Wyyzz*ky**2*kz**2 \
         +12.*Wxxyz*kx**2*ky*kz+12.*Wxyyz*ky**2*kx*kz +12.*Wxyzz*kz**2*kx*ky) \
         -5./2.*W_L2 -7./8.*W_L0
    
    W10_L0 = W10
            
    W10_L2=1.5*(W10xx*kx**2+W10yy*ky**2+W10zz*kz**2+2.*W10xy*kx*ky+2.*W10yz*ky*kz+2.*W10xz*kz*kx)-0.5*W10
            
    W10_L4=35./8.*(W10xxxx*kx**4 +W10yyyy*ky**4+W10zzzz*kz**4 \
         +4.*W10xxxy*kx**3*ky +4.*W10xxxz*kx**3*kz +4.*W10xyyy*ky**3*kx \
         +4.*W10yyyz*ky**3*kz +4.*W10xzzz*kz**3*kx +4.*W10yzzz*kz**3*ky \
         +6.*W10xxyy*kx**2*ky**2+6.*W10xxzz*kx**2*kz**2+6.*W10yyzz*ky**2*kz**2 \
         +12.*W10xxyz*kx**2*ky*kz+12.*W10xyyz*ky**2*kx*kz +12.*W10xyzz*kz**2*kx*ky) \
         -5./2.*W10_L2 -7./8.*W10_L0
    
    nBins = np.int32((NX + NY + NZ)/6.0)
    
    P_W=np.zeros((22,nBins))

    # P_W[0]=PowerCalc(rk)*kfun # Mean |k| in the bin
    P_W[0]=PowerCalc(rk) # Mean |k| in the bin
    
    P_W[1]=PowerCalc(W_L0*conj(W_L0)) - da.sum(randoms['NZ']**2*randoms['WEIGHT_FKP']**4).compute() # P00 with shot noise subtracted
    P_W[2]=PowerCalc(W_L0*conj(W_L2))*5 # P02
    P_W[3]=PowerCalc(W_L0*conj(W_L4))*9 # P04
    P_W[4]=PowerCalc(W_L2*conj(W_L2))*25 # P22
    P_W[5]=PowerCalc(W_L2*conj(W_L4))*45 # P24
    P_W[6]=PowerCalc(W_L4*conj(W_L4))*81 # P44
    
    P_W[7]=PowerCalc(W10_L0*conj(W10_L0)) - da.sum(randoms['NZ']**0*randoms['WEIGHT_FKP']**0).compute() # P00 with shot noise subtracted
    P_W[8]=PowerCalc(W10_L0*conj(W10_L2))*5 # P02
    P_W[9]=PowerCalc(W10_L0*conj(W10_L4))*9 # P04
    P_W[10]=PowerCalc(W10_L2*conj(W10_L2))*25 # P22
    P_W[11]=PowerCalc(W10_L2*conj(W10_L4))*45 # P24
    P_W[12]=PowerCalc(W10_L4*conj(W10_L4))*81 # P44
    
    P_W[13]=PowerCalc(W_L0*conj(W10_L0)) - da.sum(randoms['NZ']**1*randoms['WEIGHT_FKP']**2).compute() # P00 with shot noise subtracted
    P_W[14]=PowerCalc(W_L0*conj(W10_L2))*5 # P02
    P_W[15]=PowerCalc(W_L0*conj(W10_L4))*9 # P04
    P_W[16]=PowerCalc(W_L2*conj(W10_L0))*5 # P20
    P_W[17]=PowerCalc(W_L2*conj(W10_L2))*25 # P22
    P_W[18]=PowerCalc(W_L2*conj(W10_L4))*45 # P24
    P_W[19]=PowerCalc(W_L4*conj(W10_L0))*9 # P40
    P_W[20]=PowerCalc(W_L4*conj(W10_L2))*45 # P42
    P_W[21]=PowerCalc(W_L4*conj(W10_L4))*81 # P44
    
    P_W[1:7]/=(da.sum(randoms['W22']).compute())**2
    P_W[7:13]/=(da.sum(randoms['W10']).compute())**2
    P_W[13:]/=(da.sum(randoms['W10']).compute()*da.sum(randoms['W22']).compute())
    
    # Minor point: setting k=0 modes by hand to avoid spurious values
    P_W[1:7,0]=[1,0,0,1,0,1]; P_W[7:13,0]=[1,0,0,1,0,1]; P_W[13:,0]=[1,0,0,0,1,0,0,0,1]
    
    output_name = '/data/s4479813/FFT/SSC_window_kernel' + dire_1+ '_' + survey_name + '.npy'
    # output_name = 'SSC_window_kernel' + dire_1+ '_' + survey_name + '.npy'
    np.save(output_name, P_W)