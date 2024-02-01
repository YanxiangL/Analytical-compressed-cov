# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:32:31 2022

@author: s4479813
"""

import scipy, time, sys
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import log, log10, cos, sin, pi, cosh, sinh , sqrt, amin, amax, mean, dot, power, conj
import matplotlib
from matplotlib import pyplot as plt
from scipy.misc import derivative
from numba import jit
import pandas as pd
import seaborn as sns; sns.set_theme(style='white')
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize
from configobj import ConfigObj
import camb

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

from numpy import linalg as la

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6. https://www.sciencedirect.com/science/article/pii/0024379588902236
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        print('All eigenvalues are positive')
        return A3
    
    length = len(np.where(np.linalg.eigvals(A3) < 0.0)[0])
    print(str(length) + ' eigenvalues are negative')
    
    
    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

if __name__ == "__main__":
    #The skycut parameter from the montepython routine.
    
    diag_mat = bool(int(sys.argv[1]))
    guess = int(sys.argv[2])
    option = int(sys.argv[3])
    individual = bool(int(sys.argv[4]))
    
    # if option == 0:
    #     key = 'Anal'
    # elif option == 1:
    #     key = 'Anal_comp'
    # elif option == 2:
    #     key = 'Anal_comp_taylor'
    # else:
    #     raise Exception('Incorrect option in the config file.')
    
    if option == 0:
        key = 'Anal'
    elif option == 1:
        key = 'Anal_comp'
    elif option == 2:
        key = 'Anal_comp_taylor'    
    elif option == 3:
        key = 'Anal_guess'
    elif option == 4:
        key = 'Anal_comp_guess'
    else:
        raise Exception('Incorrect option in the config file.')
        
    if option == 1 or option == 4:
        individual = False
    
    if guess == 1:
        keyword = '_guess_' + key
    else:
        keyword = '_best_' + key
    
    # inputfiles = ['./ACM_6dFGS_input.ini', './ACM_BOSS_input.ini', './ACM_eBOSS_LRG_input.ini', './ACM_eBOSS_QSO_input.ini']
    inputfiles = ['../config/ACM_6dFGS_input.ini', '../config/ACM_BOSS_input.ini', '../config/ACM_eBOSS_LRG_input.ini', '../config/ACM_eBOSS_QSO_input.ini']

    
    # num_cov = [1, 1, 2, 2]
    
    if diag_mat == False:
        
        if individual == True:
            covmat_file = ['cov_6dFGS_analytical_rescale' + keyword + '.txt',
                      'cov_BOSS_z1_NGC_analytical_rescale' + keyword + '.txt',
                      'cov_BOSS_z1_SGC_analytical_rescale' + keyword + '.txt',
                      'cov_BOSS_z2_NGC_analytical_rescale' + keyword + '.txt',
                      'cov_BOSS_z2_SGC_analytical_rescale' + keyword + '.txt',
                      'cov_eBOSS_LRG_NGC_z1_analytical_rescale' + keyword + '.txt',
                      'cov_eBOSS_LRG_SGC_z1_analytical_rescale' + keyword + '.txt',
                      'cov_eBOSS_QSO_NGC_z1_analytical_rescale' + keyword + '.txt',
                      'cov_eBOSS_QSO_SGC_z1_analytical_rescale' + keyword + '.txt']
        
        else:
            covmat_file = ['cov_all_analytical_rescale' + keyword + '.txt']
        
        # covmat_file = ['cov_6dFGS_analytical_rescale' + keyword + '.txt',
        #           'cov_BOSS_z1_z2_NGC_SGC_analytical_rescale' + keyword + '.txt',
        #           'cov_eBOSS_LRG_NGC_z1_analytical_rescale' + keyword + '.txt',
        #           'cov_eBOSS_LRG_SGC_z1_analytical_rescale' + keyword + '.txt',
        #           'cov_eBOSS_QSO_NGC_z1_analytical_rescale' + keyword + '.txt',
        #           'cov_eBOSS_QSO_SGC_z1_analytical_rescale' + keyword + '.txt']
        
    else:
        raise Exception('Not supported at the moment.')
        covmat_file = ['cov_6dFGS_analytical_rescale_diag.txt',
                  'cov_BOSS_z1_z2_NGC_SGC_analytical_rescale_diag.txt',
                  'cov_eBOSS_LRG_NGC_z1_analytical_rescale_diag.txt',
                  'cov_eBOSS_LRG_SGC_z1_analytical_rescale_diag.txt',
                  'cov_eBOSS_QSO_NGC_z1_analytical_rescale_diag.txt',
                  'cov_eBOSS_QSO_SGC_z1_analytical_rescale_diag.txt']
    
    cov_num = 0
    cov_all = []
    redindex = 0
    for i in range(len(inputfiles)):
        configfile = inputfiles[i]
    
        pardict = input_variable(configfile)
        k_num = pardict['k_bin']
        kmax = pardict['k_max']
        Omega_cdmh2 = pardict['omega_cdm']
        Omega_bh2 = pardict['omega_b']
        h = pardict['h']
        log_10_10_As = pardict['ln_10_10_As']
        red_num = pardict['red_num']
        hem_num = pardict['hem_num']
        kbinwidth = kmax/k_num
        survey_name = pardict['name']
        
        # cov_all = []
        for j in range(red_num*hem_num):
            red_bin, hemisphere = divmod(j, red_num)
            print(j, red_num, red_bin, hemisphere)
    
            #The effective redshift of each redshift bin. Please change this accordingly if you are using a different survey. 
            redshift_all = pardict['z_eff']
            try: 
                redshift = redshift_all[red_bin]
            except:
                redshift = redshift_all
                red_bin = 0
                hemisphere = 1
            
            if (hemisphere == 0):
                dire_1 = 'NGC' + '_' + str(red_bin)
                dire_2 = 'NGC'
            else:
                dire_1 = 'SGC' + '_' + str(red_bin)
                dire_2 = 'SGC'
            
            print(dire_1)
            
            if diag_mat == False:
                # output_3 = './ACM_SPT_UV_sub_' + dire_1 + '_' + survey_name + '.npy'
                output_3 = './ACM_SPT_UV_sub_' + dire_1 + '_' + survey_name + keyword + '.npy'
            else:
                output_3 = 'cov_'+ survey_name + '_' + dire_1 + '_rescaled_diag.npy'
            
            print(output_3)
            
            covaAnl3 = np.load(output_3)
            
            covaAnl3 = nearestPD(covaAnl3)
            
            cov_all.append(covaAnl3)
            
            if individual == True:
                np.savetxt(covmat_file[redindex], cov_all[redindex])
                redindex += 1
            
        # if num_cov[i] == 1:
        #     cov_output = scipy.linalg.block_diag(*cov_all)
        #     np.savetxt(covmat_file[cov_num], cov_output)
        #     print(covmat_file[cov_num])
        #     cov_num += 1
        # else:
        #     for k in range(red_num*hem_num):
        #         cov_output = cov_all[k]
        #         np.savetxt(covmat_file[cov_num], cov_output)
        #         print(covmat_file[cov_num])
        #         cov_num += 1
    
    if individual == False:
        np.savetxt(covmat_file[0], scipy.linalg.block_diag(*cov_all))
        
                
                
                
    
    