# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:18:53 2022

@author: s4479813
"""

import numpy as np
from scipy.linalg import inv, det, block_diag
#also import all other necessary libaries from pybird
import pandas as pd
import sys
from configobj import ConfigObj
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
import os
import cmath


def Compression_vector(C_inv, P_dash, n_params):
    """
    This function generates the elements of the compression matrix from the input power spectrum and covariance matrices 
    based on the formula given in Heavens, et al (2000) (equation 11 and equation 14). 

    Parameters
    ----------
    C_inv : numpy array
        The inverse of the covariance matrix generated from the simulation.
    P_dash : list of 1d numpy arrays or 2d numpy arrays
        This list should contain the derivative of the power spectrum with respective to each free parameter (an array).
    n_params : int
        This is the total number of free parameters.

    Returns
    -------
    B: numpy array
        The compression matrix.

    """
    #Initialising the output compression matrix
    length = len(P_dash[0])
    B = []
    for i in range(n_params):
        if (i == 0): 
            B.append(np.matmul(C_inv, P_dash[i])/np.sqrt(np.matmul(P_dash[i].T, np.matmul(C_inv, P_dash[i]))))
        else:
            part_one = np.matmul(C_inv, P_dash[i])
            part_two = np.zeros(length)
            part_three = 0.0
            for j in range(i):
                part_two += np.matmul(P_dash[i].T, B[j])*B[j]
                part_three += pow(np.matmul(P_dash[i].T, B[j]), 2.0)
            part_four = np.matmul(P_dash[i].T, np.matmul(C_inv, P_dash[i]))
            
            # #
            # if (part_four - part_three <= 1e-12):
            #     B.append((part_one - part_two)/np.sqrt(1e-22))
            #     print(i)
            #     print(part_four - part_three)
            # else:
            B.append((part_one - part_two)/np.sqrt(part_four - part_three))
            # # print("Finish " + str(i+1) + " over " + str(n_params))
    
    return np.array(B)

def input_variable(input_file):
    pardict = ConfigObj(input_file)
    pardict['ln_10_10_As'] = np.float32(pardict['ln10^{10}A_s'])
    pardict['h'] = np.float32(pardict['h'])
    pardict['omega_b'] = np.float32(pardict['omega_b'])
    pardict['omega_cdm'] = np.float32(pardict['omega_cdm'])
    pardict['order'] = np.int32(pardict['order'])
    pardict['dx'] = np.array(pardict['dx'], dtype=np.float32).reshape(4)
    pardict['nl'] = np.int32(pardict['nl'])
    pardict['k_m'] = np.float32(pardict['k_m'])
    pardict['k_nl'] = np.float32(pardict['k_nl'])
    pardict['z_tot'] = np.int32(pardict['z_tot'])
    pardict['nd'] = np.array(pardict['nd'], dtype=np.float32).reshape(pardict['z_tot'])
    pardict['z_pk'] = np.array(pardict['z_pk'], dtype=np.float32).reshape(pardict['z_tot'])
    
    return pardict

if __name__ == "__main__":

    #Read in the configure files. 
    configfile = sys.argv[1]
    #This determines whether to calculates the compression matrix for all the surveys together or calculating the compression
    #matrix for each individual survey. Enter 1 for compression for individual survey and 0 otherwise.
    individual = bool(int(sys.argv[2]))#Whether we calculate the compression matrix for all redshift bins together or individually. 
    guess = bool(int(sys.argv[3])) #whether we use the guess parameters or bestfit parameters from the previous fit.
    use_sim = bool(int(sys.argv[4]))#Whether we want to use the simulated covariance matrix. 
    
    #Read in the pardict with proper formating. 
    pardict = input_variable(configfile)
    
    if use_sim == False:
        if guess == True:
            cov_keyword = '_guess_' + pardict['option']
        else:
            cov_keyword = '_best_' + pardict['option']
    
        pardict["covfile"] = '../BOSS_DR12_FullShape/cov_all_analytical_rescale' + cov_keyword + '.txt'
    
        keyword = pardict['option']
        
    else:
        keyword = 'sim_comp'
    
    if 'comp' not in keyword:
        raise Exception('Incorrect pardict option. You are not using the bestfit with the compressed covariance matrix.')
 
    #This is the name of the grid file
    gridnames = np.loadtxt(pardict["gridname"], dtype=str)
    #The location of where the grid files are being produced
    outgrids = np.loadtxt(pardict["outgrid"], dtype=str)
    #The name of the output file of the derivative of the power spectrum with respect to cosmological parameters.
    # name = pardict["code"].lower() + "-" + gridnames[redindex]
    
    skycut_all = np.array(pardict['skycuts'], dtype=np.int32)
    
    k_m = pardict['k_m']
    k_nl = pardict['k_nl']
        
    print(pardict['covfile'])
    
    cov_input = np.array(pd.read_csv(pardict['covfile'], delim_whitespace=True, header=None))
    
    # cov_all = []
    #This applies the masks to your covariance matrix and power spectrum. 
    mask_all = np.load(pardict['maskfile'], allow_pickle = True)
    mask_comb = []
    for i in range(pardict['z_tot']):
        mask = mask_all[i]
        mask_length = len(mask)
        
        if use_sim == False:
            #Assume mask including the hexadecapole. The analytical covaraince matrix doesn't support Hexadecapole
            mask_comb.append(mask[:np.int32(mask_length/3*2)])
        else:
            #The simulated covariance matrix will include the hexadecapole. 
            mask_comb.append(mask)
        
        
    mask_comb = np.concatenate(mask_comb)
    
    cov_all = np.delete(np.delete(cov_input, np.where(~mask_comb==True), axis=0), np.where(~mask_comb==True), axis=1)
    
    if guess == False:
        bestfit = np.load('../model_PS/bestfit_marg_' + pardict['option'] + '.npy')
        cosmo_params = bestfit[:4]
        bs_all = []
        for i in range(len(pardict['z_pk'])):
            bs_all.append(bestfit[4+i*10:4+(i+1)*10])
            print(bestfit[4+i*10:4+(i+1)*10])
                
    else:
        cosmo_params = np.array([pardict['ln10^{10}A_s'], pardict['h'], pardict['omega_cdm'], pardict['omega_b']])
        bs_all = []
        for i in range(len(pardict['z_pk'])):
            if pardict['do_hex']:
                bs_all.append([1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            else:
                bs_all.append([1.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5])
        
        # print(cosmo_params)
    
    #This sets the number of multipoles. 
    nmult = pardict['nl']
    nmult_grid = 3
    
    pdash = []
    
    
    pdashcosmo = []
    
    count = 0
    # for i in range(len(skycut_all)):       
       
    #     for j in range(skycut_all[i]):
    for zpk in range(len(pardict['z_pk'])):
        
        #read in the bias parameters. 
        bs = bs_all[count]
        b1, b2, b3, b4, b5, b6, b7, ce1, ce2, ce3 = bs
        
        #The derivatives are done in the order of b1, b2, b3, b4, b5, b6, b7, ce1, ce2, ce3. 
        #The growth rate of structure has already been taken into account when generating the power spectrum, so I set it to 1 here. 
        f = 1.0
        # derivative_lin = np.array([[0, f, 2.*b1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])    
        # derivative_ct = np.array([[b5/k_nl**2, b6/k_m**2, b7/k_m**2, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [b1/k_nl**2, 0, 0, f/k_nl**2, 0, 0], [0, b1/k_m**2, 0, 0, f/k_m**2, 0], [0, 0, b1/k_m**2, 0, 0, f/k_m**2]])
        # derivative_loops = np.array([[0, 1, 0, 0, 0, 2.*b1, b2, b3, b4, 0, 0, 0], [0, 0, 1, 0, 0, 0, b1, 0, 0, 2*b2, b4, 0], [0, 0, 0, 1, 0, 0, 0, b1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, b1, 0, b2, 2.*b4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        derivative_lin = np.array([[0, f, 2.*b1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])    
        derivative_ct = np.array([[b5/k_nl**2, b6/k_m**2, b7/k_m**2, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [b1/k_nl**2, 0, 0, f/k_nl**2, 0, 0], [0, b1/k_m**2, 0, 0, f/k_m**2, 0], [0, 0, b1/k_m**2, 0, 0, f/k_m**2], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        derivative_loops = np.array([[0, 1, 0, 0, 0, 2.*b1, b2, b3, b4, 0, 0, 0], [0, 0, 1, 0, 0, 0, b1, 0, 0, 2*b2, b4, 0], [0, 0, 0, 1, 0, 0, 0, b1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, b1, 0, b2, 2.*b4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        derivative_stl = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])        
        
        #The coefficients required to construct the matter power spectrum. 
        # blinear = np.array([ b1**2, b1*f, f**2 ])
        blinear = np.array([f**2, b1*f, b1**2])
        bct = np.array([ b1*b5/k_nl**2, b1*b6/k_m**2, b1*b7/k_m**2, f*b5/k_nl**2, f*b6/k_m**2, f*b7/k_m**2])
        bloop = np.array([ 1., b1, b2, b3, b4, b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4 ])
        bstl = np.array([ce1, ce2, ce3])
        
        #THe new grid name. 
        gridname = pardict["code"].lower() + "-" + gridnames[count]

        #Read in the saved linear and loop power spectra. 
        plin = np.load(os.path.join(outgrids[count], "TablePlin_%s%s.npy" % (gridname, pardict['keyword'])))
        
        ploop = np.load(os.path.join(outgrids[count], "TablePloop_%s%s.npy" % (gridname, pardict['keyword'])))    
        
        shapecrd = np.concatenate([[len(pardict["freepar"])], np.full(len(pardict["freepar"]), 2 * int(pardict["order"]) + 1)])
        padshape = [(1, 1)] * (len(shapecrd) - 1)
        
        
        plin_shape = plin.reshape((*shapecrd[1:], nmult_grid, plin.shape[-2] // nmult_grid, plin.shape[-1]))
        
        ploop_shape = ploop.reshape((*shapecrd[1:], nmult_grid, ploop.shape[-2] // nmult_grid, ploop.shape[-1]))
        
        order = int(pardict["order"])
        
        #Obtain the power spectrum at the center grid point. 
        plin = plin_shape[order, order, order, order, :, :, 1:]
        
        ploop = ploop_shape[order, order, order, order, :, :, 1:13]
        
        pct = ploop_shape[order, order, order, order, :, :, 13:19]
        
        pstl = ploop_shape[order, order, order, order, :, :, 19:]
        
        #Read in the saved derivatives of the power spectrum with respect to the cosmological parameters. The index 1 at the end means we only 
        #use the first derivative. 
        P_dash_lin = np.load(os.path.join(outgrids[count], "DerPlin_%s%s.npy" % (gridname, pardict['keyword'])), allow_pickle=True)[1]
        P_dash_loop_all = np.load(os.path.join(outgrids[count], "DerPloop_%s%s.npy" % (gridname, pardict['keyword'])), allow_pickle=True)[1]
        
        P_dash_lin = P_dash_lin[:,:,:,1:]
        P_dash_loop = P_dash_loop_all[:,:,:,1:13]
        P_dash_ct = P_dash_loop_all[:,:,:,13:19]
        
        P_dash_stl = P_dash_loop_all[:, :, :, 19:]
        
        pdashlin = np.einsum("b, kxyb -> kxy", blinear, P_dash_lin)
        pdashloop = np.einsum("b, kxyb -> kxy", bloop, P_dash_loop)
        pdashct = np.einsum("b, kxyb -> kxy", bct, P_dash_ct)
        pdashstl = np.einsum("b, kxyb -> kxy", bstl, P_dash_stl)
        
        #The derivative of the power spectrum with respect to the cosmological parameters. 
        P_dash_cosmo_grid = pdashlin + pdashloop + pdashct + pdashstl
        
        P_dash_cosmo_grid_mono = P_dash_cosmo_grid[:, 0]
        P_dash_cosmo_grid_quad = P_dash_cosmo_grid[:, 1]
        
        #The kmode for the monopole, quadrupole, and hexadecapole respectively. 
        kmode_all = np.load(pardict['kmodefile'], allow_pickle = True)[count]
        mask_all = np.load(pardict['maskfile'], allow_pickle = True)[count]
        
        length = len(kmode_all)
        
        # mask = mask_all[j*length*nmult: (j+1)*length*nmult]
        
        mask = mask_all
        
        mask_mono = mask[:length]
        mask_quad = mask[length:2*length]
        
        kmode_mono = np.delete(kmode_all, np.where(~mask_mono==True))
        kmode_quad = np.delete(kmode_all, np.where(~mask_quad==True))
        
        kmode_mono = kmode_mono.reshape(1, len(kmode_mono))
        kmode_quad = kmode_quad.reshape(1, len(kmode_quad))
        
        if nmult == 3:
            mask_hexa = mask_all[2*length:]
            
            kmode_hexa = np.delete(kmode_all, np.where(~mask_hexa == True), axis = 0)
            kmode_hexa = kmode_hexa.reshape(1, len(kmode_hexa))
        
        P_dash_cosmo_grid_mono = np.delete(P_dash_cosmo_grid_mono, np.where(~mask_mono == True), axis = -1)
        P_dash_cosmo_grid_quad = np.delete(P_dash_cosmo_grid_quad, np.where(~mask_quad == True), axis = -1)
        
        if nmult == 3:
            P_dash_cosmo_grid_hexa = P_dash_cosmo_grid[:, 2]
            P_dash_cosmo_grid_hexa = np.delete(P_dash_cosmo_grid_hexa, np.where(~mask_hexa == True), axis = -1)
            # P_dash_cosmo_bin = np.concatenate((P_dash_cosmo_grid[:,0,:length_mono], P_dash_cosmo_grid[:,1,:length_quad], P_dash_cosmo_grid[:,2,:length_hexa]), axis = -1)
            P_dash_cosmo_bin = np.concatenate((P_dash_cosmo_grid_mono, P_dash_cosmo_grid_quad, P_dash_cosmo_grid_hexa), axis = -1)
        else:
            P_dash_cosmo_bin = np.concatenate((P_dash_cosmo_grid_mono, P_dash_cosmo_grid_quad), axis = -1)
            # P_dash_cosmo_bin = np.concatenate((P_dash_cosmo_grid[:,0,:length_mono], P_dash_cosmo_grid[:,1,:length_quad]), axis = -1)
        pdashcosmo.append(P_dash_cosmo_bin)
        
        #The derivative of the power spectrum with respect ot the bias parameters. 
        
        p_dash_lin_EFT = np.einsum("kb, xyb -> kxy", derivative_lin, plin)
        p_dash_loop_EFT = np.einsum("kb, xyb -> kxy", derivative_loops, ploop)
        p_dash_ct_EFT = np.einsum("kb, xyb -> kxy", derivative_ct, pct)
        p_dash_stl_EFT = np.einsum("kb, xyb -> kxy", derivative_stl, pstl)
        
        P_dash_EFT = p_dash_lin_EFT + p_dash_loop_EFT + p_dash_ct_EFT + p_dash_stl_EFT
        
        P_dash_EFT_mono = P_dash_EFT[:, 0]
        P_dash_EFT_quad = P_dash_EFT[:, 1]
        
        P_dash_EFT_mono = np.delete(P_dash_EFT_mono, np.where(~mask_mono==True), axis = -1)
        P_dash_EFT_quad = np.delete(P_dash_EFT_quad, np.where(~mask_quad==True), axis = -1)
        
        if nmult == 3:
            P_dash_EFT_hexa = P_dash_EFT[:, 2]
            P_dash_EFT_hexa = np.delete(P_dash_EFT_hexa, np.where(~mask_hexa==True), axis = -1)
            P_dash_EFT_bin = np.concatenate((P_dash_EFT_mono, P_dash_EFT_quad, P_dash_EFT_hexa), axis = -1)
            # P_dash_EFT_bin = np.concatenate((P_dash_EFT[:,0,:length_mono], P_dash_EFT[:,1,:length_quad], P_dash_EFT[:,2,:length_hexa]), axis = -1)
        else:
            P_dash_EFT_bin = np.concatenate((P_dash_EFT_mono, P_dash_EFT_quad), axis = -1)
            # P_dash_EFT_bin = np.concatenate((P_dash_EFT[:,0,:length_mono], P_dash_EFT[:,1,:length_quad]), axis = -1)
            
        
        #The Stochastic power spectrum after applying the window
        # Pstl = np.load(os.path.join(pardict["outpk"], "all_redindex%d" % (count), "Pstl_run%s.npy" % (str(0))))
        
        # #The shot-noise has already been taken into account when generating the stochastic power spectrum. 
        # if nmult == 2:
        #     Pst_ce1, ce1_quad = np.einsum("b,lbx->lx", np.array([1.0, 0.0, 0.0]), Pstl)
        #     Pst_ce2, ce2_quad = np.einsum("b,lbx->lx", np.array([0.0, 1.0, 0.0]), Pstl)
        #     ce3_mono, Pst_ce3 = np.einsum("b,lbx->lx", np.array([0.0, 0.0, 1.0]), Pstl)
            
        #     Pst_ce1 = np.delete(Pst_ce1, np.where(~mask_mono==True), axis=-1)
        #     Pst_ce2 = np.delete(Pst_ce2, np.where(~mask_mono==True), axis=-1)
        #     Pst_ce3 = np.delete(Pst_ce3, np.where(~mask_quad==True), axis=-1)
        #     # length = len(kmode_mono) + len(kmode_quad) + len(kmode_hexa)
            
        #     ce1_quad = np.delete(ce1_quad, np.where(~mask_quad==True), axis=-1)
        #     ce2_quad = np.delete(ce2_quad, np.where(~mask_quad==True), axis=-1)
        #     ce3_mono = np.delete(ce3_mono, np.where(~mask_mono==True), axis=-1)
        #     # length = len(kmode_mono) + len(kmode_quad) + len(kmode_hexa)
            
        #     #Compute the shot-noise contribution to the power spectrum. 
        #     Pst_ce1 = Pst_ce1.reshape(1, len(Pst_ce1))
        #     Pst_ce2 = Pst_ce2.reshape(1, len(Pst_ce2))
        #     Pst_ce3 = Pst_ce3.reshape(1, len(Pst_ce3))
            
        #     ce1_quad = ce1_quad.reshape(1, len(ce1_quad))
        #     ce2_quad = ce2_quad.reshape(1, len(ce2_quad))
        #     ce3_mono = ce3_mono.reshape(1, len(ce3_mono))
            
        # else:
        #     Pst_ce1, ce1_quad, ce1_hexa = np.einsum("b,lbx->lx", np.array([1.0, 0.0, 0.0]), Pstl)
        #     Pst_ce2, ce2_quad, ce2_hexa = np.einsum("b,lbx->lx", np.array([0.0, 1.0, 0.0]), Pstl)
        #     ce3_mono, Pst_ce3, ce3_hexa = np.einsum("b,lbx->lx", np.array([0.0, 0.0, 1.0]), Pstl)
            
        #     Pst_ce1 = np.delete(Pst_ce1, np.where(~mask_mono==True), axis=-1)
        #     Pst_ce2 = np.delete(Pst_ce2, np.where(~mask_mono==True), axis=-1)
        #     Pst_ce3 = np.delete(Pst_ce3, np.where(~mask_quad==True), axis=-1)
        #     # length = len(kmode_mono) + len(kmode_quad) + len(kmode_hexa)
            
        #     ce1_quad = np.delete(ce1_quad, np.where(~mask_quad==True), axis=-1)
        #     ce2_quad = np.delete(ce2_quad, np.where(~mask_quad==True), axis=-1)
        #     ce3_mono = np.delete(ce3_mono, np.where(~mask_mono==True), axis=-1)
        #     # length = len(kmode_mono) + len(kmode_quad) + len(kmode_hexa)
            
        #     ce1_hexa = np.delete(ce1_hexa, np.where(~mask_hexa==True), axis=-1)
        #     ce2_hexa = np.delete(ce2_hexa, np.where(~mask_hexa==True), axis=-1)
        #     ce3_hexa = np.delete(ce3_hexa, np.where(~mask_hexa==True), axis=-1)
            
        #     #Compute the shot-noise contribution to the power spectrum. 
        #     Pst_ce1 = Pst_ce1.reshape(1, len(Pst_ce1))
        #     Pst_ce2 = Pst_ce2.reshape(1, len(Pst_ce2))
        #     Pst_ce3 = Pst_ce3.reshape(1, len(Pst_ce3))
            
        #     ce1_quad = ce1_quad.reshape(1, len(ce1_quad))
        #     ce2_quad = ce2_quad.reshape(1, len(ce2_quad))
        #     ce3_mono = ce3_mono.reshape(1, len(ce3_mono))
            
        #     ce1_hexa = ce1_hexa.reshape(1, len(ce1_hexa))
        #     ce2_hexa = ce2_hexa.reshape(1, len(ce2_hexa))
        #     ce3_hexa = ce3_hexa.reshape(1, len(ce3_hexa))
        
        # if nmult == 3:
        #     PS_dash_SN_1 = np.concatenate((Pst_ce1, ce1_quad, ce1_hexa), axis = -1)
        #     PS_dash_SN_2 = np.concatenate((Pst_ce2, ce2_quad, ce2_hexa), axis = -1)
        #     PS_dash_SN_3 = np.concatenate((ce3_mono, Pst_ce3, ce3_hexa), axis = -1)
        # else:
        #     PS_dash_SN_1 = np.concatenate((Pst_ce1, ce1_quad), axis = -1)
        #     PS_dash_SN_2 = np.concatenate((Pst_ce2, ce2_quad), axis = -1)
        #     PS_dash_SN_3 = np.concatenate((ce3_mono, Pst_ce3), axis = -1)
        
        # PS_dash_SN = np.concatenate((PS_dash_SN_1, PS_dash_SN_2, PS_dash_SN_3), axis=0)
        
        # PS_dash_bin_EFT = np.concatenate((P_dash_EFT_bin, PS_dash_SN), axis = 0)
        PS_dash_bin_EFT = P_dash_EFT_bin
        
#There are different ways to arrange the bias parameters as you can see in the bias_nonmarg_to_all function in 
#likelihood_class.py. Here, I assume with_stoch == True, multipoles == 2 and model == 4. In this case, cr2 is set to 0, and 
#c2 = sqrt(2)(b2 + b4) and c4 = 0, so b2 = b4. We have 8 different free parameters, b1, c2, b3, cct, cr1, ce1, ce2, ce3.
#Hence, we have to delete the derivative with respect to cr2 and calculate the derivative with respect to c2. If you are
#using a different model. Please delete the row of the coorresponding fixed parameters. 
        
        # PS_dash_c2 = np.sqrt(2.0)*PS_dash_bin_EFT[1] + np.sqrt(2.0)*PS_dash_bin_EFT[3]
        PS_dash_c2 = 1.0/np.sqrt(2.0)*PS_dash_bin_EFT[1] + 1.0/np.sqrt(2.0)*PS_dash_bin_EFT[3]

        
        if nmult == 2:
            PS_dash_bin_EFT = np.concatenate((PS_dash_bin_EFT[0], PS_dash_c2, PS_dash_bin_EFT[2], PS_dash_bin_EFT[4], 
                                              PS_dash_bin_EFT[5], PS_dash_bin_EFT[7], PS_dash_bin_EFT[8], PS_dash_bin_EFT[9]),
                                              axis = 0)
            
            PS_dash_bin_EFT = np.reshape(PS_dash_bin_EFT, (8, len(PS_dash_c2)))
            
        else:
            PS_dash_bin_EFT = np.concatenate((PS_dash_bin_EFT[0], PS_dash_c2, PS_dash_bin_EFT[2], PS_dash_bin_EFT[4], 
                                              PS_dash_bin_EFT[5], PS_dash_bin_EFT[6], PS_dash_bin_EFT[7], PS_dash_bin_EFT[8], 
                                              PS_dash_bin_EFT[9]), axis = 0)
            
            PS_dash_bin_EFT = np.reshape(PS_dash_bin_EFT, (9, len(PS_dash_c2)))
        
        
        pdash.append(PS_dash_bin_EFT)
        
        count += 1
        
    
    #The mono-, quadru-, and hexadeca- poles are already concatenate together in each redshift bin. 
    #Concatenate the power spectrum in all redshift bins together. 
    # PS_full = np.concatenate([p[i] for i in range(len(pardict["z_pk"]))], axis = -1)
    if individual == False:
        
        # cov = block_diag(*cov_all)
        cov = cov_all
        C_inv = np.linalg.inv(cov)
        
        #Concatenate the compression vectors from all redshift bins. 
        PS_dash_cosmo = np.concatenate([pdashcosmo[i] for i in range(pardict['z_tot'])], axis = -1)
        
        # PS_const_full = np.concatenate([pconst[i] for i in range(len(pardict["z_pk"]))], axis = -1)
        
        #The compression matrix will be a block diagonal matrix. 
        PS_dash_EFT = block_diag(*pdash)
        
        #Concatenate the contribution from the bias parameters and the cosmological parameters. 
        PS_dash = np.float64(np.concatenate((PS_dash_cosmo, PS_dash_EFT), axis = 0))
        
        #The total number of free sparameters is given by the number of rows in the compression matrix. 
        n_params = np.shape(PS_dash)[0]
        print("The total number of free parameters is " + str(n_params))
        #first compute the compression matrix
        CM = Compression_vector(C_inv, PS_dash, n_params)
        
        test = np.matmul(np.matmul(CM, cov), CM.T)
        
        #If the "test" matrix is close to the identity matrix, this means the compression is successful. 
        print(np.shape(CM))
        print(test)
        
        if use_sim == False:
            np.save(os.path.join(pardict["outpk"], "Compression_CM_" + keyword + '_' + pardict['keyword'] + ".npy"), CM) #Saving the compression matrix
        else:
            np.save(os.path.join(pardict["outpk"], "Compression_CM_" + 'sim' + ".npy"), CM) #Saving the compression matrix
    
    else:
        count = 0
        for i in range(len(skycut_all)):
            cov = cov_all[i]
            C_inv = np.linalg.inv(cov)
            if skycut_all[i] > 1.5:
                PS_dash_cosmo = np.concatenate([pdashcosmo[j] for j in range(count, count+skycut_all[i])], axis = -1)
                PS_dash_EFT = block_diag(*[pdash[j] for j in range(count, count+skycut_all[i])])
                count += skycut_all[i]
            else:
                PS_dash_cosmo = pdashcosmo[count]
                PS_dash_EFT = pdash[count]
                count += 1
                
            PS_dash = np.float64(np.concatenate((PS_dash_cosmo, PS_dash_EFT), axis = 0))
            
            n_params = np.shape(PS_dash)[0]
            print("The total number of free parameters is " + str(n_params))
            
            CM = Compression_vector(C_inv, PS_dash, n_params)
            
            test = np.matmul(np.matmul(CM, cov), CM.T)
            
            test_eye = np.eye(n_params)
            
            #Calculate the difference between the compressed covariance matrix and the identity matrix. In theory, if the compression
            #is successful, the compressed covariance matrix should be an identity matrix. 
            print("The maximum deivation from the identity matrix is " + str(np.max(np.abs(test - test_eye))))
            
            np.save(os.path.join(pardict["outpk"], "Compression_CM_" + pardict['names'][i] + keyword + ".npy"), CM) #Saving the compression matrix
            
        
        
    