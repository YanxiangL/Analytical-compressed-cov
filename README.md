# Analytical-compressed-cov
This repository contains the codes to calculate the analytical covariance matrix and MOPED compression for galaxy power spectrum. 

The "analytic-covariance" folder contains the code that calculates the analytical covariance matrix. The random files are too big to upload here, but the links to these files can be found in their respective papers. 

First, you need to convert the random files all to fits files because that's what Nbodykit reads in. I have uploaded a code here (create_input_fits_6dFGS.py) to convert the 6dFGS random file to the fits file. You can do the same for other surveys. 

Secondly, run the FFT_Nbodykit_openmp.py to find the all FFT kernels that will be used to calculate the Gaussian and SSC covariance matrix. I have also attached the config file for each survey here. See the comments within the config file and the code to see what you need to change if you want to apply for a different survey. You will need the same config file to calculate the Gaussian, SSC, and the full covariance matrix in the next 3 steps. 
 
Then run the window_kernel_Gaussian.py to calculate the Gaussian part of the covariance matrix. 

At the same time, you should run SSC_window_power_spectra.py to find the SSC covariance matrix. 

Lastly, run covariance_SPT.py to find the final analytical covariance matrix. 

Due to the numerical error, your covariance matrix may not be semi-positive definite. You can use the cov_txt.py provided to find the nearest symmetric semi-positive definite matrix with the algorithm from https://www.sciencedirect.com/science/article/pii/0024379588902236. 

You only need to rerun covariance_SPT.py when you change the cosmological parameters. 

All codes in this work are based on https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123517. Please reference their works as well if you are using this code. 

For the MOPED compression, you need to use the code and the config files in the "MOPED" folder. 

The best-fit power spectra are provided in the "model_PS" folder and the analytical covariance matrices are provided in the "cov" folder in the "analytic_vs_sim" folder. 

To generate the Taylor expansion grid and run the pybird code, we refer the readers to (https://github.com/cullanhowlett/pybird/tree/desi). The "third" folder is used to generate the Taylor expansion and the fitting_codes folder is used to run the fitting. 
