# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:41:20 2021

@author: Yanxiang Lai
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:03:12 2021

@author: Yansiang
"""

# Here are the expressions used to calculate the tree-level trispectrum
# These (horrible) expressions were copy-pasted from the
# mathematica notebook (Generating_T0_Z12_expressions.nb),
# which shows their derivation

import numpy as np
# from numba import jit

def InitParameters(arr):
    global be, b1, b2, b1EFT, b2EFT, b3EFT, b4EFT, g2
    [be, b1, b2, b1EFT, b2EFT, b3EFT, b4EFT, g2]=arr