'''
Created on 14 Mar 2018

@author: mjiang
ming.jiang@epfl.ch
'''

import numpy as np
import scipy.linalg as LA

def soft(alpha,th):
    '''
    soft-thresholding function
    
    @param alpha: wavelet coefficients
    
    @param th: threshold level
    
    @return: wavelet coefficients after soft threshold
    
    '''
    tmp = np.abs(alpha) - th
    tmp = (tmp + np.abs(tmp))/2.
    return np.sign(alpha) * tmp

def hard(alpha,th):
    '''
    hard-thresholding function
    
    @param alpha: wavelet coefficients 
    
    @param th: threshold level
    
    @return: wavelet coefficients after hard threshold
    
    '''
    return alpha * (alpha>0) 

def proj_sc(alpha, rad):
    '''
    scaling, projection on L2 norm
    
    @param alpha: coefficients to be processed
    
    @param rad: radius of the l2 ball
    
    @return: coefficients after projection
    
    '''
    return alpha * min(rad/LA.norm(alpha), 1)
    