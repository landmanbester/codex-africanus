'''
Created on Dec 1, 2017

@author: mjiang
ming.jiang@epfl.ch
'''

import numpy as np

def pow_method(A, At, im_size, tol, max_iter):
    '''
    Computes the spectral radius (maximum eigen value) of the operator A
    
    @param A: function handle of direct operator
    
    @param At: function handle of adjoint operator
    
    @param im_size: size of the image
    
    @param tol: tolerance of the error, stopping criterion
    
    @param max_iter: max iteration
    
    @return: spectral radius of the operator 
    '''
    x = np.random.randn(im_size[0],im_size[1])
    x /= np.linalg.norm(x,'fro')
    init_val = 1
    
    for i in np.arange(max_iter):
        y = A(x)
        x = At(y)
        val = np.linalg.norm(x,'fro')
        rel_var = np.abs(val-init_val) / init_val
        if rel_var < tol: break
        init_val = val
        x /= val
        
    return val