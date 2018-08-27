'''
Created on Nov 24, 2017

@author: mjiang
ming.jiang@epfl.ch
'''

import numpy as np
import scipy.io as sciio
import scipy.fftpack as scifft

from tools.radio import guessmatrix

class redparam:
    '''
    Fourier reduction parameters
    '''
    def __init__(self, thresholdstrategy, threshold, covmatfileexists=False):
        self.thresholdstrategy = thresholdstrategy     # 'value' or 'percent'
        self.threshold = threshold             # threshold value should be consistent with the thresholdstrategy, e.g. 1.0e-4 or 0.9(90%)
        self.covmatfileexists = covmatfileexists         # whether covariance matrix is available. For the first run, this should be False.

def operatorR(x, Phi_t, Sigma, S):
    # the embeddding operator R = \sigma * S * F * Phi^T = \sigma * S * F * Dr^T * Z^T * F^-1 * G^T
    im = Phi_t(x)
    x_red = Sigma * scifft.fft2(im).flatten()[S]
    x_red = x_red.reshape(np.size(x_red),1)
    return x_red

def operatorRt(x_red, imsize, Phi, Sigma, S):
    # the adjoint non-masked embeddding operator R^T = Phi * F^T * S^T * \sigma = G * F * Z * F^T * S^T * \sigma
    x = np.zeros(imsize).astype('complex').flatten()
    x_red = x_red.flatten() * Sigma
    x[S] = x_red
    x = Phi(np.real(scifft.ifft2(x.reshape(imsize))))
    return x

def embedOperators(Phi, Phi_t, Sigma, S, imsize):

    R = lambda x: operatorR(x, Phi_t, Sigma, S) 
    Rt = lambda x: operatorRt(x, imsize, Phi, Sigma, S)
    return R, Rt


def fourierReduction(yn, G, Gt, Phi, Phi_t, Nd, redparam):
    '''
    Fourier reduction method
    Assuming
            yn = Phi x + n, where Phi = G * A = G * F * Z
            
            Phi is the measurement operator
            G is the interpolation kernel, which is a sparse matrix
            F is the FFT operation
            Z is the zero-padding operation
    
    Suppose an embedding operator R such that 
            R(yn) = R(Phi) x + R(n), where R(yn) has much lower dimensionality than yn 
    
    @param yn: visibilities before reduction, complex column vector
    
    @param G: function handle of the interpolation kernel
    
    @param Gt: function handle of the adjoint interpolation kernel
    
    @param Phi: function handle of the measurement operator
    
    @param Phi_t: function handle of the adjoint measurement operator
    
    @param Nd: image size, tuple
    
    @param redparam: object of reduction parameters, it contains:
            redparam.thresholdstrategy: strategy to reduce the data dimensionality
                            'value' or 1: singular values will be thresholded by a predefined value
                            'percent' or 2: keep only the largest singular values according to a predefined percentage
                            
            redparam.threshold: if thresholdstrategy is 'value', threshold is the predefined value
                      if thresholdstrategy is 'percent', threshold is a percentage
            
            redparam.covmatfileexists: flag used to read available covariance matrix associated to the operator F * Phi^T
    
    @return: R, function handle of embedding operator 
             Rt, function handle of adjoint embedding operator
             ry, reduced visibilities, complex column vector
             rG, function handle of reduced interpolation kernel
             rGt, function handle of reduced adjoint interpolation kernel
             rPhi, function handle of reduced measurement operator
             rPhit, function handle of reduced adjoint measurement operator
    '''
    if not hasattr(redparam, 'thresholdstrategy'):
        redparam.thresholdstrategy = 'percent'
        print('Default reduction strategy is set to ' + redparam.thresholdstrategy)
    if not hasattr(redparam, 'threshold'):
        redparam.threshold = 0.5
        print('Default reduction level is set to ' + str(redparam.threshold*100) + '%')
    if not hasattr(redparam, 'covmatfileexists'):
        redparam.covmatfileexists = False
        print('Default covariance matrix availability is set to ' + str(redparam.covmatfileexists))
    covmatfileexists = redparam.covmatfileexists
    thresholdstrategy = redparam.thresholdstrategy
    threshold = redparam.threshold
    
    
    N = Nd[0] * Nd[1]
    
    PhitPhi = lambda x: Phi_t(Phi(x))           # Phi^T * Phi
    
    ### Parameter estimation  ###
    if covmatfileexists:
        covariancemat = sciio.mmread('data/covmat.mtx')
    else:
        covoperator = lambda x : scifft.fft2(PhitPhi(np.real(scifft.ifft2(x))))           # F * Phi^T * Phi * F^-1
        # As the embeding operator R = \sigma * F * Phi^T, when applied to the noise n,
        # it is necessary to compute the covariance of R * R^T so as to study the statistic of the new noise Rn 
        covariancemat = np.abs(guessmatrix(covoperator, N, N, diagonly=True)) 
        sciio.mmwrite('data/covmat.mtx', covariancemat)
    # Thresholding of singular values
    d = covariancemat.diagonal()
    d1 = np.copy(d)
    if thresholdstrategy == 'value' or thresholdstrategy == 1:
        ind = d<threshold 
        d1[ind] = 0
    if thresholdstrategy == 'percent' or thresholdstrategy == 2:
        dsort = np.sort(d)
        val = dsort[int(np.size(d)*(1.-threshold))]
        ind = d<val
        d1[ind] = 0
    # the mask S
    S = d1.astype('bool')
    if not d1.all():
        d1[d1<1.e-10] = 1.e-10            # avoid division by zero
    d12 = 1./np.sqrt(d1[S])
    
    ###########################################################################################################
    ############################# Main part of the Fourier Reduction ##########################################
    ############# The main idea is to conceive a robust embedding operator R    ################################
    ############# so as to reduce the observation and the measurement operator ################################
    ###########################################################################################################
    
    # the non-masked embeddding operator R = \sigma * F * Phi^T = \sigma * F * Dr^T * Z^T * F^-1 * G^T
    # the adjoint non-masked embeddding operator R^T = Phi * F^T * \sigma = G * F * Z * F^T * \sigma
    R, Rt = embedOperators(Phi, Phi_t, d12, S, Nd)                  
    
    # Apply the embedding operator to the observation and measurement operators
    # The adjoint operators are also defined
    ry = R(yn)               # R * y
    print('Ratio of compression=Ry/y:' + str(float(np.size(ry))/np.size(yn)))
    rG = lambda x: R(G(x))     # R * G 
    rGt = lambda x: Gt(Rt(x))  # G^T * R^T
    rPhi = lambda x: R(Phi(x))          # R * Phi
    rPhit = lambda x: Phi_t(Rt(x))     # Phi^T * R^T
    
    return R, Rt, ry, rG, rGt, rPhi, rPhit
###########################################################################################################
###########################################################################################################