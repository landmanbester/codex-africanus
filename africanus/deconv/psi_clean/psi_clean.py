# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sciio

from Psi.dict import *
from Psi.opt import *
from Psi.proxTools import *
from tools.maths import pow_method
from reduction.fouReduction import * 

Fs = np.fft.fftshift
iFs = np.fft.ifftshift

def psi_clean(V, Sigma, R, RH, padding_fraction=1.0, maxiter=20, tol=1e-5, pd_params):
    # get dirty image and PSF
    Sigmainv = 1.0/Sigma
    Id = RH(Sigmainv[:, None]*V).squeeze()
    npix, npix = Id.size
    PSF = RH(Sigmainv[:, None]).squeeze()
    padding = int(padding_fraction * npix)
    Id_pad = np.pad(Id, padding, mode='constant')
    PSF_pad = np.pad(PSF, padding, mode='constant')

    # get the fourier transforms for the dimensionality reduction
    Id_hat = Fs(np.fft.fft2(iFs(Id_pad)))
    PSF_hat = Fs(np.fft.fft2(iFs(PSF_pad)))

    # get diagonal of varainace
    # to do with probing, waiting on Antonio's code
    
    # set operators
    phi = lambda x: PSF_hat*Fs(np.fft.fft2(iFs(x)))
    phit = lambda x: Fs(np.fft.ifft2(iFs(x)))
    
    ############## SARA dictionary control ################
    wlt_basis = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'self']        # wavelet basis to construct SARA dictionary
    nlevel = 2              # wavelet decomposition level
    sara = SARA(wlt_basis, nlevel, imsize[0], imsize[1])
    
    ############# l2 ball bound control, very important for the optimization with constraint formulation ##########
    ####################################### Example for different settings ########################################
    ######### l2_ball_definition='value', stopping_criterion='l2-ball-percentage', stop=1.01 ######################
    ######### l2_ball_definition='sigma', stopping_criterion='sigma', bound=2., stop=2. ###########################
    ######### l2_ball_definition='chi-percentile', stopping_criterion='chi-percentile', bound=0.99, stop=0.999 ####
    l2ball = l2param(l2_ball_definition='sigma', stopping_criterion='sigma', bound=2, stop=2)
    epsilon, epsilons = util_gen_l2_bounds(yn, input_SNR, l2ball)               # input_SNR is necessary for l2 ball evaluation
    print('epsilon='+str(epsilon))

    ########### optimization parameters control ###############
    nu2 = pow_method(phi, phit, imsize, 1e-6, 200)             # Spectral radius of the measurement operator
    pd_params = optparam(positivity=True, nu1=1.0,nu2=nu2,gamma=1.e-3,tau=0.49,max_iter=500, \
                       use_reweight_steps=True, use_reweight_eps=False, reweight_begin=300, reweight_step=50, reweight_times=4, \
                       reweight_alpha=0.01, reweight_alpha_ff=0.5)
    
    # run optimisation loop
    eps = 1.0
    k = 0
    d = Id_hat
    I = np.zeros_like(Id_pad)
    while k < maxiter and eps > tol:
        # keep track of current I
        Ip = I.copy()
        # call primal dual on d
        Im, l1normIter, l2normIter, relerrorIter = forward_backward_primal_dual_minor_cycle(d, phi, phit, SARA, epsilon, epsilons, pd_params)
        I += Im

        # do full subtraction
        Vres = (V - R(I))
        Ir = RH(Sigmainv[:, None]* Vres)
        d = Fs(np.fft.fft2(iFs(Ir)))

        # maybe put the positivity constraint here?

        eps = np.abs(I - Ip).max()

    return I, Ir



