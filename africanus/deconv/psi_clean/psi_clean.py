# -*- coding: utf-8 -*-

import numpy as np

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

    # run optimisation loop
    eps = 1.0
    k = 0
    d = Id_hat
    I = np.zeros_like(Id_pad)
    while k < maxiter and eps > tol:
        # keep track of current I
        Ip = I.copy()
        # call primal dual on d
        Im = primal_dual(d, phi, phit, pd_params)
        I += Im

        # do full subtraction
        Vres = (V - R(I))
        Ir = RH(Sigmainv[:, None]* Vres)
        d = Fs(np.fft.fft2(iFs(Ir)))

        # maybe put the positivity constraint here?

        eps = np.abs(I - Ip).max()

    return I, Ir



