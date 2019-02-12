#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ...util.docs import DocstringTemplate

import numba
import numpy as np
from scipy.signal import convolve2d


def set_scales(fwhm0, max_scale=100):
    alpha0 = fwhm0 / 0.45
    alphas = [alpha0, 4 * alpha0]  # empirically determined to work pretty well
    i = 1
    while 2 * alphas[i] < max_scale:  # hardcoded for now
        alphas.append(2 * alphas[i])
        i += 1
    return np.asarray(alphas)

def set_bias(beta, alphas, dtype=np.float64):
    """
    Sets the bias for the scales
    :param beta: bias parameter
    :param alphas: array of scale sizes in pixels
    :return: 
    """
    try:
        nscales = alphas.size
    except:
        nscales = 1
    # set scale bias according to Offringa definition implemented i.t.o. inverse bias
    bias = np.ones(nscales, dtype=dtype)
    for scale in range(1, nscales):
        bias[scale] = beta**(1.0 + np.log2(alphas[scale]/alphas[1]))
    return bias


def set_scale_kernels(alphas, dtype=np.float64):
    """
    Computes Gauss pars corresponding to scales given in number of pixels
    :return: 
    """
    try:
        nscales = alphas.size
    except:
        nscales = 1

    sigmas = np.zeros(nscales, dtype=dtype)
    volumes = np.zeros(nscales, dtype=dtype)
    kernels = np.empty(nscales, dtype=object)
    for scale in range(nscales):
        # fixed convertsion factor for size of kernel
        sigmas[scale] = 3.0 * alphas[scale] / 16

        # support of Gaussian components in pixels
        half_alpha = alphas[scale]/2.0

        # set grid of x, y coordinates
        x, y = np.mgrid[-half_alpha:half_alpha:alphas[scale]*1j, -half_alpha:half_alpha:alphas[scale]*1j]

        # compute the normalisation factor
        volumes[scale] = 2*np.pi*sigmas[scale]**2

        # evaluate scale kernel
        kernels[scale] = np.exp(-(x**2 + y**2)/(2*sigmas[scale]**2))/volumes[scale]

    return sigmas, kernels, volumes

def GaussianSymmetricFT(sigma, x0, y0, u, v, rhosq, amplitude=1.0):
    """
    Gives the FT of a symmetric Gaussian analytically. Note since we will always be using this to 
    convolve with real valued inputs we only need to compute the result at the positive frequencies. 
    Note only the mean image ever gets convolved with multiple scales and we are usually convolving a 
    cube with a single scale.
    :param sigma: std deviation of Gaussian in signal space
    :param x0: center x coordinate relative to center
    :param y0: center y coordinate relative to center
    :param amp: amplitude (at delta scale) of Gaussian component (if amp.size > 1 cube must be True)
    :return: 
    """
    return amplitude * np.exp(-2.0j * np.pi * v * x0 - 2.0j * np.pi * u * y0 - 2 * np.pi ** 2 * rhosq * sigma ** 2)

def set_best_scale(dirty, kernels, bias):
    nscales = bias.size

    # get max and location for zero scale
    maxdirty = dirty.max()
    p, q = np.argwhere(dirty == maxdirty)
    iscale = 0

    # convolve with scale kernels and keep track of max
    npix = dirty.shape[0]
    for scale in range(1, nscales):
        kernel = kernels[scale]
        npixbox = kernel.shape[0]
        npad = (npix-npixbox)//2.0
        kernel = np.pad(kernel, npad, mode='constant')
        convdirty = convolve2d(dirty, kernel, mode='same')

        convmaxdirty = convdirty.max()
        # update maxdirty and position if new scale is more relevant
        if np.abs(maxdirty) < np.abs(convmaxdirty) * bias[scale]:
            maxdirty = convmaxdirty
            p, q = np.argwhere(convdirty == maxdirty)
            iscale = scale

    if iscale == 0:
        return dirty, iscale
    else:
        return convdirty, iscale

def set_convolved_psf(psf, kernel):
    npix = psf.shape[0]
    npixbox = kernel.shape[0]
    npad = (npix - npixbox)//2
    kernel = np.pad(kernel, npad, mode='constant')

    convpsf = convolve2d(psf, kernel, mode='same')
    conv2psf = convolve2d(convpsf, kernel, mode='same')

    return convpsf, conv2psf





if __name__=="__main__":
    # test set scale bias
    beta = 0.6
    FWHM0 = 5

    alphas = set_scales(FWHM0, max_scale=100)

    print(alphas)
    # bias = set_bias(beta, alphas)
    # print("Expected = ", 1, beta, beta ** 2, beta ** 3, beta ** 4)
    # print("Actual = ", bias)

    # test auto scale selection
    sigmas, kernels, volumes = set_scale_kernels(alphas)

    print(sigmas)
    print(volumes)

    import matplotlib.pyplot as plt

    for i in range(alphas.size):
        plt.imshow(kernels[i])
        plt.colorbar()
        plt.show()

