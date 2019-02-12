#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from ...util.docs import DocstringTemplate

import numba
import numpy as np
from africanus.deconv.wscms import scale_handler, gain_handler
from africanus.deconv.util import find_overlap_indices

# remember can use wsclean psf and dirty to test

def sub_minor_loop(dirty, psf, multiscale_gain, gain, maxiter=250):
    # get image sizes
    npix = dirty.shape[0]
    npixpsf = psf.shape[0]

    # assuming both are odd for now
    assert npix % 2 == 1
    assert npixpsf % 2 == 1

    halfnpixpsf = npixpsf//2

    # assuming npixpsf = 2*npix - 1 for now
    assert npix == halfnpixpsf + 1

    # get max of dirty and coordinates
    absdirty = np.abs(dirty)
    absmaxdirty = absdirty.max()
    p, q = np.argwhere(absdirty == absmaxdirty).squeeze()
    maxdirty = dirty[p, q]

    # set threshold and find the set A
    threshold = (1 - multiscale_gain) * absmaxdirty
    I = np.argwhere(absdirty > threshold).squeeze()
    Ip = I[:, 0]
    Iq = I[:, 1]
    A = dirty[Ip, Iq]

    # set component model (LB - could make this the shape of A)
    model = np.zeros_like(dirty, dtype=np.float64)
    k = 0
    while absmaxdirty > threshold and k < maxiter:
        # get shifted psf indices
        Ipp = Ip - (p - halfnpixpsf)
        Iqq = Iq - (q - halfnpixpsf)

        # subtract fraction of peak
        xtmp = maxdirty * gain
        A -= psf[Iqq, Ipp] * xtmp

        # update model
        model[p, q] += xtmp

        # get new maxdirty
        maxdirty = A.max()
        absmaxdirty = np.abs(maxdirty)
        pq = np.argwhere(np.abs(A) == absmaxdirty).squeeze()

        # read off original indices
        p = Ip[pq]
        q = Iq[pq]

        if k == 0:
            comps = ((p,q),)

        if (p,q) not in comps:
            comps += ((p,q),)

        # update counter
        k+=1

    if k == maxiter:
        print("Warning - maximum iterations reached in subminor loop")
    return model, comps


def minor_loop(dirty, psf, kernels, volumes, bias, gain=0.1, mgain=0.9, multiscale_gain=0.2, abs_threshold=1e-3, maxiter=50):
    npix = dirty.shape[0]
    npixpsf = psf.shape[0]

    # assuming this for now
    assert npixpsf == 2*npix - 1

    # get convolve psf products for zero scale
    convpsf0, conv2psf0 = scale_handler.set_convolved_psf(psf, kernels[0])

    # set threshold
    absmaxdirty = np.abs(dirty).max()
    minor_threshold = (1-mgain) * absmaxdirty
    threshold = np.maximum(minor_threshold, abs_threshold)

    # run deconvolution
    model = np.zeros_like(dirty)
    k = 0
    while absmaxdirty > threshold and k < maxiter:
        # find the most relevant scale
        convdirty, iscale = scale_handler.set_best_scale(dirty, kernels, bias)
        kernel = kernels[iscale]

        # set twice convolved psf and get gain for scale
        if iscale != 0:
            convpsf, conv2psf = scale_handler.set_convolved_psf(psf, kernel)
            scalegain = gain_handler.set_scale_gain(convpsf, convpsf0, volumes, iscale, gain)
        else:
            conv2psf = psf
            scalegain = gain

        # run subminor loop
        component_model, comps = sub_minor_loop(convdirty, conv2psf, multiscale_gain, scalegain)
        scale_model = np.zeros_like(dirty)

        # add Gaussian components to model
        nbox = kernel.shape[0]
        if iscale != 0:
            for comp in comps:
                p, q = comp

                Aedge, Bedge = find_overlap_indices(p, q, npix, nbox)

                xlow_im, xhigh_im, ylow_im, yhigh_im = Aedge
                # xlow_box, xhigh_box, ylow_box, yhigh_box = Bedge

                scale_model[xlow_im, xhigh_im, ylow_im, yhigh_im] += component_model[p,q] * kernel
        else:
            scale_model = model

        # convolved model with psf and subtract from dirty
        conv_model = convolve2d(scale_model, psf, mode='same')

        # subtract convolved model from dirty
        dirty -= conv_model

        # update global model
        model += scale_model

        absmaxdirty = dirty.max()
        k += 1

    if k == maxiter:
        print("Warning - maximum iterations reached in minor cycle")

    return model, dirty


if __name__=="__main__":
    # simulate dirty and psf
    npix = 51
    model = np.zeros((npix, npix))
    nsource = 5
    Ip = np.random.randint(5, npix-5, nsource)
    Iq = np.random.randint(5, npix-5, nsource)

    print("Ip = ", Ip)
    print("Iq = ", Iq)
    model[Ip, Iq] = 0.1 + np.random.random(nsource)
    npixpsf = 2*npix - 1
    halfnpixpsf = npixpsf//2
    sigma = 1.5
    x, y = np.mgrid[-halfnpixpsf:halfnpixpsf:1j*npixpsf, -halfnpixpsf:halfnpixpsf:1j*npixpsf]
    psf = np.exp(-(x**2 + y**2)/(2*sigma**2))
    from scipy.signal import convolve2d
    dirty = convolve2d(model, psf, mode='same')

    import matplotlib.pyplot as plt
    plt.figure('psf')
    plt.imshow(psf)
    plt.colorbar()

    plt.figure('M')
    plt.imshow(model)
    plt.colorbar()


    plt.figure('D')
    plt.imshow(dirty)
    plt.colorbar()

    modelrec = sub_minor_loop(dirty, psf, 0.5, 0.1, maxiter=50000)

    plt.figure('MR')
    plt.imshow(modelrec)
    plt.colorbar()

    plt.show()
