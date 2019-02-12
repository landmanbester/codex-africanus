#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from ...util.docs import DocstringTemplate

import numba
import numpy as np

# remember can use wsclean psf and dirty to test

def sub_minor_loop(dirty, psf, mgain, gain, maxiter=250):
    # get image sizes
    npix = dirty.shape[0]
    npixpsf = psf.shape[0]
    halfnpixpsf = npixpsf//2

    # assuming npixpsf = 2*npix - 1 for now
    assert npix == halfnpixpsf + 1

    # get max of dirty and coordinates
    maxdirty = dirty.max()
    p, q = np.argwhere(np.abs(dirty) == np.abs(maxdirty)).squeeze()

    # set threshold and find the set A
    threshold = (1 - mgain) * np.abs(maxdirty)
    I = np.argwhere(np.abs(dirty) > threshold).squeeze()
    Ip = I[:, 0]
    Iq = I[:, 1]
    A = dirty[Ip, Iq]

    pq = np.argwhere(np.abs(A) == np.abs(maxdirty)).squeeze()

    # set component model
    model = np.zeros_like(dirty, dtype=np.float64)
    k = 0
    while np.abs(maxdirty) > threshold and k < maxiter:
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
        pq = np.argwhere(np.abs(A) == np.abs(maxdirty)).squeeze()

        # read off original indices
        p = Ip[pq]
        q = Iq[pq]

        # update counter
        k+=1

    if k == maxiter:
        print("Warning - maximum iterations reached in subminor loop")
    return model, dirty


if __name__=="__main__":
    # simulate dirty and psf
    npix = 21
    dirty = np.random.randn(npix, npix)
    npixpsf = 2*npix - 1
    halfnpixpsf = npixpsf//2
    # sigma = 8.0
    # x, y = np.mgrid[-halfnpixpsf:halfnpixpsf:1j*npixpsf, -halfnpixpsf:halfnpixpsf:1j*npixpsf]
    # psf = np.exp(-(x**2 + y**2)/(2*sigma**2))
    psf = np.zeros((npixpsf, npixpsf))
    psf[halfnpixpsf, halfnpixpsf] = 1.0

    import matplotlib.pyplot as plt
    # print(psf.shape, psf[halfnpixpsf, halfnpixpsf])
    #
    # plt.imshow(psf)
    # plt.colorbar()
    # plt.show()

    model, dirty = sub_minor_loop(dirty, psf, 0.99, 0.1, maxiter=50000)

    plt.figure('M')
    plt.imshow(model)
    plt.colorbar()


    plt.figure('D')
    plt.imshow(dirty)
    plt.colorbar()

    plt.show()
