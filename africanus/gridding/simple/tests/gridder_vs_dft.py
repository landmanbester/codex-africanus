#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compares the output of the gridder with the Direct Fourier Transform
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pytest
from africanus.constants import c as lightspeed
from africanus.gridding.util import estimate_cell_size
import matplotlib.pyplot as plt

_ARCSEC2RAD = np.deg2rad(1.0/(60*60))

Fs = np.fft.fftshift
iFs = np.fft.ifftshift
FFT = np.fft.fft2
iFFT = np.fft.ifft2

def test_forward_operator():
    """
    Here we compare the output of the degridder with that of the DFT
    """
    da = pytest.importorskip("dask.array")
    from africanus.dft.dask import im_to_vis
    from africanus.filters import convolution_filter
    from africanus.filters import kaiser_bessel_filter as kbf
    from africanus.gridding.simple import degrid

    # start by simulating some uv-coverage
    nrow = 1000
    uvw = (0.1 + np.random.random(size=(nrow, 3)))*128
    uvw[:, -1] = 0.0  # set w to zero for simple gridder

    nchan = 1 # single channel test since degridder averages channels
    frequency = np.array([1e9], dtype=np.float64)
    wavelengths = lightspeed/frequency
    npix = 129
    cell_size = estimate_cell_size(uvw[:, 0], uvw[:, 1],
                                   wavelengths, factor=5, nx=npix, ny=npix).max()

    cell_size_rad = cell_size * _ARCSEC2RAD

    nover2 = npix//2
    x = np.linspace(-nover2*cell_size_rad, nover2*cell_size_rad, npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T

    # create random image and flatten for use with dft
    #image = np.abs(np.random.randn(npix, npix, nchan))  # np.zeros((npix, npix, nchan), dtype=np.float64)
    image = np.exp(-(ll**2 + mm**2)*100000)[:, :, None]
    #image = np.zeros((npix, npix, nchan), dtype=np.float64)
    #image[npix//4, npix//4, :] = 1.0
    image_flat = image.reshape(npix**2, nchan)

    # set up dask arrays
    uvw_dask = da.from_array(uvw, chunks=(250, 3))
    lm_dask = da.from_array(lm, chunks=(npix**2, 2))
    frequency_dask = da.from_array(frequency, chunks=1)
    image_dask = da.from_array(image_flat, chunks=(npix**2, 1))

    # do dft
    vis_dft = im_to_vis(image_dask, uvw_dask,
                              lm_dask, frequency_dask).compute()

    plt.figure('real')
    plt.plot(np.arange(nrow), vis_dft[:, 0].real, 'b', alpha=0.25)

    plt.figure('imag')
    plt.plot(np.arange(nrow), vis_dft[:, 0].imag, 'b', alpha=0.25)


    # get taper
    K = 3*npix
    full_sup = 51
    half_sup = (full_sup - 2)//2  # this weird convention needs to die!
    nshift = (np.arange(npix) - (npix-1)//2)/K
    beta = 2.34*full_sup
    S = kbf.kaiser_bessel_fourier(nshift, full_sup, beta)
    S = S[:, None] * S[None, :]  # outer product to get 2D version

    # apply taper and pad
    image_padded = np.pad(image / S[:, :, None], ((npix, npix), (npix, npix), (0, 0)), mode='constant')

    print(image_padded.shape, K)

    osamp = 103
    conv_filter = convolution_filter(half_sup, osamp, "kaiser-bessel")

    # get FFT
    Y = np.zeros_like(image_padded, dtype=np.complex128)
    for ch in range(nchan):
        Y[:, :, ch] = Fs(FFT(iFs(image_padded[:, :, ch])))

    # degrid
    cor = (1,)
    weights = np.ones((nrow, nchan) + cor, dtype=np.float64)
    vis_degrid = degrid(Y, uvw, weights, wavelengths, conv_filter, cell_size)

    plt.figure('real')
    plt.plot(np.arange(nrow), vis_degrid[:, 0, 0].real, 'r', alpha=0.25)

    plt.figure('imag')
    plt.plot(np.arange(nrow), vis_degrid[:, 0, 0].imag, 'r', alpha=0.25)

    plt.show()

    print(np.abs(vis_dft[:, 0] - vis_degrid[:, 0, 0]).max())

    return

def test_backward_operator():
    da = pytest.importorskip("dask.array")
    from africanus.dft.dask import vis_to_im
    from africanus.filters import convolution_filter
    from africanus.filters import kaiser_bessel_filter as kbf
    from africanus.gridding.simple import grid

    # start by simulating some uv-coverage
    nrow = 1000
    uvw = (0.1 + np.random.random(size=(nrow, 3)))*128
    uvw[:, -1] = 0.0  # set w to zero for simple gridder

    nchan = 1 # single channel test since gridder averages channels
    frequency = np.array([1e9], dtype=np.float64)
    wavelengths = lightspeed/frequency
    npix = 129
    cell_size = estimate_cell_size(uvw[:, 0], uvw[:, 1],
                                   wavelengths, factor=5, nx=npix, ny=npix).min()

    cor = (1,)
    ncor = 1
    weights = np.ones((nrow, nchan) + cor, dtype=np.float64)
    vis = np.random.randn(nrow, nchan) + 1.0j*np.random.randn(nrow, nchan)

    cell_size_rad = cell_size * _ARCSEC2RAD

    nover2 = npix//2
    x = np.linspace(-nover2*cell_size_rad, nover2*cell_size_rad, npix)
    ll, mm = np.meshgrid(x, x)
    lm = np.vstack((ll.flatten(), mm.flatten())).T

    # set up dask arrays
    uvw_dask = da.from_array(uvw, chunks=(250, 3))
    lm_dask = da.from_array(lm, chunks=(npix**2, 2))
    frequency_dask = da.from_array(frequency, chunks=1)
    vis_dask = da.from_array(vis, chunks=(250, 1))

    print("Doing dft")
    dirty_dft = vis_to_im(vis_dask, uvw_dask, lm_dask, frequency_dask).compute()

    dirty_dft = dirty_dft.reshape(npix, npix)

    print("Getting taper")
    # get taper
    K = 3*npix
    full_sup = 17
    half_sup = (full_sup - 2)//2  # this weird convention needs to die!
    nshift = (np.arange(npix) - (npix-1)//2)/K
    beta = 2.34*full_sup
    S = kbf.kaiser_bessel_fourier(nshift, full_sup, beta)
    S = S[:, None] * S[None, :]  # outer product to get 2D version

    # grid the data
    vis = vis[:, :, None]
    flags = np.zeros_like(vis, dtype=np.int8)
    osamp = 63
    conv_filter = convolution_filter(half_sup, osamp, "kaiser-bessel")
    dirty = grid(vis, uvw, flags, weights, wavelengths, conv_filter, cell_size, ny=K, nx=K)

    print(dirty.shape)

    # get FFT
    for ch in range(nchan):
        dirty[:, :, ch] = Fs(iFFT(iFs(dirty[:, :, ch])))*K**2  # K**2 because we shouldn't normalise
                                                               # the FFT for comparison with DFT

    # remove zero padding and divide by taper
    dirty = dirty[npix:-npix, npix:-npix, 0]/S

    plt.figure('dft')
    plt.imshow(dirty_dft)
    plt.colorbar()


    plt.figure('grid')
    plt.imshow(dirty.real)
    plt.colorbar()

    plt.figure('diff')
    plt.imshow((dirty_dft - dirty.real)/np.abs(dirty_dft).max())
    plt.colorbar()

    plt.show()

if __name__=="__main__":
    #test_forward_operator()
    test_backward_operator()
