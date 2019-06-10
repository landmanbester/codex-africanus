# -*- coding: utf-8 -*-
"""
Here we test the accuracy and aliasing properties of all
gridding kernels defined in africanus.gridding.filters.kernels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

from africanus.constants import c as lightspeed
from africanus.gridding.filters.kernels import kaiser_bessel, kaiser_bessel_corrector

Fs = fftpack.fftshift
iFs = fftpack.ifftshift
FFT = fftpack.fft
iFFT = fftpack.ifft


def vis_to_im(vis, uvw, lm, im_of_vis):
    for source in range(lm.shape[0]):
        l = lm[source]
        for row in range(uvw.shape[0]):
            u = uvw[row]
            real_phase = 2 * np.pi * l * u
            im_of_vis[source] += np.cos(real_phase) * vis[row].real - np.sin(real_phase) * vis[row].imag

def im_to_vis(im, uvw, lm, vis_of_im):
    for row in range(uvw.shape[0]):
        u = uvw[row]
        for source in range(lm.shape[0]):
            l = lm[source]
            real_phase = 2 * np.pi * l * u
            vis_of_im[row] += im[source] * np.exp(1.0j*real_phase)


def grid_analytic_kernel(vis, uvw, conv_filter, half_support, cell_size, grid_data):
    """
    Here we assume we have an analytic kernel so that the interpolation weights
    can be evaluated exactly (no snapping). Thus conv_filter is a function which
    takes only the frequency at which it should be evaluated where the frequencies
    are given as numbers between -half_support and half_support.
    """

    # size of the over-sampled grid
    npix = grid_data.shape[0]
    assert npix % 2 == 0  # assuming even grid for now
    half_x = npix//2
    
    # we scale the frequencies to lie on a grid between 0 and npix.
    # of course we need to make sure that we have at least half_support
    # empty pixels next to uvw.max and uvw.min. This should be taken 
    # into account when selecting the pixel size 
    # u_max = 1.0/cell_size
    # scaled_u = uvw * npix/u_max  # equivalently npix * cell_size which is Ben's scaling factor of
    scaling_factor = npix * cell_size
    for row, val in enumerate(vis):
        scaled_u = uvw[row]*scaling_factor
        # get the pixel below (this way diff is always <=0)
        disc_u = int(np.floor(scaled_u))

        # get the difference between this and the exact value
        diff = disc_u - scaled_u

        # get frequencies
        lower = -half_support + 1 + diff
        # upper = half_support + 1 + diff if diff else half_support+1
        nu = lower + np.arange(2*half_support)

        # evaluate kernel at these locations
        conv_weights = conv_filter(nu)

        # smear visibility onto grid
        lower = half_x + disc_u-half_support+1
        upper = half_x + disc_u+half_support+1 # if diff else half_x + disc_u+half_support
        for filter_i, grid_i in enumerate(range(lower, upper)):
            grid_data[grid_i] += conv_weights[filter_i] * val
    return grid_data
    

def test_oned_gridding():
    # set up image space stuffs
    cell_size = 0.0001
    npix = 128
    lm_extents = cell_size*(npix // 2)
    lm = -lm_extents + np.arange(npix) * cell_size
    print(lm)
    dft_dirty = np.zeros((npix), dtype=np.float64)
    
    # set up visibility space stuffs
    vis = np.array([1.0 + 0.0j], dtype=np.complex128)  # continuous visibility
    padding_frac = 2.0  # amount to pad the image by
    grid_size = int(npix*padding_frac)  # size of the regular grid
    npad = (grid_size - npix)//2
    assert npix + 2*npad == grid_size
    u_max = 1.0 / cell_size
    uvw = np.array([0.15*u_max], dtype=np.float64)  # continuous coordinate

    # Compute the DFT result
    vis_to_im(vis, uvw, lm, dft_dirty)

    # set up gridding kernel
    half_support = 6
    beta = 2.34  # KB factor compare to 2.34
    conv_filter = lambda nu: kaiser_bessel(nu, half_support, beta=beta)

    # grid the data
    oned_grid = np.zeros(grid_size, dtype=np.complex128)
    grid_analytic_kernel(vis, uvw, conv_filter, half_support, cell_size, oned_grid)

    # get the Fourier transform (remembering to undo the scaling that numpy applies internally)
    oned_grid_fft = (Fs(iFFT(iFs(oned_grid)))).real[npad:-npad] * grid_size #/np.sqrt(grid_size)

    # compute the taper
    taper = kaiser_bessel_corrector(half_support, npix, grid_size, beta=beta)
    # print(oned_grid_fft)
    # print(taper)

    # plt.figure('taper')
    # plt.plot(taper)
    # plt.show()

    # correct for tapering
    dirty = oned_grid_fft / taper

    plt.figure('compare')
    plt.plot(dft_dirty.squeeze().real, label='dft')
    plt.plot(dirty.real, label='grid')
    plt.legend()
    plt.show()

    SSE = np.sum((dirty - dft_dirty)**2)

    print("Total SSE = ", SSE)

def set_data_no_aliasing(nvis, nsource, npix):
    # simulation params
    uvw = -np.pi + 2*np.pi * np.random.random(nvis)
    u_max = np.abs(uvw).max()
    cell_size = 0.8/(2*u_max)  # to ensure Nyquist sampling + then some
    assert npix % 2 == 0  # ionly set up for even images
    lm_extents = cell_size*(npix // 2)
    nsource = 5
    l_source = -0.75*lm_extents + 1.5*lm_extents*np.random.random(nsource)  # no aliases
    im = 0.1 + np.random.random(nsource)  # flux
    # get dft vis
    vis = np.zeros(nvis, dtype=np.complex128)
    im_to_vis(im, uvw, l_source, vis)
    # make dirty image 
    lm = -lm_extents + np.arange(npix) * cell_size
    dirty = np.zeros(npix, dtype=np.float64)
    vis_to_im(vis, uvw, lm, dirty)
    return vis, dirty, uvw, lm

def dirty_via_gridding(vis, uvw, conv_filter, corrector, half_support, cell_size, npix, grid_size):
    npad = (grid_size - npix)//2  # must be even
    assert npix + 2*npad == grid_size
    oned_grid = np.zeros(grid_size, dtype=np.complex128)
    grid_analytic_kernel(vis, uvw, conv_filter, half_support, cell_size, oned_grid)
    dirty = (Fs(iFFT(iFs(oned_grid)))).real[npad:-npad] * grid_size
    taper = corrector(half_support, npix, grid_size)
    dirty /= taper
    return dirty

nvis = 100
nsource = 5
npix = 128
vis, dirty, uvw, lm = set_data_no_aliasing(nvis, nsource, npix)

half_support = 3
cell_size = lm[1] - lm[0]
grid_size = int(2*npix)  # padding by factor of 2
beta = 2.34
conv_filter = lambda nu: kaiser_bessel(nu, half_support, beta)
# kaiser_bessel_corrector(beta, half_support, npix, grid_size)
corrector = lambda half_support, npix, grid_size: kaiser_bessel_corrector(beta, half_support, npix, )
