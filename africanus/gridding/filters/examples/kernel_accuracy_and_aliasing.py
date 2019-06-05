# -*- coding: utf-8 -*-
"""
Here we test the accuracy and aliasing properties of all
gridding kernels defined in africanus.gridding.filters.kernels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
import numpy as np
from scipy import fftpack
import pytest

from africanus.constants import c as lightspeed
from africanus.gridding.filters.kaiser_bessel_filter import kaiser_bessel, kaiser_bessel_fourier

Fs = fftpack.fftshift
iFs = fftpack.ifftshift
FFT = fftpack.fft
iFFT = fftpack.ifft


def vis_to_im_impl(vis, uvw, lm, im_of_vis):
    # For each source
    for source in range(lm.shape[0]):
        l = lm[source]

        # For each uvw coordinate
        for row in range(uvw.shape[0]):
            u = uvw[row]

            # e^(-2*pi*(l*u + m*v + n*w)/c)
            real_phase = 2 * np.pi * l * u

            im_of_vis[source] += np.cos(real_phase) * vis[row].real - np.sin(real_phase) * vis[row].imag

    return im_of_vis

def grid(vis, uvw, conv_filter, oversample, cell_size, grid_data):

    nx = grid_data.shape[0]
    half_x = nx // 2
    half_support = (conv_filter.shape[0] // oversample) // 2

    # Similarity Theorem
    # https://www.cv.nrao.edu/course/astr534/FTSimilarity.html
    # Scale UV coordinates
    # u_max = 1.0/cell_size
    # u_cell_size = u_max/nx

    u_scale = nx * cell_size

    for r in range(vis.shape[0]):
        # put in fractions of cell size
        scaled_u = uvw[r] * u_scale

        exact_u = half_x + scaled_u
        disc_u = int(np.round(exact_u))

        frac_u = exact_u - disc_u
        # base_os_u = int(np.floor(frac_u*oversample + 0.5))
        base_os_u = -int(round(frac_u * oversample))

        # print(exact_u, disc_u, frac_u, base_os_u, frac_u * oversample)
        # print(base_os_u)
        # print(exact_u, disc_u, frac_u, base_os_u)

        lower_u = disc_u - half_support      # Inclusive
        upper_u = disc_u + half_support + 1  # Exclusive

        # print(exact_u, disc_u, lower_u, upper_u)

        # grid_data[disc_u] = vis[r]

        for ui, grid_u in enumerate(range(lower_u, upper_u)):

            # print(oversample//2 + base_os_u + ui * oversample)
            conv_weight = conv_filter[oversample//2 + base_os_u + ui * oversample]
            # print(grid_u, ui * oversample, conv_weight, vis[r] * conv_weight)
            grid_data[grid_u] += vis[r] * conv_weight

    return grid_data

def nfft_psi(x, m, sigma, N):
    n = sigma*N
    b = np.pi * (2 - 1.0/sigma)
    tmp1 = np.sqrt(m**2 - n**2*x**2)
    tmp2 = np.sqrt(n**2*x**2 - m**2)
    return np.where(np.abs(x) < m/n, np.sinh(b*tmp1)/tmp1, np.sin(b*tmp2)/tmp2)

def nfft_psihat(k, m, sigma, N):
    n = sigma*N
    b = np.pi * (2 - 1.0/sigma)
    tmp = np.sqrt(b**2 - (2*np.pi*k/n)**2)
    return np.where(np.abs(k) < n*(1-1.0/(2*sigma)), np.i0(m*tmp, 0.0))


def test_oned_gridding():
    filter_width = 5
    oversample = 3
    beta = 2.34
    cell_size = 0.0001
    grid_size = 21

    vis = np.array([1.0 + 0.0j], dtype=np.complex128)
    u_max = 1.0 / cell_size
    u_cell_size = u_max / grid_size

    os_step = u_cell_size/oversample  # cell_size / os
    posn = 0
    uvw = np.array([posn * os_step], dtype=np.float64)
    # uvw = np.array([0.0], dtype=np.float64)
    width = filter_width*oversample

    u = np.arange(width, dtype=np.float64) - width // 2
    print(u)
    # u = np.linspace(-filter_width//2+1, filter_width//2, filter_width * oversample)
    # print(u)

    # conv_filter = kaiser_bessel(u, width, beta, J=filter_width)
    conv_filter = kaiser_bessel(u, width, beta, J=filter_width)
    import matplotlib.pyplot as plt
    plt.figure('kernel')
    plt.plot(conv_filter, 'k')
    plt.show()

    oned_grid = np.zeros(grid_size, dtype=np.complex128)

    oned_grid = grid(vis, uvw, conv_filter, oversample, cell_size, oned_grid)

    # put cf on oversampled grid
    nfull = grid_size * oversample
    cffull = np.zeros(nfull)
    center = nfull//2 + posn - 1
    Icf = np.arange(center - filter_width * oversample//2, center + filter_width * oversample//2 + 1)
    cffull[Icf] = conv_filter

    # cffullhat = Fs(iFFT(iFs(cffull))).real * nfull

    # plt.figure('grid')
    # plt.plot(np.arange(grid_size), oned_grid.real, 'b')
    # plt.stem(np.arange(0, grid_size, 1.0 / oversample), np.ones(grid_size * oversample), linefmt='r--', markerfmt='')
    # plt.stem(np.arange(grid_size), np.ones(grid_size), linefmt='k-', markerfmt='')
    # plt.plot(np.arange(0, grid_size, 1.0/oversample), cffull, 'g')
    # plt.show()

    lm_extents = cell_size*(grid_size // 2)
    lm = np.linspace(-lm_extents, lm_extents, grid_size)
    dft_grid = np.zeros((oned_grid.shape[0]), dtype=oned_grid.dtype)

    vis_to_im_impl(vis, uvw, lm, dft_grid)
    oned_grid_fft = Fs(iFFT(iFs(oned_grid))).real

    oned_grid_fft *= grid_size


    # u = (np.arange(grid_size, dtype=np.float64) - ((grid_size - 1) // 2)) / (grid_size * oversample)
    u = (np.arange(grid_size, dtype=np.float64) - ((grid_size - 1) // 2)) / (grid_size)
    print(u)
    taper = kaiser_bessel_fourier(u, width, beta, J=filter_width)

    taper_opt = oned_grid_fft / dft_grid

    # taper_slow = give_cn_slow(u, conv_filter, filter_width, oversample)

    plt.figure('taper')
    plt.plot(np.arange(grid_size), taper, 'k')
    plt.plot(np.arange(grid_size), taper_opt, 'r')
    # plt.plot(np.arange(grid_size), taper_slow, 'g')
    # plt.plot(np.arange(0, grid_size, 1.0/oversample), cffullhat, 'b')
    plt.show()


    oned_grid_fft /= taper

    plt.figure('compare')
    plt.plot(dft_grid.squeeze().real, label='dft')
    plt.plot(oned_grid_fft.real, label='grid')
    plt.legend()
    plt.show()

    # plt.figure('diff')
    # plt.plot(dft_grid.squeeze().real - oned_grid_fft.real, label='real')
    # plt.plot(dft_grid.squeeze().imag - oned_grid_fft.imag, label='imag')
    # plt.legend()
    # plt.show()

    # print(np.stack([dft_grid.squeeze(), oned_grid_fft]))
    # print(dft_grid.squeeze().real - oned_grid_fft.real)

test_oned_gridding()
