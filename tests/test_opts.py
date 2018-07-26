#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `codex-africanus` package."""

import numpy as np
# from africanus.dft.kernels import im_to_vis, vis_to_im
from astropy.io import fits
import pytest
import xarrayms
import matplotlib.pyplot as plt
from africanus.opts.primaldual import primal_dual_solver as pds

# Test Power method
# from sub_opts import pow_method as pm
# import numpy as np
#
# eig = 0
#
# while eig is 0:
#     A = np.random.randn(10, 10)
#     G = A.T.dot(A)
#     eig_vals = np.linalg.eigvals(G)
#     if min(eig_vals) > 0:
#         eig = max(eig_vals)
#
# spec = pm(G.dot, G.conj().T.dot, [10,1])
#
# print(eig-spec)


# def test_pd():
"""
Tests to see if there is any great difference between the model image image and the resolved image
"""

npix = 256


def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))
    return l, m

# generate lm-coordinates
ra_pos = 3.15126500e-05
dec_pos = -0.00551471375
# source = SkyCoord(ra=ra_pos*u.degree, dec=dec_pos*u.degree, frame='icrs')
# l_val = source.galactic.l*1.5
# m_val = source.galactic.b*1.5
l_val, m_val = radec_to_lm(0, 0, ra_pos, dec_pos)
x_range = max(abs(l_val), abs(m_val))*1.5
x = np.linspace(-x_range, x_range, npix)
ll, mm = np.meshgrid(x, x)
lm = np.vstack((ll.flatten(), mm.flatten())).T

# generate frequencies
frequency = np.array([1.06e9])
ref_freq = 1#.53e9
freq = frequency/ref_freq

data_path = "/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/SSMF.MS_p0"

nrow = 500
nchan = 1

for ds in xarrayms.xds_from_ms(data_path):
    Vdat = ds.DATA.data.compute()
    uvw = ds.UVW.data.compute()[0:nrow, :]
    weights = ds.WEIGHT.data.compute()[0:nrow, 0:nchan]

vis = Vdat[0:nrow, 0:nchan, 0]# + Vdat[0:nrow, 0:nchan, 3])/2.0).reshape(nrow, nchan)


# PSF = vis_to_im(weights_dask, uvw_dask, lm_dask, frequency_dask).compute()
# PSF = vis_to_im(weights, uvw, lm, freq)

wsum = sum(weights)  # PSF.max()


#
# L = lambda image: im_to_vis(image, uvw, lm, freq)
# LT = lambda v: vis_to_im(v, uvw, lm, freq)
#
# dirty = LT(vis)

# test = L(dirty)
# print(vis - test)

# start = np.zeros_like(dirty)
# start[int(npix**2/2),] = 10
#
# cleaned = pds(start, vis, L, LT, wsum, solver='spd')
#
# plt.figure('ID')
# plt.imshow(dirty.reshape(npix, npix)/wsum)
# plt.colorbar()
#
# plt.figure('IM')
# plt.imshow(cleaned.reshape(npix, npix))
# plt.colorbar()
#
# plt.show()
#
# hdu = fits.PrimaryHDU(dirty.reshape(npix, npix))
# hdul = fits.HDUList([hdu])
# hdul.writeto('/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/dirty.fits', overwrite=True)
# hdul.close()
#
# hdu = fits.PrimaryHDU(cleaned.reshape(npix, npix))
# hdul = fits.HDUList([hdu])
# hdul.writeto('/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/recovered.fits', overwrite=True)
# hdul.close()

# def test_pd_dask
from africanus.opts.pd_dask import primal_dual_solver as pdd
from africanus.dft.dask import vis_to_im, im_to_vis
import dask.array as da

# set up dask arrays
uvw_dask = da.from_array(uvw, chunks=(1000, 3))
lm_dask = da.from_array(lm, chunks=(npix, 2))
frequency_dask = da.from_array(freq, chunks=nchan)
vis_dask = da.from_array(vis, chunks=(1000, nchan))
weights_dask = da.from_array(weights, chunks=(1000, nchan))

L_d = lambda image: im_to_vis(image, uvw_dask, lm_dask, frequency_dask).compute()
LT_d = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask).compute()/wsum

dirty = LT_d(vis_dask)

start = da.zeros_like(dirty)
start[int(npix**2/2),] = 10

cleaned = pdd(start, vis_dask, L_d, LT_d, wsum, solver='spd')

plt.figure('ID Dask')
plt.imshow(dirty.reshape(npix, npix)/wsum)
plt.colorbar()

plt.figure('IM Dask')
plt.imshow(cleaned.reshape(npix, npix))
plt.colorbar()

plt.show()

hdu = fits.PrimaryHDU(dirty.reshape(npix, npix))
hdul = fits.HDUList([hdu])
hdul.writeto('/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/dirty_dask.fits', overwrite=True)
hdul.close()

hdu = fits.PrimaryHDU(cleaned.reshape(npix, npix))
hdul = fits.HDUList([hdu])
hdul.writeto('/home/antonio/Documents/Masters/Helpful_Stuff/WSCMSSSMFTestSuite/recovered_dask.fits', overwrite=True)
hdul.close()
