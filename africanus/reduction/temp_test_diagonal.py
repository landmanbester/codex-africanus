#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from xarrayms import xds_from_ms, xds_from_table
from africanus.dft.dask import im_to_vis, vis_to_im
import matplotlib.pyplot as plt
import numpy as np
from africanus.reduction.psf_redux import F, iF, diag_probe, PSF_response, FFT, iFFT
from africanus.gridding.util import estimate_cell_size
from africanus.constants import c as lightspeed
import dask.array as da
from astropy.io import fits

_ARCSEC2RAD = np.deg2rad(1.0/(60*60))

# rad/dec to lm coordinates (straight from fundamentals)
def radec_to_lm(ra0, dec0, ra, dec):
    delta_ra = ra - ra0
    l = (np.cos(dec) * np.sin(delta_ra))
    m = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra))
    return l, m

def plot(im, stage):
    plt.figure(stage)
    plt.imshow(im.reshape([npix, npix]).real)
    plt.colorbar()

# read in data file (not on the git so must be changed!)
nrow = 1000
nchan = 1
data_path = "/home/landman/Projects/Data/MS_dir/ddfacet_test_data/WSCMSSSMFTestSuite/SSMF.MS_p0"
xds = list(xds_from_ms(data_path, chunks={"row": nrow}))[0]
spw_ds = list(xds_from_table("::".join((data_path, "SPECTRAL_WINDOW")), group_cols="__row__"))[0]
#
# vis_I = (xds.DATA.data[:, :, 0] + xds.DATA.data[:, :, 3]/2)
# vis_I = vis_I.rechunk((nrow, nchan))

# make sure we are oversampling the image enough
uvw = xds.UVW.data[0:nrow].compute()
uvw[:, 2] = 0.0  # this is for testing
freqs = spw_ds.CHAN_FREQ.data[0:nchan].compute()
wavelengths = lightspeed/freqs
cell_size = estimate_cell_size(uvw[:, 0], uvw[:, 1], wavelengths, factor=1.5).min()
cell_size_rad = _ARCSEC2RAD*cell_size



# set up fov
# this is the position of the source
ra_pos = 3.15126500e-05
dec_pos = -0.00551471375
l_val, m_val = radec_to_lm(0, 0, ra_pos, dec_pos)
fov = max(abs(l_val), abs(m_val))*1.5  # making sure the source is not at the edge of the field

# set number of pixels required to properly oversample the image
npix = int(2*fov/cell_size_rad)
if not npix%2:
    npix += 1  # make sure it is odd

print("You need to use at least npix = ", npix)

pad_factor = 1
padding = int(npix*pad_factor)
pad_pix = npix + 2*padding

x = np.linspace(-fov, fov, npix)
cell_size = x[1] - x[0]  # might have changed slightly from the recommended value
ll, mm = np.meshgrid(x, x)
lm = np.vstack((ll.flatten(), mm.flatten())).T

pad_range = fov + padding*cell_size
x_pad = np.linspace(-pad_range, pad_range, pad_pix)
ll_pad, mm_pad = np.meshgrid(x_pad, x_pad)
lm_pad = np.vstack((ll_pad.flatten(), mm_pad.flatten())).T

# normalisation factor (equal to max(PSF))
weights = xds.WEIGHT.data[0:nrow, 0:1].compute()

sqrtwsum = np.sqrt(np.sum(weights))

# plt.figure()
# plt.plot(x, x, 'rx', alpha=0.35)
# plt.plot(x_pad, x_pad, 'b+', alpha=0.35)
# plt.show()
#
# print("wsum = ", sqrtwsum**2)
# print("Diff = ", np.abs(x_pad[padding:-padding] - x).max())
#
# import sys
# sys.exit(0)

# Turn DFT into lambda functions for easy, single input access
NCPU = 8
chunkrow = nrow//NCPU
lm_dask = da.from_array(lm, chunks=(npix**2, 2))
lm_pad_dask = da.from_array(lm_pad, chunks=(pad_pix**2, 2))
#uvw_dask = xds.UVW.data[0:nrow, :].rechunk((chunkrow, 3))
uvw_dask = da.from_array(uvw, chunks=(chunkrow, 3))
frequency_dask = spw_ds.CHAN_FREQ.data.rechunk(nchan)[0:nchan]
weights_dask = da.from_array(weights, chunks=(chunkrow, nchan))

# do not call compute until the entire graph has been built
L = lambda image: im_to_vis(image, uvw_dask, lm_dask, frequency_dask)
LT = lambda v: vis_to_im(v, uvw_dask, lm_dask, frequency_dask)/sqrtwsum
L_pad = lambda image: im_to_vis(image, uvw_dask, lm_pad_dask, frequency_dask)
LT_pad = lambda v: vis_to_im(v, uvw_dask, lm_pad_dask, frequency_dask)/sqrtwsum

# # Generate FFT and DFT matrices
# R = np.zeros([nrow, pad_pix**2], dtype='complex128')
# FT = np.zeros([pad_pix**2, pad_pix**2], dtype='complex128')
# for k in range(nrow):
#     u, v, w = uvw[k]
#
#     for j in range(pad_pix**2):
#         l, m = lm_pad[j]
#         n = np.sqrt(1.0 - l ** 2 - m ** 2) - 1.0
#         R[k, j] = np.exp(-2j*np.pi*(freq[0]/c)*(u*l + v*m + w*n))/np.sqrt(wsum)
#
# delta = lm_pad[1, 0]-lm_pad[0, 0]
# F_norm = pad_pix**2
# Ffreq = np.fft.fftshift(np.fft.fftfreq(pad_pix, d=delta))
# jj, kk = np.meshgrid(Ffreq, Ffreq)
# jk = np.vstack((jj.flatten(), kk.flatten())).T
#
# for u in range(pad_pix**2):
#     l, m = lm_pad[u]
#     for v in range(pad_pix**2):
#         j, k = jk[v]
#         FT[u, v] = np.exp(-2j*np.pi*(j*l + k*m))/np.sqrt(F_norm)
#
# # for u in range(pad_pix**2):
# #     j, k = jk[u]
# #     for v in range(pad_pix**2):
# #         l, m = lm_pad[v]
# #         FT[u, v] = np.exp(-2j*np.pi*(j*l + k*m))/np.sqrt(F_norm)
#
# R.tofile('R.dat')
# FT.tofile('F.dat')
#
# R = np.fromfile('R.dat', dtype='complex128').reshape([nrow, pad_pix**2])
# FT = np.fromfile('F.dat', dtype='complex128').reshape([pad_pix**2, pad_pix**2])
#
#
# # Generate adjoint matrices (conjugate transpose)
# RH = R.conj().T  # moved the wsum to inside the loop since both operators should have it
# FH = FT.conj().T

# w = np.diag(weights.flatten())
#
# # Generate the PSF using DFT
# PSF = LT_pad(weights_dask).compute().reshape([pad_pix, pad_pix])
#
# plt.figure('PSF real')
# plt.imshow(PSF.real)
# plt.colorbar()
#
# plt.figure('PSF imag')
# plt.imshow(PSF.imag)
# plt.colorbar()

PSF_hat = FFT(LT_pad(weights_dask).reshape([pad_pix, pad_pix]))  #.compute()
# PSF_hat = FFT(LT(weights_dask).reshape(npix, npix))
# plt.figure('PSF hat real')
# plt.imshow(PSF_hat.real)
# plt.colorbar()
#
# plt.figure('PSF hat imag')
# plt.imshow(PSF_hat.imag)
# plt.colorbar()

# plt.show()
#
# import sys
# sys.exit(0)

# def PSF_probe(vec):
#     T0 = np.pad(vec.reshape([npix, npix]), [padding, padding], 'constant')
#     T1 = F(T0)
#     T2 = PSF_hat * T1
#     T3 = iF(T2).real
#     return T3[padding:-padding, padding:-padding].flatten()*np.sqrt(npix)  #pad_factor#*(pad_pix - npix)/pad_pix


# sigma = np.ones(pad_pix**2)
# P = lambda image: PSF_response(image, PSF_hat, sigma)*np.sqrt(pad_pix**2/wsum)

# # Test that RH and LT produce the same value
# test_ones = np.ones_like(weights)
# test0 = RH.dot(test_ones).real
# test1 = LT(test_ones).real
# test2 = abs(test0 - test1)
# print("Sum of difference between RH and LT: ", sum(test2.flatten()))
#
# # Test self adjointness of R RH
# gamma1 = np.random.randn(pad_pix**2)
# gamma2 = np.random.randn(weights.size)
#
# LHS = gamma2.T.dot(R.dot(gamma1)).real
# RHS = RH.dot(gamma2).T.dot(gamma1).real
#
# print("Self adjointness of R: ", np.abs(LHS - RHS))

# Test that PSF convolution and M give the same answer
# vec = np.exp(-(ll_pad**2 + mm_pad**2)*100)
vec = np.zeros([npix, npix])
vec[npix//4, npix//4] = 1.0
vec = np.pad(vec, padding, mode='constant')
vec = F(vec)
# vec[np.random.randint(0, npix), np.random.randint(0, npix)] = 1
# vec[3*npix//4, npix//3] = 1.0
# vec = np.random.randn(pad_pix, pad_pix)
# plt.figure('in real')
# plt.imshow(vec.real)
# plt.colorbar()
# plt.figure('in imag')
# plt.imshow(vec.imag)
# plt.colorbar()
# vec = np.pad(vec, padding, mode='constant')
# vec = np.ones([pad_pix, pad_pix])
vec_dask = da.from_array(vec, chunks=(pad_pix, pad_pix))
# # convolve with PSF
# vec_hat = FFT(vec_dask)
# vec_hat *= PSF_hat
# vec_convolved = iFFT(vec_hat)
#
# # now act on vec with the M op
# vec_convolved_M = LT_pad(weights_dask * L_pad(vec_dask_flat)).compute().reshape(pad_pix, pad_pix)
#
# # LB - Note the difference when w != 0
# I = slice(padding, -padding)
# plt.figure('diff')
# plt.imshow(vec_convolved.real[I, I] - vec_convolved_M.real[I, I])
# plt.colorbar()
#
# plt.figure('PSF conv')
# plt.imshow(np.abs(vec_convolved[I, I]))
# plt.colorbar()
#
# plt.figure('M op')
# plt.imshow(np.abs(vec_convolved_M[I, I]))
# plt.colorbar()
#
# plt.show()

# next we construct the dimensionally reduced covariance operator
print("Computing Mhat_op")

# get the dimensionally reduced version
Mhat_op2 = lambda x: (PSF_hat * x).compute()

res2 = Mhat_op2(vec_dask)

# Mhat_op = lambda x: FFT(LT_pad(weights_dask * L_pad(da.real(iFFT(x)).reshape(pad_pix**2, nchan))).reshape(pad_pix, pad_pix)).compute()
Mhat_op = lambda x: FFT(LT(weights_dask * L(x)).reshape(npix, npix)).compute()

# try probing operator
from africanus.reduction.psf_redux import guessmatrix
res = guessmatrix(Mhat_op, nrow, npix**2)

plt.figure('diag Mop via probing')
plt.plot(res.real, 'kx')


#res = Mhat_op(vec_dask)

# plt.figure('M op')
# plt.imshow(res.real)
# plt.colorbar()
#
# plt.figure('PSF conv')
# plt.imshow(res2.real)
# plt.colorbar()
#
# plt.figure('diff real')
# plt.imshow(res2.real - res.real)
# plt.colorbar()
#
# plt.figure('diff imag')
# plt.imshow(res2.imag - res.imag)
# plt.colorbar()
#
# # plot it
# hdu = fits.PrimaryHDU(res.real)
# hdul = fits.HDUList([hdu])
# hdul.writeto('M_op.fits', overwrite=True)
# hdul.close()
#
# # plot it
# hdu = fits.PrimaryHDU(res2.real)
# hdul = fits.HDUList([hdu])
# hdul.writeto('PSF_conv.fits', overwrite=True)
# hdul.close()

# # plot it
# tmp = PSF_hat.compute().real
# tmp /= tmp.max()
# hdu = fits.PrimaryHDU(tmp)
# hdul = fits.HDUList([hdu])
# hdul.writeto('PSF_hat.fits', overwrite=True)
# hdul.close()

# plt.figure('uv')
# plt.plot(uvw[:, 0], uvw[:, 1], 'rx')
# plt.plot(-uvw[:, 0], -uvw[:, 1], 'rx')

plt.show()
