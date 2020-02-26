#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import argparse
import numpy as np
from astropy.io import fits
import warnings
from daskms import xds_from_ms
from africanus.model.spi.examples.utils import load_fits_contiguous
iFs = np.fft.ifftshift
Fs = np.fft.fftshift

# we want to fall back to numpy if pypocketfft is not installed
# so set up functions to have the same call signatures
try:
    from pypocketfft import r2c, c2r

    def fft(x, ax, ncpu):
        return r2c(x, axes=ax, forward=True,
                   nthreads=ncpu, inorm=0)

    def ifft(y, ax, ncpu, lastsize):
        return c2r(y, axes=ax, forward=False, lastsize=lastsize,
                   nthreads=args.ncpu, inorm=2)
except:
    warnings.warn("No pypocketfft installation found. "
                  "FFT's will be performed in serial. "
                  "Install pypocketfft from "
                  "https://gitlab.mpcdf.mpg.de/mtr/pypocketfft "
                  "for optimal performance.",
                  ImportWarning)
    from numpy.fft import rfftn, irfftn
    # additional arguments will have no effect

    def fft(x, ax, ncpu):
        return rfftn(x, axes=ax)

    def ifft(y, ax, ncpu, lastsize):
        return irfftn(y, axes=ax)

def convolve_model(model, gausskern, args):
    print("Doing FFT's")
    # get padding
    _, npix_l, npix_m = model.shape
    pfrac = args.padding_frac/2.0
    npad_l = int(pfrac*npix_l)
    npad_m = int(pfrac*npix_m)
    # get fast FFT sizes
    try:
        from scipy.fftpack import next_fast_len
        nfft = next_fast_len(npix_l + 2*npad_l)
        npad_ll = (nfft - npix_l)//2
        npad_lr = nfft - npix_l - npad_ll
        nfft = next_fast_len(npix_m + 2*npad_m)
        npad_ml = (nfft - npix_m)//2
        npad_mr = nfft - npix_m - npad_ml
        padding = ((0, 0), (npad_ll, npad_lr), (npad_ml, npad_mr))
        unpad_l = slice(npad_ll, -npad_lr)
        unpad_m = slice(npad_ml, -npad_mr)
    except:
        warnings.warn("Could not determine fast fft size. "
                      "Install scipy for optimal performance.",
                      ImportWarning)
        padding = ((0, 0), (npad_l, npad_l), (npad_m, npad_m))
        unpad_l = slice(npad_l, -npad_l)
        unpad_m = slice(npad_m, -npad_m)
    ax = (1, 2)  # axes over which to perform fft
    lastsize = npix_m + np.sum(padding[-1])

    # get FT of convolution kernel
    gausskernhat = fft(iFs(np.pad(gausskern[None], padding, mode='constant'),
                           axes=ax), ax, args.ncpu)

    # Convolve model with Gaussian kernel
    convmodel = fft(iFs(np.pad(model, padding, mode='constant'), axes=ax),
                    ax, args.ncpu)

    convmodel *= gausskernhat
    return Fs(ifft(convmodel, ax, args.ncpu, lastsize),
              axes=ax)[:, unpad_l, unpad_m]

def create_parser():
    p = argparse.ArgumentParser(description='Simple spectral index fitting'
                                            'tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--image", type=str, required=True)
    p.add_argument('--output-filename', type=str,
                   help="Path to output directory. \n"
                        "Placed next to input model if outfile not provided.")
    p.add_argument('--psf-pars', default=None, nargs='+', type=float,
                   help="Beam parameters matching FWHM of restoring beam "
                        "specified as emaj emin pa. \n"
                        "By default these are taken from the fits header "
                        "of the residual image.")
    p.add_argument('--ncpu', default=0, type=int,
                   help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    p.add_argument("--circ-psf", action="store_true",
                   help="Passing this flag will convolve with a circularised "
                   "beam instead of an elliptical one")
    p.add_argument('--beam-model', default=None, type=str,
                   help="Fits beam model to use. \n"
                        "Use power_beam_maker to make power beam "
                        "corresponding to image. ")
    p.add_argument('--pb-min', type=float, default=0.05,
                   help="Set image to zero where pb falls below this value")
    return p

def main(args):

    return

if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    print(' \n ')
    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    print(' \n ')

    print("Using %i threads" % args.ncpu)

    print(' \n ')

    main(args)