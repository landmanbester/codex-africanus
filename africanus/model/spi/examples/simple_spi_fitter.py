#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import argparse
import dask
import dask.array as da
import numpy as np
from africanus.util.numba import jit
from astropy.io import fits
import warnings
from africanus.model.spi.dask import fit_spi_components
from africanus.model.spi.examples.utils import load_fits_contiguous, get_fits_freq_space_info, set_header_info, Gaussian2D
from africanus.model.spi.examples.image_convolver import convolve_model


def create_parser():
    p = argparse.ArgumentParser(description='Simple spectral index fitting'
                                            'tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--residual", type=str)
    p.add_argument('-o', '--output-filename', type=str,
                   help="Path to output directory. \n"
                        "Placed next to input model if outfile not provided.")
    p.add_argument('-pp', '--psf-pars', default=None, nargs='+', type=float,
                   help="Beam parameters matching FWHM of restoring beam "
                        "specified as emaj emin pa. \n"
                        "By default these are taken from the fits header "
                        "of the residual image.")
    p.add_argument('-cp', "--circ-psf", action="store_true",
                   help="Passing this flag will convolve with a circularised "
                   "beam instead of an elliptical one")
    p.add_argument('-th', '--threshold', default=5, type=float,
                   help="Multiple of the rms in the residual to threshold "
                        "on. \n"
                        "Only components above threshold*rms will be fit.")
    p.add_argument('-maxDR', '--maxDR', default=100, type=float,
                   help="Maximum dynamic range used to determine the "
                        "threshold above which components need to be fit. \n"
                        "Only used if residual is not passed in.")
    p.add_argument('-ncpu', '--ncpu', default=0, type=int,
                   help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    p.add_argument('-bm', '--beam-model', default=None, type=str,
                   help="Fits beam model to use. \n"
                        "Use power_beam_maker to make power beam "
                        "corresponding to image. ")
    p.add_argument('-products', '--products', default='aeikIbcm', type=str,
                   help="Outputs to write. Letter correspond to: \n"
                   "a - alpha map \n"
                   "e - alpha error map \n"
                   "i - I0 map \n"
                   "k - I0 error map \n"
                   "I - reconstructed cube form alpha and I0 \n"
                   "c - restoring beam used for convolution \n"
                   "m - convolved model \n"
                   "Default is to write all of them")
    p.add_argument('-pf', "--padding-frac", default=0.2, type=float,
                   help="Padding factor for FFT's.")
    p.add_argument('-dc', "--dont-convolve", action="store_true",
                   help="Passing this flag bypasses the convolution "
                   "by the clean beam")
    p.add_argument('-cw', "--channel_weights", default=None, nargs='+', type=float,
                   help="Per-channel weights to use during fit to frqequency axis. \n "
                   "Only has an effect if no residual is passed in (for now).")
    p.add_argument('rf', '--ref-freq', default=None, type=np.float64,
                   help='Refernce frequency at which the cube is saught. \n'
                   "Will overwrite in fits headers of output.")
    return p

def main(args):
    if args.pp is None:
        print("Attempting to take beampars from residual fits header")
        try:
            rhdr = fits.getheader(args.fitsresidual)
        except KeyError:
            raise RuntimeError("Either provide a residual with beam "
                               "information or pass them in using --beampars "
                               "argument")
        if 'BMAJ1' in rhdr.keys():
            emaj = rhdr['BMAJ1']
            emin = rhdr['BMIN1']
            pa = rhdr['BPA1']
            beampars = (emaj, emin, pa)
        elif 'BMAJ' in rhdr.keys():
            emaj = rhdr['BMAJ']
            emin = rhdr['BMIN']
            pa = rhdr['BPA']
            beampars = (emaj, emin, pa)
    else:
        beampars = tuple(args.pp)
        
    if args.cp:
        emaj = emin = (beampars[0] + beampars[1])/2.0
    print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e \n" % beampars)

    # load model image
    model = load_fits_contiguous(args.model)
    mhdr = fits.getheader(args.model)

    l_coord, m_coord, freqs, ref_freq, freq_axis = get_fits_freq_space_info(mhdr)
    nband = freqs.size
    npix_l = l_coord.size
    npix_m = m_coord.size

    if args.rf is not None and args.rf != ref_freq:
        ref_freq = args.rf
        print('Provided reference frequency does not match that of fits file. Will overwrite.')

    print("Cube frequencies:")
    with np.printoptions(precision=2):
        print(freqs)
    print("Reference frequency is %3.2e Hz \n" % ref_freq)

    # LB - new header for cubes if ref_freqs differ
    new_hdr = set_header_info(mhdr, ref_freq, freq_axis, args)

    # save next to model if no outfile is provided
    if args.o is None:
        # strip .fits from model filename 
        tmp = args.model[::-1]
        idx = tmp.find('.')
        outfile = args.model[0:-idx]
    else:
        outfile = args.o

    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')

    if not args.dc:
        print("Computing clean beam")
        # get the Gaussian convolution kernel
        gausskern = Gaussian2D(xx, yy, beampars)

        # save clean beam
        if 'c' in args.products:
            hdu = fits.PrimaryHDU(header=new_hdr)
            hdu.data = gausskern.T[:, ::-1].astype(np.float32)
            name = outfile + '.clean_psf.fits'
            hdu.writeto(name, overwrite=True)
            print("Wrote clean psf to %s \n" % name)

        # Convolve model with Gaussian restroring beam at lowest frequency
        model = convolve_model(model, gausskern, args)

        # save convolved model
        if 'm' in args.output:
            hdu = fits.PrimaryHDU(header=mhdr)
            # save it
            if freq_axis == 3:
                hdu.data = np.transpose(model, axes=(0, 2, 1))[None, :, :, ::-1]
            elif freq_axis == 4:
                hdu.data = np.transpose(model, axes=(0, 2, 1))[:, None, :, ::-1]
            name = outfile + '.convolved_model.fits'
            hdu.writeto(name, overwrite=True)
            print("Wrote convolved model to %s \n" % name)


    # set threshold
    if args.residual is not None:
        resid = load_fits_contiguous(args.fitsresidual)
        l_res, m_res, freqs_res, ref_freq, freq_axis = get_fits_freq_space_info(mhdr)
        
        if not np.array_equal(l_res, l_coord):
            raise ValueError("l coordinates of residual do not match those of model")

        if not np.array_equal(m_res, m_coord):
            raise ValueError("m coordinates of residual do not match those of model")

        if not np.array_equal(freqs, freqs_res):
            raise ValueError("Freqs of residual do not match those of model")

        rms = np.std(resid)
        rms_cube = np.std(resid.reshape(nband, npix_l*npix_m), axis=1).ravel()
        threshold = args.threshold * rms
        print("Setting cutoff threshold as %i times the rms "
            "of the residual " % args.threshold)
        del resid
    else:
        print("No residual provided. Setting  threshold i.t.o dynamic range. "
            "Max dynamic range is %i " % args.maxDR)
        threshold = model.max()/args.maxDR
        rms_cube = None

    print("Threshold set to %f Jy. \n" % threshold)

    # get pixels above threshold
    minimage = np.amin(model, axis=0)
    maskindices = np.argwhere(minimage > threshold)
    if not maskindices.size:
        raise ValueError("No components found above threshold. "
                        "Try lowering your threshold."
                        "Max of convolved model is %3.2e" % model.max())
    fitcube = model[:, maskindices[:, 0], maskindices[:, 1]].T

    # image space correction for primary beam
    if args.bm is not None:
        beam_source = beam_image[:, maskindices[:, 0], maskindices[:, 1]].T
        # correct cube
        fitcube /= beam_source

    # set weights for fit
    if rms_cube is not None:
        print("Using RMS in each imaging band to determine weights. \n")
        weights = np.where(rms_cube > 0, 1.0/rms_cube**2, 0.0)
        # normalise
        weights /= weights.max()
    else:
        if args.channel_weights is not None:
            weights = np.array(args.channel_weights)
            print("Using provided channel weights \n")
        else:
            print("No residual or channel weights provided. Using equal weights. \n")
            weights = np.ones(nband, dtype=np.float64)

    ncomps, _ = fitcube.shape
    fitcube = da.from_array(fitcube.astype(np.float64),
                            chunks=(ncomps//args.ncpu, nband))
    weights = da.from_array(weights.astype(np.float64), chunks=(nband))
    freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

    print("Fitting %i components" % ncomps)
    alpha, alpha_err, Iref, i0_err = fit_spi_components(fitcube, weights, freqsdask,
                                        np.float64(ref_freq)).compute()
    print("Done. Writing output. \n")

    alphamap = np.zeros(model[0].shape, dtype=model.dtype)
    alpha_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    i0map = np.zeros(model[0].shape, dtype=model.dtype)
    i0_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
    alpha_err_map[maskindices[:, 0], maskindices[:, 1]] = alpha_err
    i0map[maskindices[:, 0], maskindices[:, 1]] = Iref
    i0_err_map[maskindices[:, 0], maskindices[:, 1]] = i0_err

    if 'I' in args.output:
        hdu = fits.PrimaryHDU(header=mhdr)
        # get the reconstructed cube
        Irec_cube = i0map[None, :, :] * \
            (freqs[:, None, None]/ref_freq)**alphamap[None, :, :]
        # save it
        if freq_axis == 3:
            hdu.data = np.transpose(Irec_cube, axes=(0, 2, 1))[None, :, :, ::-1]
        elif freq_axis == 4:
            hdu.data = np.transpose(Irec_cube, axes=(0, 2, 1))[:, None, :, ::-1]
        name = outfile + 'Irec_cube.fits'
        hdu.writeto(name, overwrite=True)
        print("Wrote reconstructed cube to %s" % name)

    # save alpha map
    if 'a' in args.output:
        hdu = fits.PrimaryHDU(header=new_hdr)
        hdu.data = alphamap.T[:, ::-1].astype(np.float32)
        name = outfile + 'alpha.fits'
        hdu.writeto(name, overwrite=True)
        print("Wrote alpha map to %s" % name)

    # save alpha error map
    if 'e' in args.output:
        hdu = fits.PrimaryHDU(header=new_hdr)
        hdu.data = alpha_err_map.T[:, ::-1].astype(np.float32)
        name = outfile + 'alpha_err.fits'
        hdu.writeto(name, overwrite=True)
        print("Wrote alpha error map to %s" % name)

    # save I0 map
    if 'i' in args.output:
        hdu = fits.PrimaryHDU(header=new_hdr)
        hdu.data = i0map.T[:, ::-1].astype(np.float32)
        name = outfile + 'I0.fits'
        hdu.writeto(name, overwrite=True)
        print("Wrote I0 map to %s" % name)

    # save I0 error map
    if 'i' in args.output:
        hdu = fits.PrimaryHDU(header=new_hdr)
        hdu.data = i0_err_map.T[:, ::-1].astype(np.float32)
        name = outfile + 'I0_err.fits'
        hdu.writeto(name, overwrite=True)
        print("Wrote I0 error map to %s" % name)

    print(' \n ')

    print("All done here")


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
