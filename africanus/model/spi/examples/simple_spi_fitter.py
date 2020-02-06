#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import argparse
import dask
import dask.array as da
import numpy as np
from astropy.io import fits
from pyrap.tables import table
import warnings
from africanus.model.spi.dask import fit_spi_components
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


def Gaussian2D(xin, yin, GaussPar=(1., 1., 0.)):
    S0, S1, PA = GaussPar
    Smaj = np.maximum(S0, S1)
    Smin = np.minimum(S0, S1)
    A = np.array([[1. / Smin ** 2, 0],
                  [0, 1. / Smaj ** 2]])

    c, s, t = np.cos, np.sin, np.deg2rad(-PA)
    R = np.array([[c(t), -s(t)],
                  [s(t), c(t)]])
    A = np.dot(np.dot(R.T, A), R)
    sOut = xin.shape
    # only compute the result out to 5 * emaj
    extent = (5 * Smaj)**2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    ind = np.argwhere(xflat**2 + yflat**2 <= extent).squeeze()
    idx = ind[:, 0]
    idy = ind[:, 1]
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    R = np.einsum('nb,bc,cn->n', x.T, A, x)
    # need to adjust for the fact that GaussPar corresponds to FWHM
    fwhm_conv = 2*np.sqrt(2*np.log(2))
    tmp = np.exp(-fwhm_conv*R)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp
    return np.ascontiguousarray(gausskern.reshape(sOut),
                                dtype=np.float64)


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


def interpolate_beam(xx, yy, freqs, args):
    print("Interpolating beam")
    lm_source = np.vstack((xx.ravel(), yy.ravel())).T

    # # get ms info required to compute paralactic angles
    # utime = []
    # for ms_name in args.ms:
    #     ms = table(ms)
    #     times = ms.getcol('TIME')
    #     utimes, time_bin_counts = np.unique(time, return_counts=True)
    #     utime.append(np.unique())

    ntime = 1
    nant = 1
    nband = freqs.size
    parangles = np.zeros((ntime, nant,), dtype=np.float64)
    ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
    point_errs = np.zeros((ntime, nant, nband, 2), dtype=np.float64)

    if args.beammodel == "eidos":
        raise NotImplementedError("eidos is coming!!!")
    else:
        print("Loading fits beam patterns from %s" % args.beammodel)
        from glob import glob
        paths = glob(args.beammodel + '**_**.fits')
        beam_hdr = None
        for path in paths:
            if 'XX'.lower() in path[-10::]: # or 'RR'.lower() in path:
                if 're' in path[-7::]:
                    corr1_re = load_fits_contiguous(path)
                    if beam_hdr is None:
                        beam_hdr = fits.getheader(path)
                elif 'im' in path[-7::]:
                    corr1_im = load_fits_contiguous(path)
                else:
                    raise NotImplementedError("Only re/im patterns supported")
            elif 'YY'.lower() in path[-10::]: # or 'LL'.lower() in path:
                if 're' in path[-7::]:
                    corr2_re = load_fits_contiguous(path)
                elif 'im' in path[-7::]:
                    corr2_im = load_fits_contiguous(path)
                else:
                    raise NotImplementedError("Only re/im patterns supported")
        # get power beam
        beam_amp = (corr1_re**2 + corr1_im**2 + corr2_re**2 + corr2_im**2)/2.0

        # get cube in correct shape for interpolation code
        beam_amp = np.ascontiguousarray(np.transpose(beam_amp, (1, 2, 0))
                                        [:, :, :, None, None])
        # get cube info
        if beam_hdr['CUNIT1'].lower() != "deg":
            raise ValueError("Beam image units must be in degrees")
        npix_l = beam_hdr['NAXIS1']
        refpix_l = beam_hdr['CRPIX1']
        delta_l = beam_hdr['CDELT1']
        l_min = (1 - refpix_l)*delta_l
        l_max = (1 + npix_l - refpix_l)*delta_l

        if beam_hdr['CUNIT2'].lower() != "deg":
            raise ValueError("Beam image units must be in degrees")
        npix_m = beam_hdr['NAXIS2']
        refpix_m = beam_hdr['CRPIX2']
        delta_m = beam_hdr['CDELT2']
        m_min = (1 - refpix_m)*delta_m
        m_max = (1 + npix_m - refpix_m)*delta_m

        print("Supplied beam has shape ", beam_amp.shape)

        if (l_min > lm_source[:, 0].min() or m_min > lm_source[:, 1].min() or
                l_max < lm_source[:, 0].max() or m_max < lm_source[:, 1].max()):
            raise ValueError("The supplied beam is not large enough")

        beam_extents = np.array([[l_min, l_max], [m_min, m_max]])

        # get frequencies
        if beam_hdr["CTYPE3"].lower() != 'freq':
            raise ValueError(
                "Cubes are assumed to be in format [nchan, nx, ny]")
        nchan = beam_hdr['NAXIS3']
        refpix = beam_hdr['CRPIX3']
        delta = beam_hdr['CDELT3']  # assumes units are Hz
        freq0 = beam_hdr['CRVAL3']
        bfreqs = freq0 + np.arange(1 - refpix, 1 + nchan - refpix) * delta
        if bfreqs[0] > freqs[0] or bfreqs[-1] < freqs[-1]:
            warnings.warn("The supplied beam does not have sufficient "
                          "bandwidth. Beam frequencies:")
            with np.printoptions(precision=2):
                print(bfreqs)

        # interpolate beam
        from africanus.rime.fast_beam_cubes import beam_cube_dde
        # from africanus.rime.dask import beam_cube_dde
        # beam_amp = da.from_array(beam_amp, chunks=beam_amp.shape)
        # beam_extents = da.from_array(beam_extents, chunks=beam_extents.shape)
        # bfreqs = da.from_array(bfreqs, chunks=bfreqs.shape)
        # lm_source = da.from_array(lm_source, chunks=lm_source.shape)
        # parangles = da.from_array(parangles, chunks=parangles.shape)
        # point_errs = da.from_array(point_errs, chunks=point_errs.shape)
        # ant_scale = da.from_array(ant_scale, chunks=ant_scale.shape)
        # freqs = da.from_array(freqs, chunks=freqs.shape)
        beam_source = beam_cube_dde(beam_amp, beam_extents, bfreqs,
                                    lm_source, parangles, point_errs,
                                    ant_scale, freqs).squeeze() #.compute()
        # average over time/ant

        # reshape to image shape
        print("Beam shape b = ", beam_source.shape)
        beam_source = np.transpose(beam_source, axes=(1, 0))
        return beam_source.squeeze().reshape((freqs.size, *xx.shape))


def create_parser():
    p = argparse.ArgumentParser(description='Simple spectral index fitting'
                                            'tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fitsmodel", type=str, required=True)
    p.add_argument("--fitsresidual", type=str)
    p.add_argument("--ms", nargs="+", type=str, 
                   help="Mesurement sets used to make the image. \n"
                   "Used to get paralactic angles if doing primary beam correction")
    p.add_argument('--outfile', type=str,
                   help="Path to output directory. \n"
                        "Placed next to input model if outfile not provided.")
    p.add_argument('--beampars', default=None, nargs='+', type=float,
                   help="Beam parameters matching FWHM of restoring beam "
                        "specified as emaj emin pa. \n"
                        "By default these are taken from the fits header "
                        "of the residual image.")
    p.add_argument('--threshold', default=5, type=float,
                   help="Multiple of the rms in the residual to threshold "
                        "on. \n"
                        "Only components above threshold*rms will be fit.")
    p.add_argument('--maxDR', default=100, type=float,
                   help="Maximum dynamic range used to determine the "
                        "threshold above which components need to be fit. \n"
                        "Only used if residual is not passed in.")
    p.add_argument('--ncpu', default=0, type=int,
                   help="Number of threads to use. \n"
                        "Default of zero means use all threads")
    p.add_argument('--beammodel', default=None, type=str,
                   help="Fits beam model to use. \n"
                        "It is assumed that the pattern is path_to_beam/"
                        "name_corr_re/im.fits. \n"
                        "Provide only the path up to name "
                        "e.g. /home/user/beams/meerkat_lband. \n"
                        "Patterns mathing corr are determined "
                        "automatically. \n"
                        "Only real and imaginary beam models currently "
                        "supported.")
    p.add_argument('--output', default='aiIbc', type=str,
                   help="Outputs to write. Letter correspond to: \n"
                   "a - alpha map \n"
                   "i - I0 map \n"
                   "I - reconstructed cube form alpha and I0 \n"
                   "b - interpolated beam \n"
                   "c - restoring beam used for convolution \n"
                   "Default is to write all of them")
    p.add_argument("--padding_frac", default=0.2, type=float,
                   help="Padding factor for FFT's.")
    p.add_argument("--dont_convolve", action="store_true",
                   help="Passing this flag bypasses the convolution "
                   "by the clean beam")
    return p


def load_fits_contiguous(name):
    arr = fits.getdata(name).squeeze()
    # transpose spatial axes (f -> c contiguous)
    arr = np.transpose(arr, axes=(0, 2, 1))[:, ::-1]
    return np.ascontiguousarray(arr, dtype=np.float64)

# def save_fits_contiguous(arr, hdu, name):
#     hdu.data = np.transpose(arr, axes=(0, 2, 1))[:, ::-1].astype(np.float32)

def main(args):

    
    if args.beampars is None:
        print("Attempting to take beampars from residual fits header")
        try:
            rhdr = fits.getheader(args.fitsresidual)
        except KeyError:
            raise RuntimeError("Either provide a residual with beam "
                               "information or pass them in using --beampars "
                               "argument")
        emaj = rhdr['BMAJ1']
        emin = rhdr['BMIN1']
        pa = rhdr['BPA1']
        beampars = (emaj, emin, pa)
    else:
        beampars = tuple(args.beampars)
        # emaj, emin, pa = args.beampars
    print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e" % beampars)
    print(beampars[-1])

    # load model image
    model = load_fits_contiguous(args.fitsmodel)
    mhdr = fits.getheader(args.fitsmodel)

    if mhdr['CUNIT1'].lower() != "deg":
        raise ValueError("Image coordinates must be in degrees")
    npix_l = mhdr['NAXIS1']
    refpix_l = mhdr['CRPIX1']
    delta_l = mhdr['CDELT1']
    l_coord = np.arange(1 - refpix_l, 1 + npix_l - refpix_l)*delta_l

    if mhdr['CUNIT2'].lower() != "deg":
        raise ValueError("Image coordinates must be in degrees")
    npix_m = mhdr['NAXIS2']
    refpix_m = mhdr['CRPIX2']
    delta_m = mhdr['CDELT2']
    m_coord = np.arange(1 - refpix_m, 1 + npix_m - refpix_m)*delta_m

    print("Image shape = ", (npix_l, npix_m))

    # get frequencies
    if mhdr["CTYPE4"].lower() == 'freq':
        freq_axis = 4
        nband = mhdr['NAXIS4']
        refpix_nu = mhdr['CRPIX4']
        delta_nu = mhdr['CDELT4']  # assumes units are Hz
        ref_freq = mhdr['CRVAL4']
        ncorr = mhdr['NAXIS3']
    elif mhdr["CTYPE3"].lower() == 'freq':
        freq_axis = 3
        nband = mhdr['NAXIS3']
        refpix_nu = mhdr['CRPIX3']
        delta_nu = mhdr['CDELT3']  # assumes units are Hz
        ref_freq = mhdr['CRVAL3']
        ncorr = mhdr['NAXIS4']
    else:
        raise ValueError("Freq axis must be 3rd or 4th")

    if ncorr > 1:
        raise ValueError("Only Stokes I cubes supported")

    freqs = ref_freq + np.arange(1 - refpix_nu,
                                 1 + nband - refpix_nu) * delta_nu

    print("Cube frequencies:")
    with np.printoptions(precision=2):
        print(freqs)
    print("Reference frequency is %3.2e Hz " % ref_freq)

    if not args.dont_convolve:
        # get the Gaussian convolution kernel
        xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')
        gausskern = Gaussian2D(xx, yy, beampars)

        # Convolve model with Gaussian restroring beam at lowest frequency
        model = convolve_model(model, gausskern, args)

    # set threshold
    if args.fitsresidual is not None:
        resid = load_fits_contiguous(args.fitsresidual)
        rms = np.std(resid)
        rms_cube = np.std(resid.reshape(nband, npix_l*npix_m), axis=1).ravel()
        threshold = args.threshold * rms
        print("Setting cutoff threshold as %i times the rms "
              "of the residual" % args.threshold)
        del resid
    else:
        print("No residual provided. Setting  threshold i.t.o dynamic range. "
              "Max dynamic range is %i" % args.maxDR)
        threshold = model.max()/args.maxDR
        rms_cube = None

    print("Threshold set to %f Jy." % threshold)

    # get pixels above threshold
    minimage = np.amin(model, axis=0)
    maskindices = np.argwhere(minimage > threshold)
    if not maskindices.size:
        raise ValueError("No components found above threshold. "
                         "Try lowering your threshold."
                         "Max of convolved model is %3.2e" % model.max())
    fitcube = model[:, maskindices[:, 0], maskindices[:, 1]].T

    # get primary beam at source locations
    if args.beammodel is not None:
        beam_image = interpolate_beam(xx, yy, freqs, args)
        beam_source = beam_image[:, maskindices[:, 0], maskindices[:, 1]].T
        # correct cube
        fitcube /= beam_source

    # set weights for fit
    if rms_cube is not None:
        print("Using RMS in each imaging band to determine weights.")
        weights = np.where(rms_cube > 0, 1.0/rms_cube**2, 0.0)
        # normalise
        weights /= weights.max()
    else:
        print("No residual provided. Using equal weights.")
        weights = np.ones(nband, dtype=np.float64)

    ncomps, _ = fitcube.shape
    fitcube = da.from_array(fitcube.astype(np.float64),
                            chunks=(ncomps//args.ncpu, nband))
    weights = da.from_array(weights.astype(np.float64), chunks=(nband))
    freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

    print("Fitting %i components" % ncomps)
    alpha, _, Iref, _ = fit_spi_components(fitcube, weights, freqsdask,
                                           np.float64(ref_freq)).compute()
    print("Done. Writing output.")

    alphamap = np.zeros(model[0].shape, dtype=model.dtype)
    i0map = np.zeros(model[0].shape, dtype=model.dtype)
    alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
    i0map[maskindices[:, 0], maskindices[:, 1]] = Iref

    # save next to model if no outfile is provided
    if args.outfile is None:
        # find last /
        tmp = args.fitsmodel[::-1]
        idx = tmp.find('/')
        if idx != -1:
            outfile = args.fitsmodel[0:-idx]
        else:
            outfile = 'image-'
    else:
        outfile = args.outfile

    hdu = fits.PrimaryHDU(header=mhdr)
    if 'I' in args.output:
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

    if args.beammodel is not None and 'b' in args.output:
        if freq_axis == 3:
            hdu.data =  np.transpose(beam_image, axes=(0, 2, 1))[None, :, :, ::-1]
        elif freq_axis == 4:
            hdu.data =  np.transpose(beam_image, axes=(0, 2, 1))[:, None, :, ::-1]
        name = outfile + 'interpolated_beam_cube.fits'
        hdu.writeto(name, overwrite=True)
        print("Wrote interpolated beam cube to %s" % name)

    hdr_keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                'NAXIS4', 'BUNIT', 'BMAJ', 'BMIN', 'BPA', 'EQUINOX', 'BTYPE',
                'TELESCOP', 'OBSERVER', 'OBJECT', 'ORIGIN', 'CTYPE1', 'CTYPE2',
                'CTYPE3', 'CTYPE4', 'CRPIX1', 'CRPIX2', 'CRPIX3', 'CRPIX4',
                'CRVAL1', 'CRVAL2', 'CRVAL3', 'CRVAL4', 'CDELT1', 'CDELT2',
                'CDELT3', 'CDELT4', 'CUNIT1', 'CUNIT2', 'CUNIT3', 'CUNIT4',
                'SPECSYS', 'DATE-OBS']

    new_hdr = {}
    for key in hdr_keys:
        new_hdr[key] = mhdr[key]

    if freq_axis == 3:
        new_hdr["NAXIS3"] = 1
        new_hdr["CRVAL3"] = ref_freq
    elif freq_axis == 4:
        new_hdr["NAXIS4"] = 1
        new_hdr["CRVAL4"] = ref_freq

    new_hdr = fits.Header(new_hdr)

    # save alpha map
    if 'a' in args.output:
        hdu = fits.PrimaryHDU(header=new_hdr)
        hdu.data = alphamap.T[::-1].astype(np.float32)
        name = outfile + 'alpha.fits'
        hdu.writeto(name, overwrite=True)
        print("Wrote alpha map to %s" % name)

    # save I0 map
    if 'i' in args.output:
        hdu = fits.PrimaryHDU(header=new_hdr)
        hdu.data = i0map.T[::-1].astype(np.float32)
        name = outfile + 'I0.fits'
        hdu.writeto(name, overwrite=True)
        print("Wrote I0 map to %s" % name)

    # save clean beam for consistency check
    if 'c' in args.output and not args.dont_convolve:
        hdu = fits.PrimaryHDU(header=new_hdr)
        hdu.data = gausskern.T[::-1].astype(np.float32)
        name = outfile + 'clean-beam.fits'
        hdu.writeto(name, overwrite=True)
        print("Wrote clean beam to %s" % name)

    print("All done here")


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    print("Using %i threads" % args.ncpu)

    main(args)
