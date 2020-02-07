#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

import argparse
import dask
import dask.array as da
import numpy as np
from africanus.util.numba import jit
from astropy.io import fits
from pyrap.tables import table
import warnings
from africanus.model.spi.dask import fit_spi_components
from africanus.rime import parallactic_angles
from daskms import xds_from_ms
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

@jit(nopython=True, nogil=True, cache=True)
def _unflagged_counts(flags, time_idx, out):
    for i in range(time_idx.size):
            ilow = time_idx[i]
            ihigh = time_idx[i+1]
            out[i] = np.sum(~flags[ilow:ihigh])
    return out

def extract_dde_info(args, freqs):
    """
    Computes paramactic angles, antenna scaling and pointing information
    required for beam interpolation. 
    """
    # get ms info required to compute paralactic angles and weighted sum
    nband = freqs.size
    if args.ms is not None:
        utimes = []
        unflag_counts = []
        ant_pos = None
        phase_dir = None
        for ms_name in args.ms:
            # get antenna positions
            ant = table(ms_name + '::ANTENNA')
            if ant_pos is None:
                ant_pos = ant.getcol('POSITION')
            else: # check all are the same
                tmp = ant.getcol('POSITION')
                if not np.array_equal(ant_pos, tmp):
                    raise ValueError("Antenna positions not the same across measurement sets")
            
            # get phase center for field
            field = table(ms_name + '::FIELD')
            if phase_dir is None:
                phase_dir = field.getcol('PHASE_DIR')[args.field].squeeze()
            else:
                tmp = field.getcol('PHASE_DIR')[args.field].squeeze()
                if not np.array_equal(phase_dir, tmp):
                    raise ValueError('Phase direction not teh same across measurement sets')

            # get unique times and count flags
            xds = xds_from_ms(ms_name, columns=["TIME", "FLAG_ROW"], group_cols=["FIELD_ID"])[args.field]
            utime, time_idx = np.unique(xds.TIME.data.compute(), return_index=True)
            ntime = utime.size
            utimes.append(utime)
        
            flags = xds.FLAG_ROW.data.compute()
            unflag_count = _unflagged_counts(flags.astype(np.int32), time_idx, np.zeros(ntime, dtype=np.int32))
            unflag_counts.append(unflag_count)

        utimes = np.concatenate(utimes)
        unflag_counts = np.concatenate(unflag_counts)
        ntimes = utimes.size
        
        # compute paralactic angles
        parangles = parallactic_angles(utimes, ant_pos, phase_dir)

        # mean over antanna nant -> 1
        parangles = np.mean(parangles, axis=1, keepdims=True)
        nant = 1

        # beam_cube_dde requirements
        ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
        point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)

        return (da.from_array(parangles, chunks=(ntimes//args.ncpu, nant)),
                da.from_array(ant_scale, chunks=ant_scale.shape),
                da.from_array(point_errs, chunks=point_errs.shape),
                unflag_counts,
                True)
    else:
        ntimes = 1
        nant = 1
        parangles = np.zeros((ntimes, nant,), dtype=np.float64)    
        ant_scale = np.ones((nant, nband, 2), dtype=np.float64)
        point_errs = np.zeros((ntimes, nant, nband, 2), dtype=np.float64)
        unflag_counts = np.array([1])
        
        return (parangles, ant_scale, point_errs, unflag_counts, False)


def make_power_beam(args, lm_source, freqs, use_dask):
    print("Loading fits beam patterns from %s" % args.beammodel)
    from glob import glob
    paths = glob(args.beammodel + '**_**.fits')
    beam_hdr = None
    if args.corr_type == 'linear':
        corr1 = 'XX'
        corr2 = 'YY'
    elif args.corr_type == 'circular':
        corr1 = 'LL'
        corr2 = 'RR'
    else:
        raise KeyError("Unknown corr_type supplied. Only 'linear' or 'circular' supported")

    for path in paths:
        if corr1.lower() in path[-10::]:
            if 're' in path[-7::]:
                corr1_re = load_fits_contiguous(path)
                if beam_hdr is None:
                    beam_hdr = fits.getheader(path)
            elif 'im' in path[-7::]:
                corr1_im = load_fits_contiguous(path)
            else:
                raise NotImplementedError("Only re/im patterns supported")
        elif corr2.lower() in path[-10::]:
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

    if use_dask:
        return (da.from_array(beam_amp, chunks=beam_amp.shape),
                da.from_array(beam_extents, chunks=beam_extents.shape), 
                da.from_array(bfreqs, bfreqs.shape))
    else:
        return beam_amp, beam_extents, bfreqs

def interpolate_beam(ll, mm, freqs, args):
    """
    Interpolate beam to image coordinates and optionally compute average
    over time if MS is provoded
    """

    print("Interpolating beam")
    parangles, ant_scale, point_errs, unflag_counts, use_dask = extract_dde_info(args, freqs)

    lm_source = np.vstack((ll.ravel(), mm.ravel())).T
    beam_amp, beam_extents, bfreqs = make_power_beam(args, lm_source, freqs, use_dask)

    # interpolate beam
    if use_dask:
        from africanus.rime.dask import beam_cube_dde
        lm_source = da.from_array(lm_source, chunks=lm_source.shape)
        freqs = da.from_array(freqs, chunks=freqs.shape)
        beam_image = beam_cube_dde(beam_amp, beam_extents, bfreqs,
                                    lm_source, parangles, point_errs,
                                    ant_scale, freqs).compute().squeeze()
        # average over time
        beam_image = (np.sum(beam_image * unflag_counts[None, :, None], axis=1)/np.sum(unflag_counts)).squeeze()

    else:
        from africanus.rime.fast_beam_cubes import beam_cube_dde
        beam_image = beam_cube_dde(beam_amp, beam_extents, bfreqs,
                                    lm_source, parangles, point_errs,
                                    ant_scale, freqs).squeeze()
    
    

    # reshape to image shape
    beam_source = np.transpose(beam_image, axes=(1, 0))
    return beam_source.squeeze().reshape((freqs.size, *ll.shape))


def create_parser():
    p = argparse.ArgumentParser(description='Simple spectral index fitting'
                                            'tool.',
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fitsmodel", type=str, required=True)
    p.add_argument("--fitsresidual", type=str)
    p.add_argument("--ms", nargs="+", type=str, 
                   help="Mesurement sets used to make the image. \n"
                   "Used to get paralactic angles if doing primary beam correction")
    p.add_argument("--field", type=int, default=0,
                   help="Field ID")
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
    p.add_argument('--output', default='aeikIbc', type=str,
                   help="Outputs to write. Letter correspond to: \n"
                   "a - alpha map \n"
                   "e - alpha error map \n"
                   "i - I0 map \n"
                   "k - I0 error map \n"
                   "I - reconstructed cube form alpha and I0 \n"
                   "b - interpolated beam \n"
                   "c - restoring beam used for convolution \n"
                   "Default is to write all of them")
    p.add_argument("--padding_frac", default=0.2, type=float,
                   help="Padding factor for FFT's.")
    p.add_argument("--dont_convolve", action="store_true",
                   help="Passing this flag bypasses the convolution "
                   "by the clean beam")
    p.add_argument("--circularise_beam", action="store_true",
                   help="Passing this flag will convolve with a circularised "
                   "beam instead of an elliptical one")
    p.add_argument("--corr_type", type=str, default='linear',
                   help='Linear or circular feeds')
    p.add_argument("--channel_weights", default=None, nargs='+', type=float,
                   help="Per-channel weights to use during fit to frqequency axis. \n "
                   "Only has an effect if no residual is passed in (for now).")
    return p


def load_fits_contiguous(name):
    arr = fits.getdata(name).squeeze()
    # transpose spatial axes (f -> c contiguous)
    arr = np.transpose(arr, axes=(0, 2, 1))[:, ::-1]
    return np.ascontiguousarray(arr, dtype=np.float64)

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
        
    if args.circularise_beam:
        emaj = emin = (beampars[0] + beampars[1])/2.0
    print("Using emaj = %3.2e, emin = %3.2e, PA = %3.2e" % beampars)

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

    xx, yy = np.meshgrid(l_coord, m_coord, indexing='ij')

    if not args.dont_convolve:
        # get the Gaussian convolution kernel
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
        if args.channel_weights is not None:
            weights = np.array(args.channel_weights)
            print("Using provided channel weights")
        else:
            print("No residual or channel weights provided. Using equal weights.")
            weights = np.ones(nband, dtype=np.float64)

    ncomps, _ = fitcube.shape
    fitcube = da.from_array(fitcube.astype(np.float64),
                            chunks=(ncomps//args.ncpu, nband))
    weights = da.from_array(weights.astype(np.float64), chunks=(nband))
    freqsdask = da.from_array(freqs.astype(np.float64), chunks=(nband))

    print("Fitting %i components" % ncomps)
    alpha, alpha_err, Iref, i0_err = fit_spi_components(fitcube, weights, freqsdask,
                                           np.float64(ref_freq)).compute()
    print("Done. Writing output.")

    alphamap = np.zeros(model[0].shape, dtype=model.dtype)
    alpha_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    i0map = np.zeros(model[0].shape, dtype=model.dtype)
    i0_err_map = np.zeros(model[0].shape, dtype=model.dtype)
    alphamap[maskindices[:, 0], maskindices[:, 1]] = alpha
    alpha_err_map[maskindices[:, 0], maskindices[:, 1]] = alpha_err
    i0map[maskindices[:, 0], maskindices[:, 1]] = Iref
    i0_err_map[maskindices[:, 0], maskindices[:, 1]] = i0_err

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
        print("Wrote alpha map to %s" % name)

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
        print("Wrote I0 map to %s" % name)

    # save clean beam for consistency check
    if 'c' in args.output and not args.dont_convolve:
        hdu = fits.PrimaryHDU(header=new_hdr)
        hdu.data = gausskern.T[:, ::-1].astype(np.float32)
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

    GD = vars(args)
    print('Input Options:')
    for key in GD.keys():
        print(key, ' = ', GD[key])

    print("Using %i threads" % args.ncpu)

    main(args)
