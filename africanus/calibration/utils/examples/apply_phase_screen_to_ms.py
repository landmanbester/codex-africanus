# -*- coding: utf-8 -*-
"""
This example show how to apply gains to an ms in a chunked up way.
The gains should be stored in a .npy file and have the shape
expected by the corrupt_vis function.
It is assumed that the direction axis is ordered in the same way as
model_cols where model_cols is a comma separated string
"""
from time import time as timeit
from numpy.testing import assert_array_almost_equal
import argparse
from africanus.coordinates import radec_to_lm
import Tigger
from dask.diagnostics import ProgressBar
import dask.array as da
from pyrap.tables import table
from daskms import xds_from_ms, xds_to_table
from africanus.dft.dask import im_to_source_vis
from africanus.calibration.phase_only import gauss_newton
from africanus.calibration.utils import chunkify_rows
from africanus.calibration.utils.dask import corrupt_vis
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from africanus.model.coherency.dask import convert


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", help="Name of measurement set", type=str)
    p.add_argument("--sky_model", type=str, help="Tigger lsm file")
    p.add_argument("--utimes_per_chunk",  default=30, type=int,
                   help="Number of unique times in each chunk.")
    p.add_argument("--ncpu", help="The number of threads to use. "
                   "Default of zero means all", default=10, type=int)
    p.add_argument("--alpha_std", type=float, default=0.1, help="Standard deviation "
                   "of alpha values of phase screen.")
    p.add_argument("--sigma", type=float, default=0.1, help="Standard deviation of "
                   "the noise on the visibilities. ")
    p.add_argument("--phase_convention", type=str, default='CODEX', help="The convention "
                   "to use for the signa of the phase delay. Options are "
                   "'CASA' -> positive phase or 'CODEX' -> negative phase.")
    p.add_argument("--fov", type=float, help="Field of view")
    p.add_argument("--unmodelled_size", type=float, default=0.25, help="Size of the unmodelled "
                   "diffuse component. Give FWHM as a fraction of the field of view")
    p.add_argument("--unmodelled_flux_frac", type=float, default=0.1,
                   help="Unmodelled flux as a fraction of the total in the sky model")
    p.add_argument("--unmodelled_freq_range", default=None, nargs='+', type=int,
                   help="Start and end channel that unmodelled source occupies")
    return p

def busy_simple(w, a, b, c, xi):
    """
    w = half width
    a = amplitude
    b = steepness of flanks
    c = depth of trough
    xi = shifted spectral coordinate
    """
    from scipy.special import erf 
    return a*(erf(b*(w**2 - xi**2)) + 1)*(c*xi**2 + 1)/2.0


def make_diffuse(freq, uvw, args, model_flux):
    # set frequency profile
    freqp = freq - freq.min()  # set min to zero
    freqp /= freqp.max()  # set max to one
    freqp -= 0.5  # scale to lie in [-0.5, 0.5]
    chani = args.unmodelled_freq_range[0]
    chanf = args.unmodelled_freq_range[1]
    freqi = freqp[chani]
    freqf = freqp[chanf]
    delta_freq = freqf - freqi
    freq_centre = freqi + delta_freq/2.0
    xi = freqp - freq_centre  # shift
    w = delta_freq/2.0  # half width
    freq_profile = busy_simple(w, 1.0, 100, 50*w, xi)

    # set up Gaussian blob
    uv_max = np.abs(uvw[:, 0:2]).max()
    cell = 0.5/(2*uv_max)  # oversample at twice the Nyquist rate
    npix = int(args.fov/cell)
    if npix%2:
        npix += 1
    x = np.linspace(-args.fov/2.0, args.fov/2.0, npix)
    cell = x[1] - x[0]
    # assert x[-1] == args.fov/2.0
    ll, mm = np.meshgrid(x, x)
    fwhm = args.unmodelled_size * args.fov
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    unmodelled_source = np.exp(-(ll**2 + mm**2)/(2*sigma**2))
    unmodelled_flux = np.sum(unmodelled_source)
    unmodelled_source *= model_flux * args.unmodelled_flux_frac/unmodelled_flux

    # scale by frequency profile
    unmodelled_source = freq_profile[:, None, None] * unmodelled_source[None, :, :]

    # get corresponding visibilities
    n_row = uvw.shape[0]
    n_freq = freq.size
    unmodelled_vis = np.zeros((n_row, n_freq, 4), dtype=np.complex128)
    dirty = np.zeros((n_freq, npix, npix), dtype=np.float64)
    import nifty_gridder as ng
    for v in range(n_freq):
        tmp_vis = ng.dirty2ms(uvw=uvw, freq=freq[v:v+1], dirty=unmodelled_source[v], pixsize_x=cell,
                              pixsize_y=cell, epsilon=1e-7, do_wstacking=True, nthreads=args.ncpu)
        unmodelled_vis[:, v:v+1, 0] = unmodelled_vis[:, v:v+1, -1] = tmp_vis
        dirty[v] = ng.ms2dirty(uvw=uvw, freq=freq[v:v+1], ms=tmp_vis, npix_x=npix, npix_y=npix,
                                   pixsize_x=cell, pixsize_y=cell, epsilon=1e-7, do_wstacking=True,
                                   nthreads=args.ncpu)
    return unmodelled_source, unmodelled_vis, dirty


def make_screen(lm, freq, n_time, n_ant, args):
    n_dir = lm.shape[0]
    n_freq = freq.size
    # create basis matrix for plane [1, l, m]
    n_coeff = 3
    l_coord = lm[:, 0]
    m_coord = lm[:, 1]
    basis = np.hstack((np.ones((n_dir, 1), dtype=np.float64),
                       l_coord[:, None], m_coord[:, None]))
    # get coeffs
    alphas = args.alpha_std * np.random.randn(n_time, n_ant, n_coeff, 2)
    # normalise freqs
    freq_norm = freq/freq.min()
    # simulate phases
    phases = np.zeros((n_time, n_ant, n_freq, n_dir, 2),
                      dtype=np.float64)
    for t in range(n_time):
        for p in range(n_ant):
            for c in range(2):
                # get screen at source locations
                screen = basis.dot(alphas[t, p, :, c])
                # apply frequency scaling
                phases[t, p, :, :, c] = screen[None, :]/freq_norm[:, None]
    return np.exp(1.0j*phases), alphas


def simulate(args):
    # get full time column and compute row chunks
    ms = table(args.ms)
    time = ms.getcol('TIME')
    row_chunks, tbin_idx, tbin_counts = chunkify_rows(
        time, args.utimes_per_chunk)
    # convert to dask arrays
    tbin_idx = da.from_array(tbin_idx, chunks=(args.utimes_per_chunk))
    tbin_counts = da.from_array(tbin_counts, chunks=(args.utimes_per_chunk))
    n_time = tbin_idx.size
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    n_ant = np.maximum(ant1.max(), ant2.max()) + 1
    flag = ms.getcol("FLAG")
    if args.phase_convention == 'CASA':
        uvw = -ms.getcol('UVW').astype(np.float64)
    elif args.phase_convention == 'CODEX':
        uvw = ms.getcol('UVW').astype(np.float64)
    else:
        raise ValueError("Unknown sign convention for phase")
    ms.close()
    n_row, n_freq, n_corr = flag.shape
    if n_corr != 4:
        raise NotImplementedError("Only 4 correlations currently supported")

    # get phase dir
    radec0 = table(args.ms+'::FIELD').getcol('PHASE_DIR').squeeze()

    # get freqs
    freq = table(args.ms+'::SPECTRAL_WINDOW').getcol('CHAN_FREQ')[0].astype(np.float64)
    assert freq.size == n_freq

    # build source model from lsm
    lsm = Tigger.load(args.sky_model)
    n_dir = len(lsm.sources)
    model = np.zeros((n_freq, n_dir, n_corr), dtype=np.float64)
    lm = np.zeros((n_dir, 2), dtype=np.float64)
    source_names = []
    for d, source in enumerate(lsm.sources):
        # extract name
        source_names.append(source.name)
        # extract position
        radec_s = np.array([[source.pos.ra, source.pos.dec]])
        lm[d] =  radec_to_lm(radec_s, radec0)
        # get flux
        if source.flux.I:
            I0 = source.flux.I
            # get spectrum (only spi currently supported)
            tmp_spec = source.spectrum
            spi = [tmp_spec.spi if tmp_spec is not None else 0.0]
            ref_freq = [tmp_spec.freq0 if tmp_spec is not None else 1.0]
            model[:, d, 0] = I0 * (freq/ref_freq)**spi
        if source.flux.Q:
            Q0 = source.flux.Q
            # get spectrum (only spi currently supported)
            tmp_spec = source.spectrum
            spi = [tmp_spec.spi if tmp_spec is not None else 0.0]
            ref_freq = [tmp_spec.freq0 if tmp_spec is not None else 1.0]
            model[:, d, 1] = Q0 * (freq/ref_freq)**spi
        if source.flux.U:
            U0 = source.flux.U
            # get spectrum (only spi currently supported)
            tmp_spec = source.spectrum
            spi = [tmp_spec.spi if tmp_spec is not None else 0.0]
            ref_freq = [tmp_spec.freq0 if tmp_spec is not None else 1.0]
            model[:, d, 2] = U0 * (freq/ref_freq)**spi
        if source.flux.V:
            V0 = source.flux.V
            # get spectrum (only spi currently supported)
            tmp_spec = source.spectrum
            spi = [tmp_spec.spi if tmp_spec is not None else 0.0]
            ref_freq = [tmp_spec.freq0 if tmp_spec is not None else 1.0]
            model[:, d, 3] = V0 * (freq/ref_freq)**spi

    model_fluxes = np.sum(np.abs(model), axis=(1,2))
    model_flux = model_fluxes.max()

    # get unmodelled diffuse component
    if args.unmodelled_flux_frac > 0.0:
        unmodelled_source, unmodelled_vis, dirty = make_diffuse(freq, uvw, args, model_flux)
        unmodelled_vis = da.from_array(unmodelled_vis, chunks=(row_chunks, n_freq, n_corr))
        # save images TODO - save as fits files (need ref header)
        np.savez('images.npz', model=model, unmodelled_source=unmodelled_source, dirty=dirty)

    # get the gains
    jones, alphas = make_screen(lm, freq, n_time, n_ant, args)
    jones = jones.astype(np.complex128)
    jones_shape = jones.shape

    # build dask graph
    freq = da.from_array(freq, chunks=freq.shape)
    lm = da.from_array(lm, chunks=lm.shape)
    model = da.from_array(model, chunks=model.shape)
    jones_da = da.from_array(jones, chunks=(args.utimes_per_chunk,)
                             + jones_shape[1::])

    # append antenna columns
    cols = []
    cols.append('ANTENNA1')
    cols.append('ANTENNA2')
    cols.append('UVW')

    # load data in in chunks and apply gains to each chunk
    xds = xds_from_ms(args.ms, columns=cols, chunks={"row": row_chunks})[0]
    ant1 = xds.ANTENNA1.data
    ant2 = xds.ANTENNA2.data
    if args.phase_convention == 'CASA':
        uvw = -xds.UVW.data.astype(np.float64)
    elif args.phase_convention == 'CODEX':
        uvw = xds.UVW.data.astype(np.float64)
    else:
        raise ValueError("Unknown sign convention for phase")


    # get model visibilities and write to ms
    model_vis = im_to_source_vis(model, uvw, lm, freq, dtype=np.complex128)

    # convert Stokes to corr
    in_schema = ['I', 'Q', 'U', 'V']
    out_schema = [['RR', 'RL'], ['LR', 'LL']]  # TODO - get from ms
    model_vis = convert(model_vis, in_schema, out_schema)

    # apply gains
    data = corrupt_vis(tbin_idx, tbin_counts, ant1, ant2,
                       jones_da, model_vis).reshape((n_row, n_freq, n_corr))

    # assign model visibilities
    out_names = []
    for d in range(n_dir):
        xds = xds.assign(**{source_names[d]: (("row", "chan", "corr"), model_vis[:, :, d, :, :].reshape(n_row, n_freq, n_corr).astype(np.complex64))})
        out_names += [source_names[d]]

    # Assign noise free visibilities to 'CLEAN_DATA'
    xds = xds.assign(**{'CLEAN_DATA': (("row", "chan", "corr"), data.astype(np.complex64))})
    out_names += ['CLEAN_DATA']

    # get noise realisation
    if args.sigma > 0.0:
        noise = (da.random.normal(loc=0.0, scale=args.sigma, size=(n_row, n_freq, n_corr), chunks=(row_chunks, n_freq, n_corr)) + 1.0j*
                 da.random.normal(loc=0.0, scale=args.sigma, size=(n_row, n_freq, n_corr), chunks=(row_chunks, n_freq, n_corr))) /np.sqrt(2.0)
        xds = xds.assign(**{'NOISE': (("row", "chan", "corr"), noise.astype(np.complex64))})
        out_names += ['NOISE']
        noisy_data = data + noise
        xds = xds.assign(**{'DATA': (("row", "chan", "corr"), noisy_data.astype(np.complex64))})
        out_names += ['DATA']

    # add in unmodelled flux
    if args.unmodelled_flux_frac > 0.0:
        # save vis of unmodelled flux
        xds = xds.assign(**{'RESIDUAL': (("row", "chan", "corr"), unmodelled_vis.astype(np.complex64))})
        out_names += ['RESIDUAL']
        data_full = data + unmodelled_vis
        xds = xds.assign(**{'DATA_FULL': (("row", "chan", "corr"), data_full.astype(np.complex64))})
        out_names += ['DATA_FULL']
        if args.sigma > 0.0:
            noisy_data_full = data_full + noise
            xds = xds.assign(**{'NOISY_DATA_FULL': (("row", "chan", "corr"), noisy_data_full.astype(np.complex64))})
            out_names += ['NOISY_DATA_FULL']
            noisy_residual = unmodelled_vis + noise
            xds = xds.assign(**{'NOISY_RESIDUAL': (("row", "chan", "corr"), noisy_residual.astype(np.complex64))})
            out_names += ['NOISY_RESIDUAL']
        
        
    # Create a write to the table
    write = xds_to_table(xds, args.ms, out_names)

    # Submit all graph computations in parallel
    with ProgressBar():
        write.compute()

    return jones, alphas


def calibrate(args, jones, alphas):
    # simple calibration to test if simulation went as expected.
    # Note do not run on large data set

    # load data
    ms = table(args.ms)
    time = ms.getcol('TIME')
    _, tbin_idx, tbin_counts = chunkify_rows(time, args.utimes_per_chunk)
    n_time = tbin_idx.size
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    n_ant = np.maximum(ant1.max(), ant2.max()) + 1
    data = ms.getcol('DATA')  # this is where we put the data
    # we know it is pure Stokes I so we can solve using diagonals only
    data = data[:, :, (0, 3)].astype(np.complex128)
    n_row, n_freq, n_corr = data.shape
    flag = ms.getcol('FLAG')
    flag = flag[:, :, (0, 3)]

    # build source model from lsm
    lsm = Tigger.load(args.sky_model)
    n_dir = len(lsm.sources)
    model = np.zeros((n_row, n_freq, n_dir, 2), dtype=np.complex128)
    for d, source in enumerate(lsm.sources):
        # extract name
        model[:, :, d, :] = ms.getcol(source.name)[:, :, (0,3)]

    # set weights to unity
    weight = np.ones_like(data, dtype=np.float64)

    # initialise gains
    jones0 = np.ones((n_time, n_ant, n_freq, n_dir, n_corr),
                     dtype=np.complex128)

    # calibrate
    ti = timeit()
    jones_hat, jhj, jhr, k = gauss_newton(
        tbin_idx, tbin_counts, ant1, ant2, jones0, data, flag, model,
        weight, tol=1e-4, maxiter=500)
    print("%i iterations took %fs" % (k, timeit() - ti))

    # verify result
    for p in range(2):
        for q in range(p):
            diff_true = np.angle(jones[:, p] * jones[:, q].conj())
            diff_hat = np.angle(jones_hat[:, p] * jones_hat[:, q].conj())
            try:
                assert_array_almost_equal(diff_true, diff_hat, decimal=2)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    args = create_parser().parse_args()

    if args.ncpu:
        from multiprocessing.pool import ThreadPool
        import dask
        dask.config.set(pool=ThreadPool(args.ncpu))
    else:
        import multiprocessing
        args.ncpu = multiprocessing.cpu_count()

    print("Using %i threads" % args.ncpu)

    jones, alphas = simulate(args)

    calibrate(args, jones, alphas)
