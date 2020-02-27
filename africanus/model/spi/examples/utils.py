import numpy as np
from astropy.io import fits

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

def load_fits_contiguous(name):
    arr = fits.getdata(name)
    if arr.ndim == 4: 
        # figure out which axes to keep from header
        hdr = fits.getheader(name)
        if hdr["CTYPE4"].lower() == 'stokes':
            arr = arr[0]
        else:
            arr = arr[:, 0]
    # reverse last spatial axis then transpose (f -> c contiguous)
    arr = np.transpose(arr[:, :, ::-1], axes=(0, 2, 1))
    return np.ascontiguousarray(arr, dtype=np.float64)

def set_header_info(mhdr, ref_freq, freq_axis, args, beampars):
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

    new_hdr['BMAJ'] = beampars[0]
    new_hdr['BMIN'] = beampars[1]
    new_hdr['BPA'] = np.rad2deg(beampars[2])

    new_hdr = fits.Header(new_hdr)

    return new_hdr

def get_fits_freq_space_info(hdr):
    if hdr['CUNIT1'].lower() != "deg":
        raise ValueError("Image coordinates must be in degrees")
    npix_l = hdr['NAXIS1']
    refpix_l = hdr['CRPIX1']
    delta_l = hdr['CDELT1']
    l_coord = np.arange(1 - refpix_l, 1 + npix_l - refpix_l)*delta_l

    if hdr['CUNIT2'].lower() != "deg":
        raise ValueError("Image coordinates must be in degrees")
    npix_m = hdr['NAXIS2']
    refpix_m = hdr['CRPIX2']
    delta_m = hdr['CDELT2']
    m_coord = np.arange(1 - refpix_m, 1 + npix_m - refpix_m)*delta_m

    # get frequencies
    if hdr["CTYPE4"].lower() == 'freq':
        freq_axis = 4
        nband = hdr['NAXIS4']
        refpix_nu = hdr['CRPIX4']
        delta_nu = hdr['CDELT4']  # assumes units are Hz
        ref_freq = hdr['CRVAL4']
        ncorr = hdr['NAXIS3']
    elif hdr["CTYPE3"].lower() == 'freq':
        freq_axis = 3
        nband = hdr['NAXIS3']
        refpix_nu = hdr['CRPIX3']
        delta_nu = hdr['CDELT3']  # assumes units are Hz
        ref_freq = hdr['CRVAL3']
        ncorr = hdr['NAXIS4']
    else:
        raise ValueError("Freq axis must be 3rd or 4th")

    if ncorr > 1:
        raise ValueError("Only Stokes I cubes supported")

    freqs = ref_freq + np.arange(1 - refpix_nu,
                                 1 + nband - refpix_nu) * delta_nu

    return l_coord, m_coord, freqs, ref_freq, freq_axis