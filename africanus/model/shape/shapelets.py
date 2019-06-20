import numba
import numpy as np
from numpy import sqrt, exp
from africanus.constants import c as lightspeed
from africanus.constants import minus_two_pi_over_c

e = 2.7182818284590452353602874713527
square_root_of_pi = 1.77245385091

#@numba.jit(nogil=True, nopython=True, cache=True)
def hermite(n, x):
    if n==0:
        return 1
    elif n==1:
        return 2*x
    else:
        return 2*x*hermite(x,n-1)-2*(n-1)*hermite(x,n-2)


#@numba.jit(nogil=True, nopython=True, cache=True)
def factorial(n):
    if n <= 1:
        return 1
    ans = 1
    for i in range(1, n):
        ans *= i
    return ans * n

#@numba.jit(nogil=True, nopython=True, cache=True)
def basis_function(n, xx, beta, fourier=False, delta_x=None):
    if fourier:
        x = 2*np.pi*xx
        scale = 1.0/beta
    else:
        x = xx
        scale = beta
    basis_component = 1.0/np.sqrt(2.0**n * np.sqrt(np.pi) * factorial(n) * scale)
    exponential_component = hermite(n, x / scale) * np.exp(-x**2 / (2.0*scale**2))
    if fourier:
        return 1.0j**n * basis_component * exponential_component * np.sqrt(2*np.pi)/delta_x
    else:
        return basis_component * exponential_component

def phase_steer_and_w_correct(uvw, lm_source_center, frequency):
    l0, m0 = lm_source_center
    n0 = np.sqrt(1.0-l0**2-m0**2)
    u, v, w = uvw
    real_phase = minus_two_pi_over_c * frequency * (u*l0 + v*m0 + w*(n0-1))
    return np.exp(1.0j*real_phase)


#@numba.jit(nogil=True, nopython=True, cache=True)
def shapelet(coords, frequency, coeffs_l, coeffs_m, beta, delta_l, delta_m, lm, dtype=np.complex128):
    """
    shapelet: computes the shapelet model image in Fourier space
    Inputs:
        coords: coordinates in (u,v) space with shape (nrow, 3)
        frequency: frequency values with shape (nchan,)
        coeffs_l: shapelet coefficients with shape (nsrc, ncoeffs_l)
        coeffs_m: shapelet coefficients with shape (nsrc, ncoeffs_m)
        beta: characteristic shapelet size with shape (nsrc, 2)
        delta_l: pixel size in l dim
        delta_m: pixel size in m dim
        lm:source center coordinates of shape (nsource, 2)
    Returns:
        out_shapelets: Shapelet with shape (nsrc, nrow, nchan)
    """
    nrow, _ = coords.shape
    nsrc, nmax1 = coeffs_l.shape
    nmax2 = coeffs_m.shape[1]
    nchan = frequency.size
    out_shapelets = np.empty((nrow, nchan, nsrc), dtype=dtype)
    for row in range(nrow):
        u, v, w = coords[row, :]
        for chan in range(nchan):
            fu = u * 2 * np.pi * frequency[chan]/lightspeed
            fv = v * 2 * np.pi * frequency[chan]/lightspeed
            for src in range(nsrc):
                beta_u, beta_v = beta[src, :] ** (-1)
                tmp_shapelet = np.zeros(1, dtype=dtype)
                for n1 in range(nmax1):
                    for n2 in range(nmax2):
                        real_part = coeffs_l[src, n1] * basis_function(n1, fu, beta_u) \
                                    * coeffs_m[src, n2] * basis_function(n2, fv, beta_v)
                        complex_part = 1.0j**(n1 + n2)
                        tmp_shapelet += complex_part * real_part
                l, m = lm[src]
                wterm = phase_steer_and_w_correct((u,v,w), (l, m), frequency[chan])
                out_shapelets[row, chan, src] = tmp_shapelet * wterm
    return out_shapelets


def shapelet_1d(u, coeffs, fourier, delta_x=None, beta=1.0):
    """
    The one dimensional shapelet. Default is to return the
    dimensionless version. 

    Parameters
    ----------
    u : :class:`numpy.ndarray`
        Array of coordinates at which to evaluate the shapelet
        of shape (nrow)
    coeffs : :class:`numpy.ndarray`
        Array of shapelet coefficients of chape (ncoeff)
    fourier : bool
        Whether to evaluate the shapelet in Fourier space
        or in signal space
    beta : float, optional
        The scale parameter for the shapelet. If fourier is
        true the scale is 1/beta

    Returns
    -------
    out : :class:`numpy.ndarray`
        The shapelet evaluated at u of shape (nrow)
    """
    nrow = u.size
    if fourier:
        if delta_x is None:
            raise(ValueError, "You have to pass in a value for delta_x in Fourier mode")
        out = np.zeros(nrow, dtype=np.complex128)
    else:
        out = np.zeros(nrow, dtype=np.float64)
    for row, ui in enumerate(u):
        for n, c in enumerate(coeffs):
            out[row] += c * basis_function(n, ui, beta, fourier=fourier, delta_x=delta_x)
    return out 

        