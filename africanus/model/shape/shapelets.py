import numba
import numpy as np
from numpy import sqrt, exp
from africanus.constants import c as lightspeed

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
def basis_function(n, xx, beta):
    basis_component = ((2**n) * ((np.pi)**(0.5)) * factorial(n) * beta)**(-0.5)
    exponential_component = hermite(n, xx / beta) * np.exp((-0.5) * (xx**2) * (beta **(-2)))
    return basis_component * exponential_component


#@numba.jit(nogil=True, nopython=True, cache=True)
def shapelet(coords, frequency, coeffs_l, coeffs_m, beta, dtype=np.complex128):
    """
    shapelet: computes the shapelet model image in Fourier space
    Inputs:
        coords: coordinates in (u,v) space with shape (nrow, 3)
        frequency: frequency values with shape (nchan,)
        coeffs_l: shapelet coefficients with shape (nsrc, ncoeffs_l)
        coeffs_m: shapelet coefficients with shape (nsrc, ncoeffs_m)
        beta: characteristic shapelet size with shape (nsrc, 2)
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
                out_shapelets[row, chan, src] = tmp_shapelet
    return out_shapelets