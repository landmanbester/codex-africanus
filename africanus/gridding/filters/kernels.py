# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from scipy.special import jn as bessel1


def kaiser_bessel(nu, half_support, beta=2.34):
    """
    Compute a 1D Kaiser Bessel (KB) filter

    nu : filter position
    beta : KB param
    """
    alpha = beta * half_support * 2
    z = nu/half_support
    return np.i0(alpha*np.sqrt(1 - z**2))/np.i0(alpha)

def kaiser_bessel_corrector(half_support, npix, grid_size, beta=2.34):
    J = 2*half_support
    alpha = beta * J
    n = np.arange(-0.5, npix-0.5)
    eta0 = (npix-1)/2.0
    u = (n - eta0)/grid_size
    z = np.lib.scimath.sqrt((np.pi*J*u)**2 - alpha**2)
    Lambda = np.sqrt(2.0/z)*bessel1(0.5, z)
    return (np.sqrt(np.pi)*J*Lambda/(2.0*np.i0(alpha))).real

def exponential_semicircle(nu, half_support, beta):
    """
    Compute the exponential of a semi-circle (ES) filter
    from

    https://arxiv.org/pdf/1808.06736.pdf

    nu : filter position
    beta : ES param
    """
    z = nu/half_support
    return np.exp(beta * (np.sqrt(1 - z**2) -1))

def prolate_spheroidal(nu, half_support):
    """
    Compute prolate spheroidal (PS) filter using

    scipy.special.pro_ang1

    nu : filter position 
    """
    z = nu/half_support
    return pro_ang1(0, 0, R*np.pi, z)[0]

def prolate_spheroidal_rational(nu, half_support):

