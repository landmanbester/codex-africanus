#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect all utility functions for deconvolution (eg. peak finding, overlap indices etc.) here.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numba
import numpy as np

# LB - how to write a parallel dask version?
@numba.jit(nopython=True, nogil=True, cache=True)
def find_peak(residuals):
    abs_residuals = residuals
    min_peak = abs_residuals.min()
    max_peak = abs_residuals.max()

    nx, ny = abs_residuals.shape

    minx, miny = -1, -1
    maxx, maxy = -1, -1
    peak_intensity = -1

    for x in range(nx):
        for y in range(ny):
            intensity = abs_residuals[x, y]

            if intensity == min_peak:
                minx = x
                miny = y

            if intensity == max_peak:
                maxx = x
                maxy = y
                peak_intensity = intensity

    if minx == -1 or miny == -1:
        raise ValueError("Minimum peak not found")

    if maxx == -1 or maxy == -1:
        raise ValueError("Maximum peak not found")

    return maxx, maxy, minx, miny, peak_intensity

@numba.jit(nopython=True, nogil=True, cache=True)
def find_overlap_indices(x0, y0, nfull, nbox):
    """
    Routine to return the overlap indices of a 2D box of size Nbox x Nbox with an image of size Nfull x Nfull 
    where the box is centered at x0, y0 in coordinates of the image. If the box has an even
    number of pixels, we use the convention that its center is defined at Nbox//2 + 1.
    :param x0: x-coordinate center of component
    :param y0: y-coordinate center of component
    :param nfull: size of image (assuming square)
    :param nbox: size of box (assuming square)
    :return: 
    """
    # determine if box size is even or odd
    IsEven = nbox % 2 == 0
    half_nbox_low = nbox//2
    half_nbox_high = half_nbox_low if IsEven else half_nbox_low + 1

    # get extent of box in image coordinates for x-direction
    xlow = np.maximum(0, x0 - half_nbox_low)
    xhigh = np.minimum(nfull, x0 + half_nbox_high)

    # get x-extent of box overlapping image
    xlowbox = xlow - (x0 - half_nbox_low)
    xhighbox = nbox - (x0 + half_nbox_high - xhigh)

    # get the extent of box in in image coordinates for the y-direction
    ylow = np.maximum(0, y0 - half_nbox_low)
    yhigh = np.minimum(nfull, y0 + half_nbox_high)

    # get y-extent of box overlapping image
    ylowbox = ylow - (y0 - half_nbox_low)
    yhighbox = nbox - (y0 + half_nbox_high - yhigh)

    return (xlow, xhigh, ylow, yhigh), (xlowbox, xhighbox, ylowbox, yhighbox)


# if __name__=="__main__":
#     # Test find_overlap_indices
#     nfull = 101
#     nbox = 203
#     x0 = 100
#     y0 = 100
#
#     Aedge, Bedge = find_overlap_indices(x0, y0, nfull, nbox)
#
#     print("Aedge = ", Aedge, Aedge[1] - Aedge[0], Aedge[3] - Aedge[2])
#     print("Bedge = ", Bedge, Bedge[1] - Bedge[0], Bedge[3] - Bedge[2])
