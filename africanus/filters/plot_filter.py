#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

from ..filters import convolution_filter
from ..util.cmdline import parse_python_assigns
from ..util.requirements import requires_optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D  # noqa
except ImportError:
    pass


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("filter", choices=['kaiser-bessel', 'sinc'],
                   default='kaiser-bessel')
    p.add_argument("-hs", "--half-support", default=3, type=int)
    p.add_argument("-os", "--oversample", default=63, type=int)
    p.add_argument("-nn", "--no-normalise", action="store_false",
                   help="Don't normalise area under the filter to 1")
    p.add_argument("-k", "--kwargs", default="", type=parse_python_assigns,
                   help="Extra keywords arguments used to create the filter. "
                        "For example 'beta=2.3' to specify a beta shape "
                        "parameter for the Kaiser Bessel")

    return p


@requires_optional('matplotlib.pyplot', 'mpl_toolkits.mplot3d')
def _plot_filter(args):
    logging.info("Creating %s filter with half support of %d "
                 "and %d oversampling" %
                 (args.filter, args.half_support, args.oversample))

    if len(args.kwargs) > 0:
        logging.info("Extra keywords %s" % args.kwargs)

    conv_filter = convolution_filter(args.half_support,
                                     args.oversample,
                                     args.filter,
                                     normalise=not args.no_normalise,
                                     **args.kwargs)

    data = np.abs(conv_filter.filter_taps)

    X, Y = np.mgrid[-1:1:1j*data.shape[0], -1:1:1j*data.shape[1]]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, data, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    zmax = data.max()

    ax.plot([0, 0], [0, 0], zs=[0, 1.2*zmax], color='black')
    ax.plot([0, 0], [-1, 1], zs=[zmax, zmax], color='black')
    ax.plot([-1, 1], [0, 0], zs=[zmax, zmax], color='black')

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _plot_filter(create_parser().parse_args())
