-----------------------
Gridding and Degridding
-----------------------

This section contains routines for performing
non-uniform Fourier transforms required to move
between image and visibility space. In particular,
there are gridding and degridding routines where:

1.  Gridding refers to placing complex visibilities
    measured by an interferometer onto a regular grid. 
2.  Degridding refers to taking gridded visibilities
    onto the non-uniformly spaced :math:`(u,v,w)`
    coordinates.

The mapping between image and gridded visibility space
is achieved using the fast Fourier transform (FFT) and
is done separately. Furthermore, since gridding and
degridding is performed using a form of convolutional
interpolation, there is an image space correction which
needs to be performed. This operation simply correspondes
to a point-wise multiplication with some grid-corrector
function (usually refered to as the taper) in image space.


.. _convolution-filter-api:
Filters
-------

Collection of convolution filters and grid-correcting
functions (tapers) for gridding and degridding.

API
+++

.. currentmodule:: africanus.gridding.filters

.. autosummary::
    convolution_filter


.. autofunction:: convolution_filter
.. autodata:: ConvolutionFilter

.. _kaiser-bessel-filter:
Kaiser Bessel
~~~~~~~~~~~~~

The `Kaiser Bessel
<https://www.dsprelated.com/freebooks/sasp/Kaiser_Window.html>`_
function.

.. currentmodule:: africanus.gridding.filters.kaiser_bessel_filter

.. autosummary::
    kaiser_bessel
    kaiser_bessel_with_sinc
    kaiser_bessel_fourier
    estimate_kaiser_bessel_beta


.. autofunction:: kaiser_bessel
.. autofunction:: kaiser_bessel_with_sinc
.. autofunction:: kaiser_bessel_fourier
.. autofunction:: estimate_kaiser_bessel_beta


Sinc
~~~~

The `Sinc <https://en.wikipedia.org/wiki/Sinc_function>`_ function.


Simple
------

Gridding with no correction for the W-term.

Numpy
+++++

.. currentmodule:: africanus.gridding.simple

.. autosummary::
    grid
    degrid

.. autofunction:: grid
.. autofunction:: degrid


Dask
++++

.. currentmodule:: africanus.gridding.simple.dask

.. autosummary::
    grid
    degrid

.. autofunction:: grid
.. autofunction:: degrid

W Stacking
----------

**This is currently experimental**

Implements W-Stacking as described in `WSClean <wsclean_>`_.

.. currentmodule:: africanus.gridding.wstack

.. autosummary::
    w_stacking_layers
    w_stacking_bins
    w_stacking_centroids
    grid
    degrid

.. autofunction:: w_stacking_layers
.. autofunction:: w_stacking_bins
.. autofunction:: w_stacking_centroids
.. autofunction:: grid
.. autofunction:: degrid

.. _wsclean: https://academic.oup.com/mnras/article/444/1/606/1010067

Utilities
---------

.. currentmodule:: africanus.gridding.util

.. autosummary::
    estimate_cell_size

.. autofunction:: estimate_cell_size
