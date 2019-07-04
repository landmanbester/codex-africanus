#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse

import numpy as np
from astropy.io import fits
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from africanus.dft.dask import vis_to_im
from africanus.constants import arcsec2rad
import xarray as xr
from xarrayms import xds_from_ms, xds_from_table

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str)
    p.add_argument("--fitsname", type=str)
    p.add_argument("--row_chunks", default=4000, type=int)
    p.add_argument("--ncpu", default=0, type=int)
    p.add_argument("--colname", default="MODEL_DATA", type=str)
    p.add_argument('--field', default=0, type=int)
    p.add_argument("--cell_size", default=0.0, type=float)
    p.add_argument("--nx", default=0, type=int)
    p.add_argument("--ny", default=0, type=int)
    p.add_argument("--outname", type=str)
    p.add_argument("--use_rows", default=0, type=int)
    p.add_argument("--use_chan", default=0, type=int)
    return p

args = create_parser().parse_args()

if args.ncpu:
    ncpu = args.ncpu
    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(ncpu))
else:
    import multiprocessing
    ncpu = multiprocessing.cpu_count()

print("Using %i threads" % ncpu)

# set image grid
if not args.cell_size:
    # TODO - compute from uv-coverage
    raise ValueError("You need to provide the cell size")
    
if not args.nx:
    raise ValueError("You need to provide the number of x pixels")
if not args.ny:
    raise ValueError("You need to provide the number of y pixels")

# set up image coords
npix = args.nx*args.ny
cell_size_rad = args.cell_size * arcsec2rad
# need to set coordinates differently for odd and even grids
if args.nx%2:
    l = np.arange(-(args.nx//2), args.nx//2+1) * cell_size_rad
else:
    l = np.arange(-(args.nx//2), args.nx//2) * cell_size_rad
if args.ny%2:
    m = np.arange(-(args.ny//2), args.ny//2+1) * cell_size_rad
else:
    m = np.arange(-(args.ny//2), args.ny//2) * cell_size_rad
ll, mm = np.meshgrid(l, m)
lm = np.vstack((ll.flatten(), mm.flatten())).T
lm = da.from_array(lm, chunks=(npix, 2))

# get data
xds = list(xds_from_ms(args.ms, columns=['UVW', args.colname, 'FLAG', 'WEIGHT'], chunks={"row": args.row_chunks}))[0]
vis = getattr(xds, args.colname).data
uvw = xds.UVW.data
flags = xds.FLAG.data
weights = xds.WEIGHT.dat
# Get MS frequencies
spw_ds = list(xds_from_table("::".join((args.ms, "SPECTRAL_WINDOW")),
                             group_cols="__row__"))[0]
ms_freqs = spw_ds.CHAN_FREQ.data
nchan = ms_freqs.size

if args.use_rows:
    use_rows = args.use_rows
else:
    use_rows = uvw.shape[0]

if args.use_chan:
    use_chan = args.use_chan
else:
    use_chan = nchan

use_vis = vis[0:use_rows, 0:use_chan]
use_uvw = uvw[0:use_rows]
use_flags = flags[0:use_rows, 0:use_chan]
if use_chan>1:
    weights = weights[0:use_rows, None, :].compute()
    weights = np.tile(weights, (1, use_chan, 1))
    use_weights = da.from_array(weights, chunks=(args.row_chunks, use_chan, 2))
else:
    use_weights = weights[0:use_rows, None, :]


psf_graph = vis_to_im(use_weights, use_uvw, lm, ms_freqs[0:use_chan], use_flags)

dirty_graph = vis_to_im(use_weights * use_vis, use_uvw, lm, ms_freqs[0:use_chan], use_flags)

# Submit all graph computations in parallel
with ProgressBar():
    psf = psf_graph.compute()
    
with ProgressBar():    
    dirty = dirty_graph.compute()

psf = np.transpose(psf, (1, 2, 0)).reshape(use_chan, 2, args.nx, args.ny)
dirty = np.transpose(dirty, (1, 2, 0)).reshape(use_chan, 2, args.nx, args.ny)

# make the MFS image
wsums = psf[:, 0, args.nx//2, args.ny//2]
wsum = np.sum(wsums)
w = wsums/wsum  # normalised weights
# Get in Jy/beam units
dirty /= wsums[:, None, None, None]
# combine with weighted sum
mfs_dirty = np.sum(dirty * w[:, None, None, None], axis=0, keepdims=True)

StokesI = (mfs_dirty[0, 0] + mfs_dirty[0, 1])/2.0
StokesV = (mfs_dirty[0, 0] - mfs_dirty[0, 1])/2.0


import matplotlib.pyplot as plt
plt.figure('psf')
plt.imshow(psf[0, 0])
plt.colorbar()
plt.figure('I')
plt.imshow(StokesI)
plt.colorbar()
plt.figure('V')
plt.imshow(StokesV)
plt.colorbar()
plt.show()



# load in fits file (mainly for the header)
data = fits.getdata(args.fitsname)
hdr = fits.getheader(args.fitsname)
hdu = fits.PrimaryHDU(header=hdr)
hdu.data = StokesI.T[::-1]
hdu.writeto(args.outname, overwrite=True)
hdu._close()