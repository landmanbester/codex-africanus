#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from africanus.calibration.utils.dask import corrupt_vis
from africanus.calibration.utils import chunkify_rows
import xarray as xr
from xarrayms import xds_from_ms, xds_to_table
from pyrap.tables import table
import dask.array as da
from dask.diagnostics import ProgressBar
from energy import phase_energy

import argparse


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str)
    p.add_argument("--model_cols", default='MODEL_DATA', type=str)
    p.add_argument("--data_col", default='DATA', type=str)
    p.add_argument("--out_col", default='CORRECTED_DATA', type=str)
    p.add_argument("--gain_file", type=str)
    p.add_argument("--cov_file", type=str)
    p.add_argument("--utimes_per_chunk", default=32, type=int)
    p.add_argument("--ncpu", default=0, type=int)
    p.add_argument('--field', default=0, type=int)
    return p


args = create_parser().parse_args()

if args.ncpu:
    ncpu = args.ncpu
    from multiprocessing.pool import ThreadPool
    import dask
    dask.config.set(pool=ThreadPool(ncpu))
else:
    import multiprocessing
    ncpu = multiprocessing.cpu_count()

print("Using %i threads" % ncpu)


from africanus.calibration.phase_only import compute_jhr, compute_jhr_new
from africanus.calibration.utils import residual_vis
ms = table(args.ms)
time = ms.getcol('TIME')
_, tbin_idx, tbin_counts = chunkify_rows(time, 90)

vis = ms.getcol('DATA')
model1 = ms.getcol('MODEL_DATA1')
model2 = ms.getcol('MODEL_DATA2')
model3 = ms.getcol('MODEL_DATA3')
model = np.stack([model1, model2, model3], axis=2)
print("Mod = ", model.shape)
flag = ms.getcol('FLAG')
gains_true = np.load(args.gain_file)

residual = residual_vis(time_bin_indices, time_bin_counts, antenna1,
                        antenna2, gains_true, vis, flag, model)


# H = phase_energy(args.ms, args.data_col, args.model_cols, args.out_col, args.utimes_per_chunk, cov_mat_file=args.cov_file)


# test_jhr = H.make_gradient()
# res_info = test_jhr(gains_true)


# import time
# for i in range(10):
#     t1 = time.time()
#     test2 = H.AH(H._jhj * H.A_impl(gains_true))
#     print(i, time.time() - t1)
