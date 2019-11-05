# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from africanus.calibration.utils.dask import corrupt_vis, residual_vis
from africanus.calibration.phase_only.dask import compute_jhj, compute_jhr
from africanus.calibration.utils import chunkify_rows
from africanus.linalg import kronecker_tools as kt
import xarray as xr
from xarrayms import xds_from_ms, xds_to_table
import dask.array as da
from pyrap.tables import table
import argparse
import nifty5 as ift


class GradientWrapper(ift.EnergyOperator):
    def __init__(self, func, domain):
        self._domain = ift.makeDomain(domain)
        self._capability = self.ADJOINT_TIMES
        self._func = func

    def apply(self, x, mode):
        self.check_mode(x, mode)
        return ift.from_global_data(self._func(x.to_global_data()))


class AplAdjAplLOp(ift.LinearOperator):
    def __init__(self, apply, apply_adjoint, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._a, self._aadj = apply, apply_adjoint
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_mode(x, mode)
        f = self._a if mode == self.TIMES else self._aadj
        return ift.from_global_data(self._tgt(mode), f(x.to_global_data()))


class phase_energy(ift.EnergyOperator):
    def __init__(self, ms, data_col, model_cols, out_col, utimes_per_chunk, cov_mat_file=None, ncpu=0, field=0, dtype=np.complex64):
        # get unique time and indices
        self.ms_name = ms
        ms = table(self.ms_name)
        time = ms.getcol('TIME')
        ant1 = ms.getcol('ANTENNA1')
        ant2 = ms.getcol('ANTENNA2')
        self.n_ant = int(np.maximum(ant1.max(), ant2.max()) + 1)
        ms.close()
        self.row_chunks, tbin_idx, tbin_counts = chunkify_rows(
            time, utimes_per_chunk)
        self.time = np.unique(time)
        self.time -= self.time.min()
        self.time /= self.time.max()
        self.time = np.ascontiguousarray(self.time)
        self.n_time = self.time.size
        # convert to dask arrays
        self.tbin_idx = da.from_array(tbin_idx, chunks=(utimes_per_chunk))
        self.tbin_counts = da.from_array(
            tbin_counts, chunks=(utimes_per_chunk))

        if ncpu:
            from multiprocessing.pool import ThreadPool
            import dask
            dask.config.set(pool=ThreadPool(ncpu))
        else:
            import multiprocessing
            ncpu = multiprocessing.cpu_count()

        self.data_col = data_col

        # get frequencies
        spw = table(self.ms_name + '::SPECTRAL_WINDOW')
        self.freq = spw.getcol('CHAN_FREQ').squeeze()
        spw.close()
        self.freq -= self.freq.min()
        self.freq /= self.freq.max()
        self.freq = np.ascontiguousarray(self.freq)
        self.n_freq = self.freq.size

        # get model column names
        self.model_cols = model_cols.split(',')
        self.n_dir = len(self.model_cols)

        # set jones
        self.n_corr = 1

        # append antenna columns
        self.cols = []
        self.cols.append('ANTENNA1')
        self.cols.append('ANTENNA2')
        self.cols.append('FLAG')
        self.cols.append(data_col)
        for col in self.model_cols:
            self.cols.append(col)

        self._domain = ift.makeDomain(ift.UnstructuredDomain(
            (self.n_time, self.n_ant, self.n_freq, self.n_dir, self.n_corr)))

        xds = xds_from_ms(self.ms_name, columns=self.cols,
                          chunks={"row": self.row_chunks})[0]
        ant1 = xds.ANTENNA1.data
        ant2 = xds.ANTENNA2.data
        flag = xds.FLAG.data
        model = []
        for col in self.model_cols:
            model.append(getattr(xds, col).data)
        model = da.stack(model, axis=2).rechunk({2: 3})
        model = model[:, :, :, ]
        jones = da.ones((self.n_time, self.n_ant, self.n_freq, self.n_dir, self.n_corr), chunks=(
            utimes_per_chunk, -1, -1, -1, -1), dtype=dtype)

        print('Computing jhj')
        self._jhj = compute_jhj(
            self.tbin_idx, self.tbin_counts, ant1, ant2, jones, model, flag).compute()
        print("Done")

        if cov_mat_file is not None:
            cov_mat = np.load(cov_mat_file, allow_pickle=True)
            L = kt.kron_cholesky(cov_mat)
            res1 = np.zeros((self.n_time, self.n_ant, self.n_freq,
                             self.n_dir, self.n_corr), dtype=dtype)

            def A_impl(x):
                for p in range(self.n_ant):
                    for c in range(self.n_corr):
                        res1[:, p, :, :, c] = kt.kron_matvec(L, x[:, p, :, :, c]).reshape(
                            self.n_time, self.n_freq, self.n_dir)
                return res1
            LH = kt.kron_transpose(L)
            res2 = np.zeros((self.n_time, self.n_ant, self.n_freq,
                             self.n_dir, self.n_corr), dtype=dtype)

            def AH_impl(x):
                res2 = np.zeros_like(x, dtype=x.dtype)
                for p in range(self.n_ant):
                    for c in range(self.n_corr):
                        res2[:, p, :, :, c] = kt.kron_matvec(LH, x[:, p, :, :, c]).reshape(
                            self.n_time, self.n_freq, self.n_dir)
                return res2
            self.A = AplAdjAplLOp(A_impl, AH_impl, self.domain, self.domain)
            self.A_impl = A_impl
            self.AH = AH_impl
        else:
            raise NotImplementedError

    def make_gradient(self):
        xds = xds_from_ms(self.ms_name, columns=self.cols,
                          chunks={"row": self.row_chunks})[0]
        vis = getattr(xds, self.data_col).data
        ant1 = xds.ANTENNA1.data
        ant2 = xds.ANTENNA2.data
        flag = xds.FLAG.data

        model = []
        for col in self.model_cols:
            model.append(getattr(xds, col).data)
        model = da.stack(model, axis=2).rechunk({2: 3})

        resid = ...

        def energy():

        def gradient():

        return lambda x: self.AH(compute_jhr(self.tbin_idx, self.tbin_counts, ant1, ant2, x,
                                             residual_vis(
                                                 self.tbin_idx, self.tbin_counts, ant1, ant2, x, vis, flag, model),
                                             model, flag).compute())

    def _value(self):
        raise NotImplementedError

    def apply(self, x):
        self._check_input(x)
        inp = x.to_global_data()

        energy_value, gradient = daskmagic(
            inp, compute_gradient=isinstance(x, Linearization))
        y = ift.from_global_data(self.target, float(energy_value))
        if not isinstance(x, Linearization):
            return y
        jac = ift.VdotOperator(ift.from_global_data(self.domain,
                                                    gradient)).adjoint
        lin = x.new(y, jac)
        if x.want_metric:
            return y.add_metric(ift.SandwichOperator(cheese=self._jhj,
                                                     bun=self.A))
        return y
