# -*- coding: utf-8 -*-


import numpy as np
from numpy.testing import assert_array_almost_equal
import nifty_gridder as ng
from pyrap.tables import table
import argparse
from astropy.io import fits
from time import time


class Gridder(object):
    def __init__(self, data, uvw, freq, sqrtW, args):
        self.wgt = sqrtW
        self.uvw = uvw
        self.nx = args.nx
        self.ny = args.ny
        self.cell = args.cell_size * np.pi/60/60/180
        self.freq = freq
        self.precision = args.precision
        self.nthreads = args.ncpu
        self.do_wstacking = args.do_wstacking

        # freq mapping
        nchan = freq.size
        if args.channels_out is None or args.channels_out == 0:
            args.channels_out = nchan
        step = nchan//args.channels_out
        freq_mapping = np.arange(0, nchan, step)
        self.freq_mapping = np.append(freq_mapping, nchan)
        self.nband = self.freq_mapping.size - 1

        nrow = uvw.shape[0]
        self.model_data = np.zeros((nrow, nchan), dtype=data.dtype)
        self.image = np.zeros((self.nband, self.nx, self.ny), dtype=data.real.dtype)

    def dot(self, x):
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.model_data[:, Ilow:Ihigh] = ng.dirty2ms(uvw=self.uvw, freq=self.freq[Ilow:Ihigh], dirty=x[i], wgt=self.wgt[:, Ilow:Ihigh],
                                                         pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                                         nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return self.model_data

    def hdot(self, x):
        for i in range(self.nband):
            Ilow = self.freq_mapping[i]
            Ihigh = self.freq_mapping[i+1]
            self.image[i] = ng.ms2dirty(uvw=self.uvw, freq=self.freq[Ilow:Ihigh], ms=x[:, Ilow:Ihigh], wgt=self.wgt[:, Ilow:Ihigh],
                                        npix_x=self.nx, npix_y=self.ny, pixsize_x=self.cell, pixsize_y=self.cell, epsilon=self.precision,
                                        nthreads=self.nthreads, do_wstacking=self.do_wstacking, verbosity=0)
        return self.image

def init_data(args):
    print("Reading data")
    ti = time()
    uvw_full = []
    data_full = []
    weight_full = []
    freqp = None
    nrow = 0
    nvis = 0
    for ms_name in args.ms:
        print("Loading data from ", ms_name)
        ms = table(ms_name)
        if len(set(ms.getcol('FIELD_ID'))) != 1:
            raise RuntimeError("Field selection not supported yet")
        if len(set(ms.getcol('DATA_DESC_ID'))) != 1:
            raise RuntimeError("Multiple DDID's not supported yet")

        uvw = ms.getcol('UVW').astype(np.float64)
        data = ms.getcol(args.data_column).astype(np.complex128)
        _, nchan, ncorr = data.shape
        weight = ms.getcol(args.weight_column).astype(np.float64)
        if len(weight.shape) < 3:
            # print("Assuming weights were taken from less informative "
            #     "WEIGHT column. Tiling over frequency.")
            weight = np.tile(weight[:, None, :], (1, nchan, 1))
        flags = ms.getcol('FLAG')
        ms.close()
        
        # taking weighted sum to get Stokes I
        data = (weight[:, :, 0] * data[:, :, 0] + weight[:, :, ncorr-1] * data[:, :, ncorr-1])/(weight[:, :, 0] + weight[:, :, ncorr-1])
        weight = weight[:, :, 0] + weight[:, :, ncorr-1]
        flags = flags[:, :, 0] | flags[:, :, ncorr-1]  

        nrowtmp = np.sum(~flags)
        nrow += nrowtmp

        nvis += data.size

        print("Effective number of rows for ms = ", nrowtmp)
        print("Number of visibilities for ms = ", data.size)

        # only keep data where both correlations are unflagged
        data = np.where(~flags, data, 0.0j)
        weight = np.where(~flags, weight, 0.0)
        nrow = np.sum(~flags)
        spw = table(ms_name + '::SPECTRAL_WINDOW')
        freq = spw.getcol('CHAN_FREQ')[0].astype(np.float64)
        spw.close()
        if freqp is not None:
            try:
                assert (freqp == freq).all()
            except:
                raise RuntimeError("Not all MS Freqs match")

        data_full.append(data)
        weight_full.append(weight)
        uvw_full.append(uvw)
        freqp = freq

    data = np.concatenate(data_full)
    uvw = np.concatenate(uvw_full)
    weight = np.concatenate(weight_full)
    sqrtW = np.sqrt(weight)
    data *= sqrtW
    print("Time to read data = ", time()- ti)

    print("Effective number of rows total = ", nrow)
    print("Total number of visibilities = ", nvis)
    return data, uvw, sqrtW, freq

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=str, nargs='+')
    p.add_argument("--data_column", default="DATA", type=str,
                   help="The column to image.")
    p.add_argument("--weight_column", default="WEIGHT", type=str)
    p.add_argument("--nx", type=int,
                   help="Number of x pixels.")
    p.add_argument("--ny", type=int,
                   help="Number of y pixels.")
    p.add_argument('--cell_size', type=float,
                   help="Cell size in arcseconds.")
    p.add_argument("--channels_out", default=8, type=int,
                   help="Number of imaging bands to divide the ms into")
    p.add_argument("--workers", default=0, type=int,
                   help="Number of workers available. Each worker has "
                   "additional threads available.")
    return p


if __name__=="__main__":
    args = create_parser().parse_args()

    # aggregate multiple measurement sets into Stokes I visibilities
    data, uvw, sqrtW, freq = init_data(args)
    
    # nrow, nchan = data.shape
    # model_data = np.zeros((nrow, nchan), dtype=data.dtype)

    # freq mapping
    nchan = freq.size
    step = nchan//args.channels_out
    freq_mapping = np.arange(0, nchan, step)
    freq_mapping = np.append(freq_mapping, nchan)
    nband = freq_mapping.size - 1

    dirty = np.zeros((nband, args.nx, args.ny), dtype=data.real.dtype)

    
    cell_rad = args.cell_size * np.pi/60/60/180

    for i in range(nband):
        Ilow = freq_mapping[i]
        Ihigh = freq_mapping[i+1]
        dirty[i] = ng.ms2dirty(uvw=uvw, freq=freq[Ilow:Ihigh], ms=data[:, Ilow:Ihigh], wgt=sqrtW[:, Ilow:Ihigh],
                               npix_x=args.nx, npix_y=args.ny, pixsize_x=cell_rad, pixsize_y=cell_rad, epsilon=1e-7,
                               nthreads=8, do_wstacking=True, verbosity=0)

    hdu = fits.PrimaryHDU()
    hdr = hdu.header
    hdu = fits.PrimaryHDU(header=hdr)
    hdu.data = np.transpose(dirty, axes=(0, 2, 1))[:, ::-1].astype(np.float32)
    hdu.writeto('dirty.fits', overwrite=True)

