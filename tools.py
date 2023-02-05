import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import healpy as hp
import pymaster as nmt
import pyccl as ccl
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.integrate import simps

def load_fits2pd(fits_file):
    cat  = Table.read(fits_file, format='fits')
    names = [name for name in cat.colnames if len(cat[name].shape) <= 1]
    catt = cat[names].to_pandas()
    return catt

def get_nz(cat):
    z_in = np.linspace(0.0, 1.5, 100)
    nz, zz = np.histogram(cat['DNF_ZMC_SOF'], bins=100, density=None)
    zz = 0.5 * (zz[1:]+zz[:-1])
    nz_normed = nz/ simps(nz, zz)
    _p_of_z = interp1d(zz, nz_normed,bounds_error=False, fill_value=0.)
    nz_in = _p_of_z(z_in)
    return z_in, nz_in

def setup_axis(ax, xlabel=None, ylabel=None, xscale=None, yscale=None,
               fs=18, title=None):
    if xlabel: ax.set_xlabel(xlabel, fontsize=fs)
    if ylabel: ax.set_ylabel(ylabel, fontsize=fs)
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if title:  ax.set_title(title, fontsize=fs)
    return ax

# class th_cl():
#     def _init(self, cosmo, catals):
#         self.cosmo = cosmo
#         self.des_bins = len(catals)



# def cal_th_cl(cosmo, has_rsd, dndz, bias, b2=None, mag_bias=None):
