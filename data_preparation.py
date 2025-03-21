import sys, os
from os.path import join, abspath, exists, pardir
import tomlkit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table, join as tab_join
import copy
from numpy import array as a
from sklearn.neighbors import KernelDensity
import pycatch22
from tempfile import TemporaryDirectory
import subprocess
from astropy.stats import sigma_clipped_stats, sigma_clip
from supersmoother import SuperSmoother
import scipy.ndimage

from lightcurve import ASASSN_Lightcurve
from utils import read_config, n_hist_bins, hist

import astropy.units as u
from astropy.timeseries import TimeSeries, aggregate_downsample
from astropy.time import TimeDelta, Time

from tqdm import tqdm

rng = np.random.default_rng()

cfg = read_config("config.toml")
GHKSS_PATH = cfg["ghkss_path"]


def calc_cadence(times):
    return np.median(np.diff(times))
    

def read_denoised_output(filename):
    vals = []
    with open(filename) as f:
        for line in f:
            vals.append(float(line.strip()))
    return a(vals)


def denoise(mags, niter=3):
    mag_str = '\r\n'.join([str(m) for m in mags])
    denoised = []
    with TemporaryDirectory() as d:
        p = subprocess.Popen([GHKSS_PATH, "-i", f"{niter}",'-v','0'], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, cwd=abspath(d), encoding="UTF-8",text=True)
        for m in mags:
            p.stdin.write(f"{m}\n")
        p.communicate("")
        p.wait(timeout=5)
        for f in sorted(os.listdir(d)):
            denoised.append(read_denoised_output(join(d,f)))        
    return denoised

def is_oversampled(mags):
    global_min, global_max = np.min(mags), np.max(mags)
    mean_abs_difference = np.mean(np.abs(np.diff(mags)))
    return global_max - global_min > 20 * mean_abs_difference

def make_timeseries(mags,times):
    return TimeSeries(Table([a(mags),a([Time(t, format="jd") for t in times])],names=("mag","time")))

def downsample(mags,times,cadence,factor=2):
    ts = make_timeseries(mags,times)
    # print(len(ts.time),len(ts["mag"]))
    downsampled = aggregate_downsample(ts,time_bin_size=cadence*factor*u.day)
    mags, times = downsampled["mag"], downsampled.time_bin_center.jd
    not_nan = np.where(~mags.mask)
    mags = a(mags)[not_nan]
    times = times[not_nan]
    return mags, times

b = -5/np.log(10)

def copy_preprocess(lc,row):
    # make a copy and then preprocess
    new_lc = copy.deepcopy(lc)
    preprocess(new_lc,row)
    return new_lc

def preprocess(lc,row):
    lc.times -= min(lc.times)
    drop_outliers(lc,row)
    distance_normalization(lc,row)


def drop_outliers(lc,row,sigma_rough=5,sigma=3, moving_window_size=1):
    global output_array
    valid_err = (lc.mag_err < 2*np.mean(lc.mag_err))
    valid_mag = ~sigma_clip(lc.mag,sigma=sigma,masked=True).mask
    initial_valid_idx = np.where(valid_err & valid_mag)[0]
    lc.times = lc.times[initial_valid_idx]
    lc.mag = lc.mag[initial_valid_idx]
    lc.mag_err = lc.mag_err[initial_valid_idx]
    lc.fwhm = lc.fwhm[initial_valid_idx]
    if lc.abs_mag is not None:
        lc.abs_mag = lc.abs_mag[initial_valid_idx]
        lc.abs_mag_err = lc.abs_mag_err[initial_valid_idx]

    std = np.std(lc.mag)
    
    model = SuperSmoother(alpha=2)
    model.fit(lc.times, lc.mag, lc.mag_err)
    window_edges = np.linspace(min(lc.times),max(lc.times),(len(lc.times)//moving_window_size))
    fit_t = window_edges[:-1]+moving_window_size/2
    fit_mag = model.predict(fit_t)

    output_array = np.ones_like(lc.mag)
    # mask_parts = []

    out_idx = 0
    for i, mean_mag in enumerate(fit_mag):
        relevant_idx = np.where((lc.times>window_edges[i])&(lc.times<=window_edges[i+1]))[0]
        relevant_mags = lc.mag[relevant_idx]
        output_array[out_idx:out_idx+len(relevant_mags)] = np.abs(relevant_mags-mean_mag) < sigma*std
        out_idx += len(relevant_mags)

    valid_idx = output_array.astype(bool)
    lc.times = lc.times[valid_idx]
    lc.mag = lc.mag[valid_idx]
    lc.mag_err = lc.mag_err[valid_idx]
    lc.fwhm = lc.fwhm[valid_idx]
    if lc.abs_mag is not None:
        lc.abs_mag = lc.abs_mag[valid_idx]
        lc.abs_mag_err = lc.abs_mag_err[valid_idx]


def distance_normalization(lc:ASASSN_Lightcurve,row):
    d = row["EDR3_dist"]
    delta_d = row["parallax_over_error"]
    m = lc.mag
    delta_m = lc.mag_err
    
    lc.abs_mag = m - 5 * np.log10(d/10)
    # error propagation: 
    lc.abs_mag_err = lc.abs_mag * np.sqrt( np.pow(delta_d*b/d,2) + np.pow(delta_m/m,2) )
    

def denoise_and_downsample(mags,times,cadence,denoise_iter=2):
    denoised = denoise(mags,niter=denoise_iter)[-1]
    d_times = times
    if is_oversampled(denoised):
        # print("oversampled. downsampling by a factor of 2...")
        try:
            denoised, d_times = downsample(denoised,d_times,cadence,2)
        except:
            # print("Failed to downsample :(")
            pass
    else:
        pass
        # print("data is not oversampled")
    return denoised, d_times

def prepare_lc(lc,row,denoise_iter=2,do_preprocess=True):
    if do_preprocess:
        preprocess(lc,row)
    d_mag, d_times = denoise_and_downsample(lc.abs_mag,lc.times,lc.cadence,denoise_iter)
    return d_mag, d_times

def plot_prepared(mags, times, denoise_iter=2, name=None):

    p_mag, p_time = prepare_lc(mags,times,denoise_iter)
    
    fig,axes = plt.subplots(2,1)

    title = "Input Lightcurve" if name is None else f"Input: {name}"
    axes[0].set_title(title)
    axes[0].scatter(times,mags,s=1)
    axes[0].invert_yaxis()
    ylim = axes[0].get_ylim()
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_title("Processed Lightcurve")
    axes[1].scatter(p_time,p_mag,s=1)
    axes[1].invert_yaxis()
    axes[1].set_ylim(*ylim)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    

