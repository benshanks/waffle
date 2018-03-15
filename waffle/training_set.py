#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd

from sklearn import linear_model
from sklearn.cluster import *
from sklearn.neighbors import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import *

from scipy.ndimage.filters import gaussian_filter1d

import data_processing as dp
import machine_learning as ml
from pygama.utils import get_bin_centers
from pygama.waveform import Waveform
from pygama.calculators import t0_estimate

def main():

    channel = 626
    runList=np.arange(11510, 11560)

    qualified_training_data_file_name = os.path.join("training_data", "training_ch{}_run{}-{}".format(channel, runList[0], runList[-1]))

    # save_all_qualified(runList,channel, qualified_training_data_file_name)

    save_subset("32wf_set_chan{}.npz".format(channel), qualified_training_data_file_name)

def save_subset(training_set_file_name, qualified_training_data_file_name, do_plot=True):
    df_train = pd.read_hdf(qualified_training_data_file_name,key="data")

    first_dt = df_train.drift_time.min()
    last_dt = df_train.drift_time.max()

    n_bins_time = 8

    dt_bins = np.linspace(first_dt, last_dt, n_bins_time+1)

    wfs_per_bin = 4

    wfs_saved = []

    baseline_val_arr = np.empty(0)

    for b_lo, b_hi in zip(dt_bins[:-1], dt_bins[1:]):
        df_bin = df_train[(df_train.drift_time >= b_lo) & (df_train.drift_time<b_hi)]
        for i, (index, row) in enumerate(df_bin.iterrows()):
            wf = Waveform( row, amplitude=row["trap_max"], bl_slope=row["bl_slope"], bl_int=row["bl_int"], t0_estimate=row["t0_est"])
            baseline_val_arr = np.append(  baseline_val_arr, (wf.data - (row["bl_slope"]*np.arange(len(wf.data))  + row["bl_int"]))[:700]   )
            if i>=wfs_per_bin: continue
            wfs_saved.append(wf)

    np.savez(training_set_file_name, wfs=wfs_saved)

    if do_plot:
        plt.figure()
        for wf in wfs_saved:
            print(wf.amplitude)
            plt.plot(wf.data)

        plt.figure()
        plt.hist(baseline_val_arr,bins="auto")
        plt.show()

def save_all_qualified(runList, channel, file_name):

    print("loading data...", end="")
    df_all = dp.load_reduced(runList, channel)
    print(" done")

    df_ae = df_all[(df_all.ae > -10) & (df_all.ae < 40)]

    # times = [0.3,0.5,0.7,0.9,0.95]
    times = [0,0.3,0.95]
    tp_train, df_ae = ml.calculate_timepoints(df_ae, times, relative_tp=0)
    x_idx = -1
    # my_ae = calc_ae(df_ae, sigma=20)


    grid = gs.GridSpec(2, 2, height_ratios=[3, 1], width_ratios = [1,3])
    ax = plt.subplot(grid[0, 1])
    ax_y = plt.subplot(grid[0, 0], sharey = ax)
    ax_x = plt.subplot(grid[1, 1], sharex = ax)

    dt = tp_train[:,x_idx]
    df_ae["drift_time"] = dt
    df_ae["t0_est"] = tp_train[:,0]

    dt_ae = dt[(df_ae.ae>0)&(df_ae.ae<2)]
    df_cut = df_ae[(df_ae.ae>0)&(df_ae.ae<2)]

    ax_y.hist(df_ae.ae, bins="auto", histtype="step", orientation='horizontal')
    h,b,p = ax_x.hist(dt_ae, bins=100, histtype="step")
    hist,bins,p = ax_x.hist(dt_ae, bins=b, histtype="step")

    #uh, find the FWHM of the cut drift times
    bin_centers = get_bin_centers(bins)
    idxs_over_50 = hist > 0.1*np.amax(hist)
    first_dt =  bin_centers[np.argmax(idxs_over_50)]
    last_dt = bin_centers[  len(idxs_over_50) - np.argmax(idxs_over_50[::-1])  ]

    ax_x.axvline(first_dt, c="r")
    ax_x.axvline(last_dt, c="r")
    train_set = (df_ae.ae>0)&(df_ae.ae<2)&(dt >= first_dt)&(dt < last_dt)
    ax.scatter(dt[train_set], df_ae.ae[train_set], s=0.5, c="g" )
    ax.scatter(dt[~train_set], df_ae.ae[~train_set], s=0.5, c="k" )

    n_bins = 10

    dt_bins = np.linspace(first_dt, last_dt, n_bins+1)

    plt.figure()

    for b_lo, b_hi in zip(dt_bins[:-1], dt_bins[1:]):
        df_bin = df_cut[(dt_ae >= b_lo) & (dt_ae<b_hi)]
        for i, (index, row) in enumerate(df_bin.iterrows()):
            if i>=50: break
            wf = Waveform( row, )
            wf.data -= row["bl_int"] + np.arange(len(wf.data))*row["bl_slope"]
            wf.data /= row["energy_cal"]

            t0_est = np.argmax(wf.data > 0.95)

            wf_plot = wf.data[t0_est-200:t0_est+100]

            if i == 0:
                p = plt.plot(wf_plot, alpha=0.1)
            else:
                plt.plot(wf_plot, c=p[0].get_color(), alpha=0.1)



    try: os.mkdir("training_data")
    except OSError: pass

    df_train = df_ae[train_set]
    df_train.to_hdf(file_name, key="data", mode='w')

    plt.show()


if __name__ == "__main__":
    main()
