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

    from electronics_model import VelocityModel, ElectronicsModel
    v = ElectronicsModel()
    p = v.get_prior()
    print (p)
    v.perturb(p, 1)
    print (p)
    exit()

    channel = 600
    runList=np.arange(11510, 11560)
    n_bins = 10

    training_file_name = os.path.join("training_data", "training_ch{}_run{}-{}".format(channel, runList[0], runList[-1]))

    df_train = pd.read_hdf(training_file_name,key="data")

    first_dt = df_train.drift_time.min()
    last_dt = df_train.drift_time.max()

    dt_bins = np.linspace(first_dt, last_dt, n_bins+1)

    plt.figure()

    for b_lo, b_hi in zip(dt_bins[:-1], dt_bins[1:]):
        df_bin = df_train[(df_train.drift_time >= b_lo) & (df_train.drift_time<b_hi)]
        for i, (index, row) in enumerate(df_bin.iterrows()):

            if i>=50: break

            wf = Waveform( row, amplitude=row["trap_max"], bl_slope=row["bl_slope"], bl_int=row["bl_int"], t0_estimate=row["t0_est"])

            windowed_wf = wf.window_waveform(0.02)

            if i == 0:
                p = plt.plot(wf, alpha=0.1)
            else:
                plt.plot(wf, c=p[0].get_color(), alpha=0.1)

    plt.show()


if __name__ == "__main__":
    main()
