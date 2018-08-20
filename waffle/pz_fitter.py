#!/usr/local/bin/python
import os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy import signal, optimize
# from pygama.calibration import *
import pandas as pd
import pygama.filters as filt

def parse_data(chan):
    npzfile = np.load("times_chan{}.npz".format(chan))

    time = npzfile["arr_0"]
    bl_dev = npzfile["arr_1"]

    time_ms = time*10/1E6

    ms_max = 50

    cut = (time_ms<ms_max) & (time_ms>0.05) & (bl_dev > -0.05)
    time_ms = time_ms[cut]
    bl_dev=bl_dev[cut]

    return time_ms, bl_dev

def fit_data(time_ms, bl_dev):

    popt,_ = optimize.curve_fit(get_decay, time_ms, bl_dev, [4, 78, 0, 0], bounds=[(0,70,-1, -1),(10,100,1,1)])

    rc_ms, rc_us, offset,  = popt[:3]
    try:
        toffset = popt[3]
    except IndexError:
        toffset = 0

    return rc_ms, rc_us, offset, toffset

def plot_data(time_ms, bl_dev, rc_ms, rc_us, offset, toffset):
    plt.scatter(time_ms, bl_dev, s=3, c="k")
    # plt.errorbar(bc, points)
    t_plot = np.arange(0, np.amax(time_ms), 0.001)
    plt.plot(t_plot, get_decay(t_plot, rc_ms, rc_us, offset, toffset), label="{:0.2f} us, {:0.2f} ms".format(rc_us, rc_ms))
    plt.axhline(0, c="r", ls="--")

    plt.xlim(0, 15)
    plt.ylim(-.1, .05)

def get_decay(t, rc1, rc2_in, offset=0, toffset=0):
    rc2 = rc2_in/1000

    t_adj = t-toffset

    term1 = np.exp(-t_adj/rc1)/(rc1 * (rc1-rc2))
    term2 = np.exp(-t_adj/rc2)/(rc2 * (rc1-rc2))

    out = -rc1*rc2*(term1-term2)+offset
    return out
