#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil

import matplotlib.pyplot as plt
# plt.style.use('presentation')
from matplotlib import gridspec
from matplotlib.colors import LogNorm

import corner

import pandas as pd
import numpy as np
import scipy
from scipy import signal

from pysiggen import Detector

from waffle.management import FitConfiguration
from waffle.models import Model

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "brown", "deeppink", "goldenrod", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]


class ResultPlotter():
    def __init__(self, result_directory, num_samples):
        self.result_directory = result_directory

        configuration = FitConfiguration(directory=result_directory, loadSavedConfig=True)
        self.model = Model(configuration)

        self.parse_samples("sample.txt", result_directory, num_samples)


    def parse_samples(self, sample_file_name, directory, plotNum):

        #Load the data from csv (using pandas so its fast)
        sample_file_name = os.path.join(directory, sample_file_name)
        data = pd.read_csv(sample_file_name, delim_whitespace=True, header=None)
        num_samples = len(data.index)
        print( "found %d samples... " % num_samples,)

        if num_samples > plotNum: num_samples = plotNum
        print( "plotting %d samples" % num_samples)

        end_idx = len(data.index) - 1
        # end_idx = 10000

        self.plot_data = data.iloc[(end_idx - num_samples):end_idx]

        self.num_wf_params = np.int(  (len(data.columns) - self.model.num_det_params) / self.model.num_waveforms )

    def plot_waveforms(self, wf_to_plot=None):
        data = self.plot_data
        model = self.model

        bad_wf_thresh = 1000

        plt.figure(figsize=(20,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Digitizer Time [ns]")
        ax0.set_ylabel("Voltage [Arb.]")
        ax1.set_ylabel("Residual")

        num_det_params = model.num_det_params

        wf_params = np.empty(num_det_params+self.num_wf_params)
        resid_arr = np.zeros(model.wfs[0].window_length)


        for wf_idx, wf in enumerate(model.wfs):
          dataLen = wf.window_length
          t_data = np.arange(dataLen) * 10
          ax0.plot(t_data, wf.windowed_wf, color=colors[wf_idx], ls = ":")
          print ("wf %d max %d" % (wf_idx, np.amax(wf.windowed_wf)))

        for (idx) in range(len(data.index)):
            params = data.iloc[idx].as_matrix()

            wfs_param_arr = params[num_det_params:].reshape((self.num_wf_params, model.num_waveforms))
            wf_params[:num_det_params] = params[:num_det_params]
            # print(wf_params[:num_det_params])
            print("imp:{}, grad: {}".format(wf_params[num_det_params-1], wf_params[num_det_params-2]))

            for (wf_idx,wf) in enumerate(model.wfs):
                # if wf_idx < 4: continue
                # wfs_param_arr[-1,wf_idx] = 1
                wf_params[num_det_params:] = wfs_param_arr[:,wf_idx]

                fit_wf = model.make_waveform(wf.window_length,wf_params)
                if fit_wf is None:
                    continue

                # if idx == 1 and wf_idx == 0:
                #     print("\n\n")
                #     print(model.detector.lp_num)
                #     print(model.detector.lp_den)
                #     print(model.detector.hp_num)
                #     print(model.detector.hp_den)
                #     print(wf_params[num_det_params:])

                t_data = np.arange(wf.window_length) * 10
                color_idx = wf_idx % len(colors)
                ax0.plot(t_data,fit_wf, color=colors[color_idx], alpha=0.1)

                resid = fit_wf -  wf.windowed_wf
                resid_arr += resid
                ax1.plot(t_data, resid, color=colors[color_idx],alpha=0.1,)# linestyle="steps")


        ax0.set_ylim(-20, np.amax([wf.amplitude for wf in model.wfs])*1.1)
        ax0.axhline(y=0,color="black", ls=":")
        ax0.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        ax1.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.set_ylim(-bad_wf_thresh, bad_wf_thresh)

        avg_resid = resid_arr/len(model.wfs)/len(data.index)
        plt.figure(figsize=(6.5,4))
        plt.plot(avg_resid, ls="steps", color = "blue")
        plt.axhline(y=0,color="black", ls=":")
        plt.xlabel("Sample number [10s of ns]")
        plt.ylabel("Average residual [adc]")
        plt.savefig("average_residual.pdf")
