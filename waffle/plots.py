#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil

import matplotlib.pyplot as plt
# plt.style.use('presentation')
from matplotlib import gridspec
from matplotlib.colors import LogNorm

import seaborn as sns
# import corner

import pandas as pd
import numpy as np
import scipy
from scipy import signal

from waffle.management import FitConfiguration
from waffle.models import Model# #ElectronicsModel

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "brown", "deeppink", "goldenrod", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]


class PlotterBase():
    def __init__(self, result_directory, num_samples, sample_dec=1):
        self.parse_samples("sample.txt", result_directory, num_samples, sample_dec)

    def parse_samples(self, sample_file_name, directory, plotNum, sample_dec=1):

        #Load the data from csv (using pandas so its fast)
        sample_file_name = os.path.join(directory, sample_file_name)
        data = pd.read_csv(sample_file_name, delim_whitespace=True, header=None)
        num_samples = len(data.index)
        print( "found {} samples... ".format(num_samples), end='')

        if plotNum == -1:
            self.plot_data = data
        elif num_samples > plotNum:
            num_samples = plotNum
            end_idx = len(data.index) - 1
            self.plot_data = data.iloc[(end_idx - num_samples):end_idx:sample_dec]
        elif num_samples < plotNum:
            self.plot_data = data
        print( " plotting {} samples".format( len(self.plot_data) ))


class TrainingPlotter(PlotterBase):
    def __init__(self, result_directory, num_samples, sample_dec=1):
        super().__init__(result_directory, num_samples, sample_dec)

        configuration = FitConfiguration(directory=result_directory, loadSavedConfig=True)
        self.model = Model(configuration)
        self.num_wf_params = self.model.num_wf_params

        # end_idx = 10000

    def plot_waveforms(self, wf_to_plot=None, print_det_params=False):
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
        resid_arr = np.zeros(model.wfs[0].window_length)


        for wf_idx, wf in enumerate(model.wfs):
          dataLen = wf.window_length
          t_data = np.arange(dataLen) * 10
          ax0.plot(t_data, wf.windowed_wf, color=colors[wf_idx], ls = ":")
          print ("wf %d max %d" % (wf_idx, np.amax(wf.windowed_wf)))

        for (idx) in range(len(data.index)):
            params = data.iloc[idx].as_matrix()
            if print_det_params: print(params[:self.model.num_det_params])

            self.model.joint_models.apply_params(params)

            for (wf_idx,wf) in enumerate(model.wfs):
                # if wf_idx < 4: continue
                # wfs_param_arr[-1,wf_idx] = 1
                wf_params =  params[model.num_det_params + wf_idx*model.num_wf_params: model.num_det_params + (wf_idx+1)*self.num_wf_params]

                fit_wf = model.wf_models[wf_idx].make_waveform(wf.window_length,wf_params)
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
        # ax0.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.set_ylim(-bad_wf_thresh, bad_wf_thresh)

        avg_resid = resid_arr/len(model.wfs)/len(data.index)
        plt.figure(figsize=(6.5,4))
        plt.plot(avg_resid, ls="steps", color = "blue")
        plt.axhline(y=0,color="black", ls=":")
        plt.xlabel("Sample number [10s of ns]")
        plt.ylabel("Average residual [adc]")
        plt.savefig("average_residual.pdf")

    def plot_tf(self):
        em_hi_idx = self.model.joint_models.name_map["HiPassFilterModel"]
        em_lo_idx = self.model.joint_models.name_map["LowPassFilterModel"]
        em_hi = self.model.joint_models.models[em_hi_idx]
        em_lo = self.model.joint_models.models[em_lo_idx]

        f, ax = plt.subplots(1,3, figsize=(15,6))

        p = None
        for idx, row in self.plot_data.iterrows():
            hi_data = row[em_hi.start_idx: em_hi.start_idx + em_hi.num_params].values
            lo_data = row[em_lo.start_idx: em_lo.start_idx + em_lo.num_params].values

            hi_pz = em_hi.get_pz(hi_data)
            lo_pz = em_lo.get_pz(lo_data)

            for i, pole in enumerate(hi_pz["poles"]):
                if i == 0: c="b"
                else: c="g"
                ax[0].scatter(np.real(pole), np.imag(pole), c=c, alpha=0.3)

            for zero in hi_pz["zeros"]:
                ax[0].scatter(np.real(zero), np.imag(zero), c="r", alpha=0.3)

            for i, pole in enumerate(lo_pz["poles"]):
                if i == 0: c="purple"
                else: c="cyan"
                ax[0].scatter(np.real(pole), np.imag(pole), c=c, alpha=0.3)

            for zero in lo_pz["zeros"]:
                ax[0].scatter(np.real(zero), np.imag(zero), c="r", alpha=0.3)


            w_hi = np.logspace(-15, -6, 500, base=np.pi)
            w_lo = np.logspace(-6, 0, 500, base=np.pi)

            (w_hi, h,  h2) = em_hi.get_freqz(hi_data, w_hi)
            (w_lo, h3, h4) = em_lo.get_freqz(lo_data, w_lo)

            if p is None:
                p = ax[1].loglog( w_hi, np.abs(h), alpha = 0.2)
                p2 = ax[1].loglog( w_hi, np.abs(h2), alpha = 0.2)
                p3 = ax[2].loglog( w_lo, np.abs(h3), alpha = 0.2)
                p4 = ax[2].loglog( w_lo, np.abs(h4), alpha = 0.2)
            else:
                ax[1].loglog( w_hi, np.abs(h), c=p[0].get_color(), alpha = 0.2)
                ax[1].loglog( w_hi, np.abs(h2), c=p2[0].get_color(), alpha = 0.2)
                ax[2].loglog( w_lo, np.abs(h3), c=p3[0].get_color(), alpha = 0.2)
                ax[2].loglog( w_lo, np.abs(h4), c=p4[0].get_color(), alpha = 0.2)


        an = np.linspace(0,np.pi,200)
        ax[0].plot(np.cos(an), np.sin(an), c="k")
        ax[0].axis("equal")

    def plot_imp(self):
        im = self.model.imp_model
        imp_data = self.plot_data.iloc[:, self.model.imp_first_idx: self.model.imp_first_idx + im.get_num_params()]

        imp_z0 = imp_data.iloc[:,0].as_matrix()
        imp_zmax = imp_data.iloc[:,1].as_matrix()

        g = sns.jointplot(x=imp_z0, y=imp_zmax, kind="kde", stat_func=None)

        avgs = self.model.detector.imp_avg_points
        grads = self.model.detector.imp_grad_points

        # plt.figure()

        for avg in avgs:
            y1 = 2*avg - im.imp_max
            y2 = 2*avg - 0
            g.ax_joint.plot((im.imp_max, 0), (y1, y2), c="k", ls="--")
        for grad in grads:
            y1 = (grad*self.model.detector.detector_length/10) + im.imp_max
            y2 = (grad*self.model.detector.detector_length/10) + 0

            g.ax_joint.plot((im.imp_max, 0), (y1, y2), c="k", ls="--")

        g.ax_joint.set_xlim(im.imp_max, 0)
        g.ax_joint.set_ylim(im.imp_min, 0)

        #overlay the grid of calculated points

    def plot_trace(self):
        f, ax = plt.subplots(self.model.num_det_params, 1, figsize=(14,10), sharex=True)
        for i in range(self.model.num_det_params):
            tf_data = self.plot_data[i]
            ax[i].plot(tf_data, ls="steps")
        # plt.xlim(0, len(tf_data))

    def plot_detector_pair(self):
        g = sns.pairplot(self.plot_data.iloc[:, :self.model.num_det_params], diag_kind="kde")
        plt.savefig("pairplot.png")

    def plot_waveform_trace(self):
        f, ax = plt.subplots(self.num_wf_params, 1, figsize=(14,10), sharex=True)
        for i in range(self.num_wf_params):
            for j in range (self.model.num_waveforms):
                tf_data = self.plot_data[self.model.num_det_params + self.num_wf_params*j+i]
                ax[i].plot(tf_data, color=colors[j], ls="steps")


class WaveformFitPlotter(PlotterBase):

    def __init__(self, result_directory, num_samples, wf_model):
        super().__init__(result_directory, num_samples)

        self.wf_model = wf_model

    def plot_trace(self):
        f, ax = plt.subplots(self.wf_model.num_params, 1, figsize=(14,10), sharex=True)
        for i in range(self.wf_model.num_params):
            tf_data = self.plot_data[i]
            ax[i].plot(tf_data)

    def plot_waveform(self):
        data = self.plot_data
        wf_model = self.wf_model
        wf = wf_model.target_wf
        wf_idx = 0

        plt.figure(figsize=(20,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Digitizer Time [ns]")
        ax0.set_ylabel("Voltage [Arb.]")
        ax1.set_ylabel("Residual")


        dataLen = wf.window_length
        t_data = np.arange(dataLen) * 10
        ax0.plot(t_data, wf.windowed_wf, color=colors[wf_idx], ls = ":")
        print ("wf %d max %d" % (wf_idx, np.amax(wf.windowed_wf)))

        for (idx) in range(len(data.index)):
            wf_params = data.iloc[idx].as_matrix()

            fit_wf = wf_model.make_waveform(wf.window_length,wf_params)
            if fit_wf is None:
                continue

            t_data = np.arange(wf.window_length) * 10
            color_idx = wf_idx % len(colors)
            ax0.plot(t_data,fit_wf, color=colors[color_idx], alpha=0.1)

            resid = fit_wf -  wf.windowed_wf
            ax1.plot(t_data, resid, color=colors[color_idx],alpha=0.1,)# linestyle="steps")

        ax0.set_ylim(-20, wf.amplitude*1.1)
        ax0.axhline(y=0,color="black", ls=":")
        # ax0.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.set_ylim(-bad_wf_thresh, bad_wf_thresh)
