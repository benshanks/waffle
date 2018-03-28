#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil

import matplotlib.pyplot as plt
# plt.style.use('presentation')
from matplotlib import gridspec
from matplotlib.colors import LogNorm

import seaborn as sns
import corner

import pandas as pd
import numpy as np
import scipy
from scipy import signal

from pysiggen import Detector

from waffle.management import FitConfiguration
from waffle.models import Model, ElectronicsModel

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

        if plotNum == -1:
            self.plot_data = data
        elif num_samples > plotNum:
            num_samples = plotNum
            end_idx = len(data.index) - 1
            self.plot_data = data.iloc[(end_idx - num_samples):end_idx]

        self.num_wf_params = np.int(  (len(data.columns) - self.model.num_det_params) / self.model.num_waveforms )
        print( "plotting %d samples" % num_samples)


        # end_idx = 10000

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
            # print("imp:{}, grad: {}".format(wf_params[num_det_params-1], wf_params[num_det_params-2]))

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

    def plot_tf(self):
        freq_samp = 1E9
        nyq_freq = 0.5*freq_samp

        em = self.model.electronics_model

        tf_data = self.plot_data.iloc[:, self.model.tf_first_idx: self.model.tf_first_idx + em.get_num_params()]

        f, ax = plt.subplots(1,2, figsize=(12,6))

        pmag = tf_data.iloc[:,0].as_matrix()
        pphi = tf_data.iloc[:,1].as_matrix()
        pole = pmag * np.exp(1j*pphi)
        ax[0].scatter(np.real(pole), np.imag(pole), c="b", alpha=0.3)

        rc_mag = tf_data.iloc[:,2].as_matrix()
        rc_phi = tf_data.iloc[:,3].as_matrix()
        rc_mag = 1. - 10.**rc_mag
        rc_phi = 10.**rc_phi
        pole = rc_mag * np.exp(1j*rc_phi)
        ax[0].scatter(np.real(pole), np.imag(pole), c="g", alpha=0.3)

        if em.get_num_params() == 6:
            lp_zeromag = tf_data.iloc[:,4].as_matrix()
            lp_zerophi = tf_data.iloc[:,5].as_matrix()
            zero = lp_zeromag * np.exp(1j*lp_zerophi)
            ax[0].scatter(np.real(zero), np.imag(zero), c="r", alpha=0.3)

        an = np.linspace(0,np.pi,200)
        ax[0].plot(np.cos(an), np.sin(an), c="k")
        ax[0].axis("equal")

        p = None
        for i, row in tf_data.iterrows():
            if em.get_num_params() == 6:
                pmag, pphi, rc_mag, rc_phi, lp_zeromag, lp_zerophi = row.as_matrix()
                num = em.zpk_to_ba(lp_zeromag, lp_zerophi)
            else:
                pmag, pphi, rc_mag, rc_phi = row.as_matrix()
                num = [1,2,1]

            den = em.zpk_to_ba(pmag, pphi)
            num /= (np.sum(num)/np.sum(den))
            w, h = signal.freqz(num, den, worN=np.logspace(-13, 0, 500, base=np.pi), )
            w/= (np.pi /nyq_freq)

            den = em.zpk_to_ba(1. - 10.**rc_mag, 10.**rc_phi)
            num = [1,-2,1]
            w_rc, h_rc = signal.freqz(num, den, worN=np.logspace(-13, 0, 500, base=np.pi), )
            w_rc/= (np.pi /nyq_freq)

            if p is None:
                p = ax[1].loglog( w, np.abs(h), alpha = 0.2)
                p2 = ax[1].loglog( w_rc, np.abs(h_rc), alpha = 0.2)
            else:
                ax[1].loglog( w, np.abs(h), c=p[0].get_color(), alpha = 0.2)
                ax[1].loglog( w_rc, np.abs(h_rc), c=p2[0].get_color(), alpha = 0.2)

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
            ax[i].plot(tf_data)
        plt.xlim(0, len(tf_data))

    def plot_detector_pair(self):
        g = sns.pairplot(self.plot_data.iloc[:, :self.model.num_det_params], diag_kind="kde")
        plt.savefig("pairplot.png")

    def plot_waveform_trace(self):
        f, ax = plt.subplots(self.num_wf_params, 1, figsize=(14,10), sharex=True)
        for i in range(self.num_wf_params):
            for j in range (self.model.num_waveforms):
                tf_data = self.plot_data[self.model.num_det_params + self.model.num_waveforms*i+j]
                ax[i].plot(tf_data, color=colors[j])
