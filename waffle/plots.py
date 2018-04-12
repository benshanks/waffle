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
from waffle.models import *
from waffle.models import Model, PulserTrainingModel

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "brown", "deeppink", "goldenrod", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]


class PlotterBase():
    def __init__(self, result_directory, num_samples, sample_dec=1):
        self.parse_samples("sample.txt", result_directory, num_samples, sample_dec)
        self.width = 18

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
    def __init__(self, result_directory, num_samples, sample_dec=1, model_type="Model"):
        super().__init__(result_directory, num_samples, sample_dec)

        configuration = FitConfiguration(directory=result_directory, loadSavedConfig=True)

        if model_type == "Model":
            self.model = Model(configuration)
        elif model_type == "PulserTrainingModel":
            self.model = PulserTrainingModel(configuration)

        self.num_wf_params = self.model.num_wf_params

        # end_idx = 10000

    def plot_waveforms(self, print_det_params=False):
        data = self.plot_data
        model = self.model

        bad_wf_thresh = 1000

        plt.figure(figsize=(self.width,8))
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

                # em_lo_idx = self.model.joint_models.name_map["LowPassFilterModel"]
                # em_lo = self.model.joint_models.models[em_lo_idx]
                # lo_data = params[em_lo.start_idx: em_lo.start_idx + em_lo.num_params]
                # lo_pz = em_lo.get_pz(lo_data)
                #
                # if resid[130] > 0:
                #     print ("over: {}".format(lo_pz))
                # else:
                #     print ("under: {}".format(lo_pz))


        ax0.set_ylim(-20, np.amax([ np.amax(wf.data) for wf in model.wfs])*1.1)
        ax1.axhline(y=0,color="black", ls=":")
        # ax0.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.set_ylim(-bad_wf_thresh, bad_wf_thresh)

        avg_resid = resid_arr/len(model.wfs)/len(data.index)
        plt.figure(figsize=(self.width,4))
        plt.plot(avg_resid, ls="steps", color = "blue")
        plt.axhline(y=0,color="black", ls=":")
        plt.xlabel("Sample number [10s of ns]")
        plt.ylabel("Average residual [adc]")
        plt.savefig("average_residual.pdf")

    def plot_waveform_components(self):
        data = self.plot_data
        model = self.model

        plt.figure(figsize=(self.width,8))
        plt.xlabel("Digitizer Time [ns]")

        num_det_params = model.num_det_params

        # for wf_idx, wf in enumerate(model.wfs):
        #   dataLen = wf.window_length
        #   t_data = np.arange(dataLen) * 10
        #   plt.plot(t_data, wf.windowed_wf / wf.amplitude, color=colors[wf_idx], ls = ":")
        #   print ("wf %d max %d" % (wf_idx, np.amax(wf.windowed_wf)))

        for (idx) in range(len(data.index)):
            print("index {}".format(idx))
            params = data.iloc[idx].as_matrix()
            self.model.joint_models.apply_params(params)

            for (wf_idx,wf) in enumerate(model.wfs):
                # if wf_idx < 4: continue
                # wfs_param_arr[-1,wf_idx] = 1

                print("wf index {}".format(wf_idx), end="")
                wf_params =  params[model.num_det_params + wf_idx*model.num_wf_params: model.num_det_params + (wf_idx+1)*self.num_wf_params]

                h_wf = np.copy(model.wf_models[wf_idx].make_waveform(wf.window_length,wf_params, charge_type=1))
                e_wf = np.copy(model.wf_models[wf_idx].make_waveform(wf.window_length,wf_params, charge_type=-1))

                print(h_wf.shape)

                t_data = np.arange(len(h_wf))
                color_idx = wf_idx % len(colors)
                plt.plot(t_data,h_wf, color=colors[color_idx],ls="steps", alpha=0.1)
                plt.plot(t_data,e_wf, color=colors[color_idx], ls="steps", alpha=0.1)
                # plt.plot(t_data,h_wf+e_wf, color=colors[color_idx], ls="--", alpha=0.1)

        # plt.ylim(-0.05, 1.05)

    def plot_tf(self):
        plt.figure(figsize=(self.width,8))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        ax0 = plt.subplot(gs[:,0])
        ax1 = plt.subplot(gs[0,1])
        ax2 = plt.subplot(gs[0,2])

        ax_decay_rc = plt.subplot(gs[1,1])
        ax_square = plt.subplot(gs[1,2])

        for idx, row in self.plot_data.iterrows():
            mod_idx = -1

            #show the square response for the whole thing
            square = np.zeros(1000)
            square[100:] = 1

            for model in self.model.joint_models.models:
                if not isinstance(model, DigitalFilterBase): continue
                mod_idx +=1

                data = row[model.start_idx: model.start_idx + model.num_params].values

                model.apply_to_detector(data, None)
                square = model.digital_filter.apply_to_signal(square)

                pz = model.get_pz(data)

                for i, pole in enumerate(pz["poles"]):
                    ax0.scatter(np.real(pole), np.imag(pole), color=colors[mod_idx], alpha=0.3)

                for i, zero in enumerate(pz["zeros"]):
                    ax0.scatter(np.real(zero), np.imag(zero), color=colors[mod_idx], alpha=0.3)

                if isinstance(model, HiPassFilterModel):
                    w_hi = np.logspace(-15, -6, 500, base=np.pi)
                    (w_hi, h) = model.get_freqz(data, w_hi)
                    p = ax1.loglog( w_hi, np.abs(h), alpha = 0.2, color=colors[mod_idx])

                    #Pick out the effective RC constant of the decay
                    decay_const = -1/np.log(-1*model.digital_filter.den[-1])/1E9/1E-6
                    ax_decay_rc.axvline(decay_const, alpha=0.2, color=colors[mod_idx])

                elif isinstance(model, LowPassFilterModel):
                    w_lo = np.logspace(-6, 0, 500, base=np.pi)
                    (w_lo, h3) = model.get_freqz(data, w_lo)
                    p3 = ax2.loglog( w_lo, np.abs(h3), alpha = 0.2, color=colors[mod_idx])

            ax_square.plot(square, color="b", alpha=0.2)


        an = np.linspace(0,np.pi,200)
        ax0.plot(np.cos(an), np.sin(an), c="k")
        ax0.axis("equal")

        ax_decay_rc.set_xscale("log")
        ax_decay_rc.set_xlabel("RC constant (us)")


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
        f, ax = plt.subplots(self.model.num_det_params, 1, figsize=(self.width,10), sharex=True)
        for i in range(self.model.num_det_params):
            tf_data = self.plot_data[i]
            ax[i].plot(tf_data, ls="steps")
        # plt.xlim(0, len(tf_data))

    def plot_detector_pair(self):
        g = sns.pairplot(self.plot_data.iloc[:, :self.model.num_det_params], diag_kind="kde")
        plt.savefig("pairplot.png")

    def plot_waveform_trace(self):
        f, ax = plt.subplots(self.num_wf_params, 1, figsize=(self.width,10), sharex=True)
        for i in range(self.num_wf_params):
            for j in range (self.model.num_waveforms):
                tf_data = self.plot_data[self.model.num_det_params + self.num_wf_params*j+i]
                ax[i].plot(tf_data, color=colors[j], ls="steps")

class WaveformFitPlotter(PlotterBase):

    def __init__(self, result_directory, num_samples, wf_model):
        super().__init__(result_directory, num_samples)

        self.wf_model = wf_model

    def plot_trace(self):
        f, ax = plt.subplots(self.wf_model.num_params, 1, figsize=(self.width,10), sharex=True)
        for i in range(self.wf_model.num_params):
            tf_data = self.plot_data[i]
            ax[i].plot(tf_data)

    def plot_waveform(self):
        data = self.plot_data
        wf_model = self.wf_model
        wf = wf_model.target_wf
        wf_idx = 0

        plt.figure(figsize=(self.width,8))
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
