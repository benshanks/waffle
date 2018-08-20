import os, sys
import numpy as np
import numpy.random as rng
import scipy.stats as stats
from scipy import signal
import dnest4
from abc import ABC, abstractmethod

from pygama.waveform import Waveform
from siggen import PPC

from .waveform import WaveformModel
from ._model_bundle import JointModelBundle

max_float = sys.float_info.max

class Model(object):
    """
    Specify the model in Python.
    """
    def __init__(self, fit_configuration, fit_manager=None):

        self.conf = fit_configuration
        self.fit_manager = fit_manager

        #Setup detector and waveforms
        self.setup_detector()
        self.wf_models = []
        self.setup_waveforms(self.conf.wf_config, doPrint=False)

        self.changed_wfs = np.zeros(self.num_waveforms)

        #Set up all the models...
        self.joint_models = JointModelBundle(self.conf.model_conf, self.detector)
        self.num_det_params = self.joint_models.num_params

    def setup_detector(self):
        timeStepSize = self.conf.time_step_calc

        #TODO: wtf is going on with wf length?
        det = PPC( self.conf.siggen_conf_file, wf_padding=500)
        self.detector = det

    def setup_waveforms(self, wf_conf, doPrint=False):
        wfFileName = wf_conf.wf_file_name

        if os.path.isfile(wfFileName):
            print("Loading wf file {0}".format(wfFileName))
            data = np.load(wfFileName, encoding="latin1")
            wfs = data['wfs']
            self.wfs = wfs[wf_conf.wf_idxs]
            self.num_waveforms = self.wfs.size

            # self.rc1_guess = data['rc1']
            # self.rc2_guess =data['rc2']
            # self.rcfrac_guess =data['rcfrac']

        else:
          print("Saved waveform file %s not available" % wfFileName)
          exit(0)

        baselineLengths = np.zeros(wfs.size)

        for (wf_idx,wf) in enumerate(self.wfs):
          total_samples = wf_conf.num_samples
          full_samples = wf_conf.align_idx
          wf.window_waveform(time_point=wf_conf.align_percent, early_samples=wf_conf.align_idx, num_samples=total_samples)

        #   dec_idx = 150
        #   dec_factor = 20
        #   dec_samples = dec_factor*(total_samples-dec_idx)
          #
        #   wf.window_waveform(time_point=self.conf.align_percent, early_samples=self.conf.align_idx, num_samples=dec_idx+dec_samples)
        #   wf.windowed_wf = np.concatenate((wf.windowed_wf[:dec_idx], wf.windowed_wf[dec_idx::dec_factor]))
        #   wf.window_length = len(wf.windowed_wf)

          self.wf_models.append(WaveformModel(wf, align_percent=wf_conf.align_percent, detector=self.detector,
                        do_smooth=wf_conf.do_smooth, smoothing_type=wf_conf.smoothing_type))

          if doPrint:
              print( "wf %d length %d (entry %d from run %d)" % (wf_idx, wf.window_length, wf.entry_number, wf.runNumber))
          baselineLengths[wf_idx] = wf.t0_estimate

        #TODO: this doesn't work if the calc step size isn't 1 ns
        self.siggen_wf_length = np.int(  (wf_conf.align_idx - np.amin(baselineLengths) + 10)*(10  ))

        self.output_wf_length = np.int( wf_conf.num_samples + 1 )

        self.num_wf_params = self.wf_models[0].num_params

        if doPrint:
            print( "siggen_wf_length will be %d, output wf length will be %d" % (self.siggen_wf_length, self.output_wf_length))

    def from_prior(self):
        detector=self.detector

        wf_params = np.concatenate([ wf.get_prior()[:] for wf in self.wf_models ])

        prior = np.concatenate([
              self.joint_models.get_prior(), wf_params
            ])

        if False:#print out the prior to make sure i know what i'm doing
            import matplotlib.pyplot as plt
            plt.figure()

            self.apply_detector_params(prior[:self.num_det_params])
            for wf_idx, wf_model in enumerate(self.wf_models):
                print("waveform number {}".format(wf_idx))
                p = plt.plot(wf_model.target_wf.windowed_wf)
                wf_params = prior[self.num_det_params + wf_idx*self.num_wf_params: self.num_det_params + (wf_idx+1)*self.num_wf_params]
                fit_wf = wf_model.make_waveform( self.output_wf_length,   wf_params)
                plt.plot(fit_wf, c=p[0].get_color())
            plt.show()
            exit()
        return prior

    def get_wf_params(self, params, wf_idx):
        return np.concatenate(( params[:self.num_det_params], params[self.num_det_params + wf_idx*self.num_wf_params: self.num_det_params + (wf_idx+1)*self.num_wf_params]))

    def perturb(self, params):
        logH = 0.0
        num_waveforms = self.num_waveforms

        #TODO: decide whether to adjust just waveforms, or both wfs and detector params

        if rng.rand() <= 0.5:
            #adjust detector only
            reps = 1;
            if(rng.rand() < 0.5):
                reps += np.int(np.power(100.0, rng.rand()));

            for i in range(reps):
                logH += self.joint_models.perturb(params)

        else:
            #adjust at least one waveform:
            self.changed_wfs.fill(0)
            randt2 = rng.randn()/np.sqrt(-np.log(rng.rand()));
            chance = np.power(10.0, -3*np.abs(randt2));

            for wf_idx in range(num_waveforms):
                if rng.rand() <= chance:
                     self.changed_wfs[wf_idx] = True
                #make sure one waveform is changed:
            if np.any(self.changed_wfs) == 0:
                self.changed_wfs[rng.randint(num_waveforms)] = 1

            for wf_idx in range(num_waveforms):
                if self.changed_wfs[wf_idx] == 1:
                    logH += self.perturb_wf(params, wf_idx)

            #50% chance to also change detector params
            if(rng.rand() < 0.5):
                reps = 1;
                reps += np.int(np.power(100.0, rng.rand()));
                for i in range(reps):
                    logH += self.joint_models.perturb(params)

        return logH

    def perturb_wf(self, params, wf_idx, ):
        logH = self.wf_models[wf_idx].perturb(  params[self.num_det_params + wf_idx*self.num_wf_params: self.num_det_params + (wf_idx+1)*self.num_wf_params])
        return logH

    def log_likelihood(self, params):
        if np.any(np.isnan(params)): return -np.inf

        return self.fit_manager.calc_likelihood(params)

    def calc_wf_likelihood(self, det_wf_params, wf_idx ):
        try:
            self.joint_models.apply_params(det_wf_params)
        except ValueError:
            return -np.inf

        return self.wf_models[wf_idx].calc_likelihood(det_wf_params[self.num_det_params:])
