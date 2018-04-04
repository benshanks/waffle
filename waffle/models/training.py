import os, sys
import numpy as np
import numpy.random as rng
import scipy.stats as stats
from scipy import signal
import dnest4
from abc import ABC, abstractmethod

from pygama.waveform import Waveform
from siggen import PPC

from . import VelocityModel, LowPassFilterModel, HiPassFilterModel, ImpurityModel, ImpurityModelEnds, WaveformModel

max_float = sys.float_info.max

class JointModelBundle(object):
    def __init__(self, conf, detector):
        self.conf =conf
        self.detector = detector

        self.models = []
        self.index_map = {}
        self.start_map = {}
        self.name_map={}

        self.num_params = 0
        i=0
        for model_idx, model_name in enumerate(conf["model_list"]):
            model = self.append(model_name)
            self.num_params += model.num_params
            self.start_map[model_idx] = i
            self.name_map[model_name] = model_idx
            model.start_idx = i

            for j in range(model.num_params):
                self.index_map[i] = model_idx
                i +=1

    def append(self, model_name):
        #TODO: surely this can be done with introspection
        if model_name=="VelocityModel":
            model = VelocityModel(include_beta=self.conf["fit_beta"])
        elif model_name=="ImpurityModelEnds":
            model = ImpurityModelEnds(self.detector.imp_avg_lims, self.detector.imp_grad_lims, self.detector.detector_length)
        elif model_name == "HiPassFilterModel":
            model = HiPassFilterModel(order=self.conf["hp_order"])
        elif model_name == "LowPassFilterModel":
            model = LowPassFilterModel(order=self.conf["lp_order"], include_zeros=self.conf["lp_zeros"])

        self.models.append(model)
        return model

    def get_prior(self):
        priors = np.array([])
        for model in self.models:
            model_prior = model.get_prior()
            priors = np.concatenate((priors, model_prior))
        return priors

    def perturb(self, params):
        which = rng.randint(self.num_params)
        logH = 0.0

        #find model corresponding to "which"
        model_number = self.index_map[which]
        model = self.models[model_number]
        start_idx = self.start_map[model_number]

        logH += model.perturb(params[start_idx:start_idx+model.num_params], which-start_idx)

        return logH

    def apply_params(self, params):
        for (model_num, model) in enumerate(self.models):
            start_idx = self.start_map[model_num]
            model.apply_to_detector(params[start_idx:start_idx+model.num_params], self.detector)

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

          self.wf_models.append(WaveformModel(wf, align_percent=wf_conf.align_percent, detector=self.detector))

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
