import os, sys
import numpy as np
import numpy.random as rng
from scipy import signal
from scipy.interpolate import interp1d
from copy import copy
import dnest4

from pygama.waveform import Waveform

from ._parameterbase import ModelBaseClass, Parameter, JointModelBase
from ._model_bundle import JointModelBundle

class PulserEnergyModel(JointModelBase):
    def __init__(self, energy_guess):
        self.params = [
            Parameter("energy", "gaussian", energy_guess, 0.1*energy_guess, lim_lo=0.5*energy_guess, lim_hi=1.5*energy_guess),
        ]
    def apply_to_detector(self, params, detector):
        detector.energy = params

class PulserRiseTimeModel(JointModelBase):
    def __init__(self, rt_guess):
        self.params = [
            Parameter("risetime", "gaussian", rt_guess, rt_guess, lim_lo=0, lim_hi=10*rt_guess),
        ]
    def apply_to_detector(self, params, detector):
        detector.rise_time = params

class PulserGenerator(object):

    def __init__(self, wf_length):
        self.digital_filters = []
        self.output_wf = np.zeros(wf_length)
        self.interpType = "cubic"
        self.energy = 1
        self.rise_time=0

    def AddDigitalFilter(self, filter):
        self.digital_filters.append(filter)

    def make_pulser(self, align_point, align_adc, outputLength):
        energy = self.energy

        data_to_siggen_size_ratio = 10

        #make a 1E9 sampled waveform
        temp_wf_sig = np.zeros(20*outputLength)

        if self.rise_time == 0:
            temp_wf_sig[100:] = energy
        else:
            rise_time = self.rise_time
            rise_time = int(np.around(rise_time))
            temp_wf_sig[100:100+rise_time] = np.linspace(0, energy, rise_time)
            temp_wf_sig[100+rise_time:] = energy

        for filter in self.digital_filters:
            temp_wf_sig = filter.apply_to_signal(temp_wf_sig)

        #zero append a bunch to make sure i have enough zeros
        temp_wf_sig = np.concatenate( (np.zeros(10*outputLength), temp_wf_sig))

        smax_idx = np.argmax(temp_wf_sig)
        smax = temp_wf_sig[smax_idx]
        if smax == 0:
          print("bad smax")
          return None

        #linear interpolation to find the alignPointIdx: find the align percentage in the simualted array
        first_idx = np.argmax(temp_wf_sig[:smax_idx+1] > align_adc) - 1

        if first_idx+1 == len(temp_wf_sig) or first_idx <0:
            # print("bad first idx")
            #
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(temp_wf_sig)
            # plt.show()
            # exit()
            return None

        #linear interpolation as to where the true siggen align_percentage is (in the siggen wf)
        slope = (temp_wf_sig[first_idx+1] - temp_wf_sig[first_idx]) / 1.
        siggen_offset = ( align_adc -  temp_wf_sig[first_idx] ) / slope
        siggen_align = siggen_offset + first_idx

        #where (in the data wf) do we want to align?
        align_point_ceil = np.int( np.ceil(align_point) )

        start_idx_sig = siggen_align - align_point*data_to_siggen_size_ratio


        #TODO: i only really _need_ to interp between like every data_to_siggen_size_ratio sample or something right?
        self.siggen_interp_fn = interp1d(np.arange(len(temp_wf_sig)), temp_wf_sig, kind=self.interpType, copy="False", assume_sorted="True")

        num_samples_to_fill = outputLength
        offset = (align_point_ceil - align_point)*data_to_siggen_size_ratio

        sampled_idxs = np.arange(num_samples_to_fill)*data_to_siggen_size_ratio + start_idx_sig #+ siggen_offset #+ offset +

        if sampled_idxs[0] < 0 or sampled_idxs[-1] > len(temp_wf_sig) - 1:
            return None

        self.output_wf.fill(0.)

        try:
            coarse_vals =   self.siggen_interp_fn(sampled_idxs)
            self.output_wf[:num_samples_to_fill] = coarse_vals
        except ValueError:
            print( len(self.output_wf) )
            print( num_samples_to_fill)
            print( sampled_idxs)
            exit(0)

        return self.output_wf[:outputLength]


class PulserModel(ModelBaseClass):
    """
    Specify the model in Python.
    """
    def __init__(self, target_wf, pulser_gen, align_adc, align_idx=125, include_energy=True):
        self.target_wf = target_wf
        self.align_adc = align_adc
        self.pg = pulser_gen
        self.include_energy = include_energy

        self.align_sigma = 1
        self.align_idx = align_idx

        self.params = [
            Parameter("t_align", "gaussian", mean=self.align_idx, variance=self.align_sigma, lim_lo=self.align_idx-5, lim_hi=self.align_idx+5),
        ]

        if include_energy:
            wf_max = np.amax(target_wf.windowed_wf)
            self.params.append(
                Parameter("energy", "gaussian", mean=wf_max, variance=0.1*wf_max, lim_lo=0.5*wf_max, lim_hi=1.5*wf_max),
            )


    def perturb(self, params):
        logH = 0

        reps = 1
        if rng.rand() < 0.5:
            reps += np.int(np.power(100.0, rng.rand()));

        for i in range(reps):
            which = rng.randint(self.num_params)
            logH += self.perturb_param( params, which)
        return logH

    def perturb_param(self, params, which):
        #we need to treat (r,z) special.  anything else, just let it roll like normal.
        logh = super().perturb(params, which)
        return logh

    def make_waveform(self, data_len, wf_params, charge_type=None):
        if self.include_energy:
            t_align, energy =  wf_params
            self.pg.energy = energy
        else:
            assert len(wf_params) == 1
            t_align =  wf_params[:]

        pulser_model = self.pg.make_pulser(t_align, self.align_adc,  data_len)

        return pulser_model


    def calc_likelihood(self, wf_params):
        data = self.target_wf.windowed_wf
        # model_err = 0.57735027 * wf.baselineRMS
        model_err = 2.5 #TODO: get this from the waveform itself
        data_len = len(data)
        model = self.make_waveform(data_len, wf_params)

        if model is None:
            ln_like = -np.inf
        else:
            inv_sigma2 = 1.0/(model_err**2)
            ln_like = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

        return ln_like

    def log_likelihood(self, params):
        return self.calc_likelihood(params)
    def from_prior(self):
        return self.get_prior()

class PulserTrainingModel(object):

    def __init__(self, fit_configuration, fit_manager=None):

        self.conf = fit_configuration
        model_conf = copy(self.conf.model_conf)
        self.fit_manager = fit_manager

        #Setup detector and waveforms
        self.pg = PulserGenerator(self.conf.wf_conf["num_samples"])
        self.pg.interpType = fit_configuration.interpType
        self.wf_models = []
        self.setup_waveforms(self.conf.wf_config, doPrint=False)

        self.changed_wfs = np.zeros(self.num_waveforms)

        #Set up all the models...
        if self.conf.joint_energy:
            model_conf.append((PulserEnergyModel, {"energy_guess":self.wfs[0].windowed_wf.max()} ) )

        if self.conf.joint_risetime:
            model_conf.append((PulserRiseTimeModel, {"rt_guess":5} ) )

        self.joint_models = JointModelBundle(model_conf, self.pg)
        self.num_det_params = self.joint_models.num_params

    def setup_waveforms(self, wf_conf, doPrint=False):
        wfFileName = wf_conf.wf_file_name

        if os.path.isfile(wfFileName):
            print("Loading wf file {0}".format(wfFileName))
            data = np.load(wfFileName, encoding="latin1")
            wfs = data['wfs']
            self.wfs = wfs[wf_conf.wf_idxs]
            self.num_waveforms = self.wfs.size

        else:
          print("Saved waveform file %s not available" % wfFileName)
          exit(0)


        for (wf_idx,wf) in enumerate(self.wfs):
          total_samples = wf_conf.num_samples
          full_samples = wf_conf.align_idx
          wf.window_waveform(time_point=wf_conf.align_percent, early_samples=wf_conf.align_idx, num_samples=total_samples, method="value")

          self.wf_models.append(PulserModel(wf, self.pg, align_adc=wf_conf.align_percent, align_idx = wf_conf.align_idx, include_energy=(not self.conf.joint_energy)))

        self.output_wf_length = np.int( wf_conf.num_samples + 1 )

        self.num_wf_params = self.wf_models[0].num_params

        if doPrint:
            print( "siggen_wf_length will be %d, output wf length will be %d" % (self.siggen_wf_length, self.output_wf_length))

    def from_prior(self):
        wf_params = np.concatenate([ wf.get_prior()[:] for wf in self.wf_models ])
        prior = np.concatenate([
              self.joint_models.get_prior(), wf_params
            ])

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
