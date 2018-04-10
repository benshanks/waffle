import os, sys
import numpy as np
import numpy.random as rng
import scipy.stats as stats
from scipy import signal
import dnest4

from pygama.waveform import Waveform
from siggen import PPC

from ._parameterbase import ModelBaseClass, Parameter

max_float = sys.float_info.max

class WaveformModel(ModelBaseClass):
    """
    Specify the model in Python.
    """
    def __init__(self, target_wf, align_percent, detector, align_idx=125, do_smooth=True, smoothing_type="gauss"):

        self.detector = detector

        self.target_wf = target_wf
        self.align_percent = align_percent

        self.align_sigma = 1
        self.align_idx = align_idx

        self.params = [
            Parameter("r", "uniform", lim_lo=0, lim_hi=detector.detector_radius),
            Parameter("z", "uniform", lim_lo=0, lim_hi=detector.detector_length),
            Parameter("phi", "uniform", lim_lo=0, lim_hi=np.pi/4),
            Parameter("scale", "gaussian", mean=target_wf.amplitude, variance=20, lim_lo=0.5*target_wf.amplitude, lim_hi=1.5*target_wf.amplitude),
            Parameter("t_align", "gaussian", mean=self.align_idx, variance=self.align_sigma, lim_lo=self.align_idx-5, lim_hi=self.align_idx+5),
        ]

        self.do_smooth=do_smooth
        self.smoothing_type = smoothing_type
        if do_smooth:
            if smoothing_type == "gaussian":
                smooth_guess = 20
                self.params.append(Parameter("smooth", "gaussian", mean=smooth_guess, variance=10, lim_lo=1, lim_hi=100))
            elif smoothing_type == "skew":
                self.detector.smoothing_type=1
                smooth_guess = 20
                skew_guess = 0
                self.params.append(Parameter("smooth", "gaussian", mean=smooth_guess, variance=10, lim_lo=1, lim_hi=100))
                self.params.append(Parameter("skew", "gaussian", mean=skew_guess, variance=5, lim_lo=-np.inf, lim_hi=np.inf))

        self.num_params = len(self.params)

    def draw_position(self, wf_idx):
      r = rng.rand() * self.detector.detector_radius
      z = rng.rand() * self.detector.detector_length

      if not self.detector.IsInDetector(r, 0.1, z):
        return self.draw_position(wf_idx)
      else:
        return (r,z)

    def perturb(self, params):
        logH = 0

        reps = 1
        if rng.rand() < 0.5:
            reps += np.int(np.power(100.0, rng.rand()));

        for i in range(reps):
            which = rng.randint(self.num_params)
            logH += self.perturb_param(params, which)
        return logH

    def perturb_param(self, params, which):
        #we need to treat (r,z) special.  anything else, just let it roll like normal.
        logh = super().perturb(params, which)
        if which <2:
            r=params[0]
            z=params[1]
            if not self.detector.IsInDetector(r, 0.1, z):
                return self.perturb_param(params, which)
        return logh


    def get_prior(self):
        prior = super().get_prior()
        r=prior[0]
        z=prior[1]
        if not self.detector.IsInDetector(r, 0.1, z):
            return self.get_prior()
        return prior

    def make_waveform(self, data_len, wf_params, charge_type=None):
        r, z, phi, scale, maxt =  wf_params[:5]

        smooth = None
        skew=None
        if self.do_smooth:
            smooth = wf_params[5]
            if smooth < 0:
                raise ValueError("Smooth should not be below 0 (value {})".format(smooth))
            if self.smoothing_type == "skew":
                skew = wf_params[6]

        # r = rad * np.cos(theta)
        # z = rad * np.sin(theta)

        if scale < 0:
            raise ValueError("Scale should not be below 0 (value {})".format(scale))

        if not self.detector.IsInDetector(r, phi, z):
            raise ValueError("Point {},{},{} is outside detector.".format(r,phi,z))

        if charge_type is None:
                model = self.detector.MakeSimWaveform(r, phi, z, scale, maxt, self.align_percent, data_len, smoothing=smooth, skew=skew)
                # model = self.detector.GetWaveform(r, phi, z, scale)
        elif charge_type == 1:
            model = self.detector.MakeWaveform(r, phi, z,1)[0,:]
        elif charge_type == -1:
            model = self.detector.MakeWaveform(r, phi, z,-1)[0,:]
        else:
            raise ValueError("Not a valid charge type! {0}".format(charge_type))

        if model is None or np.any(np.isnan(model)):
            return None

        # if self.conf.decimate_decay_idx is not None:
        #     model = np.concatenate(( model[:decimate_decay_idx], model[decimate_decay_idx::dec_factor]))

        return model

    def calc_likelihood(self, wf_params):
        data = self.target_wf.windowed_wf
        # model_err = 0.57735027 * wf.baselineRMS
        model_err = 2.5 #TODO: get this from the waveform itself
        data_len = len(data)
        model = self.make_waveform(data_len, wf_params, )

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

    # def get_new_rad(self,rad, theta):
    #       detector = self.detector
    #       #FIND THE MAXIMUM RADIUS STILL INSIDE THE DETECTOR
    #       theta_eq = np.arctan(detector.detector_length/detector.detector_radius)
    #       theta_taper = np.arctan(detector.taper_length/detector.detector_radius)
    #       if theta <= theta_taper:
    #          z = np.tan(theta)*(detector.detector_radius - detector.taper_length) / (1-np.tan(theta))
    #          max_rad = z / np.sin(theta)
    #       elif theta <= theta_eq:
    #           max_rad = detector.detector_radius / np.cos(theta)
    #       else:
    #           theta_comp = np.pi/2 - theta
    #           max_rad = detector.detector_length / np.cos(theta_comp)
    #
    #       #AND THE MINIMUM (from PC dimple)
    #       #min_rad  = 1./ ( np.cos(theta)**2/detector.pcRad**2  +  np.sin(theta)**2/detector.pcLen**2 )
    #
    #       min_rad = 5#np.amax([detector.pcRad, detector.pcLen])
    #
    #       new_rad = rad + (max_rad - min_rad)*dnest4.randh()
    #       new_rad = dnest4.wrap(new_rad, min_rad, max_rad)
    #       return new_rad
    # def get_new_theta(self,rad,theta):
    #     detector = self.detector
    #     if rad < np.amin([detector.detector_radius - detector.taper_length, detector.detector_length]):
    #         max_val = np.pi/2
    #         min_val = 0
    #     else:
    #         if rad < detector.detector_radius - detector.taper_length:
    #             #can't possibly hit the taper
    #             min_val = 0
    #         elif rad < np.sqrt(detector.detector_radius**2 + detector.taper_length**2):
    #             #low enough that it could hit the taper region
    #             a = detector.detector_radius - detector.taper_length
    #             z = 0.5 * (np.sqrt(2*rad**2-a**2) - a)
    #             min_val = np.arcsin(z/rad)
    #         else:
    #             #longer than could hit the taper
    #             min_val = np.arccos(detector.detector_radius/rad)
    #
    #         if rad < detector.detector_length:
    #             max_val = np.pi/2
    #         else:
    #             max_val = np.pi/2 - np.arccos(detector.detector_length/rad)
    #
    #     new_theta = theta + (max_val - min_val)*dnest4.randh()
    #     new_theta = dnest4.wrap(new_theta, min_val, max_val)
    #     return new_theta
