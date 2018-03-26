import os, sys
import numpy as np
import numpy.random as rng
import scipy.stats as stats
from scipy import signal
import dnest4
from abc import ABC, abstractmethod

from pygama.waveform import Waveform
from siggen import PPC

from . import VelocityModel, ElectronicsModel, ImpurityModel, ImpurityModelEnds

max_float = sys.float_info.max

class Model(object):
    """
    Specify the model in Python.
    """
    def __init__(self, fit_configuration, fit_manager=None):

        self.conf = fit_configuration
        self.fit_manager = fit_manager

        #Setup detector and waveforms
        self.setup_waveforms(doPrint=False)
        self.setup_detector()

        self.alignidx_guess = self.conf.align_idx
        self.max_maxt = self.alignidx_guess + 5
        self.min_maxt = self.alignidx_guess - 5
        self.maxt_sigma = 1

        self.changed_wfs = np.zeros(self.num_waveforms)

        #Set up all the models...
        self.electronics_model = ElectronicsModel(include_zeros=self.conf.fit_zeros)
        self.velo_model = VelocityModel(include_beta=self.conf.fit_beta)
        self.imp_model = ImpurityModelEnds(self.detector.imp_avg_lims, self.detector.imp_grad_lims, self.detector.detector_length)

        self.tf_first_idx = 0
        self.velo_first_idx = self.tf_first_idx + self.electronics_model.get_num_params()
        self.imp_first_idx = self.velo_first_idx + self.velo_model.get_num_params()
        self.num_det_params = self.imp_first_idx  + self.imp_model.get_num_params()

    def setup_detector(self):
        timeStepSize = self.conf.time_step_calc

        #TODO: wtf is going on with wf length?
        det = PPC( self.conf.siggen_conf_file, wf_padding=500)

        self.detector = det

    def setup_waveforms(self, doPrint=False):
        wfFileName = self.conf.wf_file_name

        if os.path.isfile(wfFileName):
            print("Loading wf file {0}".format(wfFileName))
            data = np.load(wfFileName, encoding="latin1")
            wfs = data['wfs']
            self.wfs = wfs[self.conf.wf_idxs]
            self.num_waveforms = self.wfs.size

            # self.rc1_guess = data['rc1']
            # self.rc2_guess =data['rc2']
            # self.rcfrac_guess =data['rcfrac']

        else:
          print("Saved waveform file %s not available" % wfFileName)
          exit(0)

        baselineLengths = np.zeros(wfs.size)

        for (wf_idx,wf) in enumerate(wfs):
          wf.window_waveform(time_point=self.conf.align_percent, early_samples=self.conf.align_idx, num_samples=self.conf.num_samples)

          if doPrint:
              print( "wf %d length %d (entry %d from run %d)" % (wf_idx, wf.window_length, wf.entry_number, wf.runNumber))
          baselineLengths[wf_idx] = wf.t0_estimate

        #TODO: this doesn't work if the calc step size isn't 1 ns
        self.siggen_wf_length = np.int(  (self.conf.align_idx - np.amin(baselineLengths) + 10)*(10  ))

        self.output_wf_length = np.int( self.conf.num_samples + 1 )

        if doPrint:
            print( "siggen_wf_length will be %d, output wf length will be %d" % (self.siggen_wf_length, self.output_wf_length))


    def draw_position(self, wf_idx):
      r = rng.rand() * self.detector.detector_radius
      z = rng.rand() * self.detector.detector_length

      if not self.detector.IsInDetector(r, 0.1, z):
        return self.draw_position(wf_idx)
      else:
        return (r,z, None)

    def from_prior(self):
        detector=self.detector

        num_waveforms = self.num_waveforms
        rad_arr    = np.empty(num_waveforms)
        phi_arr    = np.empty(num_waveforms)
        theta_arr  = np.empty(num_waveforms)
        scale_arr  = np.empty(num_waveforms)
        t0_arr     = np.empty(num_waveforms)
        smooth_arr = np.empty(num_waveforms)
        p_arr = np.empty(num_waveforms)

        #draw waveform params for each waveform
        for (wf_idx, wf) in enumerate(self.wfs):
            (r,z,t0) = self.draw_position(wf_idx)
            rad = np.sqrt(r**2+z**2)
            theta = np.arctan(z/r)
            smooth_guess = 20
            scale = wf.amplitude

            print("wf {} amplitude: {}".format(wf_idx, scale))

            rad_arr[wf_idx] = rad
            phi_arr[wf_idx] = rng.rand() * np.pi/4
            theta_arr[wf_idx] = theta
            scale_arr[wf_idx] = 20*rng.randn() + scale
            t0_arr[wf_idx] = dnest4.wrap(self.maxt_sigma*rng.randn() + self.alignidx_guess, self.min_maxt, self.max_maxt)
            smooth_arr[wf_idx] = dnest4.wrap(rng.randn() + smooth_guess, 1, 200)

            # if self.conf.smooth_type == "gen_gaus":
            #     p_arr[wf_idx] = dnest4.wrap(10*rng.randn() + 2, 1, 20)
            # elif self.conf.smooth_type == "skew":
            #     p_arr[wf_idx] = rng.rand()*20 - 20

        prior = np.hstack([
              self.electronics_model.get_prior()[:],
              self.velo_model.get_prior()[:],
              self.imp_model.get_prior()[:],
              rad_arr[:], phi_arr[:], theta_arr[:], scale_arr[:], t0_arr[:],smooth_arr[:],
            ])

        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            wf_params = np.zeros((self.num_det_params + 6, self.num_waveforms))
            wfs_param_arr = prior[self.num_det_params:].reshape((6, self.num_waveforms))
            for wf_idx, wf in enumerate(self.wfs):
                wf_params[:self.num_det_params,wf_idx] = prior[:self.num_det_params]
                wf_params[self.num_det_params:,wf_idx] = wfs_param_arr[:,wf_idx]
                model = self.make_waveform( len(wf.windowed_wf), wf_params[:,wf_idx]  )

                p = plt.plot(wf.windowed_wf)
                if model is None: continue
                plt.plot(model, c=p[0].get_color(), ls=":")
            plt.show()
            exit()

        return prior

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
                which = rng.randint(self.num_det_params)
                logH += self.perturb_detector(params, which)

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
                    which = rng.randint(self.num_det_params)
                    logH += self.perturb_detector(params, which)

        return logH

    def perturb_detector(self, params, which):
        old_params = np.copy(params)
        logH = 0.0
        if which >=self.tf_first_idx and which < self.velo_first_idx:
            logH += self.electronics_model.perturb(params[self.tf_first_idx:self.velo_first_idx], which-self.tf_first_idx)
        elif which >= self.velo_first_idx and which < self.imp_first_idx:
            logH += self.velo_model.perturb(params[self.velo_first_idx:self.imp_first_idx], which-self.velo_first_idx)
        elif which >= self.imp_first_idx and which < self.num_det_params:
            logH += self.imp_model.perturb(params[self.imp_first_idx:self.num_det_params], which-self.imp_first_idx)
        elif which >= self.num_det_params:
            raise IndexError("detector which value %d  not supported" % which)

        return logH


    def perturb_wf(self, params, wf_idx, ):
    #do both wf and detector params in case theres strong correlation
        logH = 0.0
        num_waveforms = self.num_waveforms
        detector = self.detector

        wf_params = 6

        reps = 1
        if rng.rand() < 0.5:
            reps += np.int(np.power(100.0, rng.rand()));

        for i in range(reps):
            wf_which = rng.randint(wf_params)

            # my_which = rng.randint(len(priors) + 8)

            # if my_which < len(priors):
            #     #detector variable
            #     logH += self.perturb_detector(params, my_which)
            #
            # else:
            if wf_which < wf_params:
                #this is a waveform variable!
                # wf_which =  np.int(my_which - len(priors))

                #which idx of the global params array
                which = self.num_det_params + wf_which*num_waveforms + wf_idx

                rad_idx = self.num_det_params + wf_idx
                theta_idx =  self.num_det_params + 2*num_waveforms+ wf_idx
                self.changed_wfs[wf_idx] = 1

                if wf_which == 0:
                  theta = params[theta_idx]

                  IsInDetector = 0
                  while not(IsInDetector):
                    new_rad = self.get_new_rad(params[which], theta)
                    r = new_rad * np.cos(theta)
                    z = new_rad * np.sin(theta)
                    IsInDetector = detector.IsInDetector(r, 0,z)

                  params[which] = new_rad

                elif wf_which ==2: #theta
                    rad = params[rad_idx]

                    IsInDetector = 0
                    while not(IsInDetector):
                      new_theta = self.get_new_theta(rad, params[which])
                      r = rad * np.cos(new_theta)
                      z = rad * np.sin(new_theta)
                      IsInDetector = detector.IsInDetector(r, 0,z)
                    params[which] = new_theta

                # if wf_which == 0:
                #     params[which] += (detector.detector_radius)*dnest4.randh()
                #     params[which] = dnest4.wrap(params[which] , 0, detector.detector_radius)
                elif wf_which == 1:
                    max_val = np.pi/4
                    params[which] += np.pi/4*dnest4.randh()
                    params[which] = dnest4.wrap(params[which], 0, max_val)

                elif wf_which == 3: #scale
                    wf_guess = self.wfs[wf_idx].amplitude#self.conf.energy_guess
                    sig = 20

                    logH -= -0.5*((params[which] - wf_guess  )/sig)**2
                    params[which] += sig*dnest4.randh()
                    # params[which] = dnest4.wrap(params[which], wf_guess - 30, wf_guess + 30)
                    logH += -0.5*((params[which] - wf_guess)/sig)**2

                elif wf_which == 4: #t0
                  #gaussian around 0, sigma... 5?
                  t0_sig = self.maxt_sigma
                  logH -= -0.5*((params[which] - self.alignidx_guess )/t0_sig)**2
                  params[which] += t0_sig*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], self.min_maxt, self.max_maxt)
                  logH += -0.5*((params[which] - self.alignidx_guess)/t0_sig)**2

                elif wf_which == 5: #smooth
                  #gaussian around 10
                  smooth_guess = 20
                  sig = 20
                  logH -= -0.5*((params[which] - smooth_guess )/sig)**2
                  params[which] += sig*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], 1, 40)
                  logH += -0.5*((params[which] - smooth_guess)/sig)**2

            else:
                raise IndexError( "wf which value %d (which value %d) not supported" % (wf_which, which) )
                exit(0)

        return logH

    def get_new_rad(self,rad, theta):
          detector = self.detector
          #FIND THE MAXIMUM RADIUS STILL INSIDE THE DETECTOR
          theta_eq = np.arctan(detector.detector_length/detector.detector_radius)
          theta_taper = np.arctan(detector.taper_length/detector.detector_radius)
          if theta <= theta_taper:
             z = np.tan(theta)*(detector.detector_radius - detector.taper_length) / (1-np.tan(theta))
             max_rad = z / np.sin(theta)
          elif theta <= theta_eq:
              max_rad = detector.detector_radius / np.cos(theta)
          else:
              theta_comp = np.pi/2 - theta
              max_rad = detector.detector_length / np.cos(theta_comp)

          #AND THE MINIMUM (from PC dimple)
          #min_rad  = 1./ ( np.cos(theta)**2/detector.pcRad**2  +  np.sin(theta)**2/detector.pcLen**2 )

          min_rad = 5#np.amax([detector.pcRad, detector.pcLen])

          new_rad = rad + (max_rad - min_rad)*dnest4.randh()
          new_rad = dnest4.wrap(new_rad, min_rad, max_rad)
          return new_rad
    def get_new_theta(self,rad,theta):
        detector = self.detector
        if rad < np.amin([detector.detector_radius - detector.taper_length, detector.detector_length]):
            max_val = np.pi/2
            min_val = 0
        else:
            if rad < detector.detector_radius - detector.taper_length:
                #can't possibly hit the taper
                min_val = 0
            elif rad < np.sqrt(detector.detector_radius**2 + detector.taper_length**2):
                #low enough that it could hit the taper region
                a = detector.detector_radius - detector.taper_length
                z = 0.5 * (np.sqrt(2*rad**2-a**2) - a)
                min_val = np.arcsin(z/rad)
            else:
                #longer than could hit the taper
                min_val = np.arccos(detector.detector_radius/rad)

            if rad < detector.detector_length:
                max_val = np.pi/2
            else:
                max_val = np.pi/2 - np.arccos(detector.detector_length/rad)

        new_theta = theta + (max_val - min_val)*dnest4.randh()
        new_theta = dnest4.wrap(new_theta, min_val, max_val)
        return new_theta

    def log_likelihood(self, params):
        if np.any(np.isnan(params)): return -np.inf
        return self.fit_manager.calc_likelihood(params)

    def calc_wf_likelihood(self, wf_params, wf_idx ):
        wf = self.wfs[wf_idx]
        data = wf.windowed_wf
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

    def make_waveform(self, data_len, wf_params, charge_type=None):
        try:
            self.electronics_model.apply_to_detector(wf_params[self.tf_first_idx: self.velo_first_idx], self.detector)
            self.velo_model.apply_to_detector(wf_params[self.velo_first_idx: self.imp_first_idx], self.detector)
            self.imp_model.apply_to_detector(wf_params[self.imp_first_idx: self.num_det_params], self.detector)
        except ValueError:
            return None

        rad, phi, theta,  scale, maxt, smooth =  wf_params[self.num_det_params:]

        r = rad * np.cos(theta)
        z = rad * np.sin(theta)

        if scale < 0:
            raise ValueError("Scale should not be below 0 (value {})".format(scale))
            return None
        if smooth < 0:
            raise ValueError("Smooth should not be below 0 (value {})".format(smooth))
            return None
        if not self.detector.IsInDetector(r, phi, z):
            raise ValueError("Point {},{},{} is outside detector.".format(r,phi,z))
            return None

        # grad = wf_params[grad_idx]
        # avg_imp = wf_params[imp_avg_idx]
        # self.detector.siggenInst.SetImpurityAvg(avg_imp,grad)

        if charge_type is None:
                model = self.detector.MakeSimWaveform(r, phi, z, scale, maxt, self.conf.align_percent, data_len, smoothing=smooth)
                # model = self.detector.GetWaveform(r, phi, z, scale)
        elif charge_type == 1:
            model = self.detector.MakeRawSiggenWaveform(r, phi, z,1)
        elif charge_type == -1:
            model = self.detector.MakeRawSiggenWaveform(r, phi, z,-1)
        else:
            raise ValueError("Not a valid charge type! {0}".format(charge_type))

        if model is None or np.any(np.isnan(model)):
            return None

        return model

    def get_indices(self):
        return (self.tf_first_idx, self.num_det_params)

    def get_indices(self):
        return (self.tf_first_idx, self.velo_first_idx, 0, 0)
