import os, sys
import numpy as np
from scipy import signal

from ._parameterbase import ModelBaseClass, Parameter

class ElectronicsModel2(ModelBaseClass):
    """
    Specify the model in Python.
    """
    def __init__(self,timestep=1E-9):
        self.num_params = 4
        self.timestep=timestep

        #pretty good starting point for MJD detectors
        pole_mag = 2.57E7
        pole_phi = 145 * np.pi/180

        rc_mag = -6
        rc_phi = -5

        self.params = [
            Parameter("pole_mag", "gaussian", pole_mag, 1E7, 1E5, 0.5E9),
            Parameter("pole_phi", "uniform", lim_lo=(2./3)*np.pi, lim_hi=np.pi),
            Parameter("rc_mag", "gaussian", rc_mag, 5, lim_lo=-10, lim_hi=0),
            Parameter("rc_phi", "gaussian", rc_phi, 5, lim_lo=-10, lim_hi=0),
            # Parameter("zmag_h", "gaussian", 1, 5, lim_lo=0, lim_hi=1E9),
            # Parameter("zphi_h", "uniform", lim_lo=0, lim_hi=np.pi),
            # Parameter("zmag_l", "gaussian", 1, 5, lim_lo=0, lim_hi=1E9),
            # Parameter("zphi_l", "uniform", lim_lo=0, lim_hi=np.pi),
        ]

    def zpk_to_ba(self, pole,phi):
        return [1, -2*pole*np.cos(phi), pole**2]

    def apply_to_detector(self, params, detector):
        pmag, pphi, rc_mag, rc_phi  = params[:]
        # pmag, pphi, rc_mag, rc_phi, zmag_h,zphi_h,zmag_l,zphi_l    = params[:]

        # detector.SetTransferFunctionRC(rc1, rc2, rcfrac, digFrequency=1./self.timestep )
        detector.hp_num = [1,-2,1]
        # detector.hp_num = self.zpk_to_ba(zmag_h, zphi_h)
        detector.hp_den = self.zpk_to_ba(1. - 10.**rc_mag, np.pi * 10.**rc_phi)
        # print(detector.hp_den)

        dig = self.timestep
        (__, detector.lp_den) = signal.zpk2tf([],
                [ np.exp(dig*pmag * np.exp(pphi*1j)), np.exp(dig*pmag * np.exp(-pphi*1j))   ],1.)
        detector.lp_num = [1,2,1]
        # detector.lp_num = self.zpk_to_ba(zmag_l, zphi_l)

class ElectronicsModel(ModelBaseClass):
    """
    Specify the model in Python.
    """
    def __init__(self,timestep=1E-9):
        self.num_params = 5
        self.timestep=timestep

        #pretty good starting point for MJD detectors
        pole_mag = 2.57E7
        pole_phi = 145 * np.pi/180
        rc1 = 72
        rc2 = 2
        rcfrac = 0.995

        self.params = [
            Parameter("pole_mag", "gaussian", pole_mag, 1E7, 1E5, 0.5E9),
            Parameter("pole_phi", "uniform", lim_lo=(2./3)*np.pi, lim_hi=np.pi),
            Parameter("rc1", "gaussian", rc1, 5, lim_lo=65, lim_hi=100),
            Parameter("rc2", "gaussian", rc2, 0.25, lim_lo=0, lim_hi=10),
            Parameter("rcfrac", "gaussian", rcfrac, 0.01, lim_lo=0.99, lim_hi=1),
        ]

    def apply_to_detector(self, params, detector):
        pmag, pphi, rc1, rc2, rcfrac  = params[:]

        # detector.lp_num = [ 1.]
        # detector.lp_den = [ 1.,-1.95933813 ,0.95992564]
        # detector.hp_num = [1.0, -1.999640634643256, 0.99964063464325614]
        # detector.hp_den = [1, -1.9996247480008278, 0.99962475299714171]

        detector.SetTransferFunctionRC(rc1, rc2, rcfrac, digFrequency=1./self.timestep )
        dig = self.timestep
        (detector.lp_num, detector.lp_den) = signal.zpk2tf([],
                [ np.exp(dig*pmag * np.exp(pphi*1j)), np.exp(dig*pmag * np.exp(-pphi*1j))   ],1.)
