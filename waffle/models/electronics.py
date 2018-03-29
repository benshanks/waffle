import os, sys
import numpy as np
from scipy import signal

from ._parameterbase import JointModelBase, Parameter

class ElectronicsModel(JointModelBase):
    """
    2-pole digital filter for both HP and LP halves
    """
    def __init__(self,include_zeros=True, order_number=2):
        self.include_zeros = include_zeros
        self.order_number = order_number

        self.params = [
            #I know from experience that the lowpass poles are near (0,1)
            #(makes sense cause the amplitude response should fall off near nyquist freq)
            #just go ahead and shove the priors up near there
            Parameter("pole_mag", "uniform", lim_lo=0.9, lim_hi=1),
            Parameter("pole_phi", "uniform", lim_lo=0, lim_hi=0.1),
            # Parameter("rc_mag", "uniform", lim_lo=0, lim_hi=1),
            # Parameter("rc_phi", "uniform", lim_lo=0, lim_hi=np.pi),
            Parameter("rc_mag", "uniform", lim_lo=-10, lim_hi=-1),
            Parameter("rc_phi", "uniform", lim_lo=-10, lim_hi=-1),
        ]

        if include_zeros:
            self.params.append(
                Parameter("lp_zeromag", "uniform", lim_lo=0, lim_hi=10))
            self.params.append(
                Parameter("lp_zerophi", "uniform", lim_lo=0, lim_hi=np.pi))

        if order_number == 4:
            self.params.append(
                Parameter("pole_mag2", "uniform", lim_lo=0, lim_hi=1))
            self.params.append(    
                Parameter("pole_phi2", "uniform", lim_lo=0, lim_hi=np.pi)
                )

        self.num_params = len(self.params)

    def zpk_to_ba(self, pole,phi):
        return [1, -2*pole*np.cos(phi), pole**2]

    def apply_to_detector(self, params, detector):
        if self.include_zeros:
            pmag, pphi, rc_mag, rc_phi, lp_zeromag, lp_zerophi   = params[:]
            detector.lp_num = self.zpk_to_ba(lp_zeromag, lp_zerophi)
            if np.sum(detector.lp_num) == 0:
                raise ValueError("Zero sum low pass denominator!")
            detector.lp_den = self.zpk_to_ba(pmag, pphi)

        elif self.order_number == 4 :
            pmag, pphi, rc_mag, rc_phi, pmag2, pphi2   = params[:]
            den1 = self.zpk_to_ba(pmag, pphi)
            den2 = self.zpk_to_ba(pmag2, pphi2)
            detector.lp_num = [[1,2,1], [1,2,1]]
            detector.lp_den = [den1, den2]
            detector.lp_order = 4
        else:
            pmag, pphi, rc_mag, rc_phi   = params[:]
            detector.lp_num = [1,2,1]
            detector.lp_den = self.zpk_to_ba(pmag, pphi)

        detector.hp_num = [1,-2,1]
        # detector.hp_den = self.zpk_to_ba(rc_mag, rc_phi)
        detector.hp_den = self.zpk_to_ba(1. - 10.**rc_mag, 10.**rc_phi)


class ElectronicsModel_old(JointModelBase):
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
