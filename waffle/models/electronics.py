import os, sys
import numpy as np
from scipy import signal

from ._parameterbase import JointModelBase, Parameter
from siggen.electronics import DigitalFilter, GretinaOvershootFilter

class DigitalFilterBase(JointModelBase):
    def __init__(self, detector, order, include_zeros, pmag_lims, pphi_lims):
        assert (order==1 or order==2)
        self.order = order
        self.include_zeros = include_zeros

        self.digital_filter = DigitalFilter(order)
        detector.AddDigitalFilter(self.digital_filter)

        freq_samp = 1E9
        self.nyq_freq = 0.5*freq_samp

        if self.order == 1:
            self.params = [
                Parameter("pole_mag", "uniform", lim_lo=pmag_lims[0], lim_hi=pmag_lims[-1]),
            ]

        elif self.order == 2:
            self.params = [
                #I know from experience that the lowpass poles are near (0,1)
                #(makes sense cause the amplitude response should fall off near nyquist freq)
                #just go ahead and shove the priors up near there
                Parameter("pole_mag", "uniform", lim_lo=pmag_lims[0], lim_hi=pmag_lims[-1]),
                Parameter("pole_phi", "uniform", lim_lo=pphi_lims[0], lim_hi=pphi_lims[-1]),
            ]

    def exp_magphi(self, mag, phi=0):
        if phi == 0: return [1. - 10.**mag]
        return (1. - 10.**mag, np.pi**phi)

    def get_pz(self, params):
        self.apply_to_detector(params, None)

        return_dict = {
            "poles":[],
            "zeros":[]
        }

        try:
            return_dict["poles"].append(self.digital_filter.poles)
        except AttributeError:
            pass

        try:
            return_dict["zeros"].append(self.digital_filter.zeros)
        except AttributeError:
            pass


        return return_dict

    def apply_to_detector(self, params, detector):
        pole_params = params[:self.order]
        if self.exp: pole_params = self.exp_magphi(*pole_params)

        self.digital_filter.set_poles(*pole_params)

        if self.include_zeros:
            zero_params = params[self.order:self.order+self.order]
            self.digital_filter.set_zeros(*zero_params)

    def get_freqz(self, params, w):
        self.apply_to_detector(params, None)
        # w=np.logspace(-15, -5, 500, base=np.pi)

        num, den = self.digital_filter.num, self.digital_filter.den

        if np.sum(num) != 0: num /= np.sum(num)/np.sum(den)
        w, h = signal.freqz(num, den, worN=w )
        w/= (np.pi /self.nyq_freq)
        return w, h


class LowPassFilterModel(DigitalFilterBase):
    def __init__(self,detector, order=2, include_zeros=False, pmag_lims=None, pphi_lims=None):
        if pmag_lims is None: pmag_lims = [0.9,1]
        if pphi_lims is None: pphi_lims = [0,0.1]

        super().__init__(detector, order, include_zeros, pmag_lims, pphi_lims)
        self.exp = False

        if order == 1:
            if self.include_zeros:
                self.params.append(
                    Parameter("zero_mag", "uniform", lim_lo=-10, lim_hi=10)
                )
            else:
                self.digital_filter.num = [1,1]
        else:
            if include_zeros:
                self.params.append(
                    Parameter("zero_mag", "uniform", lim_lo=0.9, lim_hi=2))
                self.params.append(
                    Parameter("zero_phi", "uniform", lim_lo=0, lim_hi=0.5)
                )
            else:
                self.digital_filter.num = [1,2,1]

class HiPassFilterModel(DigitalFilterBase):
    def __init__(self, detector, order=2, include_zeros=False, pmag_lims=None, pphi_lims=None):
        self.exp = True
        if pmag_lims is None: pmag_lims = [-6,-1]
        if pphi_lims is None: pphi_lims = [-20,-2]

        super().__init__(detector, order, include_zeros, pmag_lims, pphi_lims)

        if order == 1:
            if self.include_zeros:
                self.params.append(
                    Parameter("zero_mag", "uniform", lim_lo=-10, lim_hi=10)
                )
            else:
                self.digital_filter.num = [1,-1]
        else:
            if include_zeros:
                self.params.append(
                    Parameter("zero_mag", "uniform", lim_lo=0.9, lim_hi=2))
                self.params.append(
                    Parameter("zero_phi", "uniform", lim_lo=0, lim_hi=0.5)
                )
            else:
                self.digital_filter.num = [1,-2,1]


class OvershootFilterModel(JointModelBase):
    def __init__(self, detector):

        self.params = [
            Parameter("pole", "gaussian", 2, 2, lim_lo=0, lim_hi=100),
            Parameter("fraction", "uniform", lim_lo=0, lim_hi=0.25)
        ]
        self.digital_filter = GretinaOvershootFilter(order=1)
        self.digital_filter.num = [1,-1]
        detector.AddDigitalFilter(self.digital_filter)

    def apply_to_detector(self, params, detector):
        freq = 1E9

        rc1_us, overshoot_frac  = params

        rc1_dig= 1E-6 * (rc1_us) * freq
        rc1_exp = np.exp(-1./rc1_dig)

        self.digital_filter.den = [1, -rc1_exp]
        self.digital_filter.overshoot_frac = overshoot_frac

# class ElectronicsModel(JointModelBase):
#     """
#     2-pole digital filter for both HP and LP halves
#     """
#     def __init__(self,order=2):
#         self.order = order
#
#         self.params = [
#             # Parameter("rc_mag", "uniform", lim_lo=0, lim_hi=1),
#             # Parameter("rc_phi", "uniform", lim_lo=0, lim_hi=np.pi),
#             Parameter("rc_mag", "uniform", lim_lo=-10, lim_hi=-1),
#             Parameter("rc_phi", "uniform", lim_lo=-10, lim_hi=-1),
#         ]
#
#         if include_zeros:
#             self.params.append(
#                 Parameter("lp_zeromag", "uniform", lim_lo=0, lim_hi=10))
#             self.params.append(
#                 Parameter("lp_zerophi", "uniform", lim_lo=0, lim_hi=np.pi))
#
#         if lp_order == 4:
#             self.params.append(
#                 Parameter("pole_mag2", "uniform", lim_lo=0, lim_hi=1))
#             self.params.append(
#                 Parameter("pole_phi2", "uniform", lim_lo=0, lim_hi=np.pi)
#                 )
#
#         if hp_order == 4:
#             self.params.append(
#                 Parameter("hp_mag2", "uniform", lim_lo=0, lim_hi=1))
#             self.params.append(
#                 Parameter("hp_phi2", "uniform", lim_lo=0, lim_hi=np.pi)
#                 )
#
#
#     def get_freqz(self, params):
#
#         w=np.logspace(-13, 0, 500, base=np.pi)
#
#         if self.lp_order == 2 :
#             if self.include_zeros:
#                 pmag, pphi, rc_mag, rc_phi, lp_zeromag, lp_zerophi   = params[:]
#                 num = self.zpk_to_ba(lp_zeromag, lp_zerophi)
#             else:
#                 pmag, pphi, rc_mag, rc_phi   = params[:]
#                 num = [1,2,1]
#             den = self.zpk_to_ba(pmag, pphi)
#             num /= (np.sum(num)/np.sum(den))
#             w, h = signal.freqz(num, den, worN=w )
#
#         elif self.order_number == 4 :
#             pmag, pphi, rc_mag, rc_phi, pmag2, pphi2   = params[:]
#             den1 = self.zpk_to_ba(pmag, pphi)
#             den2 = self.zpk_to_ba(pmag2, pphi2)
#             num = [1,2,1]
#
#             num1 = num/(np.sum(num)/np.sum(den1))
#             num2 = num/(np.sum(num)/np.sum(den2))
#
#             w, h1 = signal.freqz(num1, den1, worN=w)
#             w, h2 = signal.freqz(num2, den2, worN=w)
#
#             h = h1*h2
#
#
#
#         den_rc = em.zpk_to_ba(1. - 10.**rc_mag, 10.**rc_phi)
#         num_rc = [1,-2,1]
#         w_rc, h_rc = signal.freqz(num_rc, den_rc, worN=w, )
#
#         w/= (np.pi /nyq_freq)
#
#         return (w, h, h_rc)
#
#
#     def zpk_to_ba(self, pole,phi):
#         return [1, -2*pole*np.cos(phi), pole**2]
#
#     def apply_to_detector(self, params, detector):
#         if self.include_zeros:
#             pmag, pphi, rc_mag, rc_phi, lp_zeromag, lp_zerophi   = params[:]
#             detector.lp_num = self.zpk_to_ba(lp_zeromag, lp_zerophi)
#             if np.sum(detector.lp_num) == 0:
#                 raise ValueError("Zero sum low pass denominator!")
#             detector.lp_den = self.zpk_to_ba(pmag, pphi)
#
#         elif self.order_number == 4 :
#             pmag, pphi, rc_mag, rc_phi, pmag2, pphi2   = params[:]
#             den1 = self.zpk_to_ba(pmag, pphi)
#             den2 = self.zpk_to_ba(pmag2, pphi2)
#             detector.lp_num = [[1,2,1], [1,2,1]]
#             detector.lp_den = [den1, den2]
#             detector.lp_order = 4
#         else:
#             pmag, pphi, rc_mag, rc_phi   = params[:]
#             detector.lp_num = [1,2,1]
#             detector.lp_den = self.zpk_to_ba(pmag, pphi)
#
#         detector.hp_num = [1,-2,1]
#         # detector.hp_den = self.zpk_to_ba(rc_mag, rc_phi)
#         detector.hp_den = self.zpk_to_ba(1. - 10.**rc_mag, 10.**rc_phi)


class ElectronicsModel_old(JointModelBase):
    """
    Specify the model in Python.
    """
    def __init__(self,timestep=1E-9):
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
