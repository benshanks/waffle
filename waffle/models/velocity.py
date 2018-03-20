import os, sys
import numpy as np

from ._parameterbase import ModelBaseClass, Parameter

class VelocityModel(ModelBaseClass):

    def __init__(self,E_lo=250, E_hi=1000,include_beta=True, beta_lims=[0.1,2]):
        self.num_params = 6 if include_beta else 4
        self.E_lo = E_lo
        self.E_hi = E_hi

        #Default values are from Bruyneel NIMA 569 (Reggiani values)
        h100_lo, h100_hi, self.h100_beta = self.transform_velo_params(66333., 0.744, 181.)
        h111_lo, h111_hi, self.h111_beta = self.transform_velo_params(107270., 0.580, 100.)

        parameter_names = ["h100_{}".format(E_lo), "h111_{}".format(E_lo),
                                "h100_{}".format(E_hi), "h111{}".format(E_hi),
                                "h100_beta", "h111_beta"
                                ]

        velocity_means = np.array([h100_lo, h111_lo, h100_hi, h111_hi])
        velocity_variance = 0.1

        velo_lo_cutoff = 1
        velo_hi_cutoff = 10*velocity_means

        self.params = []
        for i in range(4):
            self.params.append( Parameter(parameter_names[i], "gaussian",
                                        mean=velocity_means[i],
                                        variance=velocity_variance*velocity_means[i],
                                        lim_lo=velo_lo_cutoff, lim_hi=velo_hi_cutoff[i])
                                )
        if include_beta:
            for i in range(4,6):
                self.params.append( Parameter(parameter_names[i], "uniform",
                                            lim_lo=beta_lims[0], lim_hi=beta_lims[1])
                                    )

    def apply_to_detector(self, params, detector):
        h_100_vlo, h_111_vlo, h_100_vhi, h_111_vhi,  = params[:4]

        if self.num_params == 6:
            h_100_beta, h_111_beta = params[4:]
        else:
            h_100_beta, h_111_beta = self.h100_beta, self.h111_beta

        h_100_mu0, h_100_beta, h_100_e0 = self.invert_velo_params(h_100_vlo, h_100_vhi, h_100_beta)
        h_111_mu0, h_111_beta, h_111_e0 = self.invert_velo_params(h_111_vlo, h_111_vhi, h_111_beta)

        detector.siggenInst.SetHoles(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 )


    def invert_velo_params(self, v_a, v_c, beta):
        E_a = self.E_lo
        E_c = self.E_hi

        psi = (E_a * v_c) / ( E_c * v_a )
        E_0 = np.power((psi**beta* E_c**beta - E_a**beta) / (1-psi**beta), 1./beta)
        mu_0 = (v_a / E_a) * (1 + (E_a / E_0)**beta )**(1./beta)

        # if v_a > v_c: print("gonna go ahead and expect a nan here")
        # if np.isnan(E_0):
        #     print("NAN HERE: {},{},{}".format(v_a/1E6, v_c/1E6,psi))
        #     a = (psi**beta* E_c**beta - E_a**beta) / (1-psi**beta)
        #     print(a)
        #     b = 1./beta
        #     print(b)
        #     print(a**b)
        #     print("\n\n\n")

        return (mu_0,  beta, E_0)

    def transform_velo_params(self, mu_0,  beta, E_0):
        E_a = self.E_lo
        E_c = self.E_hi

        v_a = (mu_0 * E_a)/np.power( 1 + (E_a/E_0)**beta ,1./beta)
        v_c = (mu_0 * E_c)/np.power( 1 + (E_c/E_0)**beta ,1./beta)

        return (v_a, v_c, beta)

##

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

##

class ImpurityModel(ModelBaseClass):
    """
    Specify the model in Python.
    """
    def __init__(self,imp_avg_lims, imp_grad_lims):
        self.num_params = 2
        self.imp_avg_lims = imp_avg_lims
        self.imp_grad_lims = imp_grad_lims

        self.params = [
            Parameter("imp_avg", "uniform", lim_lo=imp_avg_lims[0], lim_hi=imp_avg_lims[-1]),
            Parameter("imp_grad", "uniform", lim_lo=imp_grad_lims[0], lim_hi=imp_grad_lims[-1])
        ]

    def apply_to_detector(self, params, detector):
        imp_avg, imp_grad  = params[:]
        detector.siggenInst.SetImpurityAvg(imp_avg, imp_grad)
