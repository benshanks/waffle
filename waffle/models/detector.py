import os, sys
import numpy as np

from ._parameterbase import JointModelBase, Parameter

class ImpurityModel(JointModelBase):
    """
    Impurity average and gradient
    """
    def __init__(self,imp_avg_lims, imp_grad_lims):
        self.imp_avg_lims = imp_avg_lims
        self.imp_grad_lims = imp_grad_lims

        self.params = [
            Parameter("imp_avg", "uniform", lim_lo=imp_avg_lims[0], lim_hi=imp_avg_lims[-1]),
            Parameter("imp_grad", "uniform", lim_lo=imp_grad_lims[0], lim_hi=imp_grad_lims[-1])
        ]

    def apply_to_detector(self, params, detector):
        imp_avg, imp_grad  = params[:]
        detector.siggenInst.SetImpurityAvg(imp_avg, imp_grad)

class ImpurityModelEnds(JointModelBase):
    """
    Impurity at both ends of the detector
    """
    def __init__(self,imp_avg_lims, imp_grad_lims, detector_length):
        self.imp_avg_lims = imp_avg_lims
        self.imp_grad_lims = imp_grad_lims

        self.imp_max = np.ceil(-1 * imp_grad_lims[-1] * ((detector_length/10)) *100)/100
        self.imp_min = np.ceil(1 * imp_grad_lims[0] * ((detector_length/10)) *100)/100

        self.params = [
            Parameter("imp_z0", "uniform", lim_lo=self.imp_max, lim_hi=0),
            Parameter("imp_zmax", "uniform", lim_lo=self.imp_min, lim_hi=0)
        ]

    def apply_to_detector(self, params, detector):
        imp_z0, imp_zmax  = params[:]
        detector.siggenInst.SetImpurityEnds(imp_z0, imp_zmax )

class TrappingModel(JointModelBase):
    def __init__(self):
        self.params = [
            Parameter("trap_c", "gaussian", 1000, 1000, lim_lo=0, lim_hi=5000)
        ]

    def apply_to_detector(self, params, detector):
        trap_c  = params
        detector.siggenInst.SetTrapping(trap_c)
