import os, sys
import numpy as np

from ._parameterbase import ModelBaseClass, Parameter

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
