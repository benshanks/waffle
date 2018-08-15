import os, sys
import numpy as np
import numpy.random as rng

from . import *
from ._parameterbase import JointModelBase

class JointModelBundle(object):
    def __init__(self, conf, detector):
        self.conf =conf
        self.detector = detector

        self.models = []
        self.index_map = {}
        self.start_map = {}

        self.num_params = 0
        i=0
        # print(self.conf)

        for model_idx, (model_name, model_conf) in enumerate(self.conf):
            model = self.append(model_name, model_conf)
            self.num_params += model.num_params
            self.start_map[model_idx] = i
            model.start_idx = i

            for j in range(model.num_params):
                self.index_map[i] = model_idx
                i +=1

    def append(self, model_name, model_conf):
        #TODO: surely this can be done with introspection
        if model_name=="VelocityModel":
            model = VelocityModel(**model_conf)
        elif model_name=="ImpurityModelEnds":
            model = ImpurityModelEnds(self.detector, **model_conf)
        elif model_name == "HiPassFilterModel":
            model = HiPassFilterModel(self.detector, **model_conf)
        elif model_name == "LowPassFilterModel":
            model = LowPassFilterModel(self.detector, **model_conf)
        elif model_name == "DigitalFilterModel":
            model = DigitalFilterModel(self.detector, **model_conf)
        elif model_name ==  "OvershootFilterModel":
            model = OvershootFilterModel(self.detector, **model_conf)
        elif model_name ==  "OscillationFilterModel":
            model = OscillationFilterModel(self.detector, **model_conf)
        elif model_name == "AntialiasingFilterModel":
            model = AntialiasingFilterModel(self.detector, **model_conf)
        elif model_name == "FirstStageFilterModel":
            model = FirstStageFilterModel(self.detector, **model_conf)
        elif model_name ==  "TrappingModel":
            model = TrappingModel(**model_conf)
        elif issubclass(model_name, JointModelBase):
            model = model_name(**model_conf)
        else:
            raise ValueError("model_name {} is not a valid model".format(model_name))

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
