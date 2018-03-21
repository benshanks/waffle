import os, sys
import numpy as np
import numpy.random as rng
import dnest4
from abc import ABC, abstractmethod


max_float = sys.float_info.max

class Parameter(object):
    def __init__(self, name, prior_type, mean=None, variance=None, lim_lo=-max_float, lim_hi=max_float):

        if not (prior_type == "gaussian" or prior_type == "uniform"):
            raise ValueError("Parameter prior_type must be gaussian or uniform, not {}".format(prior_type))

        self.name = name
        self.prior_type = prior_type
        self.mean = mean
        self.variance = variance
        self.lim_lo = lim_lo
        self.lim_hi = lim_hi

    def draw_from_prior(self):
        if self.prior_type == "gaussian":
            _, param = self.perturb_gaussian_parameter(self.mean)
        elif self.prior_type == "uniform":
            param = rng.rand()*(self.lim_hi - self.lim_lo) + self.lim_lo
        return param

    def perturb(self, parameter):
        if self.prior_type == "gaussian":
            return self.perturb_gaussian_parameter(parameter)
        elif self.prior_type == "uniform":
            return self.perturb_uniform_parameter(parameter)

    def perturb_gaussian_parameter(self, parameter):
        mu = self.mean
        var = self.variance

        logH = 0
        logH -= -0.5*((parameter - mu)/var)**2
        parameter += var*dnest4.randh()
        if (self.lim_lo > -max_float) or (self.lim_hi<np.inf):
            parameter = dnest4.wrap(parameter, self.lim_lo, self.lim_hi)
        logH += -0.5*((parameter - mu)/var)**2

        return (logH, parameter)

    def perturb_uniform_parameter(self, parameter):
        logH = 0
        parameter += (self.lim_hi - self.lim_lo)  *dnest4.randh()
        parameter = dnest4.wrap(parameter, self.lim_lo, self.lim_hi)
        return (logH, parameter)

##

class ModelBaseClass(ABC):

    def perturb(self, params, which):
        logH = 0
        # print ("perturbing {}, self.params len is {}, param array len is {}".format(which, len(self.params), len(params)))

        if which <0 or which > self.num_params:
            raise IndexError('Which value {} in VelocityModel.perturb is beyond parameter array limit of {}'.format(which, self.num_params))

        #TODO: wow this is a confusing line!  params is a np array passed from the main model, self.params is a list of Parameter objects held by the sub-model
        logH, params[which] = self.params[which].perturb(params[which])

        return logH

    def get_prior(self):
        prior = np.zeros(self.num_params)

        for i in range(self.num_params):
            prior[i] = self.params[i].draw_from_prior()
        return prior

    def get_num_params(self):
        return len(self.params)

    @abstractmethod
    def apply_to_detector(self, params, detector):
        pass
