#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

from siggen import PPC

def generate_fields(det_info, conf_name):
    '''
    Assume the measurements are within 50 percent of being right
    Allow the max impurity at each end to go to 1.5x the measurements
    and the min to go to 0
    find the range of impurity avg and grad that would encompass
    and generate the fields
    '''


    det = PPC( conf_name, verbose=False, doInit=False)

    meshmult = 10
    imp_uncertainty_factor = 0.5

    #divide by 10 to go from 1E9 to 1E10
    imp_max = -1*np.amax((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.
    imp_min = -1*np.amin((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.

    min_avg =  np.around(  0.5*( (1+imp_uncertainty_factor)*imp_max + (1+imp_uncertainty_factor)*imp_min ), 2)
    max_grad = np.around(   (-(1+imp_uncertainty_factor)*imp_max/(det.detector_length/10) ), 2)
    min_grad = np.around(   ((1+imp_uncertainty_factor)*imp_min/(det.detector_length/10) ), 2)

    impAvgRange = np.linspace(min_avg, 0, num_avg)
    gradientRange = np.linspace(min_grad,max_grad, num_grad)

    print(impAvgRange)
    print(gradientRange)

    det.solve_fields(meshmult, impAvgRange, gradientRange)
