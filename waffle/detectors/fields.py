#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

from siggen import PPC

def generate_fields(det_info, conf_name):
    det = PPC( conf_name, verbose=False, doInit=False)

    meshmult = 10
    imp_uncertainty_factor = 2

    #divide by 10 to go from 1E9 to 1E10
    imp_max = np.amax((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.
    imp_min = np.amin((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.

    max_avg = np.round(-0.5*(imp_uncertainty_factor*imp_max + imp_uncertainty_factor*imp_min), 2)
    min_avg = np.round(-0.5*(imp_max/imp_uncertainty_factor + imp_min/imp_uncertainty_factor),2)

    max_grad = np.round((imp_uncertainty_factor*imp_max - imp_min/imp_uncertainty_factor)/(det_info["length"]/10),2)
    min_grad = np.round((imp_max/imp_uncertainty_factor - imp_uncertainty_factor*imp_min)/(det_info["length"]/10),2)

    impAvgRange = np.linspace(max_avg,min_avg, 5)
    gradientRange = np.linspace(min_grad,max_grad, 5)

    print(impAvgRange)
    print(gradientRange)

    det.solve_fields(meshmult, impAvgRange, gradientRange)
