#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

from siggen import PPC

from waffle.detectors import create_conf_file, generate_fields

def main():

    # names = ["P42664A", "P42665A"]
    names= ["B8482", "B8474"]

    for detector_name in names:

        conf_name = os.path.join(os.environ['DATADIR'], "siggen", "config_files", "{}.conf".format(detector_name))

        imp_min, imp_max = create_conf_file(detector_name, conf_name)

        # find_depleted_bounds(conf_name, -2, -5, imp_uncertainty_factor=0. )

        generate_fields(conf_name, imp_min, imp_max, num_avg=5, num_grad=5, )
        # plot_waveform(det_info, conf_name)


# def plot_waveform(det_info, conf_name):
#     det = PPC( conf_name, verbose=False)
#
#     imp_grad_lims = det.imp_grad_lims
#
#     # imp_max = -1 * imp_grad_lims[-1] * ((det.detector_length/10))
#     # imp_min = 1 * imp_grad_lims[0] * ((det.detector_length/10))
#     # print(imp_max, imp_min)
#
#     imp_max = np.amax((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.
#     imp_min = np.amin((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.
#
#     # print(1.5*imp_max, 1.5*imp_min)
#     #
#     # exit()
#
#     det.siggenInst.SetImpurityEnds(-imp_max, -imp_min)
#
#     wf = np.copy(det.GetWaveform(25,0,25))
#     plt.plot(wf)
#     plt.show()
#
#
# def find_depleted_bounds(conf_name, imp_min, imp_max, imp_uncertainty_factor=0.5 ):
#     det = PPC( conf_name, verbose=False, doInit=False)
#
#     meshmult = 2
#     nr = int(det.detector_radius*meshmult+1)
#     nz = int(det.detector_length*meshmult+1)
#
#     vals = np.nan*np.zeros((10,10))
#
#     z0_range = ((1+imp_uncertainty_factor)*imp_max,0)
#     zm_range = ((1+imp_uncertainty_factor)*imp_min, 0)
#
#     for i, z0 in enumerate(np.linspace(z0_range[0], z0_range[-1], vals.shape[0])):
#         for j, zm in enumerate(np.linspace(zm_range[0], zm_range[-1], vals.shape[1])):
#             z0 = np.around(z0,7)
#             zm = np.around(zm,7)
#
#             avg = 0.5*(z0 + zm)
#             grad = np.around((zm-z0)/(det.detector_length/10),7)
#
#             print(z0, zm, avg, grad, det.siggenInst.GetXtalHV())
#             # continue
#             efield = det.solve_field("efield", nr, nz, impurity_gradient=grad, impurity_avg=avg, xtal_HV=det.siggenInst.GetXtalHV())
#
#             # plt.figure()
#             # efld = efield[:,:,-1]
#             # im = plt.imshow(efld.T,  origin='lower',cmap='RdYlGn'
#             #              )
#             # plt.colorbar()
#             # plt.show()
#
#             e_z = efield[0,:,-1]
#             e_z[e_z==0] = -np.inf
#             # e_z[:int(np.ceil(meshmult*det_info["pc_length"]))] = -np.inf
#             max_z = np.amax(e_z)
#             vals[i,j] = max_z
#
#     plt.figure()
#     im = plt.imshow(vals.T,  origin='lower',cmap='RdYlGn',
#                  extent=[z0_range[0], z0_range[-1], zm_range[0], zm_range[-1]]
#                  )
#     plt.colorbar()
#     plt.show()
#     exit()
#
# def generate_fields(det_info, conf_name, num_avg=5, num_grad=5):
#     det = PPC( conf_name, verbose=False, doInit=False)
#
#     meshmult = 10
#     imp_uncertainty_factor = 0.5
#
#     #divide by 10 to go from 1E9 to 1E10
#     imp_max = -1*np.amax((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.
#     imp_min = -1*np.amin((det_info["impurity_tail"], det_info["impurity_seed"]) ) / 10.
#
#     min_avg =  np.around(  0.5*( (1+imp_uncertainty_factor)*imp_max + (1+imp_uncertainty_factor)*imp_min ), 2)
#     max_grad = np.around(   (-(1+imp_uncertainty_factor)*imp_max/(det.detector_length/10) ), 2)
#     min_grad = np.around(   ((1+imp_uncertainty_factor)*imp_min/(det.detector_length/10) ), 2)
#
#     impAvgRange = np.linspace(min_avg, 0, num_avg)
#     gradientRange = np.linspace(min_grad,max_grad, num_grad)
#
#     print(impAvgRange)
#     print(gradientRange)
#
#     det.solve_fields(meshmult, impAvgRange, gradientRange)


if __name__=="__main__":
    main()
