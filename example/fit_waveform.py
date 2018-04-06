#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np

import dnest4

from waffle.management import WaveformFitManager
from waffle.models import VelocityModel, LowPassFilterModel, HiPassFilterModel, ImpurityModelEnds
from siggen import PPC

chan_dict = {
600: "B8482",
692: "B8474"
}

def main(doPlot=False):

    align_point = 0.95
    wf_idx = 1

    chan = 692
    directory = "chan{}_8wfs".format(chan)

    wf_file = "training_data/chan{}_8wfs.npz".format(chan)
    conf_name = "{}.conf".format( chan_dict[chan] )

    datadir= os.environ['DATADIR']
    conf_file = datadir +"siggen/config_files/" + conf_name


    detector = PPC( conf_file, wf_padding=100)

    lp = LowPassFilterModel()
    hp = HiPassFilterModel()
    im = ImpurityModelEnds(detector.imp_avg_lims, detector.imp_grad_lims, detector.detector_length)
    vm = VelocityModel(include_beta=False)

    det_params = [ 9.76373631e-01,8.35875049e-03,-5.09732644e+00,-6.00749043e+00,
                   4.74275220e+06,3.86911389e+06,6.22014783e+06,5.22077471e+06,
                    -3.63516477e+00,-4.48184667e-01]

    lp.apply_to_detector(det_params[:2], detector)
    hp.apply_to_detector(det_params[2:4], detector)
    vm.apply_to_detector(det_params[4:8], detector)
    im.apply_to_detector(det_params[8:], detector)

    data = np.load(wf_file, encoding="latin1")
    wfs = data['wfs']

    wf = wfs[wf_idx]
    wf_directory = os.path.join(directory, "wf{}".format(wf_idx))
    if os.path.isdir(wf_directory):
        if len(os.listdir(wf_directory)) >0:
            raise OSError("Directory {} already exists: not gonna over-write it".format(wf_directory))
    else:
        os.makedirs(wf_directory)

    wf.window_waveform(time_point=align_point, early_samples=100, num_samples=125)

    fm = WaveformFitManager(wf, align_percent=align_point, detector=detector, align_idx=100)

    fm.fit(numLevels=1000, directory = wf_directory, new_level_interval=1000, numParticles=3)


if __name__=="__main__":
    main(*sys.argv[1:])
