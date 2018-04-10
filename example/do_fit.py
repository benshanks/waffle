#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np

import dnest4

from waffle.management import LocalFitManager, FitConfiguration

chan_dict = {
600: "B8482",
626: "P42574A",
640:"P42665A",
648:"P42664A",
672: "P42661A",
692: "B8474"
}

def main(chan, doPlot=False):

    # directory = "4wf_648_zero"
    # wf_file = "training_data/chan648_8wfs.npz"
    # conf_name = "P42664A.conf"

    chan = int(chan)
    directory = "8wf_overshoot4_{}".format(chan)

    wf_file = "training_data/chan{}_8wfs.npz".format(chan)
    conf_name = "{}.conf".format( chan_dict[chan] )

    wf_idxs = np.arange(0,8)
    # wf_idxs = [1,2]

    # wf_file = "16wf_set_chan{}.npz".format(chan)
    # wf_idxs = np.arange(0,16,4)
    # wf_idxs = [1,4,8,12]

    datadir= os.environ['DATADIR']
    conf_file = datadir +"siggen/config_files/" + conf_name

    wf_conf = {
        "wf_file_name":wf_file,
        "wf_idxs":wf_idxs,
        "align_idx":125,
        "num_samples":1000,
        "do_smooth":True,
        "smoothing_type":"gaussian"
    }

    model_conf = {
        "model_list": ["VelocityModel", "LowPassFilterModel", "HiPassFilterModel", "ImpurityModelEnds", "TrappingModel", "OvershootFilterModel"],
        "fit_beta":False,
        "lp_order":4,
        "hp_order":4,
        "lp_zeros":False
    }

    conf = FitConfiguration(
        conf_file,
        directory = directory,
        wf_conf=wf_conf,
        model_conf=model_conf
    )

    if doPlot:
        import matplotlib.pyplot as plt
        # conf.plot_training_set()
        fm = LocalFitManager(conf, num_threads=1)
        for wf in fm.model.wfs:
            plt.plot(wf.windowed_wf)
            print (wf.window_length)
        plt.show()
        exit()

    if os.path.isdir(directory):
        if len(os.listdir(directory)) >0:
            raise OSError("Directory {} already exists: not gonna over-write it".format(directory))
    else:
        os.makedirs(directory)

    fm = LocalFitManager(conf, num_threads=8)

    conf.save_config()
    fm.fit(numLevels=1000, directory = directory,new_level_interval=5000, numParticles=3)


if __name__=="__main__":
    main(*sys.argv[1:])
