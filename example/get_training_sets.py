#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *

def main():

    runList = np.arange(11510, 11600)

    chanList = [600,626,672, 692]

    #data processing

    proc = DataProcessor()

    # proc.tier0(runList, min_signal_thresh=100, chanList=chanList)
    # proc.load_nonlinearities_to_db(runList)
    # proc.tier1(runList, num_threads=6)

    #calibration, simple analysis
    # proc.calibrate(runList)
    # proc.ae_cut(runList, )
    # proc.baseline_cuts(runList, )

    #cut waveforms
    # proc.save_training_data(np.arange(11510,11530), chanList, "training_data/training_set.h5")

    n_waveforms = 8
    for chan in chanList:
        proc.save_subset(chan, n_waveforms, "training_data/training_set.h5", "training_data/chan{}_{}wfs.npz".format(chan, n_waveforms))


if __name__=="__main__":
    main()
