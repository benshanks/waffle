#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import matplotlib.pyplot as plt

from waffle.plots import TrainingPlotter

def main(dir_name, num_samples=2000, sample_dec=1 ):
    plotter = TrainingPlotter(dir_name, int(num_samples), int(sample_dec))

    plotter.plot_waveforms(print_det_params=False)
    # plotter.plot_waveform_components()
    plotter.plot_tf()
    # plotter.plot_imp()
    #
    # plotter.plot_detector_pair()

    plotter.plot_trace()
    # plotter.plot_waveform_trace()

    plt.show()


if __name__=="__main__":
    main(*sys.argv[1:] )
