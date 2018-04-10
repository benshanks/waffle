#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import matplotlib.pyplot as plt

from waffle.plots import TrainingPlotter

def main(dir_name, num_samples=2000, sample_dec=1 ):
    num_samples = int(num_samples)

    plotter = TrainingPlotter(dir_name, num_samples, int(sample_dec))

    # plotter.plot_waveform_components()

    # plotter.plot_imp()
    #
    # plotter.plot_detector_pair()

    if num_samples == -1:
        plotter.plot_trace()
        plotter.plot_waveform_trace()
    else:
        plotter.plot_waveforms(print_det_params=False)
        plotter.plot_tf()

    plt.show()


if __name__=="__main__":
    main(*sys.argv[1:] )
