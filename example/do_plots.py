#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import argparse
import matplotlib.pyplot as plt

from waffle.postprocessing import TrainingPlotter

def main(dir_name, num_samples=2000, sample_dec=1, print_params=False ):

    plotter = TrainingPlotter(dir_name, num_samples, sample_dec)
    # plotter.plot_waveform_components()
    # plotter.plot_imp()
    # plotter.plot_detector_pair()

    if num_samples == -1:
        plotter.plot_trace()
        plotter.plot_waveform_trace()
    else:
        plotter.plot_waveforms(print_det_params=print_params)
        plotter.plot_tf()

    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('dir_name')

    parser.add_argument('num_samples', nargs='?', default=2000, type=int)
    parser.add_argument('sample_dec', nargs='?', default=1, type=int)
    
    parser.add_argument('-p','--params', dest='print_params', action='store_true')
    parser.add_argument('-np','--no-params', dest='print_params', action='store_false')
    parser.set_defaults(print_params=False)

    args = parser.parse_args()

    main(args.dir_name, args.num_samples, args.sample_dec, args.print_params )