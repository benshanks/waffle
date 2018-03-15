#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import matplotlib.pyplot as plt

from waffle.plots import ResultPlotter

def main():

    directory = "8wf_626"

    plotter = ResultPlotter(directory, 20)

    plotter.plot_waveforms()
    plt.show()


if __name__=="__main__":
    main()
