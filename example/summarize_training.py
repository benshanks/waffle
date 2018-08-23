#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Example use:
# python give_results.py 8wf_692/ 2000
# python give_results.py 4wf_692_DS3-1/ 20000

import sys, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from waffle.management import FitConfiguration
from waffle.models import Model, PulserTrainingModel

from waffle.postprocessing import TrainingResultSummary


def main(dir_name, num_samples=2000, sample_dec=1 ):
    res = TrainingResultSummary(dir_name, int(num_samples), int(sample_dec))

    num_wf_params = res.num_wf_params
    num_joint_params = res.model.joint_models.num_params

    res.extract_model_values()
    res.summarize_params(do_plots=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('dir_name')

    parser.add_argument('num_samples', nargs='?', default=2000)
    # parser.set_defaults(num_samples=2000)

    parser.add_argument('sample_dec', nargs='?', default=1)
    # parser.set_defaults(sample_dec=1)

    args = parser.parse_args()

    main(args.dir_name, args.num_samples, args.sample_dec )
