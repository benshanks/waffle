import numpy as np
import sys, os
import pickle

class FitConfiguration(object):
    def __init__(self,
        #data files
        wf_file_name="", conf_file="",
        wf_idxs=None,
        #save path
        directory = "",
        #params for setting up & aligning waveforms
        align_idx = 200,
        num_samples = 400,
        alignType = "max",
        fit_beta = False,
        fit_zeros=False,
        lp_order=2,
        loadSavedConfig=False,
        beta_lims = [0.1, 1],
        interpType = "linear",
        time_step_calc=1#ns
    ):

        self.wf_file_name=wf_file_name
        self.siggen_conf_file=conf_file

        self.directory = directory

        self.wf_idxs = wf_idxs
        self.align_idx = align_idx

        self.fit_beta = fit_beta
        self.fit_zeros = fit_zeros
        self.lp_order=lp_order

        #velocity model reference point field
        self.E_a = 500
        self.E_lo = 250
        self.E_hi = 1000

        if not (alignType == "max" or alignType == "timepoint"):
            print ("alignType must be 'max' or 'timepoint', not {0}".format(alignType))
            exit()
        self.alignType = alignType
        self.align_percent = 0.95
        self.num_samples = num_samples

        #limits & priors for the actual fit
        self.traprc_min = 150
        self.beta_lims = beta_lims

        self.time_step_calc = time_step_calc

        if loadSavedConfig:
            self.load_config(directory)

    def save_config(self):
        saved_file=os.path.join(self.directory, "fit_params.npy")
        pickle.dump(self.__dict__.copy(),open(saved_file, 'wb'))

    def load_config(self,directory):
        saved_file=os.path.join(directory, "fit_params.npy")
        if not os.path.isfile(saved_file):
            print ("Saved configuration file {0} does not exist".format(saved_file))
            exit()

        self.__dict__.update(pickle.load(open(saved_file, 'rb')))

    def plot_training_set(self):
        import matplotlib.pyplot as plt

        if os.path.isfile(self.wf_file_name):
            print("Loading wf file {0}".format(self.wf_file_name))
            data = np.load(self.wf_file_name, encoding="latin1")
            wfs = data['wfs']
            wfs = wfs[self.wf_idxs]

            plt.figure()
            for wf in wfs:
                plt.plot(wf.data)
            plt.show()
