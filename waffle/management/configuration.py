import numpy as np
import sys, os
import pickle

class WaveformConfiguration(object):
    def __init__(self,
        #params for setting up & aligning waveforms
        wf_file_name,
        align_idx = 200,
        num_samples = 400,
        align_percent = 0.95,
        do_smoothing=True,
        wf_idxs=None,
        smoothing_type="gaussian"
    ):
        self.wf_file_name=wf_file_name
        self.align_idx = align_idx
        self.align_percent = align_percent
        self.num_samples = num_samples
        self.wf_idxs = wf_idxs
        self.do_smoothing=do_smoothing
        self.smoothing_type=smoothing_type

        #downsampling the decay portion


class FitConfiguration(object):
    def __init__(self,
        #data files
        conf_file="",

        #save path
        directory = "",

        #fit parameters
        wf_conf={},
        model_conf={},

        loadSavedConfig=False,

        time_step_calc=1,
        **kwargs
    ):

        self.siggen_conf_file=conf_file
        self.directory = directory

        self.wf_conf = wf_conf

        self.model_conf=model_conf

        self.time_step_calc=time_step_calc

        for key, value in kwargs.items():
            setattr(self, key, value)

        if loadSavedConfig:
            self.load_config(directory)
        else:
            self.wf_config = WaveformConfiguration(**wf_conf)

    def save_config(self):
        saved_file=os.path.join(self.directory, "fit_params.npy")
        pickle.dump(self.__dict__.copy(),open(saved_file, 'wb'))

    def load_config(self,directory):
        saved_file=os.path.join(directory, "fit_params.npy")
        if not os.path.isfile(saved_file):
            print ("Saved configuration file {0} does not exist".format(saved_file))
            exit()

        self.__dict__.update(pickle.load(open(saved_file, 'rb')))
        self.wf_config = WaveformConfiguration(**self.wf_conf)

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
