import numpy as np
import sys, os, shutil
import pickle
import dnest4
from multiprocessing import Pool, cpu_count

from .models import Model

def init_parallelization(conf):
    global model
    model = Model( conf,)
def WaveformLogLikeStar(a_b):
  return model.calc_wf_likelihood(*a_b)

class LocalFitManager():
    '''Does the fit using one machine -- either multicore or single-threaded'''

    def __init__(self, fit_configuration, num_threads=None):

        self.model = Model( fit_configuration, fit_manager=self)
        self.num_waveforms = self.model.num_waveforms
        self.num_det_params = self.model.num_det_params
        self.num_wf_params = self.model.num_wf_params#fit_configuration.num_wf_params

        if num_threads is None: num_threads = cpu_count()

        if num_threads > self.model.num_waveforms: num_threads = self.model.num_waveforms

        self.num_threads = num_threads

        if num_threads > 1:
            self.pool = Pool(num_threads, initializer=init_parallelization, initargs=(fit_configuration,))
        else:
            init_parallelization(fit_configuration)

    def calc_likelihood(self, params):
        num_det_params = self.num_det_params
        lnlike = 0

        #parallelized calculation
        if self.num_threads > 1:
            args = []
            for wf_idx in range(self.num_waveforms):
                args.append( [self.model.get_wf_params(params, wf_idx), wf_idx] )
                # print ("shipping {0}: {1}".format(wf_idx, wf_params[num_det_params:, wf_idx]))

            results = self.pool.map(WaveformLogLikeStar, args)
            # exit()
            for result in (results):
                lnlike += result
        else:
            for wf_idx in range(self.num_waveforms):
                result = model.calc_wf_likelihood(self.model.get_wf_params(params, wf_idx), wf_idx)
                lnlike += result
            # print (result)
        return lnlike

    def fit(self, numLevels, directory="",numPerSave=1000,numParticles=5,new_level_interval=10000 ):

      sampler = dnest4.DNest4Sampler(self.model,
                                     backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                        sep=" "))

      # Set up the sampler. The first argument is max_num_levels
      gen = sampler.sample(max_num_levels=numLevels, num_steps=200000, new_level_interval=new_level_interval,
                            num_per_step=numPerSave, thread_steps=100,
                            num_particles=numParticles, lam=10, beta=100, seed=1234)

      # Do the sampling (one iteration here = one particle save)
      for i, sample in enumerate(gen):
          print("# Saved {k} particles.".format(k=(i+1)))


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
