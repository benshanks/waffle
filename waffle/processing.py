#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import pygama
from pygama.processing import *
from pygama.calibration import *
from pygama.transforms import *
from pygama.calculators import *
from pygama.databases import *
from pygama.pulse_shape_analysis import *
import pygama.data_cleaning as pdc
from pygama.waveform import *
import pygama.decoders as dl
import pygama.filters as filt

from .pz_fitter import *

from scipy import stats
from scipy.interpolate import interp1d

class DataProcessor():

    def __init__(self, detectorChanList):
        #energy & current estimator to use for A/E
        self.energy_name = "trap_max"
        self.ecal_name = "ecal"
        self.current_name = "current_max_5"
        self.data_key = "ORGretina4MWaveformDecoder"
        self.detectorChanList = detectorChanList

        self.dt_max_param = "tp_99"


        #File hierarchy for this work:
        #Base dir is from $DATADIR environment variable
        self.mjd_data_dir = os.path.join(os.getenv("DATADIR", "."), "mjd")
        self.raw_data_dir = os.path.join(self.mjd_data_dir,"raw")
        self.t1_data_dir = os.path.join(self.mjd_data_dir,"t1")
        # self.t1_fit_dir = os.path.join(self.mjd_data_dir,"t1fit")
        self.t2_data_dir = os.path.join(self.mjd_data_dir,"t2")
        self.nl_file_name = os.path.join(self.mjd_data_dir,"NLCDB", "nonlinearities.h5")

        #TODO: in a real db?
        self.channel_info_file_name = os.path.join(self.mjd_data_dir,"analysis", "channel_info.h5")
        # self.cal_file_name = os.path.join(self.mjd_data_dir,"analysis", "calibration.h5")
        # self.psa_file_name = os.path.join(self.mjd_data_dir,"analysis", "psa.h5")
        # self.blcut_file_name = os.path.join(self.mjd_data_dir,"analysis", "blcuts.h5")

    def load_t1(self, runList):
        dfs=[]
        for runNumber in runList:
            t1_file = os.path.join(self.t1_data_dir, "t1_run{}.h5".format(runNumber))
            t1= pd.read_hdf(t1_file,key=self.data_key)
            t1["runNumber"]=runNumber
            dfs.append(t1)

            print(t1.head())
            exit()

        return df

    def load_t2(self, runList):
        dfs=[]
        for runNumber in runList:
            t2_file = os.path.join(self.t2_data_dir, "t2_run{}.h5".format(runNumber))
            t2= pd.read_hdf(t2_file,key="data")

            t2["runNumber"]=runNumber
            t2['event_number'] = t2.index
            t2.set_index(['runNumber', 'event_number'], inplace=True)

            # t2.set_index(["runNumber", "event_number"], inplace=True)
            dfs.append(t2)
        df = pd.concat(dfs, axis=0)
        # df = pd.concat(dfs, axis=0, ignore_index=True)

        return df

    def save_t2(self,df, save_dir=None):
        if save_dir is None: save_dir = self.t2_data_dir
        for runNumber, run_df in df.groupby("runNumber", group_keys=False):
            run_df.reset_index(inplace=True)
            run_df.set_index("event_number", inplace=True)

            t2_file = os.path.join(save_dir, "t2_run{}.h5".format(runNumber))
            run_df.to_hdf(t2_file, key="data", format='table', mode='w', data_columns=run_df.columns.tolist() )


    def tag_pulsers(self, df):

        #figure out what the pulser info is for each channel
        pulser_energy = []
        peak_e_err = []
        pulser_period = []
        energy_name = []
        channels = []

        for name, group in df.groupby("channel"):
            try:
                ret = pdc.find_pulser_properties(group, energy=self.energy_name)
            except IndexError:
                continue

            if ret is None:
                continue
            p_e, p_e_e, per, e = ret
            pulser_energy.append(p_e)
            peak_e_err.append(p_e_e)
            pulser_period.append(per)
            energy_name.append(e)
            channels.append(name)

        d = {
            'channel':channels,
            'pulser_energy':pulser_energy,
            'peak_e_err':peak_e_err,
            'pulser_period':pulser_period,
            'energy_name':energy_name
        }
        channel_info = pd.DataFrame.from_dict(d).set_index("channel")

        df = df.groupby(["channel", "runNumber"]).apply(pdc.tag_pulsers, chan_info=channel_info)
        # df = df.groupby("channel").apply(calibrate, energy_name="trap_max", out_name="ecal")

        return df


    def calculate_previous_event_params(self, df_chan, baseline_meas ):
        #info for each event about the _previous_ event
        #set nan for the 0th component...

        df_chan["prev_t"] = df_chan.timestamp.diff()
        # print(df_chan.timestamp)
        # df_chan["prev_t"].iloc[1:] = df_chan.timestamp.values[1:] - df_chan.timestamp.values[:-1]

        df_chan["prev_amp"] = df_chan["trap_max"].shift()
        df_chan["prev_e"] = df_chan[self.ecal_name].shift()
        df_chan["prev_bl"] = df_chan[baseline_meas].shift()

        # df_chan["prev_amp"] = np.nan
        # df_chan["prev_amp"].iloc[1:] = df_chan["trap_max"].values[:-1]

        # df_chan["prev_e"] = np.nan
        # df_chan["prev_e"].iloc[1:] = df_chan[self.ecal_name].values[:-1]

        # df_chan["prev_bl"] = np.nan
        # df_chan["prev_bl"].iloc[1:] = df_chan[baseline_meas].values[:-1]

        return df_chan


    def fit_pz(self, df, baseline_meas="bl_int"):

        #load the baseline cuts
        df_bl = pd.read_hdf(self.channel_info_file_name, key="baseline")

        pz_map = []

        for channel, df_chan in df.groupby("channel"):

            if channel not in self.detectorChanList:
                continue

            bl_cuts = df_bl.loc[channel]
            bl_norm = bl_cuts.bl_int_mean

            #drop nan's, i.e., the first event in every run
            df_chan_nona = df_chan.dropna()
            cut = (df_chan_nona.prev_e > 500) & (df_chan_nona.ecal > 100) & (df_chan_nona.isPulser==0)
            df_cut = df_chan_nona[cut]

            #make sure the previous event is at an OK baseline
            df_good = df_cut[ (df_cut.prev_bl > bl_cuts.bl_int_min) & (df_cut.prev_bl < bl_cuts.bl_int_max)   ]

            bl_good = df_good[baseline_meas]
            prev_amp = df_good["prev_amp"]
            previous_times = df_good["prev_t"]

            bl_dev = (bl_good-bl_norm)/prev_amp

            time_ms = df_good["prev_t"]*10/1E6
            ms_max = 50
            cut = (time_ms<ms_max) & (time_ms>0.05) & (bl_dev > -0.05)
            time_ms = time_ms[cut]
            bl_dev=bl_dev[cut]

            rc_ms, rc_us, offset, toffset = fit_data(time_ms, bl_dev)

            row = {"channel": channel,
                   "rc_ms": rc_ms,
                   "rc_us": rc_us,
                   "offset":offset,
                   "toffset":toffset
                   }
            pz_map.append(row)

            f = plt.figure()
            plot_data(time_ms, bl_dev, rc_ms, rc_us, offset, toffset)
            f.savefig("undershoot/channel{}.png".format(channel))
            plt.close(f)


        df_pz = pd.DataFrame(pz_map)
        df_pz.set_index("channel", drop=False, inplace=True)
        df_pz.to_hdf(self.channel_info_file_name,   key="pz", mode='a')

        return df


    def calc_baseline_cuts(self, df, baseline_meas="bl_int", settle_time = 15, cut_sigma=1):
        '''
        settle time is in ms
        '''

        bl_map = []
        try: os.mkdir("bl_plots")
        except OSError: pass

        for channel, df_chan in df.groupby("channel"):
            if channel not in self.detectorChanList:
                continue

            #drop nan's, i.e., the first event in every run
            df_chan_nona = df_chan.dropna()

            #look at all events with more than settle_time ms after the previous event for the BL to settle
            df_long = df_chan_nona[ df_chan_nona.prev_t*10/1E6 > settle_time ]

            #fit gaussians to the values to figure out normal range of bl intercept & slope

            f, ax = plt.subplots(1,2,figsize=(14,8))
            plt.suptitle("Channel {}".format(channel))
            bl_int_cuts = pdc.gaussian_cut(df_long[baseline_meas],  cut_sigma=cut_sigma, plotAxis=ax[0])
            ax[0].set_xlabel("bl intercept")
            bl_slope_cuts = pdc.gaussian_cut(df_long.bl_slope,  cut_sigma=cut_sigma, plotAxis=ax[1])
            ax[1].set_xlabel("bl slope")

            plt.savefig("bl_plots/channel{}.png".format(channel))
            plt.close(f)

            row = {"channel": channel,
                   "bl_slope_min": bl_slope_cuts[0],
                   "bl_slope_max": bl_slope_cuts[1],
                   "bl_int_min": bl_int_cuts[0],
                   "bl_int_max": bl_int_cuts[1],
                   "bl_int_mean": bl_int_cuts[2],
                   "bl_int_sigma": bl_int_cuts[3],
                   }
            bl_map.append(row)

        df_bl = pd.DataFrame(bl_map)
        df_bl.set_index("channel", drop=False, inplace=True)
        df_bl.to_hdf(self.channel_info_file_name,   key="baseline", mode='a')

    def calc_ae_cut(self, df):

        energy_name = self.energy_name
        current_name = self.current_name
        try: os.mkdir("ae_plots")
        except OSError: pass
        ae_map = []
        f = plt.figure(figsize=(12,6))

        for channel, df_chan in df.groupby("channel"):
            if channel not in self.detectorChanList:
                continue

            avse2, avse1, avse0, avse_cut, avse_mean, avse_std = get_avse_cut(df_chan[self.ecal_name], df_chan[current_name], f)

            row = {"channel": channel,
                   "current_name": current_name,
                   "energy_name": energy_name,
                   "avse0": avse0,
                   "avse1": avse1,
                   "avse2": avse2,
                   "avse_cut":avse_cut,
                   "avse_mean":avse_mean,
                   "avse_std":avse_std
                   }
            ae_map.append(row)

            plt.savefig("ae_plots/ae_channel{}.png".format(channel))

        df_ae = pd.DataFrame(ae_map)
        df_ae.set_index("channel", drop=False, inplace=True)
        df_ae.to_hdf(self.channel_info_file_name,   key="ae", mode='a')


    def calibrate(self, df, save_cal=True):

        '''only run this on one channel at a time
           only calibrates channels in self.detectorChanList
        '''


        try: os.mkdir("cal_plots")
        except OSError: pass


        channel = df.channel.unique()[0]

        if channel not in self.detectorChanList:
            df[self.ecal_name]=-1
            return df

        f1 = plt.figure(figsize=(16,6))
        m,b = calibrate_tl208(df.loc[df.isPulser==0, [self.energy_name]].values, plotFigure=f1)
        df[self.ecal_name] = m*df[self.energy_name]+b

        f1.savefig("cal_plots/cal_channel{}.png".format(channel))
        plt.close(f1)

        return df

        # chanList = np.unique(df["channel"])
        # cal_map = []
        #
        # for chan, df_chan in df.groupby("channel"):
        #     energy_chan = df_chan[energy_name]
        #
        #     print("Channel {} has {} entries...".format(chan, len(energy_chan)))
        #
        #     if len(energy_chan) < 5000:
        #         print("...skipping this channel")
        #         continue
        #
        #     linear_cal = calibrate_tl208(energy_chan, peak_energies, plotFigure=f1)
        #
        #     row = {"channel": chan,
        #            "slope": linear_cal[0],
        #            "offset": linear_cal[1]
        #            }
        #     cal_map.append(row)
        #
        #     plt.savefig("cal_plots/cal_channel{}.png".format(chan))
        #
        #     # val = input("q to quit, else to continue...")
        #     # if val == "q": exit()
        #
        # df_cal = pd.DataFrame(cal_map)
        # df_cal.set_index("channel", drop=False, inplace=True)
        # df_cal.to_hdf(self.cal_file_name,   key="cal", mode='w')
        #
        # return cal_map

    def tier0(self, runList, chan_list=None):
        process_tier_0(self.raw_data_dir, runList, output_dir=self.t1_data_dir, chan_list=chan_list)

    def tier1(self, runList, num_threads, overwrite=False):
        procs = self.get_processor_list()
        process_tier_1(self.t1_data_dir, runList, processor_list=procs, output_dir=self.t2_data_dir, num_threads=num_threads,overwrite=overwrite)

    def load_nonlinearities_to_db(self, runList):
        '''
        Reads in MJD nonlinearity text files, saves to a hdf Database
        '''

        chan_dfs = []
        for runNumber in runList:
            t1_file = os.path.join(self.t1_data_dir, "t1_run{}.h5".format(runNumber))
            channel_info = pd.read_hdf(t1_file,key="ORGretina4MWaveformDecoder")
            chan_dfs.append(channel_info)

        chan_df = pd.concat(chan_dfs, axis=0)
        chanList = np.unique(chan_df["channel"])

        # print(chan_df)
        # print(channel_info)

        NLCMap = []
        for ccc in chanList:
            try:
                # boardSN = chan_df.loc[ccc].board_id
                # boardSN = chan_df.loc[chan_df['channel'] == ccc].board_id
                boardSN = chan_df.loc[chan_df['channel'] == ccc].board_id[1]
                # NLCMap[chan] = load_nonlinearities(serial, chan, runList)
                print("For {}, board SN is {}".format(ccc,boardSN))

            except KeyError:
                print("Channel {} not there?".format(ccc))
                continue

            crate = ccc >> 9
            card = (ccc & 0xff) >>4
            channel = (ccc & 0xf)
            backwards = (crate << 9) + (card << 4) + channel

            NLCMapDir = os.path.join(os.getenv("DATADIR", "."), "mjd","NLCDB")

            #  // The board labeled SN-021h return 0x221 when probed by ORCA
            #  // Our NLC folders use the board labels so change this one.
            if (boardSN == 0x221): boardSN = 0x21;

            for run in runList:
                if (run <= 11396) or run > 18643:
                    print("This very simple code was ported only from runs 11396 to 18643, go check GAT to figure out what to do")
                    exit()

            if(boardSN > 0xf): bsnString = "{}h".format( hex(boardSN) )
            else: bsnString = "{}".format( hex(boardSN) )

            #drop the x in the 0x hex numbers
            bsnString = bsnString[0] + bsnString[2:]

            fileName1a = "%s/Boards/%s/c%dslot%d/Crate%d_GRET%d_Ch%d_part1a.dat" % (
                        NLCMapDir, bsnString, crate, card, crate, card, channel)
            fileName2a = "%s/Boards/%s/c%dslot%d/Crate%d_GRET%d_Ch%d_part2a.dat" % (
                        NLCMapDir, bsnString, crate, card, crate, card, channel)

            # print(fileName1a)
            adc1, vals1 = np.loadtxt(fileName1a, unpack=True)
            adc2, vals2 = np.loadtxt(fileName2a, unpack=True)
            if not np.array_equal(adc1, adc2):
                print("ADC values between files not the same!")
                exit(0)

            row = { "channel": ccc,
                    "adcs": adc1,
                    "nonlin1": vals1,
                    "nonlin2": vals2
            }

            NLCMap.append(row)


        df_nl = pd.DataFrame(NLCMap)
        print(df_nl)
        df_nl.set_index("channel", drop=False, inplace=True)


        df_nl.to_hdf(self.nl_file_name, key=self.data_key, mode='w')

        # map1->LoadFromCombinedFile(fileName1a);
        # NLCMaps[ddID] = map1;
        # map2->LoadFromCombinedFile(fileName2a);
        # NLCMaps2[ddID] = map2;

    def get_processor_list(self, ):
        procs = TierOneProcessorList()

        #pass energy thru to t1
        procs.AddFromTier0("channel")
        procs.AddFromTier0("energy", "onboard_energy")
        # procs.AddFromTier0("event_number")
        procs.AddFromTier0("timestamp")

        #is the wf saturated?
        procs.AddCalculator(is_saturated, {}, output_name="is_saturated")

        # Trim a few values from the beginning and end
        procs.AddTransform(trim_waveform, {"n_samples_before":15,
                                           "n_samples_after":5
                                           }, input_waveform="waveform",output_waveform="trim_wf")

        #nonlinearity_correct
        # db_path = self.nl_file_name
        # procs.AddDatabaseLookup(get_nonlinearity, {"channel": "channel", "db_path":db_path}, output_name=["nlc1", "nlc2"])
        # procs.AddTransform(nonlinearity_correct, {"time_constant_samples":190,
        #                                           "fNLCMap": "nlc1",
        #                                           "fNLCMap2":"nlc2"
        #                                         }, input_waveform="waveform", output_waveform="nlc_wf")

        #baseline remove
        procs.AddCalculator(fit_baseline, {"end_index":700}, input_waveform="waveform", output_name=["bl_slope", "bl_int"])
        procs.AddTransform(remove_baseline, {"bl_0":"bl_int", "bl_1":"bl_slope"}, input_waveform="waveform", output_waveform="blrmnlc_wf")

        #calculate max currents from baseline-removed wf with a few different sigma vals
        for sig in [5,10,15]:
            procs.AddCalculator(current_max, {"sigma":sig}, input_waveform="blrmnlc_wf", output_name="current_max_{}".format(sig))

        #estimate t0
        procs.AddCalculator(t0_estimate, {}, input_waveform="blrmnlc_wf", output_name="t0est")
        procs.AddCalculator(max_time, {}, input_waveform="blrmnlc_wf", output_name="t_max")

        #energy estimator: pz correct, calc trap
        procs.AddTransform(pz_correct, {"rc":72}, input_waveform="blrmnlc_wf", output_waveform="pz_wf")
        procs.AddTransform(trap_filter, {"rampTime":400, "flatTime":200}, input_waveform="pz_wf", output_waveform="trap_wf")

        # procs.AddCalculator(trap_max, {}, input_waveform="trap_wf", output_name="trap_max")
        procs.AddCalculator(trap_max, {"method":"fixed_time","pickoff_sample":400}, input_waveform="trap_wf", output_name="trap_ft")
        procs.AddCalculator(trap_max, {"method":"max","pickoff_sample":0}, input_waveform="trap_wf", output_name="trap_max")

        #calculate a few time points
        tps = np.array([0.3,0.4,0.5,0.6,0.7,0.8, 0.9, 0.95,0.99])
        output_names = []
        for tp in tps:
            output_names.append("tp_{:.0f}".format(tp*100))
        procs.AddCalculator(calc_timepoint, {"percentage":tps, "do_interp":True, "norm":"trap_max"}, input_waveform="blrmnlc_wf", output_name=output_names)


        # procs.AddCalculator(fit_baseline, {"start_index":1150, "end_index":-1, "order":0}, input_waveform="pz_wf", output_name="ft_mean")
        # procs.AddCalculator(fit_baseline, {"start_index":1150, "end_index":-1, "order":1}, input_waveform="pz_wf", output_name=["ft_slope", "ft_int"])

        return procs

    def apply_ae(self, df_chan, ae_chan):
        # print(df_chan["channel"].unique())

        #calc a vs e
        avse0, avse1, avse2 = ae_chan.avse0, ae_chan.avse1, ae_chan.avse2
        e = df_chan[self.ecal_name]
        a_vs_e = df_chan[ae_chan.current_name] - (avse2*e**2 + avse1*e + avse0 )

        df_chan["ae_pass"] = a_vs_e > ae_chan.avse_cut
        df_chan["ae"] = (a_vs_e - ae_chan.avse_mean)/ae_chan.avse_std

        return df_chan

    def apply_baseline_cut(self, df_chan, bl_chan, settle_time = 20):
        '''
        cut for baselines & time since last
        '''

        #cut based on time since last
        cut = df_chan.prev_t*10/1E6 > settle_time

        #Cut funny baselines
        for param in ["bl_int", "bl_slope"]:
            min_name = "{}_min".format(param)
            max_name = "{}_max".format(param)
            cut &= (df_chan[param] > bl_chan[min_name]) & (df_chan[param] < bl_chan[max_name])

        df_chan["bl_cut"] = cut
        return df_chan


    def tag_training_candidates(self, df_chan, df_bl, df_ae, settle_time=20):
        df_chan["is_training"] = 0

        min_e = 1400 #keV
        # df["ae_pass"] = np.nan
        # df["ae"] = np.nan
        # df["bl_cut"] = np.nan
        # df["drift_time"] = np.nan

        # for channel, df_chan in df.groupby("channel"):
        channel = df_chan.channel.unique()[0]
        ae_chan = df_ae.loc[channel]
        bl_chan = df_bl.loc[channel]

        #A/E cut
        avse0, avse1, avse2 = ae_chan.avse0, ae_chan.avse1, ae_chan.avse2
        e = df_chan[self.ecal_name]
        a_vs_e = df_chan[ae_chan.current_name] - (avse2*e**2 + avse1*e + avse0 )

        df_chan["ae_pass"] = a_vs_e > ae_chan.avse_cut
        df_chan["ae"] = (a_vs_e - ae_chan.avse_mean)/ae_chan.avse_std

        #baseline cut
        bl_cut = np.ones(len(df_chan), dtype=np.bool)

        #Cut funny baselines
        for param in ["bl_int", "bl_slope"]:
            min_name = "{}_min".format(param)
            max_name = "{}_max".format(param)
            bl_cut &= (df_chan[param] > bl_chan[min_name]) & (df_chan[param] < bl_chan[max_name])
        df_chan["bl_cut"] = bl_cut

        #Make a cut based on drift t0-t99 drift time

        df_chan["drift_time"] = df_chan[self.dt_max_param] - df_chan["t0est"]
        cut = df_chan["bl_cut"] & (df_chan.ae>0)&(df_chan.ae<2) & (df_chan[self.ecal_name] > min_e)

        # df_cut = df_cut[(df_cut.drift_time > df_cut.drift_time.quantile(q=0.025))   & (df_cut.drift_time < df_cut.drift_time.quantile(q=0.975)) ]
        dt_max = df_chan[cut].drift_time.quantile(q=0.99)

        cut = cut & (df_chan.drift_time < dt_max) & (df_chan.drift_time > 0)
        df_cut = df_chan[ cut ]
        dt_ae = df_cut.drift_time

        hist,bins = np.histogram(dt_ae, bins=100)

        bin_centers = get_bin_centers(bins)
        idxs_over_50 = hist > 0.1*np.amax(hist)
        first_dt =  bin_centers[np.argmax(idxs_over_50)]
        last_dt = bin_centers[  len(idxs_over_50) - np.argmax(idxs_over_50[::-1]) -1  ]

        dt_cut = (df_chan.drift_time >= first_dt) & (df_chan.drift_time <= last_dt)

        #Make a cut based on t50-t99 drift time (good for PC events)
        pc_fig = plt.figure()
        df_chan["t50_99"] = df_chan.tp_99 - df_chan.tp_50
        df_pc = df_chan[ cut & (df_chan.drift_time >= first_dt) & (df_chan.drift_time <= last_dt) ]
        cut_lo, cut_hi, __, __ = pdc.gaussian_cut(df_pc.t50_99, cut_sigma=3, plotAxis = plt.gca())
        plt.xlabel("t50-t99 time")

        t50_99_cut = (df_chan.t50_99 > cut_lo) & (df_chan.t50_99 < cut_hi)

        training_cut = cut & dt_cut & t50_99_cut & (df_chan.prev_t > settle_time)
        weak_cut = df_chan["bl_cut"] & (df_chan.ae>-10)&(df_chan.ae<40) & (df_chan[self.ecal_name] > min_e) & (df_chan.drift_time < dt_max) & (df_chan.drift_time > 0)

        df_chan["is_training"] = training_cut

        if True:
            f1 = plt.figure()
            grid = gs.GridSpec(2, 2, height_ratios=[3, 1], width_ratios = [1,3])
            ax = plt.subplot(grid[0, 1])
            ax_y = plt.subplot(grid[0, 0], sharey = ax)
            ax_x = plt.subplot(grid[1, 1], sharex = ax)

            ax_y.hist(df_cut.ae, bins="auto", histtype="step", orientation='horizontal')

            ax_x.plot(bin_centers, hist, ls="steps")

            ax_x.axvline(first_dt, c="r")
            ax_x.axvline(last_dt, c="r")

            ax.scatter(df_chan.drift_time[weak_cut&~training_cut], df_chan.ae[weak_cut&~training_cut], s=0.5, c="k" )
            ax.scatter(df_chan.drift_time[training_cut], df_chan.ae[training_cut], s=0.5, c="g" )

            ax_x.set_xlabel("Drift time [t0-95%]")
            ax_y.set_ylabel("A vs E")

            try: os.mkdir("training_plots")
            except OSError: pass
            f1.savefig("training_plots/chan{}_timepoints".format(channel))
            pc_fig.savefig("training_plots/chan{}_t50-99".format(channel))
            plt.close(f1)
            plt.close(pc_fig)
            # f2.savefig("training_plots/chan{}_waveforms".format(channel))

        return df_chan


    def save_training_data(self, runList, file_name, chanList=None, settle_time=20):
        '''
        settle_time in ms is minimum time since previous event (on same channel)
        '''
        if chanList is None: chanList = self.detectorChanList

        training_df = []

        for runNumber in runList:

            t1_file = os.path.join(self.t1_data_dir,"t1_run{}.h5".format(runNumber))
            t2_file = os.path.join(self.t2_data_dir, "t2_run{}.h5".format(runNumber))

            df = pd.read_hdf(t2_file,key="data")
            tier1 = pd.read_hdf(t1_file,key=self.data_key)
            tier1 = tier1.drop({"channel", "energy", "timestamp"}, axis=1)

            df = df.join(tier1, how="inner")

            df_train = df.loc[df.is_training==1]
            training_df.append(df_train )

        df_train = pd.concat(training_df, axis=0)
        df_train.to_hdf(file_name, key="data", mode='w')

        #TODO: if the multisampling params changed in the middle of this run range, you're hosed.
        g4 = dl.Gretina4MDecoder(t1_file)

        if True:
            for channel, df_chan in df_train.groupby("channel"):
                n_bins = 10
                dt_bins = np.linspace(df_chan.drift_time.min(), df_chan.drift_time.max(), n_bins+1)

                f2 = plt.figure(figsize=(12,8))
                for b_lo, b_hi in zip(dt_bins[:-1], dt_bins[1:]):
                    df_bin = df_chan[(df_chan.drift_time >= b_lo) & (df_chan.drift_time<b_hi)]
                    for i, (index, row) in enumerate(df_bin.iterrows()):
                        if i>=50: break

                        wf=g4.parse_event_data(row)
                        wf_full = wf.get_waveform()
                        wf_full -= row["bl_int"] + np.arange(len(wf_full))*row["bl_slope"]
                        wf_full /= row[self.ecal_name]

                        # t95_int = int(row[self.dt_max_param])
                        t95_int = np.argmax(wf_full > 0.95)
                        wf_plot = wf.data[t95_int-200:t95_int+100]
                        if i == 0:
                            p = plt.plot(wf_plot, alpha=0.1)
                        else:
                            plt.plot(wf_plot, c=p[0].get_color(), alpha=0.1)

                # plt.show()
                # exit()

                try: os.mkdir("training_plots")
                except OSError: pass
                f2.savefig("training_plots/chan{}_waveforms".format(channel))
                plt.close(f2)

    def save_subset(self, channel, n_waveforms, training_data_file_name, output_file_name, exclude_list = [], do_plot=True):

        df_train = pd.read_hdf(training_data_file_name,key="data")
        df_train = df_train[df_train.channel == channel]

        first_dt = df_train.drift_time.min()
        last_dt = df_train.drift_time.max()

        n_bins_time = n_waveforms

        dt_bins = np.linspace(first_dt, last_dt, n_bins_time+1)

        wfs_per_bin = 1

        wfs_saved = []

        for b_lo, b_hi in zip(dt_bins[:-1], dt_bins[1:]):
            df_bin = df_train[(df_train.drift_time >= b_lo) & (df_train.drift_time<b_hi)]
            for i, (index, row) in enumerate(df_bin.iterrows()):
                if index in exclude_list: continue

                t1_file = os.path.join(self.t1_data_dir, "t1_run{}.h5".format(row.runNumber))
                g4 = dl.Gretina4MDecoder(t1_file)
                wf=g4.parse_event_data(row)

                wf.training_set_index = index
                wf.amplitude = row.trap_max
                wf.bl_slope = row.bl_slope
                wf.bl_int = row.bl_int
                wf.t0_estimate = row.t0est
                wf.tp_50 = row.tp_50

                wfs_saved.append(wf)
                break

        np.savez(output_file_name, wfs=wfs_saved)

        if do_plot:
            try: os.mkdir("training_plots")
            except OSError: pass

            f, ax = plt.subplots(1,2, figsize=(12,8))
            # print ("Channel {} set:".format(channel))
            for wf in wfs_saved:
                # print("  index {}".format(wf.training_set_index))
                wf_window = wf.window_waveform()
                ax[0].plot( wf_window )
                ax[1].plot( wf_window / wf.amplitude )

            plt.savefig("training_plots/chan{}_{}wf_set.png".format(channel, n_waveforms))

            # plt.figure()
            # plt.hist(baseline_val_arr,bins="auto")
            # plt.show()

# def fit_tail(wf_data):
#     '''
#     try to fit out the best tail parameters to flatten the top
#     '''
#
#     from scipy import optimize
#     max_idx = np.argmax(wf_data)
#
#
#     def min_func(x):
#         rc_decay, overshoot_decay, overshoot_pole_rel, energy = x
#         # rc_decay, overshoot_decay, overshoot_pole_rel, energy, long_rc = x
#
#         rc_num, rc_den = filt.rc_decay(rc_decay*10)
#         wf_proc1 = signal.lfilter(rc_den, rc_num, wf_data)
#
#         # long_rc_num, long_rc_den = filt.rc_decay(long_rc*1000)
#         # wf_proc1 = signal.lfilter(long_rc_den, long_rc_num, wf_proc1)
#
#         overshoot_num, overshoot_den = filt.gretina_overshoot(overshoot_decay, overshoot_pole_rel)
#         wf_proc = signal.lfilter(overshoot_den, overshoot_num, wf_proc1)
#
#         tail_data = wf_proc[max_idx:]
#         flat_line = np.ones(len(tail_data))*energy*1000
#         return np.sum((tail_data-flat_line)**2)
#
#     wf_max = wf_data.max()/1000
#
#     res = optimize.minimize(min_func, [7.2, 2, -4, wf_max], method="Powell")
#     print(res["x"])
#     rc1, rc2, f, e = res["x"]
#
#     rc1*=10
#     rc_num, rc_den = filt.rc_decay(rc1)
#     wf_proc1 = signal.lfilter(rc_den, rc_num, wf_data)
#
#     # long_rc*=1000
#     # long_rc_num, long_rc_den = filt.rc_decay(long_rc)
#     # wf_proc1 = signal.lfilter(long_rc_den, long_rc_num, wf_proc1)
#
#     overshoot_num, overshoot_den = filt.gretina_overshoot(rc2, f)
#     wf_proc = signal.lfilter(overshoot_den, overshoot_num, wf_proc1)
#
#     return wf_proc, (e*1000), res["fun"]
