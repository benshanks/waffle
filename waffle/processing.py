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
from pygama.data_cleaning import *
from pygama.waveform import *

from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import describe, skewnorm, exponnorm, norm
from scipy.interpolate import spline, interp1d

class DataProcessor():

    def __init__(self):
        #energy & current estimator to use for A/E
        self.energy_name = "trap_max"
        self.current_name = "current_max_5"

        #File hierarchy for this work:
        #Base dir is from $DATADIR environment variable
        self.mjd_data_dir = os.path.join(os.getenv("DATADIR", "."), "mjd")
        self.raw_data_dir = os.path.join(self.mjd_data_dir,"raw")
        self.t1_data_dir = os.path.join(self.mjd_data_dir,"t1")
        self.t2_data_dir = os.path.join(self.mjd_data_dir,"t2")
        self.nl_file_name = os.path.join(self.mjd_data_dir,"NLCDB", "nonlinearities.h5")
        self.cal_file_name = os.path.join(self.mjd_data_dir,"analysis", "calibration.h5")
        self.psa_file_name = os.path.join(self.mjd_data_dir,"analysis", "psa.h5")
        self.blcut_file_name = os.path.join(self.mjd_data_dir,"analysis", "blcuts.h5")

    def baseline_cuts(self, runList, chanList=None):

        dfs = []
        for runNumber in runList:
            t2_file = os.path.join(self.t2_data_dir, "t2_run{}.h5".format(runNumber))
            tier2 = pd.read_hdf(t2_file,key="data")
            dfs.append(tier2)

        df = pd.concat(dfs, axis=0)
        cal = pd.read_hdf(self.cal_file_name,key="cal")

        if chanList is None: chanList = np.unique(cal["channel"])
        bl_map = []
        try: os.mkdir("bl_plots")
        except OSError: pass
        for channel in chanList:
            cal_chan = cal.loc[channel]
            df_chan = df[ df["channel"] == channel]

            f = plt.figure()
            bl_int_cuts = gaussian_cut(df_chan.bl_int,  cut_sigma=1.5, plotFigure=f)
            plt.xlabel("bl intercept")
            plt.savefig("bl_plots/bl_int_channel{}.png".format(channel))
            plt.close()

            f2 = plt.figure()
            bl_slope_cuts = gaussian_cut(df_chan.bl_slope,  plotFigure=f2)
            plt.xlabel("bl slope")
            plt.savefig("bl_plots/bl_slope_channel{}.png".format(channel))
            plt.close()

            row = {"channel": channel,
                   "bl_slope_min": bl_slope_cuts[0],
                   "bl_slope_max": bl_slope_cuts[1],
                   "bl_int_min": bl_int_cuts[0],
                   "bl_int_max": bl_int_cuts[1]
                   }
            bl_map.append(row)

        df_bl = pd.DataFrame(bl_map)
        df_bl.set_index("channel", drop=False, inplace=True)
        df_bl.to_hdf(self.blcut_file_name,   key="bl", mode='w')

    def ae_cut(self, runList, chanList=None):
        energy_name = self.energy_name
        current_name = self.current_name

        dfs = []
        for runNumber in runList:
            t2_file = os.path.join(self.t2_data_dir, "t2_run{}.h5".format(runNumber))
            tier2 = pd.read_hdf(t2_file,key="data")
            dfs.append(tier2)
        df = pd.concat(dfs, axis=0)
        cal = pd.read_hdf(self.cal_file_name,key="cal")

        if chanList is None: chanList = np.unique(cal["channel"])

        f = plt.figure(figsize=(12,6))

        ae_map = []
        try: os.mkdir("ae_plots")
        except OSError: pass
        for channel in chanList:
            cal_chan = cal.loc[channel]
            df_chan = df[ df["channel"] == channel]

            energy_cal = df_chan[energy_name]*cal_chan.slope + cal_chan.offset
            current = df_chan[current_name]

            avse2, avse1, avse0, avse_cut, avse_mean, avse_std = get_avse_cut(energy_cal, current, f)

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
        df_ae.to_hdf(self.psa_file_name,   key="ae", mode='w')


    def calibrate(self, runList):
        energy_name = self.energy_name

        dfs = []
        for runNumber in runList:
            t2_file = os.path.join(self.t2_data_dir, "t2_run{}.h5".format(runNumber))
            tier2 = pd.read_hdf(t2_file,key="data")
            dfs.append(tier2)

        df = pd.concat(dfs, axis=0)
        chanList = np.unique(df["channel"])

        peak_energies = [238, 510.77, 583.191, 860.564, 2614.533]
        cal_map = []

        try: os.mkdir("cal_plots")
        except OSError: pass

        plt.ion()
        f1 = plt.figure(figsize=(16,6))

        for chan in chanList:
            df_chan = df[ df["channel"] == chan ]
            energy_chan = df_chan[energy_name]

            print("Channel {} has {} entries...".format(chan, len(energy_chan)))

            if len(energy_chan) < 5000:
                print("...skipping this channel")
                continue

            linear_cal = calibrate_tl208(energy_chan, peak_energies, plotFigure=f1)

            row = {"channel": chan,
                   "slope": linear_cal[0],
                   "offset": linear_cal[1]
                   }
            cal_map.append(row)

            plt.savefig("cal_plots/cal_channel{}.png".format(chan))

            # val = input("q to quit, else to continue...")
            # if val == "q": exit()

        df_cal = pd.DataFrame(cal_map)
        df_cal.set_index("channel", drop=False, inplace=True)
        df_cal.to_hdf(self.cal_file_name,   key="cal", mode='w')

        return cal_map

    def tier0(self, runList, min_signal_thresh, chanList=None):
        process_tier_0(self.raw_data_dir, runList, output_dir=self.t1_data_dir, min_signal_thresh=min_signal_thresh, chanList=chanList)

    def tier1(self, runList, num_threads):
        procs = self.get_processor_list()
        process_tier_1(self.t1_data_dir, runList, processor_list=procs, output_dir=self.t2_data_dir, num_threads=num_threads)

    def load_nonlinearities_to_db(self, runList):
        '''
        Reads in MJD nonlinearity text files, saves to a hdf Database
        '''

        chan_dfs = []
        for runNumber in runList:
            t1_file = os.path.join(self.t1_data_dir, "t1_run{}.h5".format(runNumber))
            channel_info = pd.read_hdf(t1_file,key="channel_info")
            chan_dfs.append(channel_info)

        chan_df = pd.concat(chan_dfs, axis=0)
        chanList = np.unique(chan_df["channel"])

        NLCMap = []
        for ccc in chanList:
            try:
                boardSN = channel_info.loc[ccc].board_id
                # NLCMap[chan] = load_nonlinearities(serial, chan, runList)
                # print(boardSN)
            except KeyError:
                print("Channel {} not there?".format(chan))

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
        df_nl.set_index("channel", drop=False, inplace=True)


        df_nl.to_hdf(self.nl_file_name, key="data", mode='w')

        # map1->LoadFromCombinedFile(fileName1a);
        # NLCMaps[ddID] = map1;
        # map2->LoadFromCombinedFile(fileName2a);
        # NLCMaps2[ddID] = map2;

    def get_processor_list(self, ):
        procs = TierOneProcessorList()

        #pass energy thru to t1
        # procs.AddFromTier0("energy")
        procs.AddFromTier0("channel")
        procs.AddFromTier0("energy", "onboard_energy")

        #is the wf saturated?
        procs.AddCalculator(is_saturated, {}, output_name="is_saturated")

        #nonlinearity_correct
        db_path = self.nl_file_name
        procs.AddDatabaseLookup(get_nonlinearity, {"channel": "channel", "db_path":db_path}, output_name=["nlc1", "nlc2"])
        procs.AddTransform(nonlinearity_correct, {"time_constant_samples":190,
                                                  "fNLCMap": "nlc1",
                                                  "fNLCMap2":"nlc2"
                                                }, input_waveform="waveform", output_waveform="nlc_wf")

        #baseline remove
        procs.AddCalculator(fit_baseline, {"end_index":700}, input_waveform="nlc_wf", output_name=["bl_slope", "bl_int"])
        procs.AddTransform(remove_baseline, {"bl_0":"bl_int", "bl_1":"bl_slope"}, input_waveform="nlc_wf", output_waveform="blrmnlc_wf")

        #calculate max currents from baseline-removed wf with a few different sigma vals
        for sig in [1,3,5,7]:
            procs.AddCalculator(current_max, {"sigma":sig}, input_waveform="blrmnlc_wf", output_name="current_max_{}".format(sig))

        #calculate a few time points (50%, 90%, 95%)
        tps = np.array([0.4,0.5,0.6,0.7,0.8, 0.9, 0.95,0.99])
        output_names = []
        for tp in tps:
            output_names.append("tp_{:.0f}".format(tp*100))

        procs.AddCalculator(calc_timepoint, {"percentage":tps, "do_interp":True}, input_waveform="blrmnlc_wf", output_name=output_names)

        #estimate t0
        # procs.AddTransform(savgol_filter, {"window_length":47, "order":2}, input_waveform="blrmnlc_wf", output_waveform="sg_wf")
        # procs.AddCalculator(t0_estimate, {}, input_waveform="sg_wf", output_name="t0est")

        #energy estimator: pz correct, calc trap
        procs.AddTransform(pz_correct, {"rc_1":72, "rc_2":2, "rc1_frac":0.99}, input_waveform="blrmnlc_wf", output_waveform="pz_wf")
        procs.AddTransform(trap_filter, {"rampTime":400, "flatTime":200}, input_waveform="pz_wf", output_waveform="trap_wf")

        procs.AddCalculator(trap_max, {}, input_waveform="trap_wf", output_name="trap_max")
        procs.AddCalculator(trap_max, {"method":"fixed_time","pickoff_sample":400}, input_waveform="trap_wf", output_name="trap_ft")

        # procs.AddCalculator(fit_baseline, {"start_index":1150, "end_index":-1, "order":0}, input_waveform="pz_wf", output_name="ft_mean")
        # procs.AddCalculator(fit_baseline, {"start_index":1150, "end_index":-1, "order":1}, input_waveform="pz_wf", output_name=["ft_slope", "ft_int"])

        return procs

    def save_training_data(self, runList, chanList, file_name):
        df_all_runs = []
        for runNumber in runList:
            t1_file = os.path.join(self.t1_data_dir,"t1_run{}.h5".format(runNumber))
            t2_file = os.path.join(self.t2_data_dir, "t2_run{}.h5".format(runNumber))
            df = pd.read_hdf(t2_file,key="data")
            tier1 = pd.read_hdf(t1_file,key="data")
            df = df.drop({"channel", "energy", "timestamp"}, axis=1)
            df = df.join(tier1)
            df_all_runs.append(df)

        df = pd.concat(df_all_runs, axis=0)
        df_cal = pd.read_hdf(self.cal_file_name,key="cal")
        df_bl = pd.read_hdf(self.blcut_file_name,key="bl")
        df_ae_all = pd.read_hdf(self.psa_file_name, key="ae")



        if chanList is None: chanList = np.unique(df_cal["channel"])

        all_training_data = []
        for channel in chanList:

            df_chan = df[df.channel == np.int(channel)]
            bl_chan = df_bl.loc[channel]
            cal_chan = df_cal.loc[channel]
            ae_chan = df_ae_all.loc[channel]

            #Cut funny baselines
            for param in ["bl_int", "bl_slope"]:
                min_name = "{}_min".format(param)
                max_name = "{}_max".format(param)
                df_chan = df_chan[ (df_chan[param] > bl_chan[min_name]) & (df_chan[param] < bl_chan[max_name]) ]

            #energy calibrate
            energy_name = ae_chan.energy_name
            energy_cal = df_chan[energy_name]*cal_chan.slope + cal_chan.offset
            df_chan["energy_cal"] = energy_cal

            #calc a vs e
            avse0, avse1, avse2 = ae_chan.avse0, ae_chan.avse1, ae_chan.avse2
            e = df_chan.energy_cal
            a_vs_e = df_chan[ae_chan.current_name] - (avse2*e**2 + avse1*e + avse0 )

            df_chan["ae_pass"] = a_vs_e > ae_chan.avse_cut
            df_chan["ae"] = (a_vs_e - ae_chan.avse_mean)/ae_chan.avse_std

            df_reduced = df_chan[ (df_chan.energy_cal > 1400) ]

            df_ae = df_reduced[(df_reduced.ae > -10) & (df_reduced.ae < 40)]

            # times = [0.3,0.5,0.7,0.9,0.95]
            times = [0,0.3,0.95]
            tp_train, df_ae = calculate_timepoints(df_ae, times, relative_tp=0)
            x_idx = -1

            dt = tp_train[:,x_idx]
            df_ae["drift_time"] = dt
            df_ae["t0_est"] = tp_train[:,0]

            dt_ae = dt[(df_ae.ae>0)&(df_ae.ae<2)]
            df_cut = df_ae[(df_ae.ae>0)&(df_ae.ae<2)]


            hist,bins = np.histogram(dt_ae, bins=100)

            bin_centers = get_bin_centers(bins)
            idxs_over_50 = hist > 0.1*np.amax(hist)
            first_dt =  bin_centers[np.argmax(idxs_over_50)]
            last_dt = bin_centers[  len(idxs_over_50) - np.argmax(idxs_over_50[::-1])  ]

            train_set = (df_ae.ae>0)&(df_ae.ae<2)&(dt >= first_dt)&(dt < last_dt)

            all_training_data.append(df_ae[train_set] )

            if True:
                f1 = plt.figure()
                grid = gs.GridSpec(2, 2, height_ratios=[3, 1], width_ratios = [1,3])
                ax = plt.subplot(grid[0, 1])
                ax_y = plt.subplot(grid[0, 0], sharey = ax)
                ax_x = plt.subplot(grid[1, 1], sharex = ax)

                ax_y.hist(df_ae.ae, bins="auto", histtype="step", orientation='horizontal')

                ax_x.plot(bin_centers, hist, ls="steps")

                ax_x.axvline(first_dt, c="r")
                ax_x.axvline(last_dt, c="r")

                ax.scatter(dt[train_set], df_ae.ae[train_set], s=0.5, c="g" )
                ax.scatter(dt[~train_set], df_ae.ae[~train_set], s=0.5, c="k" )

                ax_x.set_xlabel("Drift time [t0-95%]")
                ax_y.set_ylabel("A vs E")

                n_bins = 10
                dt_bins = np.linspace(first_dt, last_dt, n_bins+1)

                f2 = plt.figure()

                for b_lo, b_hi in zip(dt_bins[:-1], dt_bins[1:]):
                    df_bin = df_cut[(dt_ae >= b_lo) & (dt_ae<b_hi)]
                    for i, (index, row) in enumerate(df_bin.iterrows()):
                        if i>=50: break
                        wf = Waveform( row, )
                        wf.data -= row["bl_int"] + np.arange(len(wf.data))*row["bl_slope"]
                        wf.data /= row["energy_cal"]

                        t0_est = np.argmax(wf.data > 0.95)

                        wf_plot = wf.data[t0_est-200:t0_est+100]

                        if i == 0:
                            p = plt.plot(wf_plot, alpha=0.1)
                        else:
                            plt.plot(wf_plot, c=p[0].get_color(), alpha=0.1)

                try: os.mkdir("training_plots")
                except OSError: pass
                f1.savefig("training_plots/chan{}_timepoints".format(channel))
                f2.savefig("training_plots/chan{}_waveforms".format(channel))

        df_train = pd.concat(all_training_data, axis=0)
        df_train.to_hdf(file_name, key="data", mode='w')


    def save_subset(self, channel, n_waveforms, training_data_file_name, output_file_name, exclude_list = [], do_plot=True):

        df_train = pd.read_hdf(training_data_file_name,key="data")
        df_train = df_train[df_train.channel == channel]

        first_dt = df_train.drift_time.min()
        last_dt = df_train.drift_time.max()

        n_bins_time = n_waveforms

        dt_bins = np.linspace(first_dt, last_dt, n_bins_time+1)

        wfs_per_bin = 1

        wfs_saved = []

        baseline_val_arr = np.empty(0)

        for b_lo, b_hi in zip(dt_bins[:-1], dt_bins[1:]):
            df_bin = df_train[(df_train.drift_time >= b_lo) & (df_train.drift_time<b_hi)]
            for i, (index, row) in enumerate(df_bin.iterrows()):
                if index in exclude_list: continue
                wf = Waveform( row, amplitude=row["trap_max"], bl_slope=row["bl_slope"], bl_int=row["bl_int"], t0_estimate=row["t0_est"])
                wf.training_set_index = index
                baseline_val_arr = np.append(  baseline_val_arr, (wf.data - (row["bl_slope"]*np.arange(len(wf.data))  + row["bl_int"]))[:700]   )

                wfs_saved.append(wf)
                break

        np.savez(output_file_name, wfs=wfs_saved)

        if do_plot:
            try: os.mkdir("training_plots")
            except OSError: pass

            plt.figure()
            # print ("Channel {} set:".format(channel))
            for wf in wfs_saved:
                # print("  index {}".format(wf.training_set_index))
                plt.plot(wf.data)

            plt.savefig("training_plots/chan{}_{}wf_set.png".format(channel, n_waveforms))

            # plt.figure()
            # plt.hist(baseline_val_arr,bins="auto")
            # plt.show()

def calculate_timepoints(df, time_points, relative_tp=0.5):

    tp_calc = np.zeros((len(df), len(time_points)))
    max_idx = np.zeros(len(df), dtype=np.int)
    min_val = np.zeros(len(df))
    max_val = np.zeros(len(df))

    for i, (index, row) in enumerate(df.iterrows()):
        wf_cent = get_waveform(row, doInterp=False)
        max_idx[i] = np.argmax(wf_cent)
        min_val[i] = np.amin(wf_cent)
        max_val[i] = np.amax(wf_cent)
        if max_idx[i] > 350 or min_val[i] > 0.001 or max_val[i] < 0.99:
            continue

        if relative_tp == 0:
            tp_rel = t0_estimate(wf_cent)
        else:
            tp_rel = calc_timepoint(wf_cent, percentage=relative_tp, baseline=0, do_interp=True, doNorm=False)
        for j,tp in enumerate(time_points):
            if tp == 0:
                tp_calc[i,j] = t0_estimate(wf_cent)
            else:
                # tp_calc[i,j] = calc_timepoint(wf_cent, percentage=tp, baseline=0, do_interp=True,doNorm=False) - tp_rel
                approx_idx = np.argmax(wf_cent > tp)
                f = interp1d(wf_cent[approx_idx-5:approx_idx+5], np.arange(approx_idx-5, approx_idx+5))
                tp_calc[i,j] = f(tp)- tp_rel

        if np.any(tp_calc[i,:] < -195):
            plt.ion()
            plt.figure()
            plt.plot(wf_cent)
            print("This waveform is messing things up!!  minval is {}".format(min_val[i]))
            inp = input("q to quit, else to continue")
            if inp == "q": exit()
            plt.ioff()

    return tp_calc[(max_idx<250)&(min_val<0.01)&(max_val>0.99)], df[(max_idx<250)&(min_val<0.01)&(max_val>0.99)]

def get_waveform(row, align_tp=0.5, doInterp=True):
    import pygama.transforms as pgt

    wf = Waveform( row, )

    wf.data -= row["bl_int"] + np.arange(len(wf.data))*row["bl_slope"]
    wf.data /= row["trap_max"]

    tp = calc_timepoint(wf.data, align_tp, do_interp=True, doNorm=False)

    if doInterp:
        offset = tp - np.floor(tp)
        wf_interp = pgt.interpolate(wf.data, offset)
        wf_cent = pgt.center(wf_interp, int(np.floor(tp)), 200, 200)
    else:
        wf_cent = pgt.center(wf.data, int(np.floor(tp)), 200, 200)

    return wf_cent
