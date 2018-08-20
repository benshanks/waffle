#!/usr/bin/env python
# -*- coding: utf-8 -*-

from waffle.processing import *

def main():

    runList = np.arange(11510, 11600)

    mjdList = [
    582,583,580, 581,578, 579,
    692 ,693 ,648, 649 ,640, 641 ,
    610, 610,608, 609, 664, 665,
    #624, 625, 628, 629,688, 689, 694, 695, 614, 615,
    672, 673,
    632, 633,626, 627, 690, 691,
    600, 601, 598, 599,594, 595, 592, 593,
    ]

    #only take high gain channels for now
    chanList = [chan for chan in mjdList if chan%2==0]

    # chanList = [578, 582]

    #data processing

    proc = DataProcessor(detectorChanList=chanList)

    #Pygama processing
    runList = np.arange(11510,11515)
    proc.tier0(runList, chanList)
    df = proc.tier1(runList, num_threads=4, overwrite=True)
    exit()

    #Load all runs into one common DF
    df = proc.load_t2(runList)

    df = proc.tag_pulsers(df)
    df = df.groupby("channel").apply(proc.calibrate)
    df = df.groupby(["runNumber","channel"]).apply(proc.calculate_previous_event_params, baseline_meas="bl_int")

    proc.calc_baseline_cuts(df, settle_time=25) #ms
    proc.fit_pz(df)
    proc.calc_ae_cut(df )

    calculate cut of good training waveforms
    df_bl = pd.read_hdf(proc.channel_info_file_name, key="baseline")
    df_ae = pd.read_hdf(proc.channel_info_file_name, key="ae")
    df = df.groupby("channel").apply(proc.tag_training_candidates, df_bl=df_bl,df_ae=df_ae)

    proc.save_t2(df)

    proc.save_training_data(runList, "training_data/training_set.h5")

    n_waveforms = 8
    for chan in chanList:
        proc.save_subset(chan, n_waveforms, "training_data/training_set.h5", "training_data/chan{}_{}wfs.npz".format(chan, n_waveforms))


if __name__=="__main__":
    main()
