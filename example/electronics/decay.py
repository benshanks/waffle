 #!/usr/local/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from siggen import PPC
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d

from pygama.filters import rc_decay
from siggen.electronics import DigitalFilter

wf_length = 1000
det = PPC( os.path.join(os.environ["DATADIR"],  "siggen", "config_files", "bege.config"), wf_padding=1000, maxWfOutputLength=wf_length)
imp_avg = -2
imp_grad = 1.2
det.siggenInst.SetImpurityAvg(imp_avg, imp_grad)

nyq_freq = 0.5*1E9

def main():
    # poles(det)
    oscillation(det)
    # two_rc(det)

def two_rc(det):
    '''
    WHATS IT LOOK LIKE WITH A 72 US AND 2 MS DECAY?
    '''

    lowpass = DigitalFilter(2)
    lowpass.num = [1,2,1]
    lowpass.set_poles(0.975, 0.007)

    hipass = DigitalFilter(1)
    hipass.num, hipass.den = rc_decay(82, 1E9)

    det.AddDigitalFilter(lowpass)
    det.AddDigitalFilter(hipass)

    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, wf_length, smoothing=20))
    wf_compare = np.copy(wf_proc)

    f, ax = plt.subplots(2,1,figsize=(15,8))
    ax[0].plot (wf_compare,  color="r")

    # hipass2 = DigitalFilter(1)
    # hipass2.num, hipass2.den = rc_decay(2000, 1E9)
    # hipass.num, hipass.den = rc_decay(74.75, 1E9)
    # det.AddDigitalFilter(hipass2)

    mag = 1.-10.**-5.22
    phi = np.pi**-13.3
    det.RemoveDigitalFilter(hipass)
    hipass = DigitalFilter(2)
    hipass.num = [1,-2,1]
    hipass.set_poles(mag, phi)
    det.AddDigitalFilter(hipass)

    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, wf_length, smoothing=20))

    ax[0].plot (wf_proc,  color="g")
    ax[1].plot (wf_compare-wf_proc,  color="g")

    plt.show()
    exit()

def overshoot(det):
    '''
    How do I get me a decaying overshoot?
    '''

    lowpass = DigitalFilter(2)
    lowpass.num = [1,2,1]
    lowpass.set_poles(0.975, 0.007)

    hipass = DigitalFilter(1)
    hipass.num, hipass.den = rc_decay(82, 1E9)

    det.AddDigitalFilter(lowpass)
    # det.AddDigitalFilter(hipass)

    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, wf_length, smoothing=20))
    wf_compare = np.copy(wf_proc)

    f, ax = plt.subplots(2,2,figsize=(15,8))
    # plt.figure()
    ax[0,0].plot (wf_compare,  color="r")
    cmap = cm.get_cmap('viridis')

    new_filt = DigitalFilter(1)
    det.AddDigitalFilter(new_filt)

    p_mags = 1 - np.logspace(-4, -2, 100, base=10)
    z_mags = 1 - np.logspace(-4, -2, 20, base=10)

    for zero_mag in [5E-4]:#z_mags:
        for pole_mag in np.logspace(-7, -5, 100, base=10):#p_mags:
            zero_mag = 1-zero_mag
            pole_mag = zero_mag - pole_mag

            if zero_mag == pole_mag: continue

            color = "b"
            # color = cmap( (mag2 - mags[0])/(mags[-1] - mags[0]) )
            new_filt.set_zeros(zero_mag, 0)
            new_filt.set_poles(pole_mag, 0)

            # print (new_filt.num, np.sum(new_filt.num))
            # print (new_filt.den, np.sum(new_filt.den))
            # exit()

            wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=20))

            try:
                if wf_proc[155] < 1.00001: continue
                if np.amax(wf_proc) > 1.1: continue
                print( zero_mag, pole_mag)
                p = ax[0,0].plot (wf_proc)
                color = p[0].get_color()
                ax[1,0].plot (wf_proc-wf_compare,  color=color)
            except (TypeError, IndexError) as e:
                continue

            w,h2 = get_freq_resp(new_filt, w=np.logspace(-15, 0, 500, base=np.pi))
            ax[0,1].loglog( w, h2, color=color)

            p[0] = ax[1,1].scatter(zero_mag, 0, color=color)
            ax[1,1].scatter(pole_mag, 0, color=color ,marker="x")
            an = np.linspace(0,np.pi,200)
            ax[1,1].plot(np.cos(an), np.sin(an), c="k")
            ax[1,1].plot(np.cos(an), -np.sin(an), c="k")
            ax[1,1].axis("equal")

    plt.show()

def oscillation(det):
    '''
    How do I get me a decaying oscillation?
    '''

    lowpass = DigitalFilter(2)
    lowpass.num = [1,2,1]
    lowpass.set_poles(0.975, 0.007)

    hipass = DigitalFilter(1)
    hipass.num, hipass.den = rc_decay(82, 1E9)

    det.AddDigitalFilter(lowpass)
    # det.AddDigitalFilter(hipass)

    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, wf_length, smoothing=20))
    wf_compare = np.copy(wf_proc)

    f, ax = plt.subplots(2,2,figsize=(15,8))
    # plt.figure()
    ax[0,0].plot (wf_compare,  color="r")
    cmap = cm.get_cmap('viridis')

    new_filt = DigitalFilter(2)
    det.AddDigitalFilter(new_filt)
    new_filt.num = [1,2,1]

    p_mags = 1 - np.logspace(-3, -2, 20, base=10)
    p_phis = [0.5*np.pi**-3]
    # p_phis = np.logspace(-5, 1, 4, base=np.pi)
    # z_mags = 1 - np.logspace(-4, -2, 20, base=10)

    pole_phi = 0.5*np.pi**-3
    pole_mag = 0.995

    for i in range(3):

    # for pole_phi in p_phis:
    #     for pole_mag in p_mags:
            # zero_mag = 1-zero_mag
            # pole_mag = zero_mag + pole_mag
            # if zero_mag == pole_mag: continue

            color = get_color(cmap, pole_mag, p_mags)
            if i == 0:
                color = "k"
            elif i == 1:
                new_filt.set_zeros(0.99, 0.01)
                color = "b"
            elif i == 2:
                new_filt.set_zeros(0.995, 0.001)
                color = "g"
            new_filt.set_poles(pole_mag, pole_phi)

            # print (new_filt.num, np.sum(new_filt.num))
            # print (new_filt.den, np.sum(new_filt.den))
            # exit()

            wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=20))

            try:
                # if wf_proc[155] < 1.00001: continue
                # # if np.amax(wf_proc) > 1.1: continue
                # print( zero_mag, pole_mag)
                p = ax[0,0].plot (wf_proc,  color=color)
                # color = p[0].get_color()
                ax[1,0].plot (wf_proc-wf_compare,  color=color)
            except (TypeError, IndexError) as e:
                continue

            w,h2 = get_freq_resp(new_filt, w=np.logspace(-15, 0, 500, base=np.pi))
            ax[0,1].loglog( w, h2, color=color)
            ax[0,1].axvline(pole_phi/ (np.pi /nyq_freq))

            # p[0] = ax[1,1].scatter(zero_mag, 0, color=color)
            ax[1,1].scatter(pole_mag*np.cos(pole_phi), pole_mag*np.sin(pole_phi), color=color ,marker="x")
            an = np.linspace(0,np.pi,200)
            ax[1,1].plot(np.cos(an), np.sin(an), c="k")
            ax[1,1].plot(np.cos(an), -np.sin(an), c="k")
            ax[1,1].axis("equal")

    plt.show()

def poles(det):

    lowpass = DigitalFilter(2)
    lowpass.num = [1,2,1]
    lowpass.set_poles(0.975, 0.007)

    mag = 1.-10.**-7
    phi = np.pi**-13.3

    hipass = DigitalFilter(2)
    hipass.num = [1,-2,1]
    hipass.set_poles(mag, phi)

    det.AddDigitalFilter(lowpass)
    det.AddDigitalFilter(hipass)



    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=20))
    wf_compare = np.copy(wf_proc)

    f, ax = plt.subplots(2,2,figsize=(15,8))
    ax[0,0].plot (wf_compare,  color="r")


    cmap = cm.get_cmap('viridis')

    phis = np.logspace(-20, -8, 100, base=np.pi)
    mags = 1 - np.logspace(-7.5, -6.5, 100, base=10)
    # phis = np.pi - phis

    w,h = get_freq_resp(hipass)
    ax[0,1].loglog( w, h, color="r")

    # for phi2 in phis:
        # color = cmap( (phi2 - phis[0])/(phis[-1] - phis[0]) )
        # mag2=mag
    for mag2 in mags:
        phi2=phi
        color = cmap( (mag2 - mags[0])/(mags[-1] - mags[0]) )

        hipass.set_poles(mag2, phi2)

        wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=20))

        try:
            ax[0,0].plot (wf_proc, label="{}, {}".format(phi,mag), color=color)
            ax[1,0].plot (wf_proc-wf_compare, label="{}, {}".format(phi,mag), color=color)
        except TypeError:
            continue

        w,h2 = get_freq_resp(hipass)
        ax[0,1].loglog( w, h2, color=color)


    # plt.legend()
    plt.show()

def get_freq_resp(hipass1, w=None):
    freq_samp = 1E9
    nyq_freq = 0.5*freq_samp

    if w is None:
        w = np.logspace(-15, -7, 500, base=np.pi)

    w, h = signal.freqz(hipass1.num/ (np.sum(hipass1.num)/np.sum(hipass1.den)), hipass1.den, worN=w)

    # if p_mag2 is not None:
    #     den2 = em.zpk_to_ba(p_mag2, p_phi2)
    #     w, h2 = signal.freqz(num, den2, worN=w)
    #     h *= h2

    w/= (np.pi /nyq_freq)

    return w, np.abs(h)

def get_color(cmap, val, val_list):
    color = cmap( (val - val_list[0])/(val_list[-1] - val_list[0]) )
    return color

if __name__ == "__main__":
    main()
