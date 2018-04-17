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

    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, wf_length, smoothing=None))
    wf_compare = np.copy(wf_proc)

    f, ax = plt.subplots(3,2,figsize=(15,8))
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

    pole_phi = 5E6 * (np.pi /nyq_freq)
    pole_mag = 0.995

    for i in range(3):

    # for pole_phi in p_phis:
    #     for pole_mag in p_mags:
            # zero_mag = 1-zero_mag
            # pole_mag = zero_mag + pole_mag
            # if zero_mag == pole_mag: continue

            color = get_color(cmap, pole_mag, p_mags)
            if i == 0:
                zero_mag, zero_phi = np.nan, np.nan
                color = "k"
            elif i == 1:
                zero_mag, zero_phi = pole_mag, 4E6 * (np.pi /nyq_freq)
                new_filt.set_zeros(zero_mag, zero_phi)
                color = "b"
            elif i == 2:
                zero_mag, zero_phi = pole_mag, 6E6 * (np.pi /nyq_freq)
                new_filt.set_zeros(zero_mag, zero_phi)
                color = "g"
            elif i == 3:
                zero_mag, zero_phi = 1.001, pole_phi
                new_filt.set_zeros(zero_mag, zero_phi)
                color = "purple"
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
            ax[1,1].scatter(zero_mag*np.cos(zero_phi), zero_mag*np.sin(zero_phi), color=color ,marker="o")
            an = np.linspace(0,np.pi,200)
            ax[1,1].plot(np.cos(an), np.sin(an), c="k")
            ax[1,1].plot(np.cos(an), -np.sin(an), c="k")
            ax[1,1].axis("equal")

            wf1 = wf_proc[np.argmax(wf_proc>1):]
            xf,power = signal.periodogram(wf1, fs=1E8, detrend="linear")
            ax[2,0].loglog(xf,power, color=color )

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
