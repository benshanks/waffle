 #!/usr/local/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from siggen import PPC
from scipy import signal
from waffle.models.electronics import HiPassFilterModel
em = HiPassFilterModel()

det = PPC( os.path.join(os.environ["DATADIR"],  "siggen", "config_files", "bege.config"), wf_padding=100)
imp_avg = -2
imp_grad = 1.2
det.siggenInst.SetImpurityAvg(imp_avg, imp_grad)
det.lp_order=2

det.lp_num = [1,2,1]
det.lp_den = em.zpk_to_ba(0.975, 0.007)

det.hp_order = 2
mag = 1.-10.**-5.145
phi = np.pi**-13.3
det.hp_num = [1,-2,1]
det.hp_den = em.zpk_to_ba(mag, phi)



def main():
    skew()

def skew():
    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=25))
    wf_compare = np.copy(wf_proc)

    det.smoothing_type = 1

    f, ax = plt.subplots(2,1,figsize=(15,8), sharex=True)
    ax[0].plot (wf_compare,  color="r")

    cmap = cm.get_cmap('viridis')

    # sigmas = np.linspace(0.1,30, 100)
    # color = cmap( (sigma - sigmas[0])/(sigmas[-1] - sigmas[0]) )
    skews = np.linspace(-5,5, 21)

    for skew in skews:
        color = cmap( (skew - skews[0])/(skews[-1] - skews[0]) )

        wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=25, skew=skew))

        try:
            ax[0].plot (wf_proc, label="{}, {}".format(phi,mag), color=color)
            ax[1].plot (wf_proc-wf_compare, label="{}, {}".format(phi,mag), color=color)
        except TypeError:
            continue

    plt.show()

def gaussian():

    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=None))
    wf_compare = np.copy(wf_proc)

    f, ax = plt.subplots(2,1,figsize=(15,8), sharex=True)

    ax[0].plot (wf_compare,  color="r")

    cmap = cm.get_cmap('viridis')

    sigmas = np.linspace(0.1,30, 100)

    for sigma in sigmas:
        color = cmap( (sigma - sigmas[0])/(sigmas[-1] - sigmas[0]) )

        wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=sigma))

        try:
            ax[0].plot (wf_proc, label="{}, {}".format(phi,mag), color=color)
            ax[1].plot (wf_proc-wf_compare, label="{}, {}".format(phi,mag), color=color)
        except TypeError:
            continue

    plt.show()

def zeros(det):
    det.lp_order=2

    det.lp_num = [1,2,1]
    det.lp_den = em.zpk_to_ba(0.975, 0.007)

    det.hp_order = 2


    mag = 1.-10.**-5.145
    phi = np.pi**-13.3
    det.hp_num = [1,-2,1]
    det.hp_den = em.zpk_to_ba(mag, phi)

    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=20))
    wf_compare = np.copy(wf_proc)

    f, ax = plt.subplots(2,2,figsize=(15,8))
    ax[0,0].plot (wf_compare,  color="r")

    cmap = cm.get_cmap('viridis')

    phis = np.logspace(-20, -8, 100, base=np.pi)
    mags = 1-np.logspace(-7, 7, 100, base=10)
    # phis = np.pi - phis

    w,h = get_freq_resp(mag, phi)
    ax[0,1].loglog( w, h, color="r")

    # for phi2 in phis:
        # color = cmap( (phi2 - phis[0])/(phis[-1] - phis[0]) )
        # mag2=mag
    for mag2 in mags:
        phi2= np.pi**-13

        color = cmap( (mag2 - mags[0])/(mags[-1] - mags[0]) )

        det.hp_num = em.zpk_to_ba(mag2, phi2)
        det.hp_num /= np.sum(det.hp_num)/np.sum(det.hp_den)

        wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=20))

        w,h2 = get_freq_resp(mag, phi, mag2, phi2)
        ax[0,1].loglog( w, h2, color=color)

        try:
            ax[0,0].plot (wf_proc, label="{}, {}".format(phi,mag), color=color)
            ax[1,0].plot (wf_proc-wf_compare, label="{}, {}".format(phi,mag), color=color)
        except TypeError:
            continue


    # plt.legend()
    plt.show()


def get_freq_resp(p_mag, p_phi, z_mag=None, z_phi=None, p_mag2=None, p_phi2=None):
    freq_samp = 1E9
    nyq_freq = 0.5*freq_samp

    den1 = em.zpk_to_ba(p_mag, p_phi)

    if z_mag is None:
        num = np.array([1,-2,1])
    else:
        num = em.zpk_to_ba(z_mag, z_phi)
        num /= np.sum(num)/np.sum(den1)

    w =np.logspace(-15, -7, 500, base=np.pi)

    w, h = signal.freqz(num, den1, worN=w)

    if p_mag2 is not None:
        den2 = em.zpk_to_ba(p_mag2, p_phi2)
        w, h2 = signal.freqz(num, den2, worN=w)
        h *= h2

    w/= (np.pi /nyq_freq)

    return w, np.abs(h)


if __name__ == "__main__":
    main()
