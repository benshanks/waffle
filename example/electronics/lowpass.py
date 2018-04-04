#!/usr/local/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt

from siggen import PPC
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import cm

from waffle.models.electronics import LowPassFilterModel
em = LowPassFilterModel()

def main():
    # det = Detector("GEM", "config_files/ortec4b.config", verbose=False)
    # det = PPC("conf/P42574A_imp.config")
    det = PPC( os.path.join(os.environ["DATADIR"],  "siggen", "config_files", "bege.config"), wf_padding=1000)
    imp_avg = -2
    imp_grad = 1.2
    det.siggenInst.SetImpurityAvg(imp_avg, imp_grad)
    det.hp_order=0


    # freq_resp(det)
    # poles(det)
    zeros(det)
    plt.show()

def zeros(det):


    phi = 0.007
    mag = 0.975

    det.lp_num = [1,2,1]
    det.lp_den = em.zpk_to_ba(mag, phi)

    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 200, smoothing=20))
    wf_compare = np.copy(wf_proc)

    f, ax = plt.subplots(2,2,figsize=(12,7))

    ax[0,0].plot(wf_compare, c="k")
    w,h = get_freq_resp(mag, phi)
    ax[0,1].loglog(w, h, color="k")

    cmap = cm.get_cmap('viridis')
    # phis = np.linspace(0.00001, np.pi, 100)
    zphi = 0.5
    mags = np.linspace(0.9,1.5,100)
    for zmag in mags:
        det.lp_num = em.zpk_to_ba(zmag, zphi)#[1,0,-zero**2]
        wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 200, smoothing=20))

        w,h = get_freq_resp(mag, phi, zmag, zphi)

        # color = cmap( (phi - phis[0])/(phis[-1] - phis[0]) )
        color = cmap( (zmag - mags[0])/(mags[-1] - mags[0]) )

        ax[0,0].plot(wf_proc, color=color)
        ax[1,0].plot(wf_compare-wf_proc, color=color)
        ax[0,1].loglog(w, h, color=color)


def poles(det):
    wf = np.copy(det.GetWaveform(25,0,25))

    phis = np.linspace(0.00001, 0.1, 200)
    mags = np.linspace(0, 0.99, 20)

    for phi in phis:
        for mag in [0.99]:#mags:
            color = cmap( (phi - phis[0])/(phis[-1] - phis[0]) )

            det.lp_num = [1,2,1]
            det.lp_den = em.zpk_to_ba(mag, phi)

            # wf_proc = signal.lfilter(det.lp_num, det.lp_den, wf)
            # wf_proc /= (np.sum(det.lp_num)/np.sum(det.lp_den))
            # if wf_proc is None:
            #     print(mag,phi)
            #     continue

            # plt.figure()

            # plt.show()
            # exit()

            #
            # # color = cmap( (mag - mags[0])/(mags[-1] - mags[0]) )
            #
            # plt.plot (wf_proc, color = color)


    # plt.legend()

# def freq_resp(det):
#     cmap = cm.get_cmap('viridis')
#
#     phis = np.linspace(0.00001, np.pi, 5)
#     mags = np.linspace(0, 0.99, 5)
#
#     for z_mag in mags:
#         for z_phi in phis:
#             for p_mag in mags:
#                 for p_phi in phis:
#                     w, h = get_freq_resp(z_mag, z_phi, p_mag, p_phi)
#
#                     color = cmap( (z_mag - mags[0])/(mags[-1] - mags[0]) )
#                     plt.loglog(w, h, color = color)

def get_freq_resp( p_mag, p_phi, z_mag=None, z_phi=None):
    freq_samp = 1E9
    nyq_freq = 0.5*freq_samp

    den = em.zpk_to_ba(p_mag, p_phi)

    if z_mag is not None:
        num = em.zpk_to_ba(z_mag, z_phi)
    else:
        num = [1,2,1]
    num /= (np.sum(num)/np.sum(den))


    w, h = signal.freqz(num, den, worN=np.logspace(-8, 0, 500, base=np.pi))
    w/= (np.pi /nyq_freq)
    return w, np.abs(h)
    # plt.plot(w, np.abs(h), color = color)

if __name__=="__main__":
    main()
