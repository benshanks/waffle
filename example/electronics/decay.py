#!/usr/local/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from siggen import PPC
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d

from waffle.models.electronics import HiPassFilterModel

em = HiPassFilterModel()

def main():
    # det = Detector("GEM", "config_files/ortec4b.config", verbose=False)
    # det = PPC("conf/P42574A_imp.config")
    det = PPC( os.path.join(os.environ["DATADIR"],  "siggen", "config_files", "bege.config"), wf_padding=100)
    imp_avg = -2
    imp_grad = 1.2
    det.siggenInst.SetImpurityAvg(imp_avg, imp_grad)

    plot_wfs(det)


def plot_wfs(det):
    phi = 1E-5

    mag = -5
    mag = 1-10.**mag
    phi = 1E-6

    wf = np.copy(det.GetWaveform(25,0,25))

    f, ax = plt.subplots(1,2,figsize=(15,8))

    cmap = cm.get_cmap('viridis')

    phis = np.logspace(-10, -5, 100)
    # phis = np.pi - phis

    w,h = get_freq_resp(mag, phi)
    ax[1].loglog( w, h, color="r")
    num, den = get_tf(mag, phi)
    wf_proc1 = signal.lfilter(num, den, wf)

    for phi2 in phis:

        color = cmap( (phi2 - phis[0])/(phis[-1] - phis[0]) )

        mag2= 1 - 10E-8
        num2, den2 = get_tf(mag2, phi2)

        w,h2 = get_freq_resp(mag2, phi2)
        ax[1].loglog( w, h2, color=color)

        # continue


        wf_proc = signal.lfilter(num2, den2, wf_proc1)
        if wf_proc is None:
            print(mag,phi)
            continue

        ax[0].plot (wf_proc-wf_proc1, label="{}, {}".format(phi,mag), color=color)




    # plt.legend()
    plt.show()



def get_tf(pole_mag, pole_phi):

    num = [1,-2,1]
    den = em.zpk_to_ba(pole_mag, pole_phi)

    return num, den

def get_freq_resp(p_mag, p_phi):
    freq_samp = 1E9
    nyq_freq = 0.5*freq_samp

    num = np.array([1,-2,1])
    den1 = em.zpk_to_ba(p_mag, p_phi)

    w =np.logspace(-15, -7, 500, base=np.pi)

    w, h = signal.freqz(num, den1, worN=w)
    w/= (np.pi /nyq_freq)

    return w, np.abs(h)


if __name__ == "__main__":
    main()
