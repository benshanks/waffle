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



def main():
    # poles(det)
    # zeros(det)
    two_rc(det)

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

def get_freq_resp(hipass1):
    freq_samp = 1E9
    nyq_freq = 0.5*freq_samp

    w = np.logspace(-15, -7, 500, base=np.pi)

    w, h = signal.freqz(hipass1.num, hipass1.den, worN=w)

    # if p_mag2 is not None:
    #     den2 = em.zpk_to_ba(p_mag2, p_phi2)
    #     w, h2 = signal.freqz(num, den2, worN=w)
    #     h *= h2

    w/= (np.pi /nyq_freq)

    return w, np.abs(h)


if __name__ == "__main__":
    main()
