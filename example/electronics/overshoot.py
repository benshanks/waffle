 #!/usr/local/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from siggen import PPC
from scipy import signal
from waffle.models.electronics import HiPassFilterModel
from pygama.filters import rc_decay

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

    f, ax = plt.subplots(2,1,figsize=(15,8), sharex=True)
    ax[0].plot (wf_compare,  color="r")

    det.overshoot = 1
    det.overshoot_num, det.overshoot_den = rc_decay(1.7, 1E9)

    for frac in np.linspace(0.01,0.05, 5):
        det.overshoot_frac = frac
        wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=25))

        ax[0].plot(wf_proc)
        ax[1].plot(wf_proc - wf_compare)

    plt.show()



if __name__ == "__main__":
    main()
