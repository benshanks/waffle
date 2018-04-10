 #!/usr/local/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from siggen import PPC
from scipy import signal
from siggen.electronics import DigitalFilter, GretinaOvershootFilter
from pygama.filters import rc_decay

det = PPC( os.path.join(os.environ["DATADIR"],  "siggen", "config_files", "bege.config"), wf_padding=100)
imp_avg = -2
imp_grad = 1.2
det.siggenInst.SetImpurityAvg(imp_avg, imp_grad)

def main():
    skew()

def skew():

    lowpass = DigitalFilter(2)
    lowpass.num = [1,2,1]
    lowpass.set_poles(0.975, 0.007)

    hipass = DigitalFilter(2)
    hipass.num = [1,-2,1]
    hipass.set_poles(1.-10.**-7, np.pi**-13.3)

    det.AddDigitalFilter(lowpass)
    det.AddDigitalFilter(hipass)

    wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=25))
    wf_compare = np.copy(wf_proc)

    f, ax = plt.subplots(2,1,figsize=(15,8), sharex=True)
    ax[0].plot (wf_compare,  color="r")

    overshoot = GretinaOvershootFilter(1)
    overshoot.num, overshoot.den = rc_decay(1.7, 1E9)
    det.AddDigitalFilter(overshoot)

    for frac in np.linspace(0.01,0.05, 5):
        overshoot.overshoot_frac = frac
        wf_proc = np.copy(det.MakeSimWaveform(25, 0, 25, 1, 125, 0.95, 1000, smoothing=25))

        ax[0].plot(wf_proc)
        ax[1].plot(wf_proc - wf_compare)

    plt.show()



if __name__ == "__main__":
    main()
