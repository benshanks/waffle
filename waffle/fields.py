#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
from siggen import PPC
import os

def main():
    # det_name = "bege.config"
    # wp_name = "fields/bege_wpot.field"
    # ef_name= "fields/bege_ev.field"

    det_name = "P42574A_smaller.config"
    wp_name = "fields/P42574A_smaller_wpot.field"
    ef_name= "fields/P42574A_smaller_ev.field"

    det = PPC( "conf/" + det_name, verbose=False, doInit=False)

    meshmult = 10
    xtal_HV = 3500
    impAvgRange = np.linspace(-3,0, 5)
    gradientRange = np.linspace(0,2, 5)

    wp_mat, efld = det.solve_fields(meshmult, impAvgRange, gradientRange, wp_name =wp_name , ef_name=ef_name)

    # nr = int(det.detector_radius*meshmult+1)
    # nz = int(det.detector_length*meshmult+1)
    # efield = det.solve_field("efield", nr, nz, impurity_gradient=gradientRange[0], impurity_avg=impAvgRange[0], xtal_HV=xtal_HV)
    #
    # plotfield = efield[:,:,3]
    #
    # # plotfield[plotfield<1E-6] = np.nan
    # # plotfield[plotfield==1] = np.nan
    # # levels = np.linspace(0, 3000, 15 )
    # levels = np.linspace(-500, 500, 5 )
    # plt.contourf(plotfield.T, origin="lower", extent=[0,det.detector_radius,0,det.detector_length], levels=levels, extend="both", cmap="bwr")
    # # plotfield[plotfield>0] = 1
    # # plotfield[plotfield<=0] = -1
    # # plt.imshow(plotfield.T, origin="lower", extent=[0,det.detector_radius,0,det.detector_length])
    # plt.colorbar()
    # plt.show()

if __name__=="__main__":
    main()
