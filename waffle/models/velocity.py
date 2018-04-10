import os, sys
import numpy as np

from ._parameterbase import JointModelBase, Parameter

class VelocityModel(JointModelBase):

    def __init__(self,E_lo=250, E_hi=1000,include_beta=True, beta_lims=[0.1,2]):
        self.E_lo = E_lo
        self.E_hi = E_hi

        #Default values are from Bruyneel NIMA 569 (Reggiani values)
        h100_lo, h100_hi, self.h100_beta = self.transform_velo_params(66333., 0.744, 181.)
        h111_lo, h111_hi, self.h111_beta = self.transform_velo_params(107270., 0.580, 100.)

        parameter_names = ["h100_{}".format(E_lo), "h111_{}".format(E_lo),
                                "h100_{}".format(E_hi), "h111{}".format(E_hi),
                                "h100_beta", "h111_beta"
                                ]

        velocity_means = np.array([h100_lo, h111_lo, h100_hi, h111_hi])
        velocity_variance = 0.1

        velo_lo_cutoff = 1
        velo_hi_cutoff = 10*velocity_means

        self.params = []
        for i in range(4):
            self.params.append( Parameter(parameter_names[i], "gaussian",
                                        mean=velocity_means[i],
                                        variance=velocity_variance*velocity_means[i],
                                        lim_lo=velo_lo_cutoff, lim_hi=velo_hi_cutoff[i])
                                )
        if include_beta:
            for i in range(4,6):
                self.params.append( Parameter(parameter_names[i], "uniform",
                                            lim_lo=beta_lims[0], lim_hi=beta_lims[1])
                                    )

    def apply_to_detector(self, params, detector):
        h_100_vlo, h_111_vlo, h_100_vhi, h_111_vhi,  = params[:4]

        if self.num_params == 6:
            h_100_beta, h_111_beta = params[4:]
        else:
            h_100_beta, h_111_beta = self.h100_beta, self.h111_beta

        h_100_mu0, h_100_beta, h_100_e0 = self.invert_velo_params(h_100_vlo, h_100_vhi, h_100_beta)
        h_111_mu0, h_111_beta, h_111_e0 = self.invert_velo_params(h_111_vlo, h_111_vhi, h_111_beta)

        detector.siggenInst.SetHoles(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 )


    def invert_velo_params(self, v_a, v_c, beta):
        E_a = self.E_lo
        E_c = self.E_hi

        psi = (E_a * v_c) / ( E_c * v_a )
        E_0 = np.power((psi**beta* E_c**beta - E_a**beta) / (1-psi**beta), 1./beta)
        mu_0 = (v_a / E_a) * (1 + (E_a / E_0)**beta )**(1./beta)

        # if v_a > v_c: print("gonna go ahead and expect a nan here")
        # if np.isnan(E_0):
        #     print("NAN HERE: {},{},{}".format(v_a/1E6, v_c/1E6,psi))
        #     a = (psi**beta* E_c**beta - E_a**beta) / (1-psi**beta)
        #     print(a)
        #     b = 1./beta
        #     print(b)
        #     print(a**b)
        #     print("\n\n\n")

        return (mu_0,  beta, E_0)

    def transform_velo_params(self, mu_0,  beta, E_0):
        E_a = self.E_lo
        E_c = self.E_hi

        v_a = (mu_0 * E_a)/np.power( 1 + (E_a/E_0)**beta ,1./beta)
        v_c = (mu_0 * E_c)/np.power( 1 + (E_c/E_0)**beta ,1./beta)

        return (v_a, v_c, beta)
