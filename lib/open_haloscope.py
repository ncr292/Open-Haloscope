# -*- coding: utf-8 -*-
# Main file with the varius tools used in the Open Haloscope project
# Written by Nicolò Crescini

from tqdm.notebook import tqdm

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng() 

from scipy import constants as c
from scipy.signal import periodogram, welch



class OHAxion():
    ## Axions and friends
    # The axion class has all the parameter assumed for the DFSZ axion model, and features
    # some useful functions that are used to estimate the haloscope sensitivity.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_dm = 0.3e9 / (0.01**3)    # dark matter density in eV per cubic meter
        self.v_a = 1e-3 * c.c            # relative speed of the dark matter wind

        self.DFSZ_u = 1.0/3.0            # DFSZ axion upper limit
        self.DFSZ_l = 2.0e-5             # DFSZ axion upper limit
        self.gx = 8.943e-11              # coupling constant


    # print the axion and dark matter parameters
    def dm_axion_parameters(self):
        print('Local dark matter density: ' + str(self.n_dm * (0.01**3) / 1e9) + ' GeV/cm3' )
        print('Relative speed of the dark matter wind: ' + str(self.v_a / c.c) + 'c = ' + str(self.v_a) + ' m/s' )
        print('DFSZ axion coupling: ' + str(self.gx) + '/mass' + 
              ', with upper limit ' + str(np.round(self.DFSZ_u, 1)) + ' and lower limit ' + str(self.DFSZ_l))


    # mass-coupling relation
    def g_x(self, C_ae, m_a):
        return self.gx * C_ae * m_a


    # calculate the effective magnetic field for dark matter axions of mass m_a
    def effective_field(self, m_a):
        g_p = OHAxion.g_x(self, self.DFSZ_u, m_a)
        b_eff = g_p / (2 * c.e) * np.sqrt( self.n_dm * c.hbar / c.c) * self.v_a
        return np.sqrt(c.hbar) * b_eff


    # get the value of g corresponding to a mium detected fiel b_a
    def g_limit(self, b_a):
        g_invert = (2 * c.e) / np.sqrt( self.n_dm * c.hbar / c.c) / self.v_a
        return b_a * g_invert / np.sqrt(c.hbar)


    # plot the DFSZ model
    def plot_DFSZ_axion(self, mass):
        ## QCD Axion band:
        DFSZ_u = self.DFSZ_u
        DFSZ_l = self.DFSZ_l

        plt.loglog(mass, OHAxion.g_x(self, DFSZ_l, mass), '-', lw=3.5, c='black', alpha=0.5)
        plt.loglog(mass, OHAxion.g_x(self, DFSZ_u, mass), '-', lw=3.5, c='black', alpha=0.5)
        plt.fill_between(mass, OHAxion.g_x(self, DFSZ_l, mass), y2=OHAxion.g_x(self, DFSZ_u, mass), facecolor='goldenrod', zorder=0, alpha=0.3)
        plt.text(4*np.min(mass),  10*np.min(OHAxion.g_x(self, DFSZ_l, mass)), 'DFSZ axion', rotation=20)

        return




class OHMethods():
    ## Haloscope methods
    # This is the haloscope class, it has all the experimental parameters

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f0 = 4e9
        self.Q = 100


    def update_haloscope(self, params):
        self.f0 = params['f0']
        self.Q = params['Q']
        return


    # mixing an input signal with a local oscillator
    def mixer(self):
        return 


    # lock-in measurement
    def lockin(self, t, signal, reference, tau=1e-3):
        mixed_signal = signal * reference
        
        n_out = int( np.max(t) / tau )
        points_in_tau = int( tau / (t[1] - t[0]) )
        output_signal = np.zeros((n_out))
        
        for i in range(n_out):
            n_start = i * points_in_tau
            n_end = (i+1) * points_in_tau
            output_signal[i] = np.sum( mixed_signal[n_start:n_end] )
            
        taus = tau * np.linspace(0, n_out, n_out)
        
        return taus, np.sqrt( output_signal )




class OHUtils():
    ## Utilities class
    # Class used for some handy functions that always help

    # useful variables
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test = 1

    def dB(x):
        return 10 * np.log10(x)

    # unit conversions
    def eV_to_Hz(m):
        f = m * c.e / c.hbar  / (2 * c.pi)
        return f
