# -*- coding: utf-8 -*-
# Main file with the varius tools used in the Open Haloscope project
# Written by Nicolò Crescini

from tqdm.notebook import tqdm

import numpy as np
from numpy.random import default_rng
rng = default_rng() 

from scipy import constants as c
from scipy.signal import periodogram, welch


# axion related functions
class OHAxion():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_dm = 0.3e9 / (0.01**3)    # dark matter density in eV per cubic meter
        self.v_a = 1e-3 * c.c            # relative speed of the dark matter wind

    def dm_axion_parameters(self):
        print('Local dark matter density: ' + str(self.n_dm * (0.01**3) / 1e9) + ' GeV/cm3' )
        print('Relative speed of the dark matter wind: ' + str(self.v_a / c.c) + 'c = ' + str(self.v_a) + ' m/s' )



    def effective_field(self, m_a):
        b = g / (2 * c.e) * np.sqrt( (self.n_dm / m_a) * c.hbar / (m_a * c.c)) * m_a * self.v_a
        return b

    # get the value of g corresponding to a mium detected fiel b_a and the axion mass m_a
    def g_limit(self, b_a, m_a):
        g = b_a / (2 * c.e) * np.sqrt( (self.n_dm / m_a) * c.hbar / (m_a * c.c)) * m_a * self.v_a
        return g

    # unit conversions
    def eV_to_Hz(self, m):
        f = m * c.e / c.hbar  / (2 * c.pi)
        return f




# methods
class OHMethods():

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.library = True

    # mixing an input signal with a local oscillator
    def mixer():
        return 0

    # lock-in measurement
    def lockin(t, signal, reference, tau=1e-3):
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



# utilities
class OHUtils():

    # class used for some handy functions that always help

    # useful variables
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test = True


    def dB(x):
        return 10 * np.log10(x)





