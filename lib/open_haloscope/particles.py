# -*- coding: utf-8 -*-
# Particles.
# This is the place to add all the properties of particles which can be searched with the haloscope.
# The properties can be used to estimate the sensitivity, extract upper limits, get experimentally 
# relevant numbers and so on.


import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as c

from .utils import OHUtils


class Particle():
    def __init__(self, *args, **kwargs):
        # initialisation of the particle parameters, like coupling constants or other useful properties.
        self.mass = 125e9 # eV
        self.interaction_constant = 0

    # unit conversion
    def eV_to_Hz(self, m):
        f = m * c.e / c.hbar / (2 * c.pi)
        return f

    def Hz_to_eV(self, f):
        m = f / c.e * c.hbar * (2 * c.pi)
        return m


class DMAxion(Particle):
    ## Axions and friends
    # The axion class has all the parameter assumed for the DFSZ axion model, and features
    # some useful functions that are used to estimate the haloscope sensitivity.

    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self.n_dm = 0.4e9 / (0.01**3)    # dark matter density in eV per cubic meter
        self.v_a = 1e-3 * c.c            # relative speed of the dark matter wind

        self.DFSZ_u = 1.0/3.0            # DFSZ axion upper limit
        self.DFSZ_l = 2.0e-5             # DFSZ axion upper limit
        self.gx = 8.943e-11              # coupling constant


    # print the axion and dark matter parameters
    def print_axion_parameters(self):
        print('Local dark matter density: ' + str(self.n_dm * (0.01**3) / 1e9) + ' GeV/cm3' )
        print('Relative speed of the dark matter wind: ' + str(self.v_a / c.c) + 'c = ' + str(self.v_a) + ' m/s' )
        print('DFSZ axion coupling: ' + str(self.gx) + '/mass' + 
              ', with upper limit ' + str(np.round(self.DFSZ_u, 1)) + ' and lower limit ' + str(self.DFSZ_l))

    # mass-coupling relation
    def g_x(self, C_ae, m_a):
        return self.gx * C_ae * m_a

    # calculate the effective magnetic field for dark matter axions of mass m_a
    def effective_field(self, m_a):
        g_p = DMAxion.g_x(self, self.DFSZ_u, m_a)
        b_eff = g_p / (2 * c.e) * np.sqrt( self.n_dm * c.hbar / c.c) * self.v_a
        return np.sqrt(c.hbar) * b_eff

    # return the value of g_p given a measured field limit
    def field_to_g_p(self, b):
        g_p = b * (2 * c.e) / np.sqrt( self.n_dm * c.hbar / c.c) / self.v_a
        return g_p / np.sqrt(c.hbar)

    # get the value of g corresponding to a mium detected fiel b_a
    def g_limit(self, b_a):
        g_invert = (2 * c.e) / np.sqrt( self.n_dm * c.hbar / c.c) / self.v_a
        return b_a * g_invert / np.sqrt(c.hbar)

    # plot the DFSZ model, inspired from the code of Ciaran O'Hare
    def plot_DFSZ_axion(self, mass):
        ## QCD Axion band:
        DFSZ_u = self.DFSZ_u
        DFSZ_l = self.DFSZ_l

        plt.loglog(mass, DMAxion.g_x(self, DFSZ_l, mass), '-', lw=3.5, c='black', alpha=0.5)
        plt.loglog(mass, DMAxion.g_x(self, DFSZ_u, mass), '-', lw=3.5, c='black', alpha=0.5)
        plt.fill_between(mass, DMAxion.g_x(self, DFSZ_l, mass), y2=DMAxion.g_x(self, DFSZ_u, mass), facecolor='goldenrod', zorder=0, alpha=0.3)
        plt.text(4*np.min(mass),  100*np.min(DMAxion.g_x(self, DFSZ_l, mass)), 'DFSZ axion', rotation=0)

        return

    # calculate and plot an upper limit extracted by a magnetic field measurement
    def exclusion_plot(self, frequency, magnetic_field_std):
        # from the residuals upper limit one can extract the exclusion plot corresponding to the
        # measurements. So with frequency and magnetic field limit as input this function calculates
        # the limit which can be used in an exclusion plot.

        m_a = self.Hz_to_eV(frequency)
        g_p = self.field_to_g_p(magnetic_field_std)

        return m_a, g_p


class RelicNeutrino(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self.mass = 1e-4                # eV
        self.interaction_constant = 0.0 # eV


class DarkPhoton(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self.mass = 0.0                 # eV
        self.interaction_constant = 0.0 # eV


class GravitationalWave(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(self)
        self.mass = 0.0                 # eV
        self.interaction_constant = 0.0 # eV











