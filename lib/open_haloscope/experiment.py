# -*- coding: utf-8 -*-
# Main file with the varius tools used in the Open Haloscope project
# Written by Nicolò Crescini

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy import constants as c


class Experiment():
    def __init__(self, experiment_json: str):
        # This is the experiment class, it has all the experimental parameters
        self.property = 0
        self.sampling_frequency = 125e6
        self._initialiseExperiment(experiment_json)

    def _initialiseExperiment(self, experiment_json):
        # initialisation of the experimental parameters
        self.experiment_parameters = 0

        with open(experiment_json) as json_file:
            self.experiment_parameters = json.load(json_file)


    def mixer(self, signal_a, signal_b):
        # mixing an input signal with a local oscillator
        if len(signal_a) == len(signal_b):
            signal_c = signal_a * signal_b
        else:
            raise TypeError("The inputs should be two arrays of the same length.")

        return signal_c


    def lockin(self, 
               signal, 
               reference_frequency, 
               sampling_frequency,
               filter_tau = 1e-3, 
               filter_order = 5
               ):

        # lock-in amplifier
        points = len(signal)
        numeric_time = np.linspace(0, points-1, points)        
        numeric_frequency = frequency / sampling_frequency

        sine = np.sin(2*np.pi * numeric_frequency * numeric_time)
        cosine = np.cos(2*np.pi * numeric_frequency * numeric_time)

        # definition of the filter
        sos = butter(filter_order, filter_tau, fs=1, output='sos')
        # mixing and filtering
        x = sosfilt(sos, signal * sine)[-1]
        y = sosfilt(sos, signal * cosine)[-1]

        # r = np.abs(x + 1j*y)
        # theta = np.angle(x + 1j*y)
        
        return x + 1j*y


class FermionicHaloscope(Experiment):
    def __init__(self, experiment_json):
        super().__init__(experiment_json)

        self._initialiseExperiment(experiment_json)
        
        self.sensitivity_parameters = {"f0": self.experiment_parameters['f0'],       # frequency of the resonance 
                                       "Q": self.experiment_parameters['Q'],         # quality factor of the resonance
                                       "An": self.experiment_parameters['An'],       # amplitude/rt(Hz) of the noise
                                       "Ap": self.experiment_parameters['Ap'],       # amplitude of the pump
                                       "beta1": self.experiment_parameters['beta1'], # coupling of antenna 1
                                       "beta2": self.experiment_parameters['beta2']  # coupling of antenna 2
                                       }

    def get_sensitivity(self, f0, Q, An, Ap, beta1, beta2):
        """
        This function return the sensitivity of the haloscope given some specified quantities.
        Supposedly, the expected (or already measured) parameters are in the experiment .json file,
        which can be loaded and passed to this function to calculate the expected sensitivity.
        """

        # electron gyromagnetic ration in MHz/T
        gamma = c.physical_constants['electron gyromag. ratio in MHz/T'][0] * 1e6
        B0 = f0 / gamma
        eta = (1 + beta2) / 2 * np.sqrt((1 + beta1) / (beta1 * beta2))

        b = 2 * eta * B0 / (np.pi * Q) * (An / Ap)
        return b

    def analyse_run(self):
        return 0


class Haloscope(Experiment):
    def __init__(self, experiment_json):
        super().__init__(experiment_json)
        self.f0 = 0
        self.Q = 0

        self._initialiseExperiment(experiment_json)

    def analyse_run(self):
        return 0


class FerromegnaticHaloscope(Experiment):
    def __init__(self, experiment_json):
        super().__init__(experiment_json)
        self.f0 = 0
        self.Q = 0

        self._initialiseExperiment(experiment_json)

    def analyse_run(self):
        return 0


