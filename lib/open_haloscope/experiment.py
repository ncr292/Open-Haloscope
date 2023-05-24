# -*- coding: utf-8 -*-
# Experiment.
# Here one can defined different types of experiments which search for the particles of particles.py.
# In general, these classes contain methods and functions which are the bread and butter of experimentalists.
# If a function is general purpose, like a lock-in amplifier, it is good to add it directly to the Experiment() 
# class, as it will be inherited by all the other experiments. On the other hand, if a function is specific of
# one experiment, or haloscope, then it is better to keep it within the corresponding experiment class. For example
# the function to analyse a run, being a specific process which depends on the haloscope, is only a function of a 
# specific experiment class,  like FermionicHaloscope().
#
# Written by Nicolò Crescini

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy import constants as c
from datetime import datetime

# open haloscope functions
from .utils import OHUtils

# qcodes functions
import qcodes as qc
from qcodes.dataset import (do0d,
                            do1d,
                            Measurement,
                            experiments,
                            initialise_or_create_database_at,
                            load_or_create_experiment,
                            load_by_run_spec,
                            plot_dataset
                            )

# qcodes dummy instrument for time
from qcodes.tests.instrument_mocks import DummyInstrument
chrono = DummyInstrument('chrono', gates=['t0', 't1'])


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

    def lockin(self, signal, reference_frequency, sampling_frequency, filter_tau = 1e-3, filter_order = 5):

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

    # useful functions
    def lorentzian(self, x, x0, a, gamma):
        return a * gamma**2 / ( gamma**2 + ( x - x0 )**2)


class FermionicHaloscope(Experiment):
    def __init__(self, experiment_json):
        super().__init__(experiment_json)

        self.instruments = []
        self.sampling_frequency
        self.station = None
        self.data_path = ''
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

    def initialise_haloscope(self, instruments_list, decimation = 4, trigger = 0.0):
        ## initialisation of the experimental apparatus
        # here, one needs to add all the intruments used in the setup, which will
        # then be used to set-up the experiment before starting the operation.

        print('Loading instrumentation')

        for instrument in instruments_list:
            self.instruments.append(instrument)
            if instrument.name == 'redpitaya': red = instrument

            print(' ', instrument.name, 'added to the experiment')

        # rf configuration
        print('\nSetup the radiofrequency lines')

        # input
        red.ADC_averaging('OFF')
        red.ADC_decimation(decimation)

        red.ADC_trigger_level(trigger)
        print(' inputs configured, trigger level =', str(trigger), 'V, decimation =', str(decimation))
        self.sampling_frequency = red.FS / red.ADC_decimation()
        print(' resulting sampling frequency =', str(self.sampling_frequency / 1e6), 'MHz')

        # outputs
        red.align_channels_phase()
        red.OUT_trigger()
        red.OUT1_status('OFF')
        red.OUT2_status('OFF')
        print(' output generators triggered, phase aligned and turned off')

        print('\nStarting UART communication')
        red.UART_init()
        print(' testing sensors')
        print(' temperature =', str(red.temperature()), 'K')
        print(' pressure =', str(red.pressure()), 'bar')
        print(' magnetic field =', str(red.magnetic_field()), 'V')
        print(' photoresistance =', str(red.photoresistance()), 'V')
        print(' acceleration =', str(red.acceleration()), 'm/s^2')

        print('\nConfiguring data storage.')
        self.data_path += os.path.join(OHUtils.get_runs_folder(), datetime.today().strftime('%Y-%m-%d')) 
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        print(' data are stored in', self.data_path)

        # QCodes configuration
        self.station = qc.Station()
        for instrument in instruments_list:
                    self.station.add_component(instrument)  
        self.station.add_component(chrono)
        print(' QCodes station and database configured')

        print('\nHaloscope initialised. Good luck, dark matter hunter.')

    def characterise(self, db_name = 'experiment_characterisation.db', monitoring_time = 30, time_points = 301):
        # launch a characterisation measurement of the haloscope, which consists in a transmission measurement
        # of its two channels, and some time to monitor its sensors.

        # create a database which contains the characterisation measurements
        db_path = os.path.join(self.data_path, db_name)
        initialise_or_create_database_at(db_path)

        exp = load_or_create_experiment(experiment_name="kakapo", sample_name="characterisation")

        # measure s21 of both channels 
        meas = Measurement(exp=exp, station=self.station, name='spectroscopy')


        # check sensors stability
        temperature = self.station.redpitaya.temperature # need to add an if to check that the instrument is there
        pressure = self.station.redpitaya.pressure
        magnetic_field = self.station.redpitaya.magnetic_field
        photoresistance = self.station.redpitaya.photoresistance
        acceleration = self.station.redpitaya.acceleration

        meas = Measurement(exp=exp, station=self.station, name='temperture')
        do1d(chrono.t0, 0, monitoring_time, time_points, monitoring_time/time_points, temperature)



        print('Done.')
        pass

    def prepare_for_operation(self):

        print('\nHaloscope ready for research.')
        return 0

    def run(self):

        print('\nRun completed.')
        return 0

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


class FerromagneticHaloscope(Experiment):
    def __init__(self, experiment_json):
        super().__init__(experiment_json)
        self.f0 = 0
        self.Q = 0

        self._initialiseExperiment(experiment_json)

    def analyse_run(self):
        return 0


