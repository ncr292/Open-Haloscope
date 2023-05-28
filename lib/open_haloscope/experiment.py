# -*- coding: utf-8 -*-
# Experiment.
# Here one can defined different types of experiments which search for the particles of particles.py.
# In general, these classes contain methods and functions which are the bread and butter of experimentalists.
# If a function is general purpose, like a lock-in amplifier, it is good to add it directly to the Experiment() 
# class, as it will be inherited by all the other experiments. On the other hand, if a function is specific of
# one experiment, or haloscope, then it is better to keep it within the corresponding experiment class. For example
# the function to analyse a run, being a specific process which depends on the haloscope, is only a function of a 
# specific experiment class,  like FermionicHaloscope().


import os
import json
from datetime import datetime

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.optimize import curve_fit
from scipy import constants as c

import matplotlib.pyplot as plt
# plot options
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size=14)

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
        self.buffer_length = 2**14
        self.station = None        
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
        self.data_path = ''
        self._initialiseExperiment(experiment_json)
        
        self.haloscope_name = self.experiment_parameters['haloscope_name']
        self.sensitivity_parameters = {"f0": self.experiment_parameters['f0'],       # frequency of the resonance 
                                       "Q": self.experiment_parameters['Q'],         # quality factor of the resonance
                                       "f1": self.experiment_parameters['f2'],       # freuquency of resonance 1
                                       "k1": self.experiment_parameters['k1'],       # linewidth of resonance 1
                                       "f2": self.experiment_parameters['f2'],       # frequency of resonance 2
                                       "k2": self.experiment_parameters['k2'],       # linewidth of resonance 2
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
        self.buffer_length = red.ADC_buffer_size()
        print(' resulting sampling frequency =', str(self.sampling_frequency / 1e6), 'MHz')
        print(' buffer length =', str(self.buffer_length),'samples, i.e.', str(self.buffer_length/self.sampling_frequency),'s')

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

    def characterise(self, db_name = 'experiment_characterisation.db', monitoring_time = 30, time_points = 61, start_frequency=2e6, stop_frequency=10e6, frequency_points=101, rbw=10e3, probe_power=0.001, averages=2, plot = False):
        # launch a characterisation measurement of the haloscope, which consists in a transmission measurement
        # of its two channels, and some time to monitor its sensors. The default sampling time for the sensors is 1/s.

        print('Characterisation data')

        # create a database which contains the characterisation measurements
        db_path = os.path.join(self.data_path, db_name)
        initialise_or_create_database_at(db_path)
        print(' characterisation database created in ', self.data_path)

        exp = load_or_create_experiment(experiment_name=self.haloscope_name, sample_name="characterisation")

        # measure s21 of both channels 
        print('\nInitiating spectroscopy in the span', str(start_frequency/1e6), 'to', str(stop_frequency/1e6), 'MHz.')
        meas = Measurement(exp=exp, station=self.station)

        self.station.redpitaya.vna_start(start_frequency)
        self.station.redpitaya.vna_stop(stop_frequency)
        self.station.redpitaya.vna_points(frequency_points)
        self.station.redpitaya.vna_rbw(rbw)

        self.station.redpitaya.vna_amplitude(probe_power)
        self.station.redpitaya.vna_averages(averages)

        print(' channel 1')
        dataset_tx1 = do0d(self.station.redpitaya.TX1, measurement_name='spectroscopy ch1');
        print(' channel 2')
        dataset_tx2 = do0d(self.station.redpitaya.TX2, measurement_name='spectroscopy ch2');

        # check sensors stability
        print('\nInitiating sensors stability check for', str(monitoring_time), 's')

        print(' temperature')
        t_wait = 0.9 # 1 sample per second minus 0.1 seconds of delay set by the arduino
        dataset_t = do1d(chrono.t0, 0, 100, monitoring_time // 1, t_wait, self.station.redpitaya.temperature, measurement_name='sensors - t');
        print(' pressure')
        dataset_p = do1d(chrono.t0, 0, 100, monitoring_time // 1, t_wait, self.station.redpitaya.pressure, measurement_name='sensors - p');
        print(' magnetic_field')
        dataset_b = do1d(chrono.t0, 0, 100, monitoring_time // 1, t_wait, self.station.redpitaya.magnetic_field, measurement_name='sensors - B');
        print(' photoresistance')
        dataset_l = do1d(chrono.t0, 0, 100, monitoring_time // 1, t_wait, self.station.redpitaya.photoresistance, measurement_name='sensors - light');
        #print(' acceleration')
        #dataset_t = do1d(chrono.t0, 0, 100, monitoring_time // 1, t_wait, self.station.redpitaya.acceleration)

        # getting the data
        f1 = dataset_tx1[0].get_parameter_data()['redpitaya_TX1']['redpitaya_frequency_axis']
        tx1 = dataset_tx1[0].get_parameter_data()['redpitaya_TX1']['redpitaya_TX1']

        f2 = dataset_tx2[0].get_parameter_data()['redpitaya_TX2']['redpitaya_frequency_axis']
        tx2 = dataset_tx2[0].get_parameter_data()['redpitaya_TX2']['redpitaya_TX2']

        time = np.linspace(0, monitoring_time, monitoring_time // 1)
        t = dataset_t[0].get_parameter_data()['redpitaya_temperature']['redpitaya_temperature']
        p = dataset_p[0].get_parameter_data()['redpitaya_pressure']['redpitaya_pressure']
        b = dataset_b[0].get_parameter_data()['redpitaya_magnetic_field']['redpitaya_magnetic_field']
        l = dataset_l[0].get_parameter_data()['redpitaya_photoresistance']['redpitaya_photoresistance']

        # fitting the resonances
        # resonance 1
        x0 = np.argmax(tx1)
        start = x0 - frequency_points//10
        stop = x0 + frequency_points//10

        popt1, _ = curve_fit(self.lorentzian, f1[start:stop], tx1[start:stop]**2 / np.max(tx1**2), p0=[f1[x0], 1, f1[x0]/100])
        self.experiment_parameters['f1'] = popt1[0]
        self.experiment_parameters['k1'] = popt1[2]

        # resonance 2
        x0 = np.argmax(tx2)
        start = x0 - frequency_points//10
        stop = x0 + frequency_points//10

        popt2, _ = curve_fit(self.lorentzian, f2[start:stop], tx2[start:stop]**2 / np.max(tx2**2), p0=[f2[x0], 1, f2[x0]/100])
        self.experiment_parameters['f2'] = popt2[0]
        self.experiment_parameters['k2'] = popt2[2]

        # optional plotting
        if plot == True:
            plt.figure(figsize=(12, 6))
            ax0 = plt.subplot(2,1,1)  
            ax1 = plt.subplot(2,4,5)    
            ax2 = plt.subplot(2,4,6)
            ax3 = plt.subplot(2,4,7)
            ax4 = plt.subplot(2,4,8)

            # transmissions
            f_plot = np.linspace(start_frequency, stop_frequency, 100001)
            ax0.set_xlabel('Frequency (MHz)')
            ax0.set_ylabel('Transmission (a.u.)')
            ax0.semilogy(f1/1e6, tx1 / np.max(tx1))
            ax0.semilogy(f_plot/1e6, np.sqrt(self.lorentzian(f_plot, *popt1)), alpha=0.5)
            ax0.semilogy(f2/1e6, tx2 / np.max(tx2))
            ax0.semilogy(f_plot/1e6, np.sqrt(self.lorentzian(f_plot, *popt2)), alpha=0.5)

            # sensors
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Temperature (K)')
            ax1.plot(time, t, 'o')

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Pressure (bar)')
            ax2.plot(time, p, 'o')

            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('B-field (V)')
            ax3.plot(time, b, 'o')

            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Photoresistance (V)')
            ax4.plot(time, l, 'o')

            plt.tight_layout()


        print('\nHaloscope parameters acquired.')

    def prepare_for_operation(self):

        print('\nHaloscope ready for research.')
        return 0

    def run(self):

        print('\nRun completed.')
        return 0

    def generate_simulated_run_data(self, f1=5e6, a1=1, f2=6e6, a2=1, phase_noise=1e-8, amplitude_noise=1e-5, number_of_traces=1000, axion_signal=True, aa=1e-8, fa=2e3, common_noise_frequency = 1e3, common_noise_amplitude = 1e-5):
        # This function generates fake data which look like the ones of a real experimental run, it can be 
        # used to test the analysis routines with different parameters. One can add the axion signal with specific
        # frequency and amplitude (only applied on data1 for simplicity), and there is a common noise to evaluate
        # its rejection. 

        fs = self.sampling_frequency
        trace_length = self.buffer_length

        t = np.linspace(0,trace_length / fs, trace_length)

        data1 = np.zeros((trace_length, number_of_traces))
        data2 = np.zeros((trace_length, number_of_traces))

        # axionic signal which can go in data1
        axion = aa * np.sin(2*np.pi * fa * t)

        # common noise
        common = common_noise_amplitude * np.sin(2*np.pi * common_noise_frequency * t)


        for i in range(number_of_traces):

            # noise
            amplitude_noise_trace = np.random.normal(scale = amplitude_noise, size = trace_length)
            phase_noise_trace = np.random.normal(scale = phase_noise, size = trace_length)

            # channel 1
            if axion_signal == True:
                data1[:,i] = a1 * np.sin(2*np.pi*f1 * t + 2*np.pi * np.random.uniform() + axion + phase_noise_trace + common) + amplitude_noise_trace
            else:
                data1[:,i] = a1 * np.sin(2*np.pi*f1 * t + 2*np.pi * np.random.uniform() + phase_noise_trace + common) + amplitude_noise_trace

            # channel 2
            data2[:,i] = a2 * np.sin(2*np.pi*f2 * t + 2*np.pi * np.random.uniform() + phase_noise_trace + common) + amplitude_noise_trace


        return t, np.rot90(data1), np.rot90(data2)

    def analyse_run(self, data1, data2):
        # Simple analysis routine to extract the limit on the effective magnetic field starting from the run data
        # of the fermionic interferometer.

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


