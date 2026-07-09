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
import time
import contextlib
from datetime import datetime
#from tqdm import tqdm
from tqdm.notebook import tqdm

import numpy as np
from scipy.signal import butter, sosfilt, periodogram
from scipy.optimize import curve_fit
from scipy import stats
from scipy import constants as c

from joblib import Parallel, delayed

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
                            load_by_run_spec
                            )

# qcodes dummy instrument for time
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
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

    def update_json_file(self, path_json, update_dict):
        json_file = open(path_json, "r") # Open the JSON file for reading
        data = json.load(json_file) # Read the JSON into the buffer
        json_file.close() # Close the JSON file

        ## Working with buffered content
        update_data = data | update_dict

        ## Save our changes to JSON file
        json_file = open(path_json, "w+")
        json_file.write(json.dumps(update_data, indent=2))
        json_file.close()

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

    def dB(self, x):
        return 10 * np.log10(x)

    # useful functions
    def lorentzian(self, x, x0, a, gamma):
        return a * gamma**2 / ( gamma**2 + ( x - x0 )**2)

    def background(self, frequency, ap, kp, ar, kr, c):
        # function which models the background of a fermionic haloscope

        pump = self.lorentzian(frequency, 0, ap, kp)
        res = self.lorentzian(frequency, 0, ar, kr)
        cst = c * np.ones(len(frequency))

        return (pump + cst) / res


class FermionicHaloscope(Experiment):
    """
    Main class of open-haloscope, which contains all the function to characterise and operate the haloscope,
    as well as the analysis functions.
    """
    def __init__(self, experiment_json):
        super().__init__(experiment_json)

        self.instruments = []
        self.data_path = ''
        self.logs_path = ''
        self._initialiseExperiment(experiment_json)
        
        self.haloscope_name = self.experiment_parameters['haloscope_name']
        self.sensitivity_parameters = {"f": self.experiment_parameters['f'],         # frequency of the resonance 
                                       "Q": self.experiment_parameters['Q'],         # quality factor of the resonance
                                       "An": self.experiment_parameters['An'],       # amplitude of the noise
                                       "Ap": self.experiment_parameters['Ap'],       # amplitude of the pump
                                       }

    def get_b_sensitivity(self, f, Q, An, Ap, beta1=1, beta2=1):
        """
        This function return the sensitivity of the haloscope given some specified quantities.
        Supposedly, the expected (or already measured) parameters are in the experiment .json file,
        which can be loaded and passed to this function to calculate the expected sensitivity.
        """

        # electron gyromagnetic ration in MHz/T
        gamma = c.physical_constants['electron gyromag. ratio in MHz/T'][0] * 1e6
        B0 = f / gamma
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
        red.IN1_gain('LV')
        red.IN2_gain('LV')
        print(' input gain set to', red.IN1_gain())

        red.ADC_averaging('OFF')
        red.ADC_decimation(decimation)

        red.ADC_trigger_level(trigger)
        print(' trigger level =', str(trigger), 'V, decimation =', str(decimation))
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

        print('\nConfiguring data storage')
        self.data_path += os.path.join(OHUtils.get_runs_folder(), datetime.today().strftime('%Y-%m-%d')) 
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        print(' data are stored in', self.data_path)

        self.logs_path += os.path.join(OHUtils.get_logs_folder(), datetime.today().strftime('%Y-%m-%d')) 
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        print(' logs are stored in', self.logs_path)

        # QCodes configuration
        self.station = qc.Station()
        for instrument in instruments_list:
                    self.station.add_component(instrument)  
        self.station.add_component(chrono)
        print(' QCodes station, QCodes database and logfiles configured')

        print('\nHaloscope initialised. Good luck, dark matter hunter.')

    def characterise(self, db_name = 'experiment_characterisation.db', monitoring_time = 30, time_points = 61, start_frequency=2e6, stop_frequency=10e6, frequency_points=101, rbw=10e3, probe_power=0.001, averages=2, plot = False):
        # launch a characterisation measurement of the haloscope, which consists in a transmission measurement
        # of its two channels, and some time to monitor its sensors. The default sampling time for the sensors is 1/s.

        print('Characterisation data')

        if self.station.redpitaya.IN1_gain() != 'LV':
            print('Warning: the input gain is', IN1_gain(), '. For characterisation purposes, the gain should be LV.')

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
        dataset_t = do1d(chrono.t0, 0, 100, monitoring_time // 1, t_wait, self.station.redpitaya.temperature, measurement_name='sensors - t', show_progress=False);
        print(' pressure')
        dataset_p = do1d(chrono.t0, 0, 100, monitoring_time // 1, t_wait, self.station.redpitaya.pressure, measurement_name='sensors - p', show_progress=False);
        print(' magnetic_field')
        dataset_b = do1d(chrono.t0, 0, 100, monitoring_time // 1, t_wait, self.station.redpitaya.magnetic_field, measurement_name='sensors - B', show_progress=False);
        print(' photoresistance')
        dataset_l = do1d(chrono.t0, 0, 100, monitoring_time // 1, t_wait, self.station.redpitaya.photoresistance, measurement_name='sensors - light', show_progress=False);
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

    def prepare_for_operation(self, a1=1, a2=1):
        # Function which sets all the haloscope parameters preparing it for a run.
        
        f1 = self.experiment_parameters['f1']
        f2 = self.experiment_parameters['f2']

        print('Preparing generators')
        self.station.redpitaya.OUT1_frequency(f1)
        self.station.redpitaya.OUT1_amplitude(a1)
        print(' frequency of output 1 set to', str(f1/1e6), 'MHz')
        print(' amplitude of output 1 set to', str(a1), 'V')

        self.station.redpitaya.OUT2_frequency(f2)
        self.station.redpitaya.OUT2_amplitude(a2)
        print(' frequency of output 2 set to', str(f2/1e6), 'MHz')
        print(' amplitude of output 2 set to', str(a2), 'V\n')

        print(' generators setup for operation and turned off.')
        self.station.redpitaya.OUT1_status('OFF')
        self.station.redpitaya.OUT2_status('OFF')

        print('\nPreparing acquisition')
        self.station.redpitaya.IN1_gain('HV')
        self.station.redpitaya.IN2_gain('HV')
        print(' input 1 gain set to', self.station.redpitaya.IN1_gain())
        print(' input 2 gain set to', self.station.redpitaya.IN2_gain())
        print(' the sampling frequency is', str(self.sampling_frequency/1e6), 'MHz\n')

        print(' calibrating the duty cycle')
        self.station.redpitaya.estimate_duty_cycle()
        print(' the resulting duty cycle is', str(self.station.redpitaya.duty_cycle()))

        print(' input channels configured')

        print('\nHaloscope ready for research.')

    def run(self, run_time, data_saver_periodicity=10):
        # Data acquisition function to acquire the data of a scientific run.
        # This function will save the data in a database, separate from the characterisation one,
        # in blocks of data_saver_periodicity duration. Upon saving a data block, all the sensors
        # reads are saved in a log file.

        t0 = time.time()

        # load the log file of the runs
        logfile_run = os.path.join(os.path.dirname(self.logs_path), 'runs.txt')

        with open(logfile_run, "r") as rlog:
            last_line = rlog.readlines()[-1]
            run_number = int(last_line.split('\t')[0]) + 1
        run_number = str(run_number)

        # generate new run name
        run_name = 'RUN_' + run_number
        print('Starting ' + run_name)

        # record starting date on the runs.txt logfile
        now = datetime.now()
        with open(logfile_run, "a") as rlog:
            rlog.write('\n')
            rlog.write(run_number + '\t' + now.strftime("%d/%m/%Y %H:%M:%S"))

        # create a database which contains the characterisation measurements
        db_name = run_name + '_experiment_data.db'
        db_path = os.path.join(self.data_path, db_name)
        initialise_or_create_database_at(db_path)
        print(' run database created in', self.data_path)

        # create a sensor logfile for the run
        log_name = run_name + '_log.txt'
        with open(os.path.join(self.logs_path, log_name), 'w') as sensor_log:
            sensor_log.write('# temperature (K)\tpressure (bar)\tmagnetic field (V)\tphotoresistance (V)\tacceleration (m/s^2)\n')
            sensor_log.write(str(self.station.redpitaya.temperature()) + '\t')
            sensor_log.write(str(self.station.redpitaya.pressure()) + '\t')
            sensor_log.write(str(self.station.redpitaya.magnetic_field()) + '\t')
            sensor_log.write(str(self.station.redpitaya.photoresistance()) + '\t')
            sensor_log.write(str(self.station.redpitaya.acceleration()) + '\n')

        # create experiment for the run
        exp = load_or_create_experiment(experiment_name=self.haloscope_name, sample_name=run_name)

        # measure
        print('\nInitiating data acquisition')
        meas = Measurement(exp=exp, station=self.station)

        print(' f1 =', self.station.redpitaya.OUT1_frequency(), 'MHz \tf2 =', self.station.redpitaya.OUT2_frequency(), 'MHz')
        print(' a1 =', self.station.redpitaya.OUT1_amplitude(), 'V \t\ta2 =', self.station.redpitaya.OUT2_amplitude(), 'V')

        # turn on the generators
        self.station.redpitaya.OUT1_status('OFF')
        self.station.redpitaya.OUT2_status('OFF')
        print('\n outputs ready\n')

        # initiate the real acquisition
        waveforms_number = self.station.redpitaya.estimate_waveform_number(duration=data_saver_periodicity/2)
        self.station.redpitaya.number_of_waveforms(waveforms_number)

        data_blocks_number = run_time // data_saver_periodicity

        print(' estimated number of waveforms per data block:', str(waveforms_number))
        print(' estimated number of data blocks:', str(data_blocks_number))

        index = 0
        pbar = tqdm(total=data_blocks_number)

        while data_blocks_number > index:
            index += 1

            # suppress annoying output
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):

                # channel 1
                self.station.redpitaya.OUT1_status('ON')
                do0d(self.station.redpitaya.IN1, measurement_name='data ch1', log_info='Channel 1');
                self.station.redpitaya.OUT1_status('OFF')

                # channel 2
                self.station.redpitaya.OUT2_status('ON')
                do0d(self.station.redpitaya.IN2, measurement_name='data ch2', log_info='Channel 2');
                self.station.redpitaya.OUT2_status('OFF')

            #time.sleep(data_saver_periodicity)
            # add sensor data to the logs
            with open(os.path.join(self.logs_path, log_name), 'a') as sensor_log:
                sensor_log.write(str(self.station.redpitaya.temperature()) + '\t')
                sensor_log.write(str(self.station.redpitaya.pressure()) + '\t')
                sensor_log.write(str(self.station.redpitaya.magnetic_field()) + '\t')
                sensor_log.write(str(self.station.redpitaya.photoresistance()) + '\t')
                sensor_log.write(str(self.station.redpitaya.acceleration()) + '\n')

            pbar.update(1)

        pbar.close()

        # turn off the generators
        self.station.redpitaya.OUT1_status('OFF')
        self.station.redpitaya.OUT2_status('OFF')
        print('\n outputs off\n')

        # calculate the run's duration
        now = datetime.now()
        with open(logfile_run, "a") as rlog:
            rlog.write('\t' + now.strftime("%d/%m/%Y %H:%M:%S"))

        run_duration = time.time() - t0
        print(' total number of acquired waveforms=', str(data_blocks_number*waveforms_number))
        print(' actual duration of the run=', str(run_duration),'s')

        time.sleep(1)
        print('\nRun completed.')

    def generate_simulated_run_data(self, 
                                    f1=5e6, 
                                    a1=1, 
                                    f2=6e6, 
                                    a2=1, 
                                    phase_noise=1e-8, 
                                    amplitude_noise=1e-5, 
                                    number_of_traces=1000, 
                                    axion_signal=True, 
                                    aa=1e-8, 
                                    fa=2e3, 
                                    common_noise_frequency = 1e3, 
                                    common_noise_amplitude = 1e-5
                                    ):
        # This function generates fake data which look like the ones of a real experimental run, it can be 
        # used to test the analysis routines with different parameters. One can add the axion signal with specific
        # frequency and amplitude (only applied on data1 for simplicity), and there is a common noise to evaluate
        # its rejection. 

        fs = self.sampling_frequency
        trace_length = self.buffer_length

        t = np.linspace(0, trace_length / fs, trace_length)

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
                data1[:,i] = a1 * np.sin(2*np.pi*f1 * t + 2*np.pi * 0*np.random.uniform() + axion + phase_noise_trace + common) + amplitude_noise_trace
            else:
                data1[:,i] = a1 * np.sin(2*np.pi*f1 * t + 2*np.pi * 0*np.random.uniform() + phase_noise_trace + common) + amplitude_noise_trace

            # channel 2
            data2[:,i] = a2 * np.sin(2*np.pi*f2 * t + 2*np.pi * 0*np.random.uniform() + phase_noise_trace + common) + amplitude_noise_trace

        return t, np.rot90(data1), np.rot90(data2)
        
        
    """
    def analyse_run(self, 
                    run, 
                    blocks_to_analyse=1, 
                    mode='amplitude_noise_rejection', 
                    window='blackman', 
                    return_magnetic_field=True, 
                    lorentzian_normalization=False,
                    fold=True
                   ):
        # Simple analysis routine to extract the limit on the effective magnetic field starting from the run data
        # of the fermionic interferometer. From the raw data this function outputs an averaged interferometric measurement
        # which can be used to calculate an upper limit on the axion signal.

        print('Loading data')
        RUN = run.split('_')[1]

        # open logfile and find the date corresponding to the run number
        logfile_run = os.path.join(os.path.dirname(self.logs_path), 'runs.txt')
        
        with open(logfile_run, "r") as rlog:
            for ln in rlog:
                if ln.startswith(RUN):
                    date = ln.split('\t')[1][6:10] + '-' + ln.split('\t')[1][3:5] + '-' + ln.split('\t')[1][:2]
            db_date = os.path.join(self.data_path[:-10], date)

        # create a database which contains the characterisation measurements
        db_name = run + '_experiment_data.db'
        db_path = os.path.join(db_date, db_name)
        print(' loading data from', db_path)
        initialise_or_create_database_at(db_path)
        print(' ' + run + ' data loaded')

        print('\nInterferometric down-conversion')
        f1 = self.experiment_parameters['f1']
        f2 = self.experiment_parameters['f2']
        print(' north arm frequency', str(f1/1e6),'MHz, east arm frequency', str(f2/1e6), 'MHz')

        if mode == 'phase_noise_rejection':
            delta = np.pi / 1.00
        if mode == 'amplitude_noise_rejection':
            delta = np.pi / 2.00
        print(' ' + mode + ' mode selected')

        BUFFER = self.buffer_length
        
        t = np.linspace(0, BUFFER-1, BUFFER)
        downconversion_frequency = (f1 + f2)/2
        print(' down-conversion frequency set to', str(downconversion_frequency/1e6), 'MHz')
        
        downconverted_frequency_origin = (f2 - f1)/2
        downconversion_signal = np.sin(2*np.pi * downconversion_frequency * t / self.sampling_frequency + delta)
        index = 0
        interference_spectrum_matrix = np.zeros((BUFFER // 2 + 1, blocks_to_analyse))
        
        print('\nAnalysis and averaging')
        print(' in progress')
        pbar = tqdm(total=blocks_to_analyse)
        
        while index < blocks_to_analyse:
            dataset1 = load_by_run_spec(captured_run_id=2*index + 1)
            data1 = dataset1.get_parameter_data('redpitaya_IN1')['redpitaya_IN1']['redpitaya_IN1']
            dataset2 = load_by_run_spec(captured_run_id=2*index + 2)
            data2 = dataset2.get_parameter_data('redpitaya_IN2')['redpitaya_IN2']['redpitaya_IN2']

            #print('data1 hash:', hash(data1.tobytes()) if hasattr(data1, 'tobytes') else [hash(d.tobytes()) for d in data1])
            #print('data2 hash:', hash(data2.tobytes()) if hasattr(data2, 'tobytes') else [hash(d.tobytes()) for d in data2])
            
            if len(data1) != len(data2):
                raise ValueError('The length of the two datasets are different.')
            else:
                L = len(data1)

            # extract Ap from raw time traces before mixing
            # peak amplitude = (max - min), averaged over all waveforms and both arms
            Ap1 = np.mean([(np.max(data1[w]) - np.min(data1[w])) for w in range(L)])
            Ap2 = np.mean([(np.max(data2[w]) - np.min(data2[w])) for w in range(L)])
            Ap_measured = (Ap1 + Ap2) / 2
                
            interference = np.zeros((BUFFER, L))
            for waveform_index in np.arange(L):
                interference[:, waveform_index] = self.mixer(
                    data1[waveform_index] + data2[waveform_index],
                    downconversion_signal
                )
                ps = periodogram(interference[:, waveform_index], fs=self.sampling_frequency, window=window, scaling='spectrum')
                interference_f = ps[0] - downconverted_frequency_origin
                if return_magnetic_field == True:
                    _, ps_tesla = self.volt_to_magnetic_field(interference_f, ps[1] / L, Ap=Ap_measured, lorentzian_normalization=lorentzian_normalization)
                    interference_spectrum_matrix[:, index] += ps_tesla
                else:
                    interference_spectrum_matrix[:, index] += ps[1] / L
            index += 1
            pbar.update(1)
            
        pbar.close()

        if fold:
            interference_f, interference_spectrum_matrix = self.fold_spectrum_matrix(
                interference_f, interference_spectrum_matrix
            )
    
        return interference_f, interference_spectrum_matrix
    """


    def load_run_data(self, run, blocks_to_analyse, remove_block_initial_waveform = True):
        """
        Load raw time traces from a run.
    
        Returns
        -------
        data1 : list of np.ndarray
            data1[i] has shape (n_waveforms, buffer_length)
        data2 : list of np.ndarray
            Same for the second channel.
        """
    
        print("Loading data")
    
        RUN = run.split('_')[1]
    
        logfile_run = os.path.join(os.path.dirname(self.logs_path), "runs.txt")
    
        with open(logfile_run) as rlog:
            for ln in rlog:
                if ln.startswith(RUN):
                    date = (
                        ln.split('\t')[1][6:10] + '-'
                        + ln.split('\t')[1][3:5] + '-'
                        + ln.split('\t')[1][:2]
                    )
    
        db_date = os.path.join(self.data_path[:-10], date)
    
        db_name = run + "_experiment_data.db"
        db_path = os.path.join(db_date, db_name)
    
        initialise_or_create_database_at(db_path)
    
        data1_blocks = []
        data2_blocks = []
    
        for index in tqdm(range(blocks_to_analyse), desc="Loading"):
    
            dataset1 = load_by_run_spec(captured_run_id=2 * index + 1)
            dataset2 = load_by_run_spec(captured_run_id=2 * index + 2)
    
            data1 = dataset1.get_parameter_data(
                "redpitaya_IN1"
            )["redpitaya_IN1"]["redpitaya_IN1"]
    
            data2 = dataset2.get_parameter_data(
                "redpitaya_IN2"
            )["redpitaya_IN2"]["redpitaya_IN2"]

            if remove_block_initial_waveform:
                # sometimes it is necessary to discard the first waveform of each data block
                data1 = data1[1:]
                data2 = data2[1:]
    
            data1_blocks.append(np.asarray(data1))
            data2_blocks.append(np.asarray(data2))

        data1_full = np.concatenate(data1_blocks, axis=0)
        data2_full = np.concatenate(data2_blocks, axis=0)
    
        return data1_full, data2_full


    def compute_interference_spectrum(
        self,
        data1,
        data2,
        mode='amplitude_noise_rejection',
        window='blackman',
        return_magnetic_field=True,
        lorentzian_normalization=False,
        ):
        """
        Down-convert and compute the interference spectrum for a full set of
        concatenated waveforms (as returned by load_run_data).
    
        Parameters
        ----------
        data1, data2 : np.ndarray, shape (n_waveforms, buffer_length)
            Output of load_run_data.
    
        Returns
        -------
        interference_f : np.ndarray, shape (buffer_length // 2 + 1,)
        interference_spectrum_matrix : np.ndarray, shape (buffer_length // 2 + 1, n_waveforms)
        """
    
        if len(data1) != len(data2):
            raise ValueError('The length of the two datasets are different.')
        L = len(data1)
    
        f1 = self.experiment_parameters['f1']
        f2 = self.experiment_parameters['f2']
        print(' north arm frequency', str(f1/1e6),'MHz, east arm frequency', str(f2/1e6), 'MHz')
    
        if mode == 'phase_noise_rejection':
            delta = np.pi / 1.00
        elif mode == 'amplitude_noise_rejection':
            delta = np.pi / 2.00
        print(' ' + mode + ' mode selected')
    
        BUFFER = self.buffer_length
        t = np.linspace(0, BUFFER - 1, BUFFER)
        downconversion_frequency = (f1 + f2) / 2
        downconverted_frequency_origin = (f2 - f1) / 2
        downconversion_signal = np.sin(
            2 * np.pi * downconversion_frequency * t / self.sampling_frequency + delta
        )
        print(' down-conversion frequency set to', str(downconversion_frequency/1e6), 'MHz')

    
        # peak amplitude = (max - min), averaged over all waveforms and both arms
        Ap1 = np.mean([(np.max(data1[w]) - np.min(data1[w])) for w in range(L)])
        Ap2 = np.mean([(np.max(data2[w]) - np.min(data2[w])) for w in range(L)])
        Ap_measured = (Ap1 + Ap2) / 2
        print(' measured arm drive amplitude', Ap_measured, 'V')
    
        interference_spectrum_matrix = np.zeros((BUFFER // 2 + 1, L))
        interference_f = None
    
        print('\nAnalysis and averaging')
        print(' in progress')
    
        for waveform_index in tqdm(range(L), desc="Interfering"):
            interference = self.mixer(
                data1[waveform_index] + data2[waveform_index],
                downconversion_signal
            )
            ps = periodogram(interference, fs=self.sampling_frequency, window=window, scaling='spectrum')
            interference_f = ps[0] - downconverted_frequency_origin
    
            if return_magnetic_field:
                _, ps_tesla = self.volt_to_magnetic_field(
                    interference_f, ps[1],
                    Ap=Ap_measured,
                    lorentzian_normalization=lorentzian_normalization
                )
                interference_spectrum_matrix[:, waveform_index] = ps_tesla
            else:
                interference_spectrum_matrix[:, waveform_index] = ps[1] / L
    
        return interference_f, interference_spectrum_matrix
        
        
        
    def fold_spectrum_matrix(self, frequency, matrix):
        """
        Fold a two-sided spectrum (or spectrum-difference) matrix around zero
        frequency, combining each positive-frequency bin with its mirrored
        negative-frequency bin.
    
        Parameters
        frequency : np.ndarray, shape (n_bins,)
        matrix    : np.ndarray, shape (n_bins, n_blocks)
    
        Returns
        folded_frequency : np.ndarray, shape (n_fold,)
        folded_matrix    : np.ndarray, shape (n_fold, n_blocks)
        """
        zero_f = np.min(np.abs(frequency))
        zero_index, = np.where(np.isclose(np.abs(frequency), zero_f))
        zero_index = zero_index[0]
    
        folded_frequency = np.linspace(zero_f, -frequency[0], zero_index)
        folded_matrix = (
            matrix[0:zero_index, :] + np.flip(matrix[zero_index:2 * zero_index, :], axis=0)
        ) / 2
    
        return folded_frequency, folded_matrix

    
    def vet_candidates(self, frequency, signal_matrix, background_matrix,
                       sigma_threshold=3.0, outlier_fraction=0.6, neighbor_bins=2):
        """
        Vet candidate signal bins from a per-bin two-sample comparison.
    
        Parameters
        frequency         : np.ndarray, shape (n_bins,), frequency axis in Hz
        signal_matrix     : np.ndarray, shape (n_bins, n_blocks)
        background_matrix : np.ndarray, shape (n_bins, n_blocks)
        sigma_threshold   : float, detection threshold in sigma (default 3.0)
        outlier_fraction  : float, minimum fraction of signal blocks above
                            background mean to pass outlier check (default 0.6)
        neighbor_bins     : int, number of bins on each side to check for
                            broad spectral features (default 2)
    
        Returns
        candidates : list of dict, one entry per bin above threshold,
                     each with keys: bin, frequency, snr, excess, sigma,
                     frac_above_bkg, check1_positive, check2_not_outlier,
                     check3_isolated, flagged
        """
        assert signal_matrix.shape == background_matrix.shape, \
            "Signal and background matrices must have the same shape."
    
        n_bins, n_blocks = signal_matrix.shape
    
        sig_mean = np.mean(signal_matrix, axis=1)
        bkg_mean = np.mean(background_matrix, axis=1)
        sig_var  = np.var(signal_matrix, axis=1, ddof=1)
        bkg_var  = np.var(background_matrix, axis=1, ddof=1)
    
        excess = sig_mean - bkg_mean
        sigma  = np.sqrt(sig_var / n_blocks + bkg_var / n_blocks)
        snr    = excess / sigma
    
        candidates = []
        for b in range(n_bins):
            if snr[b] < sigma_threshold:
                continue
    
            # check 1: excess must be positive -- negative means more power
            # in background, which cannot be a signal
            check1 = excess[b] > 0
    
            # check 2: excess must not be driven by a few outlier blocks --
            # a real signal should lift the majority of blocks above the
            # background mean uniformly
            frac_above = np.mean(signal_matrix[b, :] > bkg_mean[b])
            check2 = frac_above > outlier_fraction
    
            # check 3: excess must be isolated in a single bin -- a broad
            # feature spanning multiple bins is a spectral artifact, not
            # a narrow axion-like signal
            neighbor_snrs = [
                snr[b + d]
                for d in range(-neighbor_bins, neighbor_bins + 1)
                if d != 0 and 0 <= b + d < n_bins
            ]
            check3 = not any(ns > sigma_threshold for ns in neighbor_snrs)
    
            flagged = check1 and check2 and check3
    
            candidates.append({
                'bin'               : b,
                'frequency'         : frequency[b],
                'snr'               : snr[b],
                'excess'            : excess[b],
                'sigma'             : sigma[b],
                'frac_above_bkg'    : frac_above,
                'check1_positive'   : check1,
                'check2_not_outlier': check2,
                'check3_isolated'   : check3,
                'flagged'           : flagged,
            })
    
        return candidates
    

    # parallelised injection test
    def _single_trial(
        self,
        background_matrix,
        injected_power,
        inj_bin,
        sigma_threshold,
        outlier_fraction,
        neighbor_bins,
        confidence_level,
    ):
        n_bins, n_blocks = background_matrix.shape
    
        idx_sig = np.random.randint(0, n_blocks, n_blocks)
        idx_bkg = np.random.randint(0, n_blocks, n_blocks)
    
        sim_background = background_matrix[:, idx_bkg]
        sim_signal = background_matrix[:, idx_sig].copy()
    
        sim_signal[inj_bin, :] += injected_power
    
        dummy_f = np.arange(n_bins)
    
        _, ul, excess, sigma = self.calculate_upper_limit(
            dummy_f,
            sim_signal,
            sim_background,
            confidence_level=confidence_level,
        )
    
        candidates = self.vet_candidates(
            dummy_f,
            sim_signal,
            sim_background,
            sigma_threshold=sigma_threshold,
            outlier_fraction=outlier_fraction,
            neighbor_bins=neighbor_bins,
        )
    
        detected = any(
            (c["flagged"] and c["bin"] == inj_bin)
            for c in candidates
        )
    
        return excess[inj_bin], ul[inj_bin], detected

    
    def injection_test_parallel(
        self,
        background_matrix,
        injected_power,
        injection_bins=None,
        sigma_threshold=3.0,
        outlier_fraction=0.6,
        neighbor_bins=2,
        confidence_level=0.95,
        n_trials=500,
        n_jobs=-1,
    ):
        n_bins, _ = background_matrix.shape
    
        if injection_bins is None:
            injection_bins = np.arange(n_bins)
    
        recovery_bias = np.full(n_bins, np.nan)
        recovery_sigma = np.full(n_bins, np.nan)
        detection_fraction = np.full(n_bins, np.nan)
        mean_upper_limit = np.full(n_bins, np.nan)
    
        for inj_bin in tqdm(injection_bins, leave=False):
    
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._single_trial)(
                    background_matrix,
                    injected_power,
                    inj_bin,
                    sigma_threshold,
                    outlier_fraction,
                    neighbor_bins,
                    confidence_level,
                )
                for _ in range(n_trials)
            )
    
            excess_vals, ul_vals, detected_vals = zip(*results)
    
            excess_vals = np.array(excess_vals)
            ul_vals = np.array(ul_vals)
            detected_vals = np.array(detected_vals)
    
            recovery_bias[inj_bin] = np.mean(excess_vals) - injected_power
            recovery_sigma[inj_bin] = np.std(excess_vals)
            detection_fraction[inj_bin] = np.mean(detected_vals)
            mean_upper_limit[inj_bin] = np.mean(ul_vals)
    
        return recovery_bias, recovery_sigma, detection_fraction, mean_upper_limit
        
        

    def calculate_upper_limit(self, frequency, signal_matrix,
                              background_matrix, confidence_level=0.95):
        """
        Compute a per-bin Bayesian upper limit on excess power from two power spectra
        matrices, using a truncated Gaussian posterior with a flat non-negative
        prior on the signal power. Assumes signal_matrix and background_matrix
        have already been folded (if desired) by analyse_run -- this function
        is purely statistical and has no notion of frequency structure.
        """
        assert signal_matrix.shape == background_matrix.shape, \
            "Signal and background matrices must have the same shape."
    
        n_bins, n_blocks = signal_matrix.shape
    
        excess = np.mean(signal_matrix, axis=1) - np.mean(background_matrix, axis=1)
        sigma  = np.sqrt(
            np.var(signal_matrix, axis=1, ddof=1) / n_blocks +
            np.var(background_matrix, axis=1, ddof=1) / n_blocks
        )
    
        upper_limit = np.zeros(n_bins)
        for i in range(n_bins):
            a = -excess[i] / sigma[i]
            dist = stats.truncnorm(a=a, b=np.inf, loc=excess[i], scale=sigma[i])
            upper_limit[i] = dist.ppf(confidence_level)
    
        return frequency, upper_limit, excess, sigma
    
        

    # experiment-specific functions
    def volt_to_magnetic_field(self, frequency, voltage_spectrum, Ap, lorentzian_normalization=False):
        
        f = (self.experiment_parameters['f1'] + self.experiment_parameters['f2']) / 2
        k = (self.experiment_parameters['k1'] + self.experiment_parameters['k2']) / 2
        #Q = f / k
        
        gamma = c.physical_constants['electron gyromag. ratio in MHz/T'][0] * 1e6
        volt_to_tesla = 2 * k / (np.pi * gamma * Ap)
    
        magnetic_spectrum = volt_to_tesla * np.sqrt(voltage_spectrum)
        resonance = self.lorentzian(frequency, 0, 1, k)

        if lorentzian_normalization == True:        
            if voltage_spectrum.ndim == 2:
                magnetic_spectrum = magnetic_spectrum / resonance[:, np.newaxis]
            else:
                magnetic_spectrum = magnetic_spectrum / resonance            
    
        return frequency, magnetic_spectrum

        
"""
    def calculate_residuals_upper_limit(self, frequency, signal, background, sigmas=2):
        # from an analysed run this function extracts the upper limit of no field for a given 
        # confidence level. Default is 2sigma = 90% C.L. and signal and background need to be defined
        # in the same frequency range.

        zero_f = np.min(np.abs(frequency))
        zero_index, = np.where(np.isclose(np.abs(frequency), zero_f))
        # compute the frequency axis
        f = np.linspace(zero_f, -frequency[0], zero_index[0])

        # as an alternative to the background one can use a fitting function with the model of the background
        residuals = signal - background
        folded_residuals = (residuals[0:zero_index[0]] + np.flip(residuals[zero_index[0]:2*zero_index[0]])) / 2

        # sigmas confidence level
        sigma = sigmas * np.abs(folded_residuals)

        return f, sigma
"""


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


