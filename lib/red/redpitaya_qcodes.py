# -*- coding: utf-8 -*-
# This is a Qcodes driver for Redpitaya board written for the Alca Haloscope project.
# Written by Nicolò Crescini with inspiration from a version by Arpit Ranadive and Martina Esposito.

import time
import binascii
from typing import Tuple

import numpy as np
from scipy.signal import periodogram, butter, sosfilt
import matplotlib.pyplot as plt

from tqdm import tqdm, trange 

from qcodes import validators as vals
from qcodes.instrument import ( Instrument,
                                InstrumentChannel,
                                InstrumentModule,
                                ManualParameter,
                                MultiParameter,
                                VisaInstrument,
                                )

from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter, ParamRawDataType


class GeneratedSetPoints(Parameter):
    # A parameter that generates a setpoint array from start, stop and num points parameters.

    def __init__(self, start_param, stop_param, num_points_param, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_param = start_param
        self._stop_param = stop_param 
        self._num_points_param = num_points_param

    def get_raw(self):
        return np.linspace(self._start_param(), self._stop_param() -1, self._num_points_param())


class IN1_data(ParameterWithSetpoints):
    # Formats and outputs the raw data acquired by the Redpitaya

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):
        waveforms_number = self._instrument.number_of_waveforms()
        sampling_frequency = self._instrument.sampling_frequency()
        decimation = self._instrument.ADC_decimation()
        points = self._instrument.waveform_points()

        # acquire data from channel 1
        raw_data = self._instrument.get_data(1, waveforms_number, data_type='BIN')

        try:
            data = np.reshape(raw_data, (self._instrument.number_of_waveforms(), 
                                         self._instrument.BUFFER_SIZE))
        except:
            raise NameError(raw_data)
        
        return data


class IN2_data(ParameterWithSetpoints):
    # Formats and outputs the raw data acquired by the Redpitaya

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):
        waveforms_number = self._instrument.number_of_waveforms()
        sampling_frequency = self._instrument.sampling_frequency()
        decimation = self._instrument.ADC_decimation()
        points = self._instrument.waveform_points()

        # acquire data from channel 2
        raw_data = self._instrument.get_data(2, waveforms_number, data_type='BIN')

        try:
            data = np.reshape(raw_data, (self._instrument.number_of_waveforms(), 
                                         self._instrument.BUFFER_SIZE))
        except:
            raise NameError(raw_data)
        
        return data


class TX1_trace(ParameterWithSetpoints):
    ## Formats and outputs the raw data acquired by the Redpitaya
    # Note: one needs to set the decimation before using the VNA, so to use a 
    # large decimation for the low frequency traces and viceversa.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):
        start_frequency = self._instrument.vna_start()
        stop_frequency = self._instrument.vna_stop()
        number_of_points = self._instrument.vna_points()
        number_of_averages = self._instrument.vna_averages()

        # initalise the trace array
        frequency = np.linspace(start_frequency, stop_frequency, number_of_points)
        magnitude = np.zeros(number_of_points)
        #phase = np.zeros(number_of_points)

        # first turn on the ADC and the sources
        self._instrument.ADC_data_format('ASCII')
        self._instrument.ADC_trigger_level(0.0)

        self._instrument.OUT_trigger()
        self._instrument.OUT1_status('ON')

        # rest
        time.sleep(0.2)

        # then measure the points from channel 1
        with tqdm(total = number_of_averages * number_of_points) as pbar:
            for avg in range(number_of_averages):
                for point in range(number_of_points):
                    m0, p0 = self._instrument.spectrscopy(1, frequency[point])
                    magnitude[point] += m0
                    #phase[point] += p0
                    # update progress bar
                    pbar.update(1)

        magnitude = magnitude / number_of_averages
        #phase = phase / number_of_averages

        # eventually turn off the source
        self._instrument.OUT1_status('OFF')
        
        return magnitude


class TX2_trace(ParameterWithSetpoints):
    ## Formats and outputs the raw data acquired by the Redpitaya
    # Note: one needs to set the decimation before using the VNA, so to use a 
    # large decimation for the low frequency traces and viceversa.
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):
        start_frequency = self._instrument.vna_start()
        stop_frequency = self._instrument.vna_stop()
        number_of_points = self._instrument.vna_points()
        number_of_averages = self._instrument.vna_averages()

        # initalise the trace array
        frequency = np.linspace(start_frequency, stop_frequency, number_of_points)
        magnitude = np.zeros(number_of_points)
        #phase = np.zeros(number_of_points)

        # first turn on the ADC and the sources
        self._instrument.ADC_data_format('ASCII')
        self._instrument.ADC_trigger_level(0.0)


        self._instrument.OUT_trigger()
        self._instrument.OUT2_status('ON')

        # then measure the points from channel 2
        with tqdm(total = number_of_averages * number_of_points) as pbar:
            for avg in range(number_of_averages):
                for point in range(number_of_points):
                    #time.sleep(0.2)
                    m0, p0 = self._instrument.spectrscopy(2, frequency[point])
                    magnitude[point] += m0
                    #phase[point] += p0
                    # update progress bar
                    pbar.update(1)

        magnitude = magnitude / number_of_averages
        #phase = phase / number_of_averages

        # eventually turn off the sources
        self._instrument.OUT2_status('OFF')
        
        return magnitude


class VNA1_trace(MultiParameter):
    ## Formats and outputs the raw data acquired by the Redpitaya
    # Note: one needs to set the decimation before using the VNA, so to use a 
    # large decimation for the low frequency traces and viceversa.
    #
    # Right now this is not working, as it is not possible to consistently measure the phase.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self) -> Tuple[ParamRawDataType, ...]:
        start_frequency = self._instrument.vna_start()
        stop_frequency = self._instrument.vna_stop()
        number_of_points = self._instrument.vna_points()
        number_of_averages = self._instrument.vna_averages()

        # initalise the trace array
        frequency = np.linspace(start_frequency, stop_frequency, number_of_points)
        magnitude = np.zeros(number_of_points)
        phase = np.zeros(number_of_points)

        # first turn on the ADC and the sources
        self._instrument.ADC_data_format('ASCII')
        self._instrument.ADC_trigger_level(0.0)

        self._instrument.OUT_trigger()
        self._instrument.OUT1_status('ON')

        # rest
        time.sleep(0.2)

        # then measure the points from channel 1
        with tqdm(total = number_of_averages * number_of_points) as pbar:
            for avg in range(number_of_averages):
                for point in range(number_of_points):
                    m0, p0 = self._instrument.spectrscopy(1, frequency[point])
                    magnitude[point] += m0
                    phase[point] += p0
                    # update progress bar
                    pbar.update(1)

        magnitude = magnitude / number_of_averages
        phase = phase / number_of_averages

        # eventually turn off the source
        self._instrument.OUT1_status('OFF')
        
        return magnitude, phase


class VNA2_trace(MultiParameter):
    ## Formats and outputs the raw data acquired by the Redpitaya
    # Note: one needs to set the decimation before using the VNA, so to use a 
    # large decimation for the low frequency traces and viceversa.
    #
    # Right now this is not working, as it is not possible to consistently measure the phase.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):
        start_frequency = self._instrument.vna_start()
        stop_frequency = self._instrument.vna_stop()
        number_of_points = self._instrument.vna_points()
        number_of_averages = self._instrument.vna_averages()

        # initalise the trace array
        frequency = np.linspace(start_frequency, stop_frequency, number_of_points)
        magnitude = np.zeros(number_of_points)
        phase = np.zeros(number_of_points)

        # first turn on the ADC and the sources
        self._instrument.ADC_data_format('ASCII')
        self._instrument.ADC_trigger_level(0.0)


        self._instrument.OUT_trigger()
        self._instrument.OUT2_status('ON')

        # then measure the points from channel 2
        with tqdm(total = number_of_averages * number_of_points) as pbar:
            for avg in range(number_of_averages):
                for point in range(number_of_points):
                    #time.sleep(0.2)
                    m0, p0 = self._instrument.spectrscopy(2, frequency[point])
                    magnitude[point] += m0
                    phase[point] += p0
                    # update progress bar
                    pbar.update(1)

        magnitude = magnitude / number_of_averages
        phase = phase / number_of_averages

        # eventually turn off the sources
        self._instrument.OUT2_status('OFF')
        
        return magnitude, phase


class GetTemperature(Parameter):
    ## Measures the temperature from the Arduino module
    # Notes: 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):  

        # start UART communication
        #self._instrument.UART_init()
        #self._instrument.UART_setup()

        # the Aruino is configured to return the pressure when prompted with a 'p'
        t = str( ord('t') )

        # set the length of the message and send it, since it is a single letter the length is 1
        self._instrument.UART_data_length(1)
        self._instrument.UART_write(t)   
        time.sleep(0.1)
        
        # set the length of the response, which is 7 bytes (e.g. 1023.51 = 7 characters) plus stop byte and new line
        self._instrument.UART_data_length(7)
        t_string = self._instrument.UART_read() 
        
        # format the response into a number
        t_string = t_string[1:-1].split(',')
        t_string = list(map(int, t_string[0:5]))
        t_string = "".join( list(map(chr, t_string)))

        # pressure
        t = float(t_string)

        # close UART communication
        #self._instrument.UART_release()

        celsius_to_K = 273.15
        return t + celsius_to_K


class GetPressure(Parameter):
    ## Measures the pressure from the Arduino module
    # Notes: 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):  

        # start UART communication
        #self._instrument.UART_init()
        #self._instrument.UART_setup()

        # the Aruino is configured to return the pressure when prompted with a 'p'
        p = str( ord('p') )

        # set the length of the message and send it, since it is a single letter the length is 1
        self._instrument.UART_data_length(1)
        self._instrument.UART_write(p)   

        # set the length of the response, which is 7 bytes (e.g. 1023.51 = 7 characters) plus stop byte and new line
        self._instrument.UART_data_length(9)
        p_string = self._instrument.UART_read() 

        # format the response into a number
        p_string = p_string[1:-1].split(',')
        p_string = list(map(int, p_string[0:7]))
        p_string = "".join( list(map(chr, p_string)))

        # pressure
        p = float(p_string)

        # close UART communication
        #self._instrument.UART_release()

        hPa_to_bar = 1e-3
        return hPa_to_bar*p


class GetMagneticField(Parameter):
    ## Measures the magnetic field on the Hall probe connected to the Arduino
    # Notes: 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):  

        # start UART communication
        #self._instrument.UART_init()
        #self._instrument.UART_setup()

        # the Aruino is configured to return the pressure when prompted with a 'p'
        b = str( ord('b') )

        # set the length of the message and send it, since it is a single letter the length is 1
        self._instrument.UART_data_length(1)
        self._instrument.UART_write(b)   

        # set the length of the response, which is 7 bytes (e.g. 1023.51 = 7 characters) plus stop byte and new line
        self._instrument.UART_data_length(9)
        b_string = self._instrument.UART_read() 

        # format the response into a number
        b_string = b_string[1:-1].split(',')
        b_string = list(map(int, b_string[0:7]))
        b_string = "".join( list(map(chr, b_string)))

        # magnetic field
        pin_read_to_voltage = 5.0 / 1024
        b = float(b_string) * pin_read_to_voltage

        # close UART communication
        #self._instrument.UART_release()

        return b # V


class GetPhotoresistance(Parameter):
    ## Measures the light on the photoresistance connected to the Arduino
    # Notes: 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):  

        # start UART communication
        #self._instrument.UART_init()
        #self._instrument.UART_setup()

        # the Aruino is configured to return the pressure when prompted with a 'p'
        l = str( ord('l') )

        # set the length of the message and send it, since it is a single letter the length is 1
        self._instrument.UART_data_length(1)
        self._instrument.UART_write(l)   

        # set the length of the response, which is 7 bytes (e.g. 1023.51 = 7 characters) plus stop byte and new line
        self._instrument.UART_data_length(9)
        l_string = self._instrument.UART_read() 

        # format the response into a number
        l_string = l_string[1:-1].split(',')
        l_string = list(map(int, l_string[0:7]))
        l_string = "".join( list(map(chr, l_string)))

        # light
        l = float(l_string)

        # close UART communication
        #self._instrument.UART_release()

        pin_read_to_voltage = 5.0 / 1024
        return pin_read_to_voltage * l


class GetAcceleration(Parameter):
    ## Measures the acceleration from the Arduino module
    # Notes: 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):  

        # start UART communication
        #self._instrument.UART_init()
        #self._instrument.UART_setup()

        # the Aruino is configured to return the pressure when prompted with a 'x', 'y' or 'z'
        ain = ['x', 'y', 'z']
        aout = np.zeros(3)

        for ind in range(3):
            a = str( ord(ain[ind]) )

            # set the length of the message and send it, since it is a single letter the length is 1
            self._instrument.UART_data_length(1)
            self._instrument.UART_write(a)   

            # set the length of the response, which is 7 bytes (e.g. 1023.51 = 7 characters) plus stop byte and new line
            self._instrument.UART_data_length(9)
            a_string = self._instrument.UART_read() 

            # format the response into a number
            a_string = a_string[1:-1].split(',')
            a_string = list(map(int, a_string[0:7]))
            a_string = "".join( list(map(chr, a_string)))

            # pressure
            aout[ind] = float(a_string)

        # close UART communication
        #self._instrument.UART_release()

        return aout[0], aout[1], aout[2]


class Redpitaya(VisaInstrument):
    ## main driver
    # a general issue is the triggering in binary mode
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r\n', **kwargs)
        
        # Connect to the Redpitaya board using the appropriate communication protocol
        self._address = address
        # Connect using wireless or ethernet
        # For wireless or ethernet connection use TCPIP::address::port::SOCKET
        # 

        # Sampling frequency
        self.FS = 125000000.0
        # Buffer size
        self.BUFFER_SIZE = 2**14

        # see https://redpitaya.readthedocs.io/en/latest/appsFeatures/remoteControl/remoteControl.html
        # for more detail of the commands and of their limits
        
        
        # analog outs
        input_pin = [ 'AIN0', 'AIN1', 'AIN2', 'AIN3' ]
        output_pin = [ 'AOUT0', 'AOUT1', 'AOUT2', 'AOUT3' ]
        max_get_voltage = 3.3
        max_set_voltage = 1.8
        
        for pin in output_pin:
            # get the voltage for any of the analog pins
            self.add_parameter( name=pin,
                                label='Read input/output voltage on pin',
                                vals=vals.Numbers(-max_get_voltage, max_get_voltage),
                                unit='V',
                                set_cmd='ANALOG:PIN ' + pin + ',' + '{:.12f}',
                                get_cmd='ANALOG:PIN? ' + pin,
                                get_parser=float
                                )        
            
        for pin in input_pin:
            # set the voltage for the output pins
            self.add_parameter( name=pin,
                                label='Set output voltage on pin',
                                vals=vals.Numbers(-max_set_voltage, max_set_voltage),
                                unit='V',
                                set_cmd=None,
                                get_cmd='ANALOG:PIN? ' + pin,
                                get_parser=float
                                )  
            
        # UART communication
        # handling parameters
        self.add_parameter( name='UART_bits',
                            label='Sets the character size in bits.',
                            vals=vals.Enum('CS6', 'CS7', 'CS8'),
                            unit='',
                            set_cmd='UART:BITS ' + '{}',
                            get_cmd='UART:BITS?'
                            )

        self.add_parameter( name='UART_speed',
                            label='Speed of the UART connection.',
                            vals=vals.Enum(1200,2400,4800,9600,19200,38400,57600,115200,230400,576000,921000,1000000,1152000,1500000,2000000,2500000,3000000,3500000,4000000),
                            unit='',
                            set_cmd='UART:SPEED ' + '{}',
                            get_cmd='UART:SPEED?'
                            )

        self.add_parameter( name='UART_stop_bit',
                            label='Length of the stop bit.',
                            vals=vals.Enum('STOP1', 'STOP2'),
                            unit='',
                            set_cmd='UART:STOPB ' + '{}',
                            get_cmd='UART:STOPB?'
                            )

        self.add_parameter( name='UART_parity',
                            label='Sets parity check mode.',
                            vals=vals.Enum('NONE', 'EVEN', 'ODD', 'MARK', 'SPACE'),
                            unit='',
                            set_cmd='UART:PARITY ' + '{}',
                            get_cmd='UART:PARITY?'
                            )

        self.add_parameter( name='UART_timeout',
                            label='Sets the timeout.',
                            vals=vals.Numbers(0,255),
                            unit='',
                            set_cmd='UART:TIMEOUT ' + '{}',
                            get_cmd='UART:TIMEOUT?',
                            get_parser = int
                            )

        self.add_parameter( name='UART_data_length',
                            vals=vals.Numbers(0,255),
                            unit='',
                            initial_value=1,
                            set_cmd=None,
                            get_cmd=None,
                            get_parser = int            
                            )

        # SPI communication
        # handling parameters
        """
        self.add_parameter( name='SPI_mode',
                            label='Mode for SPI.',
                            vals=vals.Enum('LISL', 'LIST', 'HISL', 'HIST'),
                            unit='',
                            set_cmd='SPI:SET:MODE ' + '{}',
                            get_cmd='SPI:SET:MODE?'
                            )

        self.add_parameter( name='SPI_csmode',
                            label='CS mode for SPI.',
                            vals=vals.Enum('NORMAL', 'HIGH'),
                            unit='',
                            set_cmd='SPI:SET:CSMODE ' + '{}',
                            get_cmd='SPI:SET:CSMODE?'
                            )

        self.add_parameter( name='SPI_speed',
                            label='Speed of the SPI connection.',
                            vals=vals.Numbers(1,100000000),
                            unit='',
                            set_cmd='SPI:SET:SPEED ' + '{}',
                            get_cmd='SPI:SET:SPEED?',
                            get_parser=int
                            )

        self.add_parameter( name='SPI_word',
                            label='Length of the word in bits.',
                            vals=vals.Numbers(7,np.inf),
                            unit='',
                            set_cmd='SPI:SET:WORD ' + '{}',
                            get_cmd='SPI:SET:WORD?',
                            get_parser=int
                            )

        # I2C communication
        # handling parameters
        self.add_parameter( name='I2C_dev',
                            label='Device address on the I2C bus in dec format.',
                            #vals=vals.Enum('LISL', 'LIST', 'HISL', 'HIST'),
                            unit='',
                            set_cmd=None,
                            get_cmd='I2C:DEV?'
                            )

        self.add_parameter( name='I2C_fmode',
                            label='Enables forced bus operation even if the device is in use.',
                            vals=vals.Enum('ON', 'OFF'),
                            unit='',
                            set_cmd='I2C:FMODE ' + '{}',
                            get_cmd='I2C:FMODE?'
                            )
        """


        ## signal generators
        min_frequency = 1
        max_frequency = 50e6
        max_voltage = 1

        # output generators
        outputs = ['1', '2']
        for out in outputs:
            self.add_parameter( name='OUT' + out + '_status',
                                label='Status of the generator 1',
                                vals=vals.Enum('ON', 'OFF'),
                                set_cmd='OUTPUT' + out + ':STATE ' + '{}',
                                get_cmd='OUTPUT' + out + ':STATE?'
                                )

            self.add_parameter( name='OUT' + out + '_function',
                                label='Output function of the generator',
                                vals=vals.Enum('SINE', 'SQUARE', 'TRIANGLE', 'SAWU', 'SAWD', 'PWM', 'ARBITRARY', 'DC', 'DC_NEG'),
                                set_cmd='SOUR' + out + ':FUNC ' + '{}',
                                get_cmd='SOUR' + out + ':FUNC?'
                                )        

            self.add_parameter( name='OUT' + out + '_frequency',
                                label='Frequency of the generator',
                                vals=vals.Numbers(min_frequency, max_frequency),
                                unit='Hz',
                                set_cmd='SOUR' + out + ':FREQ:FIX ' + '{:.12f}',
                                get_cmd='SOUR' + out + ':FREQ:FIX?',
                                get_parser=float
                                )

            self.add_parameter( name='OUT' + out + '_phase',
                                label='Phase of the generator',
                                vals=vals.Numbers(-360, 360),
                                unit='deg',
                                set_cmd='SOUR' + out + ':PHAS  ' + '{:.12f}',
                                get_cmd='SOUR' + out + ':PHAS?',
                                get_parser=float
                                )

            self.add_parameter( name='OUT' + out + '_amplitude',
                                label='Amplitude of the generator',
                                vals=vals.Numbers(-max_voltage, max_voltage),
                                unit='V',
                                set_cmd='SOUR' + out + ':VOLT ' + '{:.12f}',
                                get_cmd='SOUR' + out + ':VOLT?',
                                get_parser=float
                                )     

            self.add_parameter( name='OUT' + out + '_offset',
                                label='Amplitude offset of the generator',
                                vals=vals.Numbers(-max_voltage, max_voltage),
                                unit='V',
                                set_cmd='SOUR' + out + ':VOLT:OFFS ' + '{:.12f}',
                                get_cmd='SOUR' + out + ':VOLT:OFFS?',
                                get_parser=float
                                )  

            self.add_parameter( name='OUT' + out + '_duty_cycle',
                                label='Duty cycle of the generator',
                                vals=vals.Numbers(0, 1),
                                unit='',
                                set_cmd='SOUR' + out + ':DCYC ' + '{:.12f}',
                                get_cmd='SOUR' + out + ':DCYC?',
                                get_parser=float
                                )  

            self.add_parameter( name='OUT' + out + '_awg',
                                label='Arbitrary waveform for the output',
                                vals=vals.Arrays(min_value=-max_voltage, max_value=max_voltage ,shape=(self.BUFFER_SIZE,)),
                                unit='',
                                set_cmd='SOUR' + out + ':TRAC:DATA:DATA ' + '{:.12f}',
                                get_cmd='SOUR' + out + ':TRAC:DATA:DATA?',
                                get_parser=str
                                ) 

            self.add_parameter( name='OUT' + out + '_type',
                                label='Set output to pulsed or continuous',
                                vals=vals.Enum('BURST', 'CONTINUOUS'),
                                set_cmd='SOUR' + out + ':BURS:STAT ' + '{}',
                                get_cmd='SOUR' + out + ':BURS:STAT?'
                                )

            self.add_parameter( name='OUT' + out + '_pulse_cycle',
                                label='Set the number of periods in a pulse',
                                vals=vals.Numbers(1, 50000),
                                unit='',
                                set_cmd='SOUR' + out + ':DCYC ' + '{:.12f}',
                                get_cmd='SOUR' + out + ':DCYC?',
                                get_parser=float
                                )         

            self.add_parameter( name='OUT' + out + '_pulse_repetition',
                                label='Set the number of repeated pulses (65536 = inf)',
                                vals=vals.Numbers(1, 65536),
                                unit='',
                                set_cmd='SOUR' + out + ':BURS:NOR ' + '{:.12f}',
                                get_cmd='SOUR' + out + ':BURS:NOR?',
                                get_parser=float
                                ) 

            self.add_parameter( name='OUT' + out + '_pulse_period',
                                label='Set the duration of a single pulse in microseconds',
                                vals=vals.Numbers(1, 500e6),
                                unit='',
                                set_cmd='SOUR' + out + ':BURS:INT:PER ' + '{:.12f}',
                                get_cmd='SOUR' + out + ':BURS:INT:PER?',
                                get_parser=float
                                ) 

            self.add_parameter( name='OUT' + out + '_trigger_source',
                                label='Set the trigger source for the output',
                                vals=vals.Enum('EXT_PE', 'EXT_NE', 'INT', 'GATED'),
                                set_cmd='SOUR' + out + ':TRIG:SOUR ' + '{}',
                                get_cmd='SOUR' + out + ':TRIG:SOUR?'
                                )
        
        
        ## acquisition
        self.add_parameter( name='ADC_decimation',
                            label='Set the decimation factor, each sample is the average of skipped samples if decimation > 1',
                            vals=vals.Enum(1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536),
                            unit='',
                            initial_value=1,
                            set_cmd='ACQ:DEC ' + '{}',
                            get_cmd='ACQ:DEC?',
                            get_parser=int
                            )
        
        self.add_parameter( name='ADC_averaging',
                            label='Enable/disable averaging',
                            vals=vals.Enum('ON', 'OFF'),
                            set_cmd='ACQ:AVG ' + '{}',
                            get_cmd='ACQ:AVG?'
                            )
        
        # trigger
        self.add_parameter( name='ADC_trigger',
                            label='Disable triggering, trigger immediately or set trigger source and edge',
                            vals=vals.Enum('DISABLED', 'NOW', 'CH1_PE', 'CH1_NE', 'CH2_PE', 'CH2_NE', 'EXT_PE', 'EXT_NE', 'AWG_PE', 'AWG_NE'),
                            set_cmd='ACQ:TRIG ' + '{}',
                            get_cmd='ACQ:TRIG:STAT?'
                            )
        
        self.add_parameter( name='ADC_trigger_delay',
                            label='Trigger delay (in units of samples)',
                            vals=vals.Numbers(0, np.inf),
                            set_cmd='ACQ:TRIG:DLY ' + '{}',
                            get_cmd='ACQ:TRIG:DLY?'
                            )
        
        self.add_parameter( name='ADC_trigger_delay_ns',
                            label='Trigger delay (in nanoseconds)',
                            vals=vals.Numbers(0, np.inf),
                            unit='ns',
                            set_cmd='ACQ:TRIG:DLY:NS ' + '{:.12f}',
                            get_cmd='ACQ:TRIG:DLY:NS?',
                            get_parser=float 
                            )        
        
        self.add_parameter( name='ADC_trigger_level',
                            label='The trigger level in volts',
                            vals=vals.Numbers(-np.inf, np.inf),
                            unit='V',
                            set_cmd='ACQ:TRIG:LEV ' + '{:.12f}',
                            get_cmd='ACQ:TRIG:LEV?',
                            get_parser=float 
                            )       

        
        # data pointers
        self.add_parameter( name='ADC_trigger_pointer',
                            label='Returns the position where the trigger event happened',
                            set_cmd=None,
                            get_cmd='ACQ:TPOS?',
                            get_parser=int
                            )
        
        self.add_parameter( name='ADC_write_pointer',
                            label='Returns the current position of the write pointer',
                            set_cmd=None,
                            get_cmd='ACQ:WPOS?',
                            get_parser=int
                            )
    
    
    
        ## inputs parameters
        # data
        self.add_parameter( name='ADC_data_units',
                            label='Select units in which the acquired data will be returned',
                            vals=vals.Enum('RAW', 'VOLTS'),
                            set_cmd='ACQ:DATA:UNITS ' + '{}',
                            get_cmd='ACQ:DATA:UNITS?'
                            )
        
        self.add_parameter( name='ADC_data_format',
                            label='Select formats in which the acquired data will be returned',
                            vals=vals.Enum('BIN', 'ASCII'),
                            set_cmd='ACQ:DATA:FORMAT ' + '{}',
                            get_cmd=None
                            )
        
        self.add_parameter( name='ADC_buffer_size',
                            label='Returns the buffer size',
                            get_cmd='ACQ:BUF:SIZE?',
                            get_parser=int
                            )
        
        inputs = ['1', '2']
        for inp in inputs:
            self.add_parameter( name='IN' + inp + '_gain',
                                label='Set the gain to HIGH or LOW, which refers to jumper settings on the fast analog inputs',
                                vals=vals.Enum('HV', 'LV'),
                                set_cmd='ACQ:SOUR' + inp + ':GAIN ' + '{}',
                                get_cmd='ACQ:SOUR' + inp + ':GAIN?'
                                )
            
            self.add_parameter( name='IN' + inp + '_read_buffer',
                                label='Read the full buffer',
                                get_cmd='ACQ:SOUR' + inp + ':DATA?'
                                )

        self.add_parameter( name='ADC_output',
                            label='Returns the output',
                            get_cmd=self.get_output
                            )

        self.add_parameter( 'sampling_frequency',
                            initial_value=self.FS,
                            set_cmd=None,
                            get_cmd=None
                            )


        ## Parameters for a run data acquisition
        # length of the acquisition
        self.add_parameter( 'acquisition_length',
                            initial_value=1,
                            unit='s',
                            label='Length of the acquisition',
                            # the minimum value is 1e-6s which gives 125 points
                            vals=vals.Numbers(1e-6, np.inf),
                            set_cmd=None,
                            get_cmd=None
                            )

        # starting waveform (useless, but qcodes wants it)
        self.add_parameter( 'waveform_start',
                            unit='',
                            initial_value=0,
                            vals=vals.Numbers(0, np.inf),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=int
                            )

        # number of points in a single waveform
        self.add_parameter( 'waveform_points',
                            unit='',
                            initial_value=int(self.BUFFER_SIZE),
                            vals=vals.Numbers(1, np.inf),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=int
                            )

        # duration of a single waveform
        self.add_parameter( 'waveform_length',
                            unit='s',
                            initial_value=float(self.BUFFER_SIZE * self.ADC_decimation() / self.FS),
                            vals=vals.Numbers(0, np.inf),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=float
                            )

        # number of waveforms in the run
        self.add_parameter( 'number_of_waveforms',
                            unit='',
                            initial_value=1,
                            vals=vals.Numbers(0, np.inf),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=int
                            )

        ## time axis
        self.add_parameter( 'time_start',
                            unit='',
                            initial_value=0.0,
                            vals=vals.Numbers(0.0, np.inf),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=int
                            )

        self.add_parameter( 'time_axis',
                            unit='s',
                            label='Time axis',
                            parameter_class=GeneratedSetPoints,
                            start_param = self.time_start,
                            stop_param=self.waveform_length,
                            num_points_param=self.waveform_points,
                            snapshot_value=False,
                            vals=vals.Arrays(shape=(self.waveform_points.get_latest,))
                            )

        ## multiple waveforms axis
        self.add_parameter( 'waveform_axis',
                            unit='',
                            label='Waveform number',
                            parameter_class=GeneratedSetPoints,
                            start_param = self.waveform_start,
                            stop_param=self.number_of_waveforms,
                            num_points_param=self.number_of_waveforms,
                            snapshot_value=False,
                            vals=vals.Arrays(shape=(self.number_of_waveforms.get_latest,))
                            )

        self.add_parameter( 'duty_cycle',
                            initial_value=0.0046, # this value is obtained with a decimation of 4
                            set_cmd=None,
                            get_cmd=None
                            )


        ## measured waveform 1
        self.add_parameter( 'IN1',
                            unit='V',
                            setpoints=(self.waveform_axis, self.time_axis,),
                            label='Input 1',
                            parameter_class=IN1_data,
                            vals=vals.Arrays(shape=(self.number_of_waveforms.get_latest,self.waveform_points.get_latest,))
                            )

        ## measured waveform 2
        self.add_parameter( 'IN2',
                            unit='V',
                            setpoints=(self.waveform_axis,self.time_axis,),
                            label='Input 2',
                            parameter_class=IN2_data,
                            vals=vals.Arrays(shape=(self.number_of_waveforms.get_latest,self.waveform_points.get_latest,))
                            )
        

        ## Parameters for a the VNA
        # all the settings of the vna mode are here
        self.add_parameter( 'vna_amplitude',
                            initial_value=0.1,
                            unit='V',
                            label='Amplitude of the probe tone',
                            vals=vals.Numbers(0, 1),
                            set_cmd=None,
                            get_cmd=None,
                            get_parser=float
                            )

        self.add_parameter( 'vna_start',
                            unit='Hz',
                            initial_value=1e5,
                            label='Start frequency of the VNA trace',
                            vals=vals.Numbers(min_frequency, max_frequency),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=float
                            )

        self.add_parameter( 'vna_stop',
                            unit='Hz',
                            initial_value=1e5,
                            label='Stop frequency of the VNA trace',
                            vals=vals.Numbers(min_frequency, max_frequency),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=float
                            )

        self.add_parameter( 'vna_points',
                            unit='',
                            initial_value=101,
                            label='Number of points in a VNA trace',
                            vals=vals.Numbers(1, np.inf),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=int
                            )

        self.add_parameter( 'vna_rbw',
                            unit='Hz',
                            initial_value=1e5,
                            label='Resolution bandwidth of the VNA trace',
                            vals=vals.Numbers(self.FS / self.BUFFER_SIZE / self.ADC_decimation(), self.FS),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=float
                            )

        self.add_parameter( 'vna_averages',
                            unit='',
                            initial_value=1,
                            label='Resolution bandwidth of the VNA trace',
                            vals=vals.Numbers(1, np.inf),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=int
                            )

        ## measured traces
        self.add_parameter( 'frequency_axis',
                            unit='Hz',
                            label='Frequency',
                            parameter_class=GeneratedSetPoints,
                            start_param = self.vna_start,
                            stop_param=self.vna_stop,
                            num_points_param=self.vna_points,
                            snapshot_value=False,
                            vals=vals.Arrays(shape=(self.vna_points.get_latest,))
                            )

        # trace of channel 1
        self.add_parameter( 'VNA1',
                            parameter_class=VNA1_trace,
                            names = ('VNA1_mag', 'VNA1_phase'),
                            units=('dB', 'rad'),
                            setpoint_units=(("Hz",), ("Hz",)),
                            labels=('Channel 1 magnitude', 'Channel 1 phase'),
                            setpoints=((self.frequency_axis(),), (self.frequency_axis(),)),
                            vals=vals.Arrays(shape=((self.vna_points.get_latest,), (self.vna_points.get_latest,))),
                            shapes=((self.vna_points(),), (self.vna_points(),),),
                            )


        # trace of channel 2
        self.add_parameter( 'VNA2',
                            parameter_class=VNA2_trace,
                            names = ('VNA2_mag', 'VNA2_phase'),
                            units=('dB', 'rad'),
                            setpoint_units=(("Hz",), ("Hz",)),
                            labels=('Channel 2 magnitude', 'Channel 2 phase'),
                            setpoints=((self.frequency_axis(),), (self.frequency_axis(),)),
                            vals=vals.Arrays(shape=((self.vna_points.get_latest,), (self.vna_points.get_latest,))),
                            shapes=((self.vna_points(),), (self.vna_points(),),),
                            )

        # transmission of channel 1
        self.add_parameter( 'TX1',
                            parameter_class=TX1_trace,
                            unit='a.u.',
                            label='Channel 1 transmission',
                            vals=vals.Arrays(shape=(self.vna_points.get_latest,)),
                            setpoints=(self.frequency_axis,),
                            )

        # transmission of channel 2
        self.add_parameter( 'TX2',
                            parameter_class=TX2_trace,
                            unit='a.u.',
                            label='Channel 2 transmission',
                            vals=vals.Arrays(shape=(self.vna_points.get_latest,)),
                            setpoints=(self.frequency_axis,),
                            )

        ## sensors data     
        self.add_parameter( 'temperature',
                            parameter_class=GetTemperature,
                            unit='K',
                            label = 'Temperature',
                            vals=vals.Numbers(0, np.inf),
                            )

        self.add_parameter( 'pressure',
                            parameter_class=GetPressure,
                            unit='bar',
                            label = 'Pressure',
                            vals=vals.Numbers(0, np.inf),
                            )
        
        self.add_parameter( 'magnetic_field',
                            parameter_class=GetMagneticField,
                            unit='V',
                            label = 'Magnetic field',
                            vals=vals.Numbers(-np.inf, np.inf),
                            )

        self.add_parameter( 'photoresistance',
                            parameter_class=GetPhotoresistance,
                            unit='V',
                            label = 'Light',
                            vals=vals.Numbers(0, np.inf),
                            )

        self.add_parameter( 'acceleration',
                            parameter_class=GetAcceleration,
                            unit='m/s^2',
                            label = 'Acceleration',
                            vals=vals.Numbers(0, np.inf),
                            )

        # good idea to call connect_message at the end of your constructor.
        # this calls the 'IDN' parameter that the base Instrument class creates 
        # for every instrument  which serves two purposes:
        # 1) verifies that you are connected to the instrument
        # 2) gets the ID info so it will be included with metadata snapshots later.
        self.connect_message() 
        
        
    ## functions
    # signal generators
    def align_channels_phase(self):
        # Align the phase of the outputs
        self.write('PHAS:ALIGN')

    def OUT_trigger(self):
        # Triggers immediately both the outputs
        self.write('SOUR:TRIG:INT')

    def OUT_reset(self):
        # Reset both the outputs
        self.write('GEN:RST')

    def OUT1_trigger(self):
        # Triggers immediately channel 1
        self.write('SOUR1:TRIG:INT')

    def OUT2_trigger(self):
        # Triggers immediately channel 2
        self.write('SOUR2:TRIG:INT')

    def reset(self):
        # Reset both the outputs
        self.write('GEN:RST')


    ## acquisition
    # analog-to-digital converter
    def ADC_start(self):
        # Start the acquisition
        self.write('ACQ:START')
        
    def ADC_stop(self):
        # Stop the acquisition
        self.write('ACQ:STOP')
        
    def ADC_reset(self):
        # Stops the acquisition and sets all parameters to default values
        self.write('ACQ:RST')

    def ADC_trigger_full(self):
        # Returns 1 if the buffer is full of data. Otherwise returns 0
        return self.ask('ACQ:TRIG:FILL?')

    def get_output(self):
        # Returns the output, which is useful to check for 'ERR!' messages
        return self.ask('OUTPUT:DATA?')

    def ADC_read_N_from_A(self, channel, size: int, pointer: int):
        # read N samples from the buffer of the Redpitaya starting from the pointer

        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA:STA:N? ' + str(pointer) + ',' + str(size)
        raw_data = self.ask(scpi_string)

        return raw_data

    def ADC_read_N_from_A_bin(self, channel, size: int, pointer: int):
        # read N binary samples from the buffer of the Redpitaya starting from the pointer

        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA:STA:N? ' + str(pointer) + ',' + str(size)
        raw_data = self.visa_handle.query_binary_values( scpi_string, 
                                                         datatype='f', 
                                                         is_big_endian=True,
                                                         expect_termination=False, 
                                                         data_points=size,
                                                         container=np.ndarray
                                                         )
        return raw_data

    def ADC_read_A_to_B(self, channel, pointer_A: int, pointer_B: int):
        # read B-A samples from the buffer of the Redpitaya starting from pointer A

        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA:STA:END? ' + str(pointer_A) + ',' + str(pointer_B)
        raw_data = self.ask(scpi_string)

        return raw_data

    def ADC_read_N_after_trigger(self, channel, size: int):
        # read N samples from the buffer of the Redpitaya starting from the trigger
        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA:OLD:N? ' + str(size)
        raw_data = self.ask(scpi_string)

        data_string = np.array( raw_data[1:-1].split(',') )
        data_line = data_string.astype(float)

        return data_line

    def ADC_read_N_after_trigger_bin(self, channel, size: int):
        # read N binary samples from the buffer of the Redpitaya starting from the trigger

        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA:OLD:N? ' + str(size)
        raw_data = self.visa_handle.query_binary_values( scpi_string, 
                                                         datatype='f', 
                                                         is_big_endian=True,
                                                         expect_termination=False, 
                                                         data_points=size,
                                                         container=np.ndarray
                                                         )

        return raw_data

    def ADC_read_N_before_trigger(self, channel, size: int):
        # read N samples from the buffer of the Redpitaya before the trigger

        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA:LAT:N? ' + str(size)
        raw_data = self.ask(scpi_string)

        return raw_data

    def ADC_read_buffer_bin(self, channel):
        # read the whole buffer of the selected channel

        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA?'
        raw_data = self.visa_handle.query_binary_values( scpi_string, 
                                                         datatype='f', 
                                                         is_big_endian=True,
                                                         expect_termination=False, 
                                                         data_points=self.BUFFER_SIZE,
                                                         container=np.ndarray
                                                         )

        return raw_data


    # composite acquisition functions
    def get_data(self, channel, waveforms_number, data_type='BIN', disable_pbar=True):
    # This is the core part of the measurement, with a defined measurement length it
    # keeps reading and emptying the Redpitaya buffer, concatenating the waveforms.

        #DEC = self.ADC_decimation()
        BLOCK = self.BUFFER_SIZE
        self.ADC_data_format(data_type)

        t = 0.0
        index = 0
        duration = self.estimate_acquisition_length()
        self.ADC_trigger_level(0.0)
        
        """
        Ci sono ancora diversi problemi.
        Ad esempio il duty cycle fa schifo.
        """

        if data_type == 'ASCII':    
            # One can choose the data format to be used
            # ascii is immediately readable but slow
            data = ''
            time.sleep(0.2)

            t0 = time.time()
            pbar = tqdm(total=waveforms_number, disable=disable_pbar)

            if channel == 1:
                self.ADC_trigger('CH1_PE')
            if channel == 2:
                self.ADC_trigger('CH2_PE')

            while index < waveforms_number:
                index += 1

                # trigger the ADC and refill the buffer
                self.ADC_start()

                # read all the data
                while 1:
                    if self.ADC_trigger() == 'TD':
                        break

                # this reduced the maximum duty cycle to 75%            
                time.sleep(1.5 * BLOCK / self.FS)

                data += self.ADC_read_N_after_trigger(channel, BLOCK)[1:-1] + ','

                # increment the waveform number
                #self.number_of_waveforms(index)
                #trash = self.get_output()

                pbar.update(1)
                self.ADC_stop()

            pbar.close()
            time.sleep(0.2)

            # update the time which passed from the beginning of the run and the wavefor index
            t = time.time() - t0
            self.acquisition_length(t)

            data_string = np.array( data[1:-1].split(',') )
            data_line = data_string.astype(float)

            return data_line    

        elif data_type == 'BIN':
            # One can choose the data format to be used
            # binary is fast, using it improves the duty cycle
            data = np.array([])
            time.sleep(0.2)

            t0 = time.time()
            pbar = tqdm(total=waveforms_number, disable=disable_pbar)

            while index < waveforms_number:
                index += 1

                # starting the ADC
                self.ADC_start()

                # read all the data
                if channel == 1:
                    self.ADC_trigger('CH1_PE')
                if channel == 2:
                    self.ADC_trigger('CH2_PE')

                while 1:
                    if self.ADC_trigger() == 'TD':
                        self.ADC_trigger('DISABLED')
                        break

                # this reduced the maximum duty cycle to 50%
                #time.sleep(1 * BLOCK / self.FS * self.ADC_decimation())

                #new_waveform = self.ADC_read_N_after_trigger_bin(channel, BLOCK)
                new_waveform = self.ADC_read_buffer_bin(channel)
                data = np.append(data, new_waveform)

                #self.number_of_waveforms(index)
                #self.ADC_stop()

                pbar.update(1)

            pbar.close()
            time.sleep(0.2)
            self.ADC_stop()

            # update the time which passed from the beginning of the run and the wavefor index
            t = time.time() - t0
            self.acquisition_length(t)

            # go back to ascii at the end
            self.ADC_data_format('ASCII')

            return data

        else:
            raise NameError('Format must be BIN or ASCII')

    def spectrscopy(self, channel, frequency):
    # This function is used in the VNA classes and, as of the name, it used for spectroscopy.
    # It considers a tone at a single frequency, outputs the ratio between input and output amplitude.
        rbw = self.vna_rbw()
        DEC = self.ADC_decimation()
        min_rbw = self.FS / self.BUFFER_SIZE / DEC
        amp = self.vna_amplitude()

        BLOCK = int(self.FS / rbw / self.ADC_decimation())
        if BLOCK > self.BUFFER_SIZE:
            raise MemoryError('Resolution bandwidth too low to handle. \nWith this decimation the minimum is ' + min_rbw + 'Hz.')

        if channel == 1:
            self.OUT1_amplitude(amp)
            self.OUT1_frequency(frequency)
            self.ADC_trigger('CH1_NE')

        elif channel == 2:
            self.OUT2_amplitude(amp)
            self.OUT2_frequency(frequency)
            self.ADC_trigger('CH2_NE')

        else:
            raise NameError('Invalid channel.')

        # phase and magnitude are extracted from the digital lockin
        self.ADC_start()

        # read data
        while 1:
            if self.ADC_trigger() == 'TD':
                break
        
        # this reduced the maximum duty cycle to 75%  
        time.sleep(1.5 * BLOCK / self.FS)
        waveform = self.ADC_read_N_after_trigger(channel, BLOCK)
        #print(waveform)

        self.ADC_stop()

        # save the output in a complex number
        z = self.lockin(waveform - np.mean(waveform), frequency, DEC)

        return 2*np.abs(z) / amp, np.angle(z)


    ## UART protocol
    def UART_init(self):
        # initialises the API for working with UART
        self.write('UART:INIT')

    def UART_setup(self):
        # apply the setup to the UART
        self.write('UART:SETUP')

    def UART_write(self, message:str):
        # writes data to UART, <self.UART_data_length()> is the length of data sent to UART
        self.write('UART:WRITE' + str(self.UART_data_length()) + ' ' + message)

    def UART_read(self):
        # reads data from UART, <self.UART_data_length()> is the length of data received from UART
        message = self.ask('UART:READ' + str(self.UART_data_length()))
        return message

    def UART_release(self):
        # releases all used resources
        self.write('UART:RELEASE')

    """
    ## SPI protocol
    def SPI_init(self):
        # initializes the API for working with SPI
        self.write('SPI:INIT')

    def SPI_init_dev(self, path: str):
        # initializes the API for working with SPI. <path> - Path to the SPI device
        # on some boards, it may be different from the standard: /dev/spidev1.0 
        self.write('SPI:INIT:DEV "' + path + '"')
        
    def SPI_release(self):
        # releases all used resources
        self.write('SPI:RELEASE')

    def SPI_default_settings(self):
        # sets the settings for SPI to default values
        self.write('SPI:SET:DEF')

    def SPI_set_settings(self):
        # sets the specified settings for SPI
        # executed after specifying the parameters of communication
        self.write('SPI:SET:SET')

    def SPI_get_settings(self):
        # gets the specified settings for SPI
        self.write('SPI:SET:GET')

    def SPI_create_message_queue(self, N):
        # creates a message queue for SPI (reserves the space for data buffers)
        # once created, they need to be initialized
        # N - the number of messages in the queue
        # the message queue can operate within a single CS state switch
        self.write('SPI:MSG:CREATE ' + str(N))

    def SPI_delete_messages(self):
        # deletes all messages and data buffers allocated for them
        self.write('SPI:MSG:DEL')

    def SPI_get_message_size(self):
        # returns the length of the message queue
        return self.ask('SPI:MSG:SIZE?')

    def SPI_send_message(self, N:int, M:int, message:str):
        # sets data for the write buffer for the specified message
        # N - index of message 0 <= N < msg queue size
        # M - TX buffer length
        # Sends <m> ‘bytes’ from message N, no data is received
        self.write('SPI:MSG' + str(N) + ':TX' + str(M) + message)

    def SPI_send_cs_message(self, N:int, M:int, message:str):
        # sets data for the write buffer for the specified message
        # N - index of message 0 <= N < msg queue size
        # M - TX buffer length
        # Sends <m> ‘bytes’ from message N, no data is received
        self.write('SPI:MSG' + str(N) + ':TX' + str(M) + ':CS' + message)

    def SPI_send_receive_message(self, N:int, M:int, message_tx:str):
        # sets data for the write buffer for the specified message
        # N - index of message 0 <= N < msg queue size
        # M - TX buffer length
        # sends <m> ‘bytes’ from message N, and receives the same amount of data from the dataline
        message_rx = self.ask('SPI:MSG' + str(N) + ':TX' + str(M) + ':RX' + message_tx)
        return message_rx

    def SPI_send_receive_cs_message(self, N:int, M:int, message_tx:str):
        # sets data for the write buffer for the specified message
        # N - index of message 0 <= N < msg queue size
        # M - TX buffer length
        # sends <m> ‘bytes’ from message N, and receives the same amount of data from the dataline
        message_rx = self.ask('SPI:MSG' + str(N) + ':TX' + str(M) + ':RX:CS' + message_tx)
        return message_rx

    def SPI_receive_message(self, N:int, M:int):
        # initializes a buffer for reading the specified message
        # N - index of message 0 <= N < msg queue size
        # M - RX buffer length
        # receives M ‘bytes’ into message N, no data is transmitted
        message_rx = self.ask('SPI:MSG' + str(N) + ':RX' + str(M))
        return message_rx

    def SPI_receive_cs_message(self, N:int, M:int):
        # initializes a buffer for reading the specified message
        # N - index of message 0 <= N < msg queue size
        # M - RX buffer length
        # receives M ‘bytes’ into message N, no data is transmitted
        message_rx = self.ask('SPI:MSG' + str(N) + ':RX' + str(M) + ':CS')
        return message_rx

    def SPI_get_message_read_buffer(self, N):
        # returns a read buffer for the specified message
        buffer = self.ask('SPI:MSG' + str(N) + ':RX?')
        return buffer

    def SPI_get_message_write_buffer(self, N):
        # returns a read buffer for the specified message
        buffer = self.ask('SPI:MSG' + str(N) + ':TX?')
        return buffer

    def SPI_message_cs_mode_settings(self, N):
        # returns the setting for CS mode for the specified message
        settings = self.ask('SPI:MSG' + str(N) + ':CS?')
        return settings

    def SPI_pass(self):
        # sends the prepared messages to the SPI device
        self.write('SPI:PASS')

    ## I2C protocol
    def I2C_set_dev(self, addr:int, path:str):
        # initializes settings for I2C. <path> - Path to the I2C device <addr>
        print('I2C:DEV' + str(addr) + ' "' + path + '"')
        self.write('I2C:DEV' + str(addr) + ' "' + path + '"')

    """

    ## helpers
    # useful functions
    def lockin(self, signal, frequency, decimation):
        points = len(signal)
        numeric_frequency = frequency / self.FS * decimation

        numeric_time = np.linspace(0, points-1, points)
        sine = np.sin(2*np.pi * numeric_frequency * numeric_time)
        cosine = np.cos(2*np.pi * numeric_frequency * numeric_time)

        # definition of the filter
        sos = butter(5, numeric_frequency/4, fs=1, output='sos')
        # mixing and filtering
        x = sosfilt(sos, signal * sine)[-1]
        y = sosfilt(sos, signal * cosine)[-1]

        #plt.plot(np.abs(x + 1j*y), 'o')
        #plt.plot(np.angle(x + 1j*y), 'o')
        
        return x + 1j*y

    def format_output_status(self, status_in):
        # to return ON and OFF when asked for the status of outputs
        if status_in == '0': 
            status_out = 'OFF'
        elif status_in == '1': 
            status_out = 'ON'
        
        return status_out

    def estimate_waveform_number(self, duration=None):
        # estimate the number of waveforms in a run (needed to intialise the axis of Qcodes)
        if (duration == None):
            n = self.acquisition_length()*self.duty_cycle() / self.waveform_length() / self.ADC_decimation()
        else:
            n = duration*self.duty_cycle() / self.waveform_length() / self.ADC_decimation()

        #self.number_of_waveforms(int(n))
        return int(n)

    def estimate_acquisition_length(self):
        # estimate the number of waveforms in a run (needed to intialise the axis of Qcodes)
        t = self.number_of_waveforms() / self.duty_cycle() * self.waveform_length() * self.ADC_decimation()

        self.acquisition_length(t)
        return t

    def estimate_duty_cycle(self, waveforms_sample=30):
        # calculated instrument duty cycle
        self.number_of_waveforms(waveforms_sample)

        self.OUT1_status('ON')
        sample_data = self.IN1()
        self.OUT1_status('OFF')

        acquired_points = self.number_of_waveforms() * self.BUFFER_SIZE
        estimated_points = self.acquisition_length() * self.FS / self.ADC_decimation()

        duty_cycle = acquired_points / estimated_points

        self.duty_cycle(duty_cycle)
        return duty_cycle

