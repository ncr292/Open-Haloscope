# -*- coding: utf-8 -*-
# This is a Qcodes driver for Redpitaya board written for the Alca Haloscope project.
# Written by Nicolò Crescini taking inspiration from a version by Arpit Ranadive and Martina Esposito.

import time
import binascii

import numpy as np
from tqdm import tqdm

from qcodes import validators as vals
from qcodes.instrument import ( Instrument,
                                InstrumentChannel,
                                InstrumentModule,
                                ManualParameter,
                                MultiParameter,
                                VisaInstrument,
                                )

from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter


"""
DEVELOPMENT NOTES
- the decimation, as defined in IN1_out, is acquired at the import and does not update if changed.
  This needs to be fixed as now to use the decimation, one needs to set it and reimport the driver.


"""



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
        duration = self._instrument.acquisition_length()
        sampling_frequency = self._instrument.sampling_frequency()
        decimation = self._instrument.ADC_decimation()
        points = self._instrument.waveform_points()

        # acquire data from channel 1
        raw_data = self._instrument.get_data(1, duration, data_type='BIN')

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
        duration = self._instrument.acquisition_length()
        sampling_frequency = self._instrument.sampling_frequency()
        decimation = self._instrument.ADC_decimation()
        points = self._instrument.waveform_points()

        # acquire data from channel 2
        raw_data = self._instrument.get_data(2, duration, data_type='BIN')

        try:
            data = np.reshape(raw_data, (self._instrument.number_of_waveforms(), 
                                         self._instrument.BUFFER_SIZE))
        except:
            raise NameError(raw_data)
        
        return data



class Redpitaya(VisaInstrument):
    ## main driver
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r\n', **kwargs)
        
        # Connect to the Redpitaya board using the appropriate communication protocol
        self._address = address
        # Connect using TCP or USB
        # For wireless connection use TCPIP::address::port::SOCKET
        # For USB connection use ***to be tested***

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
        
        for pin in (input_pin + output_pin):
            # get the voltage for any of the analog pins
            self.add_parameter( name='get_'+pin,
                                label='Read input/output voltage on pin',
                                vals=vals.Numbers(-max_get_voltage, max_get_voltage),
                                unit='V',
                                set_cmd=None,
                                get_cmd='ANALOG:PIN? ' + pin,
                                get_parser=float
                                )        
            
        for pin in output_pin:
            # set the voltage for the output pins
            self.add_parameter( name='set_'+pin,
                                label='Set output voltage on pin',
                                vals=vals.Numbers(-max_set_voltage, max_set_voltage),
                                unit='V',
                                set_cmd='ANALOG:PIN ' + pin + ',' + '{:.12f}',
                                get_cmd=None,
                                )  
            
        # digital outputs
        # In progress...

            
        ## signal generators
        min_frequency = 1
        max_frequency = 50e6
        max_voltage = 1
        awg_array_size = 16384

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
                                vals=vals.Arrays(min_value=-max_voltage, max_value=max_voltage ,shape=(awg_array_size,)),
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
                            get_cmd='ACQ:BUF:SIZE?'
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
                            initial_value=0,
                            vals=vals.Numbers(0, np.inf),
                            get_cmd=None,
                            set_cmd=None,
                            get_parser=int
                            )

        ## time axis
        self.add_parameter( 'time_axis',
                            unit='s',
                            label='Time axis',
                            parameter_class=GeneratedSetPoints,
                            start_param = 0,
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
                            start_param = 0,
                            stop_param=self.number_of_waveforms,
                            num_points_param=self.number_of_waveforms,
                            snapshot_value=False,
                            vals=vals.Arrays(shape=(self.number_of_waveforms.get_latest,))
                            )

        self.add_parameter( 'duty_cycle',
                            initial_value=self.estimated_duty_cycle(),
                            set_cmd=None,
                            get_cmd=self.estimated_duty_cycle
                            )


        ## measured waveform 1
        self.add_parameter( 'IN1',
                            unit='V',
                            setpoints=(self.waveform_axis,self.time_axis),
                            label='Input 1',
                            parameter_class=IN1_data,
                            vals=vals.Arrays(shape=(self.number_of_waveforms.get_latest,self.waveform_points.get_latest,))
                            )

        ## measured waveform 2
        self.add_parameter( 'IN2',
                            unit='V',
                            setpoints=(self.waveform_axis,self.time_axis),
                            label='Input 2',
                            parameter_class=IN2_data,
                            vals=vals.Arrays(shape=(self.number_of_waveforms.get_latest,self.waveform_points.get_latest,))
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

    def get_output(self):
        # Returns the output, which is useful to check for 'ERR!' messages
        return self.ask('OUTPUT:DATA?')


    def ADC_read_N_from_A(self, channel, size: int, pointer: int):
        # read N samples from the buffer of the Redpitaya starting from the pointer

        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA:STA:N? ' + str(pointer) + ',' + str(size)
        raw_data = self.ask(scpi_string)

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

        return raw_data

    def ADC_read_N_after_trigger_bin(self, channel, size: int):
        # read N binary samples from the buffer of the Redpitaya starting from the trigger

        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA:OLD:N? ' + str(size)
        raw_data = self.visa_handle.query_binary_values( scpi_string, 
                                                         datatype='f', 
                                                         is_big_endian=True,
                                                         expect_termination=False, 
                                                         data_points=self.BUFFER_SIZE,
                                                         container=np.ndarray
                                                         )

        return raw_data

    def ADC_read_N_before_trigger(self, channel, size: int):
        # read N samples from the buffer of the Redpitaya before the trigger

        scpi_string = 'ACQ:SOUR' + str(channel) + ':DATA:LAT:N? ' + str(size)
        raw_data = self.ask(scpi_string)

        return raw_data


    # composite acquisition functions

    def get_data(self, channel, duration, data_type='ASCII'):
    # This is the core part of the measurement, with a defined measurement length it
    # keeps reading and emptying the Redpitaya buffer, concatenating the waveforms.

        DEC = self.ADC_decimation()
        BLOCK = self.BUFFER_SIZE
        self.ADC_data_format(data_type)

        t = 0.0
        index = 0

        """
        Ci sono ancora diversi problemi.
        Ad esempio il duty cycle fa schifo.
        """

        if data_type == 'ASCII':    
            # One can choose the data format to be used
            # ascii is immediately readable but slow
            data = ''

            self.ADC_start()
            time.sleep(0.2)

            t0 = time.time()
            self.ADC_write_pointer(0)
            pbar = tqdm(total=100)

            while t < duration:
                index += 1

                # trigger the ADC and refill the buffer
                self.ADC_write_pointer(0)
                self.ADC_trigger('NOW')

                # read all the data
                data += self.ADC_read_N_after_trigger(channel, BLOCK)[1:-1] + ','

                self.number_of_waveforms(index)
                self.ADC_write_pointer(0)

                # update the time which passed from the beginning of the run
                t = time.time() - t0
                pbar.update(int(100 * t / duration - pbar.n))

            pbar.close()
            time.sleep(0.2)
            self.ADC_stop()

            data_string = np.array( data[1:-1].split(',') )
            data_line = data_string.astype(float)

            return data_line    


        elif data_type == 'BIN':
            # One can choose the data format to be used
            # binary is fast, using it improves the duty cycle
            data = np.array([])

            self.ADC_start()
            time.sleep(0.2)

            t0 = time.time()
            self.ADC_write_pointer(0)
            pbar = tqdm(total=100)

            while t < duration:
                index += 1

                # trigger the ADC and refill the buffer
                self.ADC_write_pointer(0)
                self.ADC_trigger('NOW')

                # read all the data
                new_waveform = self.ADC_read_N_after_trigger_bin(channel, BLOCK)
                data = np.append(data, new_waveform)
                #print(data)
                
                self.number_of_waveforms(index)
                self.ADC_write_pointer(0)

                # update the time which passed from the beginning of the run
                t = time.time() - t0
                pbar.update(int(100 * t / duration - pbar.n))

            pbar.close()
            time.sleep(0.2)
            self.ADC_stop()

            # go back to ascii at the end
            self.ADC_data_format('ASCII')

            return data

        else:
            raise NameError('Format must be BIN or ASCII')

        

    ## helpers
        
    def format_output_status(self, status_in):
        # to return ON and OFF when asked for the status of outputs
        if status_in == '0': 
            status_out = 'OFF'
        elif status_in == '1': 
            status_out = 'ON'
        
        return status_out

    def estimated_duty_cycle(self):
        duty_cycle = self.number_of_waveforms() * (self.BUFFER_SIZE / self.FS * self.ADC_decimation())

        return duty_cycle