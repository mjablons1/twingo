# after moving under conda environment import importlib does not make util available for some reason
from importlib import util as importlib_util
import numpy as np
from threading import Lock
from PyQt5 import QtCore  # for signal generation
from time import sleep

from model import config
from model.pftltools import FunctionGenerator, wave_bytes_to_ndarray, \
    ndarray_to_wave_bytes, nearest_supported_sample_rates, clip_list

# This makes the code compatible with python>=3.4 onwards only
nidaqmx_is_present = importlib_util.find_spec("nidaqmx") is not None
pyaudio_is_present = importlib_util.find_spec("pyaudio") is not None

if nidaqmx_is_present:
    import nidaqmx as ni
    from nidaqmx import stream_readers  # has to be done this way. nidaqmx.stream_readers is otherwise not found
    from nidaqmx import stream_writers

if pyaudio_is_present:
    import pyaudio

    pa = pyaudio.PyAudio()


class Error(Exception):
    """Base class for other exceptions"""
    pass


class CouldNotInitStreamingDevFcnGen(Error):
    """Exception raised when user tries to run init_fcn_generator before setting up the streaming device properly"""
    pass


def io_streaming_device_discovery():
    """
    Finds all supported sound devices present in the system and matches their names to the iostreamingdevice models.
    This to makes it possible to instantiate the correct model with reference by name:
    my_daq = devices_found['Dev1']('Dev1')
    """

    devices_found = None

    if nidaqmx_is_present:
        try:
            system = ni.system.System.local()
            devices_found = dict.fromkeys(system.devices.device_names, NiDaqStreamingDevice)
        except OSError:
            print("NIDAQmx system drivers are probably not present in your system.\n "
                  "If you do not wish to use NI Harware anyway please uninstall the nidaqmx package from this environment to prevent this warning.")

    if pyaudio_is_present:
        # check if default I/O devices are present
        try:
            def_input_device_name = pa.get_default_input_device_info()["name"]
            def_output_device_name = pa.get_default_output_device_info()["name"]
        except IOError:
            print('No default audio I/O devices could be found in your system.')
            return

        dev_name = def_input_device_name + ' / ' + def_output_device_name
        pa_device = {dev_name: PyAudioSoundStreamingDevice}
        devices_found.update(pa_device)

    # merege the two dictionaries:
    # device_list = [ni_devices_found, pa_devices_found]
    # devices_found = {}
    # for dict_element in device_list:
    #    devices_found.update(dict_element)
    # this replaces above code but its compatible from python 3.5 onwards only.
    # devices_found = {**ni_devices_found, **pa_devices_found}

    #print('Following devices can be supported:')
    #print(devices_found)
    #print('\n')

    return devices_found


class StereoStreamingDeviceBase(QtCore.QObject):
    _nr_of_active_chans = 2
    input_frame_ready_signal = QtCore.pyqtSignal()
    monitor_ready_signal = QtCore.pyqtSignal()

    def __init__(self, device_name):
        QtCore.QObject.__init__(self)
        self.read_lock = Lock() # TODO perhaps some more complex inheritance is required to have these locks actually work
        self.mutex = QtCore.QMutex() # TODO perhaps some more complex inheritance is required to have these locks actually work

        self.device_name = device_name
        self.STANDARD_SAMPLE_RATES = [192000.0, 176400.0, 96000.0, 88200.0, 48000.0, 44100.0, 22050.0]
        self.CM_INPUT_FRAME_LEN = 2048
        self.CM_OUTPUT_FRAME_LEN = 2048
        self.CM_OUTPUT_FRAMES_PER_BUFFER = 4
        self.CM_INPUT_FRAMES_PER_BUFFER = 4
        self.UPSAMPLE_RATIO = 10
        self.QUEUE_INPUT_FRAMES_ENABLED = True
        self.FILTER_ORDER = 4

        self.limits = self.Limits() # this is just namespace class to separate available HW properties / settings

        self._monitor_frame_counter = None
        self.monitor_lock = Lock()

        # self.OUTPUT_HW_BUFFER_LEN =
        self.output_filter_enabled = False
        self.input_filter_enabled = False
        self._acq_mode = None

        self.output_filter_coefs = None
        self.input_filter_coefs = None

        # user adjustable settings that must be added after instantiation and will be used for next acq/gen
        self.ao_a_name = None
        self.ao_b_name = None
        self.ao_min_val = None
        self.ao_max_val = None
        self.ao_fs = None
        self.ao_terminal_config = None  # not used on NI6211
        self.sw_output_buffer_size = None
        self._output_frame_len = None

        self.ai_a_name = None
        self.ai_b_name = None
        self.ai_min_val = None
        self.ai_max_val = None
        self.ai_fs = None
        self.ai_terminal_config = None  # for NI devices just the string representation (.name) of the config enum has to be stored here
        self.sw_input_buffer_size = None
        # self.hw_input_buffer_size = None
        # self.input_frames_per_buffer = None
        self._input_frame_len = None

        self.input_frame = None
        self.input_time_base = None
        self.function_gen = None
        self.finite_frame_len_sec = config.default_finite_frame_len_sec
        # self.freq1 = None
        # self.freq0 = None
        self.function_type = None
        self._frames_per_monitor = None
        self._monitor_storage = None

        self.cm_measurement_is_running = False
        self.io_start = None
        self.io_stop = None

    class Limits:
        # Below attributes are placed into this class just to be able to create separate namespace for them.
        # To be populated immediately upon init of a specific model based on questioning the HW driver.
        # Note: Adjustable limits should be given as list where first element(s) will be used for default
        # configuration
        def __init__(self):
            self.ao_physical_chans = None
            self.ao_max_rate = None
            self.ao_min_rate = None
            self.ao_voltage_rngs = None

            self.ai_physical_chans = None
            self.ai_max_single_chan_rate = None
            self.ai_max_two_chan_rate = None
            self.ai_min_rate = None
            self.ai_voltage_rngs = None

            self.terminal_configs = None
            self.supported_input_rates = None
            self.supported_output_rates = None
            self.supported_monitor_frame_lengths = None

    def get_nchans(self):
        return self._nr_of_active_chans

    def get_supported_sampling_rates(self):
        pass

    def get_supported_monitor_frame_lengths(self):
        return [2 ** n for n in
                range(int(np.log2(self.CM_INPUT_FRAME_LEN)), int(np.log2(config.max_monitor_frame_len)) + 1)]

    def _set_limits(self):
        """ populate self.limits attributes here"""
        self.limits.supported_monitor_frame_lengths = self.get_supported_monitor_frame_lengths()
        pass

    def set_default_config(self):
        """
        This is the config to run in the init of any specific instance. It will set the I/O parameters to default but
        within device limits
        """
        self.ao_a_name = self.limits.ao_physical_chans[0]
        self.ao_b_name = self.limits.ao_physical_chans[1]
        self.ao_fs = self.limits.supported_output_rates[0]
        self.ao_terminal_config = self.limits.terminal_configs[0]
        self.ao_max_val = self.limits.ao_voltage_rngs[0]
        self.ao_min_val = self.limits.ao_voltage_rngs[1]

        self.ai_a_name = self.limits.ai_physical_chans[0]
        self.ai_b_name = self.limits.ai_physical_chans[1]
        self.ai_fs = self.limits.supported_input_rates[0]
        self.ai_terminal_config = self.limits.terminal_configs[0]
        self.ai_max_val = self.limits.ai_voltage_rngs[0]
        self.ai_min_val = self.limits.ai_voltage_rngs[1]

        self.function_gen = FunctionGenerator(sample_rate=self.ao_fs,
                                              amplitude=self.ai_max_val,
                                              nr_of_chan=self._nr_of_active_chans,
                                              freq1=self.ao_fs // 100)
        self.set_mode_to_continuous()
        self.set_monitor(config.default_monitor_len)
        # set filters to default:
        self.set_filters()

    def start_continuous_acq_n_gen(self):
        self.cm_measurement_is_running = True # NOTE this line is just reminder to the dev
        pass

    def stop_continuous_acq_n_gen(self):
        self.cm_measurement_is_running = False

    def start_finite_acq_n_gen(self):
        pass

    def get_ao_buffer_level_prc(self):
        pass

    def get_ai_buffer_level_prc(self):
        pass

    def set_monitor(self, size):
        self._monitor_frame_counter = 0
        self._frames_per_monitor = size // self._input_frame_len
        # self.input_queue = Queue(maxsize=self._frames_per_monitor)
        # print('Queue maxsize set to: {}'.format(self.input_queue.maxsize))
        self._monitor_storage = np.zeros((self._frames_per_monitor, self._nr_of_active_chans, self._input_frame_len))
        #print('Monitor storage preallocated shape is {}'.format(self._monitor_storage.shape))

    def _put_monitor_frame(self):
        if self._monitor_frame_counter <= self._frames_per_monitor - 1:
            self.monitor_lock.acquire()  # TODO it seems like its still not thread-safe, something's not working
            self.read_lock.acquire()
            self._monitor_storage[self._monitor_frame_counter][:][:] = self.input_frame
            self.read_lock.release()
            self._monitor_frame_counter += 1
            self.monitor_lock.release()
        else:
            self.monitor_ready_signal.emit()
            self._monitor_frame_counter = 0
        # NOTE: Notice that this if clause skips over the frame which causes emit. This is wanted behavior because it
        # decimates signal emits at _frames_per_monitor = 1 by half , reducing the refresh rate / recalculation

    def set_ao_fs(self, fs):
        self.ao_fs = fs
        self.function_gen.set_sample_rate(self.ao_fs)
        if self._acq_mode == 'finite':
            self.set_mode_to_finite(self.finite_frame_len_sec)
        elif self._acq_mode == 'continuous':
            self.set_mode_to_continuous()

    def set_ai_fs(self, fs):
        self.ai_fs = fs
        if self._acq_mode == 'finite':
            self.set_mode_to_finite(self.finite_frame_len_sec)
        elif self._acq_mode == 'continuous':
            self.set_mode_to_continuous()

    def get_mode(self):
        return self._acq_mode

    @property # TODO decorate other function properties to remove getter / setter pattern and simplify usage
    def input_frame_len(self):
        return self._input_frame_len

    @property
    def output_frame_len(self):
        return self._output_frame_len

    # def set_start_frequency(self, freq0):
    #    self.freq0 = freq0
    #    self.function_gen.set_start_frequency(freq0)
    # HERE ADD ANY CALCULATION OF IIR FILTER COEFFICIENTS OR FIR FRAME

    # def set_frequency(self, freq1):
    #    self.freq1 = freq1
    #    self.function_gen.set_frequency(freq1)
    # HERE ADD ANY CALCULATION OF IIR FILTER COEFFICIENTS OR FIR FRAME

    def set_mode_to_finite(self, fm_meas_time_sec):
        print('iostreamingdevice: switching to finite mode')
        self._acq_mode = 'finite'
        self.finite_frame_len_sec = fm_meas_time_sec
        self._input_frame_len = int(fm_meas_time_sec * self.ai_fs)
        self.input_frame = np.zeros((self._nr_of_active_chans, self._input_frame_len), dtype=np.float64)
        self._set_input_time_base()
        self._output_frame_len = int(fm_meas_time_sec * self.ao_fs)
        self.function_gen.set_output_frame_len(self._output_frame_len)
        self.io_start = self.start_finite_acq_n_gen
        self.io_stop = lambda: print('\n>> WARNING << You have tried to stop a finite measurement but this type of '
                                     'measurement is blocking and always stops by itself. Use set_mode_to_finite('
                                     'measurement_time_sec) method to specify another measurement length if '
                                     'required.\n')

        #print('Input_frame_len changed to: {}'.format(self._input_frame_len))

    def set_mode_to_continuous(self):
        print('iostreamingdevice: switching to continuous mode')
        self._acq_mode = 'conitnuous'
        self._input_frame_len = self.CM_INPUT_FRAME_LEN
        self.input_frame = np.zeros((self._nr_of_active_chans, self._input_frame_len), dtype=np.float64)
        self.sw_input_buffer_size = self._input_frame_len * self.CM_INPUT_FRAMES_PER_BUFFER
        self._set_input_time_base()
        self._output_frame_len = self.CM_OUTPUT_FRAME_LEN
        self.sw_output_buffer_size = self._output_frame_len * self.CM_OUTPUT_FRAMES_PER_BUFFER
        self.function_gen.set_output_frame_len(self._output_frame_len)
        self.io_start = self.start_continuous_acq_n_gen
        self.io_stop = self.stop_continuous_acq_n_gen

        #print('Input_frame_len changed to: {}'.format(self._input_frame_len))
        #print('sw_input_buffer_size changed to: {}'.format(self.sw_input_buffer_size))
        #print('sw_output_buffer_size changed to: {}'.format(self.sw_output_buffer_size))

    def _set_input_time_base(self):
        time_series = np.arange(self._input_frame_len, dtype=np.float64) / self.ai_fs
        self.input_time_base = np.tile(time_series, (self._nr_of_active_chans, 1))

    def set_filters(self):
        pass
        # self.output_filter_coefs = sig.bessel(self.FILTER_ORDER, self.freq1 / (self.ao_fs / 2), btype='low', analog=False, output='ba',
        #                                      fs=self.ao_fs)
        # self.input_filter_coefs = sig.bessel(self.FILTER_ORDER, self.freq1 / (self.ai_fs / 2), btype='low', analog=False, output='ba',
        #                                     fs=self.ai_fs)

    def get_monitor(self):
        # return monitor_frame by reshaping back from 3 to 2 dimensions (Z(frame),Y(channel),X(timeseries)) to [[ChA],[ChB]] timeseries data)
        # this simple task looks surprisingly complex in numpy but should still be computationally efficient...
        self.monitor_lock.acquire()
        monitor_window = self._monitor_storage.swapaxes(0, 1).reshape((self._nr_of_active_chans, -1), order='C')
        self.monitor_lock.release()
        return monitor_window


class NiDaqStreamingDevice(StereoStreamingDeviceBase):  # this is model
    def __init__(self, device_name):
        super().__init__(device_name)
        print('Instantiating NiDaqStreamingDevice...')

        # overwrite the base class constants:
        self.CM_INPUT_FRAME_LEN = 2048  # use 2**n only
        self.CM_INPUT_FRAMES_PER_BUFFER = 10
        self.CM_OUTPUT_FRAME_LEN = self.CM_INPUT_FRAME_LEN * 2  # It seems that NIDAQ USB has issues with uneven timing for forth and back communications over USB at high sample rate and in combination with multiplexed input and non-multiplexed output this means the data rate capability of output is up to 2x that of the input. Setting 2x frame length on output helps to balance the transfer frequency out and results in 2x larger output buffer.
        self.CM_OUTPUT_FRAMES_PER_BUFFER = 10

        #self.FILTER_ORDER = 4
        self.AA_OUTPUT_FILTER_ORDER = 8

        self.ATTEMPT_OVERRIDE_DEFAULT_INPUT_OVERWRITE_BEHAVIOUR = False
        self.INPUT_OVERWRITE = False # for this property to take effect you must set applicable OVERRIDE to true
        self.ATTEMPT_OVERRIDE_DEFAULT_INPUT_BUFFER_SIZE = True  # ON NI6211 the default buffer size settings seems too small. Overwrite errors at high sampling rates are quite frequent.

        self.ATTEMPT_OVERRIDE_DEFAULT_UNDERFLOW_BEHAVIOR_TO_PAUSE = False #E.g. this is not implemented on NI6211
        self.ATTEMPT_OVERRIDE_DEFAULT_OUTPUT_REGENERATION_MODE = False
        self.ALLOW_OUTPUT_REGENERATION = True # On NI6211 output regen is default but I leave this default to true as fail-safe for adventurous users who will swithch OVERRIDE to True
        self.ATTEMPT_OVERRIDE_DEFAULT_OUTPUT_BUFFER_SIZE = True  # ON NI6211 the default buffer size settings seems too small. Its possible to run into buffer underflow condition at high sampling rates.

        self.ATTEMPT_OVERRIDE_DEFAULT_USB_XFER = False
        self.AO_USB_XFER_REQ_SIZE = 32768
        self.AO_USB_XFER_REQ_COUNT = 4

        self._stream_reader = None
        self._stream_writer = None
        self._ai_task = None
        self._ao_task = None

        system = ni.system.System.local()
        self.device_info = system.devices[self.device_name]

        self._set_limits()
        self.set_default_config()

    def _set_limits(self):
        self.limits.supported_monitor_frame_lengths = self.get_supported_monitor_frame_lengths()
        self.limits.ao_physical_chans = [chan.name for chan in self.device_info.ao_physical_chans]
        self.limits.ao_min_rate = self.device_info.ao_min_rate
        self.limits.ao_max_rate = self.device_info.ao_max_rate
        self.limits.ao_voltage_rngs = self.device_info.ao_voltage_rngs[::-1]
        #print(self.limits.ao_voltage_rngs)

        self.limits.ai_physical_chans = [chan.name for chan in self.device_info.ai_physical_chans]
        self.limits.ai_max_single_chan_rate = self.device_info.ai_max_single_chan_rate

        self.limits.ai_max_two_chan_rate = self.device_info.ai_max_single_chan_rate / self._nr_of_active_chans  # TODO here we assume the output is multiplexed but it would be worthwhile to try checking if output supports parallel conversion.
        self.limits.ai_min_rate = self.device_info.ai_min_rate
        self.limits.ai_voltage_rngs = self.device_info.ai_voltage_rngs[::-1]
        #print(self.limits.ai_voltage_rngs)

        self.limits.terminal_configs = [terminal_config.name for terminal_config in
                                        ni.constants.TerminalConfiguration]  # TODO for the moment I find no implementation of nidaqmx property to check supported terminal configurations , therefore all possible types (including non supported will be listed). Handle later by try except. Alternative is to run a test here and reject the types that cause an exception, for this a temporary task would be required..

        self.limits.supported_input_rates, self.limits.supported_output_rates = self.get_supported_sampling_rates()

    def set_filters(self):
        # super().set_filters()
        # ensure that already the function generator output signal does not exceed the input sampling rate capability
        self.function_gen.set_aa_filter(wn=0.7 * self.ai_fs / self.ao_fs, order=self.AA_OUTPUT_FILTER_ORDER)

    def get_supported_sampling_rates(self):
        #print('Fetching supported sampling rates list.')
        info_task = ni.Task()
        info_task.ai_channels.add_ai_voltage_chan(self.limits.ai_physical_chans[0])
        # Task must have I/O channel added before sample clock timebase rate can be queried
        try:
            supported_input_rates = nearest_supported_sample_rates(info_task.timing.samp_clk_timebase_rate,
                                                                   self.STANDARD_SAMPLE_RATES, duplicates=False)
        except ni.DaqError as err:
            print(err)
            print('Could not obtain sample clock timebase rate information.\n'
                  '  Using analog output max sampling rate as a proxy.')
            supported_input_rates = nearest_supported_sample_rates(self.ao_max_rate,
                                                                   self.STANDARD_SAMPLE_RATES, duplicates=False)

        supported_output_rates = supported_input_rates.copy()

        clip_list(supported_input_rates, (self.limits.ai_min_rate, self.limits.ai_max_two_chan_rate))
        clip_list(supported_output_rates, (self.limits.ao_min_rate, self.limits.ao_max_rate))

        # append maxima to the top of the list to make them the default values:
        supported_input_rates.insert(0, self.limits.ai_max_two_chan_rate)
        supported_output_rates.insert(0, self.limits.ao_max_rate)

        # remove potential duplicate entries
        supported_input_rates = sorted(list(set(supported_input_rates)), reverse=True)
        supported_output_rates = sorted(list(set(supported_output_rates)), reverse=True)

        info_task.close()
        return supported_input_rates, supported_output_rates

    def start_continuous_acq_n_gen(self):
        self._ao_task = ni.Task()
        self._ai_task = ni.Task()
        ai_args = {'min_val': self.ai_min_val,
                   'max_val': self.ai_max_val,
                   'terminal_config': ni.constants.TerminalConfiguration[self.ai_terminal_config]}

        self._ai_task.ai_channels.add_ai_voltage_chan(self.ai_a_name, **ai_args)
        self._ai_task.ai_channels.add_ai_voltage_chan(self.ai_b_name, **ai_args)

        self._ai_task.timing.cfg_samp_clk_timing(rate=self.ai_fs, sample_mode=ni.constants.AcquisitionType.CONTINUOUS,
                                                 samps_per_chan=self._input_frame_len)

        self._ai_task.triggers.start_trigger.cfg_dig_edge_start_trig("ao/StartTrigger",
                                                                     trigger_edge=ni.constants.Edge.RISING)

        # Below is a bit clumsy EAFP but we have to be careful writing to properties because they may or may not be
        # present depending on device type while the "double locking" system with both OVERRIDE and property needing
        # change requires user to really know what he is doing. In most cases if some control attribute is not
        # implemented in a device, it can work without issues just with the default settings though. On NI6211,
        # this code is debugged with, only the default output buffer had to be changed to prevent underflow at high
        # UBS data rates. I leave this code though for users to experiment in case of issues with other DAQ models.

        if self.ATTEMPT_OVERRIDE_DEFAULT_INPUT_BUFFER_SIZE:
            try:
                self._ai_task.in_stream.input_buf_size = self.sw_input_buffer_size
            except ni.errors.DaqError as exc:
                print(exc)
                print('ATTEMPT_OVERRIDE_DEFAULT_INPUT_BUFFER_SIZE failed')

        if self.ATTEMPT_OVERRIDE_DEFAULT_INPUT_OVERWRITE_BEHAVIOUR is True:
            try:
                if self.INPUT_OVERWRITE is False:
                    self._ai_task.in_stream.over_write = ni.constants.OverwriteMode.DO_NOT_OVERWRITE_UNREAD_SAMPLES
                elif self.INPUT_OVERWRITE is True:
                    self._ai_task.in_stream.over_write = ni.constants.OverwriteMode.OVERWRITE_UNREAD_SAMPLES
            except ni.errors.DaqError as exc:
                print(exc)
                print('ATTEMPT_OVERRIDE_DEFAULT_INPUT_OVERWRITE_BEHAVIOUR failed')

        ao_args = {'min_val': self.ao_min_val,
                   'max_val': self.ao_max_val}

        ao_chan_a = self._ao_task.ao_channels.add_ao_voltage_chan(self.ao_a_name, **ao_args)
        ao_chan_b = self._ao_task.ao_channels.add_ao_voltage_chan(self.ao_b_name, **ao_args)

        if self.ATTEMPT_OVERRIDE_DEFAULT_USB_XFER is True:
            try:
                ao_chan_a.ao_usb_xfer_req_count = self.AO_USB_XFER_REQ_COUNT
                ao_chan_b.ao_usb_xfer_req_count = self.AO_USB_XFER_REQ_COUNT
                ao_chan_a.ao_usb_xfer_req_size = self.AO_USB_XFER_REQ_SIZE
                ao_chan_b.ao_usb_xfer_req_size = self.AO_USB_XFER_REQ_SIZE
            except ni.errors.DaqError as exc:
                print(exc)
                print('ATTEMPT_OVERRIDE_DEFAULT_USB_XFER failed')

        self._ao_task.timing.cfg_samp_clk_timing(rate=self.ao_fs, samps_per_chan=self._output_frame_len, #TODO bug on dev1
                                                 sample_mode=ni.constants.AcquisitionType.CONTINUOUS)

        if self.ATTEMPT_OVERRIDE_DEFAULT_OUTPUT_REGENERATION_MODE is True:
            try:
                if self.ALLOW_OUTPUT_REGENERATION is False:
                    self._ao_task.out_stream.regen_mode = ni.constants.RegenerationMode.DONT_ALLOW_REGENERATION  # prevents DAQ card from repeating output, it just waits for more data to be added to the buffer- > but on cards that cant handle to automatically pause output generation when out of buffer this setting will case random and unexpected crash!
                elif self.ALLOW_OUTPUT_REGENERATION is True:
                    self._ao_task.out_stream.regen_mode = ni.constants.RegenerationMode.ALLOW_REGENERATION
            except ni.errors.DaqError as exc:
                print(exc)
                print('ATTEMPT_OVERRIDE_DEFAULT_OUTPUT_REGENERATION_MODE failed')

        if self.ATTEMPT_OVERRIDE_DEFAULT_UNDERFLOW_BEHAVIOR_TO_PAUSE is True:
            try:
                self._ao_task.timing.implicit_underflow_behavior = ni.constants.UnderflowBehavior.AUSE_UNTIL_DATA_AVAILABLE  # SIC!
            except ni.errors.DaqError as exc:
                print(exc)
                print('Could not OVERRIDE_DEFAULT_UNDEFLOW_BEHAVIOR')
                if self._ao_task.out_stream.regen_mode == ni.constants.RegenerationMode.DONT_ALLOW_REGENERATION:
                    print('WARNING: Your device does not support pause in case of output underflow while auto '
                          'regeneration of the AO buffer is not allowed.\n  '
                          'In case of output buffer underflow due to system resources the application will crash!\n  '
                          'To prevent this warning and risk of crash its highly recommended to set _out_stream.regen_'
                          'mode to ALLOW_REGENERATION.\n  '
                          'This will allow output glitches in theory but will prevent '
                          'application from crashing on output buffer underflow.')

        if self.ATTEMPT_OVERRIDE_DEFAULT_OUTPUT_BUFFER_SIZE is True:
            try:
                self._ao_task.out_stream.output_buf_size = self.sw_output_buffer_size  # seems like buffer is not calculating correct and we must call out explicitly on NI6211. Not setting this property leads to a chain of very confusing warnings/glitches
                # print('Output buffer size set to {}'.format(self.sw_output_buffer_size))
            except ni.errors.DaqError as exc:
                print(exc)

        self._stream_reader = stream_readers.AnalogMultiChannelReader(self._ai_task.in_stream)
        self._stream_writer = stream_writers.AnalogMultiChannelWriter(self._ao_task.out_stream)

        # fill AO buffer with data
        self.function_gen.generate()
        buffer_frame = self.function_gen.output_frame

        for frames in range(self.CM_OUTPUT_FRAMES_PER_BUFFER - 1): # TODO UGLY
            self.function_gen.generate()
            buffer_frame = np.append(buffer_frame, self.function_gen.output_frame, axis=1)
        self._ao_task.write(buffer_frame)

        self._ai_task.register_every_n_samples_acquired_into_buffer_event(self._input_frame_len,
                                                                          self.reading_task_callback)

        self._ao_task.register_every_n_samples_transferred_from_buffer_event(self._output_frame_len,
                                                                             self.writing_task_callback)

        self._ai_task.start()  # arm but do not trigger the acquisition
        self._ao_task.start()  # trigger both generation and acquisition simultaneously

        self.cm_measurement_is_running = True

    def reading_task_callback(self, task_idx, event_type, num_samples, callback_data=None):
        """This callback is called to read out the data from the buffer.
        This callback is for working with the task callback register_every_n_samples_acquired_into_buffer_event.
        This function arguments must follow prototype defined in nidaqxm documentation.

        Args:
            task_idx (int): Task handle index value
            event_type (nidaqmx.constants.EveryNSamplesEventType): ACQUIRED_INTO_BUFFER
            num_samples (int): The number_of_samples parameter contains the value you passed in the sample_interval
             parameter of register_every_n_samples_acquired_into_buffer_event function.
            callback_data (object)[None]: The callback_data parameter contains the value you passed in the callback_data
             parameter of register_every_n_samples_acquired_into_buffer_event function.
        """

        # print('before read {}'.format(self._ai_task._in_stream.curr_read_pos))

        self.input_frame = np.zeros((self._nr_of_active_chans, self._input_frame_len),
                                    dtype=np.float64)
        # TODO if we do not re-init the storage the following frames contain glitches between L/R and queued data is
        #  not phase continuous. I have no clue why but it seems that once written into, the array format becomes
        #  modified.
        self.read_lock.acquire()
        self._stream_reader.read_many_sample(self.input_frame, num_samples, timeout=ni.constants.WAIT_INFINITELY)
        self.read_lock.release()
        self.input_frame_ready_signal.emit()
        self._put_monitor_frame()

        # The callback function must return to prevent
        # raising an exception: 'TypeError: an integer is required (got type NoneType)'
        return 0

    def writing_task_callback(self, task_idx, event_type, num_samples, callback_data=None):
        """This callback is called to write new data to the buffer.
        This callback is for working with the task callback register_every_n_samples_transferred_from_buffer_event.
        This function arguments must follow prototype defined in nidaqxm documentation.

        Args:
            task_idx (int): Task handle index value
            event_type (nidaqmx.constants.EveryNSamplesEventType): TRANSFERRED_FROM_BUFFER
            num_samples (int): The number_of_samples parameter contains the value you passed in the sample_interval
             parameter of register_every_n_samples_transferred_from_buffer_event.
            callback_data (object)[None]: The callback_data parameter contains the value you passed in the callback_data
             parameter of register_every_n_samples_transferred_from_buffer_event.
        """
        self.function_gen.generate()
        self._stream_writer.write_many_sample(self.function_gen.output_frame, timeout=3)

        return 0  # the callback must return to prevent raising an exception

    def stop_continuous_acq_n_gen(self):
        self._ao_task.stop()
        self._ai_task.stop()
        self._ao_task.close()
        self._ai_task.close()
        self.cm_measurement_is_running = False

    def start_finite_acq_n_gen(self):
        self.function_gen.reset_phase()
        self.function_gen.generate()

        with ni.Task() as ao_task, ni.Task() as ai_task:  # enter drop down info to arguments while checking one by one that code still works

            ao_args = {'min_val': self.ao_min_val,
                       'max_val': self.ao_max_val}

            ao_task.ao_channels.add_ao_voltage_chan(self.ao_a_name, **ao_args)
            ao_task.ao_channels.add_ao_voltage_chan(self.ao_b_name, **ao_args)


            ao_task.timing.cfg_samp_clk_timing(self.ao_fs,
                                               samps_per_chan=self._output_frame_len,
                                               sample_mode=ni.constants.AcquisitionType.FINITE)

            ai_args = {'min_val': self.ai_min_val,
                       'max_val': self.ai_max_val,
                       'terminal_config': ni.constants.TerminalConfiguration[self.ai_terminal_config]}

            ai_task.ai_channels.add_ai_voltage_chan(self.ai_a_name, **ai_args)
            ai_task.ai_channels.add_ai_voltage_chan(self.ai_b_name, **ai_args)

            ai_task.timing.cfg_samp_clk_timing(self.ai_fs,
                                               samps_per_chan=self._input_frame_len,
                                               sample_mode=ni.constants.AcquisitionType.FINITE)

            ai_task.triggers.start_trigger.cfg_dig_edge_start_trig("ao/StartTrigger",
                                                                   trigger_edge=ni.constants.Edge.RISING)

            #ao_task.write(np.ascontiguousarray(self.function_gen.output_frame), auto_start=False)
            ao_task.write(self.function_gen.output_frame, auto_start=False)
            ai_task.start()  # arms ai but does not trigger
            ao_task.start()  # triggers both ao an ai simultaneously.

            ai_task.wait_until_done(timeout=self.finite_frame_len_sec + 0.1)
            self.input_frame = np.array(ai_task.read(number_of_samples_per_channel=self._input_frame_len))
            # NOTE: returned type is a list of lists, for compatibility we should convert to ndarray - same format as in
            # continuous mode

            # TODO implement upsampling at the experiment level instead
            # if self.input_upsampling_enabled is True:
            #     # upsample the input signal
            #     upsampled_fs = self.ai_fs * self.UPSAMPLE_RATIO  # is this really required? the next time you run without upsampling will be with wrong ai_fs unless you correct for that
            #     self.input_frame, self.input_time_base = sig.resample(self.input_frame, upsampled_fs,
            #                                                           t=self.input_time_base, axis=1, window=None)

    def get_ao_buffer_level_prc(self):
        return int((self.sw_output_buffer_size - self._ao_task.out_stream.space_avail) / self.sw_output_buffer_size * 100)

    def get_ai_buffer_level_prc(self):
        return int(self._ai_task.in_stream.avail_samp_per_chan / self.sw_input_buffer_size * 100)


class PyAudioSoundStreamingDevice(StereoStreamingDeviceBase):
    def __init__(self, hardware_id):
        super().__init__(hardware_id)
        print('Instantiating PyAudioSoundStreamingDevice')
        self.READ_OFFSET_MSEC = 15
        self.input_info = pa.get_default_input_device_info()
        self.output_info = pa.get_default_output_device_info()
        #print(self.input_info)
        #print(self.output_info)

        self._in_stream = None  # later consider renaming to _stream_reader and placing both libraries property already in the base class
        self._out_stream = None

        self._set_limits()
        self.set_default_config()

    def _set_limits(self):
        self.limits.supported_monitor_frame_lengths = self.get_supported_monitor_frame_lengths()
        self.limits.supported_input_rates, self.limits.supported_output_rates = self.get_supported_sampling_rates()

        # question driver about device properties
        self.limits.ao_physical_chans = \
            ['Ch:' + str(chan_index) for chan_index in range(self.output_info['maxOutputChannels'])]
        self.limits.ao_min_rate = min(self.limits.supported_output_rates)
        self.limits.ao_max_rate = max(self.limits.supported_output_rates)
        self.limits.ao_voltage_rngs = config.default_pyAudio_paFloat32_level_range
        self.limits.ai_physical_chans = \
            ['Ch:' + str(chan_index) for chan_index in range(self.input_info['maxInputChannels'])]
        self.limits.ai_min_rate = min(self.limits.supported_input_rates)
        self.limits.ai_max_single_chan_rate = max(self.limits.supported_input_rates)
        self.limits.ai_max_two_chan_rate = max(self.limits.supported_input_rates)
        self.limits.ai_voltage_rngs = config.default_pyAudio_paFloat32_level_range
        self.limits.terminal_configs = ['N/A']

    def get_supported_sampling_rates(self):
        input_rates_supported = []
        output_rates_supported = []
        for rate in self.STANDARD_SAMPLE_RATES:
            input_rates_supported.append(pa.is_format_supported(rate, input_device=self.input_info['index'],
                                                                input_channels=self._nr_of_active_chans,
                                                                input_format=pyaudio.paFloat32))
            output_rates_supported.append(pa.is_format_supported(rate, output_device=self.output_info['index'],
                                                                 output_channels=self._nr_of_active_chans,
                                                                 output_format=pyaudio.paFloat32))
        supported_input_rates = [rate for (rate, supported) in
                                 zip(self.STANDARD_SAMPLE_RATES, input_rates_supported) if supported]
        supported_output_rates = [rate for (rate, supported) in
                                  zip(self.STANDARD_SAMPLE_RATES, output_rates_supported) if supported]

        supported_input_rates.insert(0, self.input_info['defaultSampleRate'])
        supported_output_rates.insert(0, self.output_info['defaultSampleRate'])

        return supported_input_rates, supported_output_rates

    def start_continuous_acq_n_gen(self):
        # self.input_frame = np.array([[], []])
        self._in_stream = pa.open(start=False, format=pyaudio.paFloat32,
                                  channels=self._nr_of_active_chans,
                                  rate=int(self.ai_fs),
                                  input=True,
                                  stream_callback=self.reading_callback,
                                  frames_per_buffer=self.CM_INPUT_FRAME_LEN)

        self._out_stream = pa.open(start=False, format=pyaudio.paFloat32,
                                   channels=self._nr_of_active_chans,
                                   rate=int(self.ao_fs),
                                   output=True,
                                   stream_callback=self.writing_callback,
                                   frames_per_buffer=self.CM_OUTPUT_FRAME_LEN)

        print('Starting continuous acq')
        self._out_stream.start_stream()
        # self.wait_event.wait(self.READ_OFFSET_MSEC / 1000)
        sleep(self.READ_OFFSET_MSEC / 1000)
        self._in_stream.start_stream()
        self.cm_measurement_is_running = True

    def stop_continuous_acq_n_gen(self):
        self._in_stream.stop_stream()
        self._out_stream.stop_stream()
        self._in_stream.close()
        self._out_stream.close()
        self.cm_measurement_is_running = False

    def reading_callback(self, in_data, frame_count, time_info, status):
        # USE FRAME COUNT AND self.CM_INPUT_FRAME_LEN to predict roughly how many frames an allocated storage np.array should have and udpdate it with converted data from here. Later clip away any trailing NONEs
        self.read_lock.acquire()
        #self.mutex.lock()
        self.input_frame = wave_bytes_to_ndarray(in_data, self._nr_of_active_chans, np.float32)
        #self.mutex.unlock()
        self.read_lock.release()
        self.input_frame_ready_signal.emit()
        self._put_monitor_frame()

        return None, status

        # if self.QUEUE_INPUT_FRAMES_ENABLED:
        #     try:
        #         self.input_queue.put_nowait(self.input_frame)
        #     except Full:
        #         self.monitor_ready_signal.emit()
        # return None, status

    def writing_callback(self, in_data, frame_count, time_info, status):

        self.function_gen.generate()
        out_data = ndarray_to_wave_bytes(self.function_gen.output_frame, np.float32)

        return out_data, status

    def start_finite_acq_n_gen(self):
        self.input_frame = np.array([[]]*self._nr_of_active_chans)
        self.function_gen.reset_phase()
        self.function_gen.generate()

        self._in_stream = pa.open(start=False, format=pyaudio.paFloat32,
                                  channels=self._nr_of_active_chans,
                                  rate=int(self.ai_fs),
                                  input=True,
                                  stream_callback=self.finite_reading_callback,
                                  frames_per_buffer=self.CM_INPUT_FRAME_LEN)

        self._out_stream = pa.open(start=False, format=pyaudio.paFloat32,
                                   channels=self._nr_of_active_chans,
                                   rate=int(self.ao_fs),
                                   output=True,
                                   stream_callback=self.finite_writing_callback,
                                   frames_per_buffer=self.function_gen.chunk_len)

        latency_offset = self._out_stream.get_output_latency() + self._in_stream.get_input_latency()

        self._out_stream.start_stream()
        sleep(self.READ_OFFSET_MSEC / 1000)
        self._in_stream.start_stream()

        sleep(self.finite_frame_len_sec + latency_offset)

        self._out_stream.close()
        self._in_stream.close()
        raw_frame_len = max(self.input_frame.shape)
        #offset = raw_frame_len - self._input_frame_len
        # print(
        #     'output signal len is {},'
        #     ' raw bytes data len is{},'
        #     ' frame_len is {},'
        #     ' complete input frame len{},'
        #     ' offset is {}'.format(
        #         max(self.function_gen.output_frame.shape),
        #         self._output_frame_len, self._input_frame_len,
        #         max(self.input_frame.shape), offset))
        self.input_frame = self.input_frame[:self._nr_of_active_chans, -self._input_frame_len:]

    def finite_reading_callback(self, in_data, frame_count, time_info, status):
        # we need reading by callback else simultaneous, blocking out and in stream would block each other
        this_frame = wave_bytes_to_ndarray(in_data, self._nr_of_active_chans, np.float32)
        self.input_frame = np.append(self.input_frame, this_frame, axis=1)  # we use this callback in combination with
        # sleep (blocking mode) therefore its acceptable to use append as we truly do not know the size of the output
        # we will end up with.
        return None, status

    def finite_writing_callback(self, in_data, frame_count, time_info, status):
        chunk, complete = self.function_gen.next_chunk()
        chunk_bytes = ndarray_to_wave_bytes(chunk, np.float32)

        if complete:
            status = pyaudio.paComplete
        else:
            status = pyaudio.paContinue
        return chunk_bytes, status  # the first returned data will be written into the output buffer

    def get_ao_buffer_level_prc(self):
        return 'n.a.' # TODO I don't think that pyAudio has got an interface for this kind of info

    def get_ai_buffer_level_prc(self):
        return 'n.a.'


if __name__ == "__main__":

    # only needed to run example:
    import pyqtgraph as pg
    import sys


    def update_time_plot():
        plot_data_item_A.setData(daq.input_time_base[0], daq.input_frame[0])
        plot_data_item_B.setData(daq.input_time_base[1], daq.input_frame[1])


    def update_monitor_plot():
        monitor_window = daq.get_monitor()
        monitor_plot_data_item_A.setData(monitor_window[0])
        monitor_plot_data_item_B.setData(monitor_window[1])

    try:
        print('\n######## SYSTEM SOUND DEVICE LIST #########')
        for index in range(pa.get_device_count()):
            print(pa.get_device_info_by_index(index))
        print('######### END OF DEVICE LIST ##########\n')
    except NameError:
        pass

    devices_name_to_model_dict = io_streaming_device_discovery()
    if len(devices_name_to_model_dict) == 0:
        print('No I/O streaming devices were found')
    else:
        device_names = [name for name in devices_name_to_model_dict.keys()]
        for i in range(len(device_names)):
            print('<' + str(i) + '> ' + device_names[i] + '\n')
        dev_index = input("<Select> streaming device from the list.>>")
        device_name = device_names[int(dev_index)]
        daq_model_type = devices_name_to_model_dict[device_name]
        daq = daq_model_type(device_name)

        # add changes to default streaming device config here:
        if daq.device_name == 'Dev1':
            daq.ai_terminal_config = 'RSE'

        #for key, value in daq.__dict__.items():
        #    print(key, '=', value)

        app = pg.QtGui.QApplication(sys.argv)
        win = pg.GraphicsWindow(title=daq.device_name)
        plot_item1 = win.addPlot(title="Input frame (s)")
        plot_data_item_A = plot_item1.plot(pen=1)
        plot_data_item_B = plot_item1.plot(pen=2)
        win.nextRow()
        plot_item2 = win.addPlot(title="Monitor window (pts)")
        monitor_plot_data_item_A = plot_item2.plot(pen=1)
        monitor_plot_data_item_B = plot_item2.plot(pen=2)

        plot_timer_frame = QtCore.QTimer()
        plot_timer_frame.timeout.connect(update_time_plot)
        daq.monitor_ready_signal.connect(update_monitor_plot)

        daq.start_continuous_acq_n_gen()
        plot_timer_frame.start(100)
        app.exec_()
        daq.stop_continuous_acq_n_gen()
