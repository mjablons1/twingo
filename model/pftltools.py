import numpy as np
from scipy import signal as sig
from threading import Thread, Event, Timer, Lock
from time import time as now

class Error(Exception):
    """Base class for other exceptions"""
    pass


class FilterUnstable(Error):
    """Exception raised when unstable filter is constructed"""
    pass


class NyquistFrequencyExceeded(Error):
    """Exception raised when required signal frequency exceeds nyquist frequency for given sampling rate."""
    pass


class FrequencyRangeIncorrect(Error):
    """Exception raised when frequency range for ESS method is incorrectly set."""
    pass


#class ClipWarning(Error):
#    """Exception raised if data in list or array had to be clipped to prevent exceeding the limits"""
#    pass

# def check_str_input_contains_positive_float(func):
#     def func_wrapper(input_text):
#         try:
#             number = float(input_text)
#         except TypeError as exc:
#             print(exc)
#         if number <= 0:
#             raise NumberIsNotPositive
#         return func(input_text)
#     return func_wrapper


def filterfilter(input_signal, fs, f0, f1, filter_order):
    '''
    Does forward and backward filtering using Butterworth bandpass filter of specified order from f0 to f1.
    fs must represent the input signal sampling rate. Because the filtering is done forward and backward the resulting
    order is twice that the input specified.
    '''
    LO_cutoff_norm = f0 / (fs / 2)
    HI_cutoff_norm = f1 / (fs / 2)
    # create filter for audible frequencies:
    b, a = sig.butter(filter_order, [LO_cutoff_norm, HI_cutoff_norm], 'bandpass', analog=False, output='ba')

    # Check if filter is stable.
    # If not, raise error preventing potentially dangerous output signal from reaching hardware.
    if not np.all(np.abs(np.roots(a)) < 1):
        raise FilterUnstable('Filter with cutoff at {} Hz and {} Hz is unstable given'
                                  ' sample frequency {} Hz'.format(f0, f1, fs))
    # filter the signal according set frequency limits:
    output_signal = sig.filtfilt(b, a, input_signal, method="gust")

    return output_signal


def rescale(sig_array, amp):
    """
    :param <ndarray> sig_array: numpy array
    :param <float> amp: new max value
    :return <ndarray>: rescaled to within +/- amp
    """
    sig_array /= np.max(np.abs(sig_array))
    sig_array *= amp
    return sig_array


def limit(value, limits):
    """
    :param <float> value: value to limit
    :param <list>/<tuple> limits: (min, max) limits to which restrict the value
    :return <float>: value from within limits, if input value readily fits into the limits its left unchanged. If value exceeds limit on either boundary its set to that boundary.
    """
    if value < limits[0]:
        value = limits[0]
    elif value > limits[1]:
        value = limits[1]
    else:
        pass
    return value


def clip_list(values, limits):  #TODO this is unnecessary because numpy already implements clip that is high performance

    checksum = sum([abs(value) for value in values])
    for i in range(len(values)):
        values[i] = limit(values[i], limits)

    if sum([abs(value) for value in values]) != checksum:
        clipping_occurred = True
    else:
        clipping_occurred = False

    return clipping_occurred
    #raise ClipWarning('Warning: Some values were clipped to the limits.')


def optimal_noverlap(win_name,win_len):
    """
    This function is intended to support scipy.signal.stft calls with noverlap parameter.
    :param win_name: (str) name of the window (has to follow scipy.signal.windows naming)
    :param win_len: (int) lenght of the FFT window
    :return : (int) optimal overlap in points for the given window type
    """

    window_to_overlap_coef = {'hann': 0.5,
                              'hamming': 0.75,
                              'blackmanharris': 0.75,
                              'blackman': 2/3,
                              'flattop': 0.75,
                              'boxcar': 0}
    try:
        noverlap = int(win_len*window_to_overlap_coef[win_name])
    except KeyError as exc:
        print(exc)
        print('The window you have selected is not recognized or does not have optimal overlap. Setting window overlap to default 75%.')
        noverlap = int(win_len*0.75)

    return noverlap


def wave_bytes_to_ndarray(interleaved_in_data, nr_of_channels, in_data_format):
    """ interleaved_in_data - bytes"""
    interleaved_ndarray = np.frombuffer(interleaved_in_data, in_data_format)
    deinterleaved_chans_tuple = tuple([interleaved_ndarray[i::nr_of_channels] for i in range(nr_of_channels)])
    deinterleaved_out_data = np.stack(deinterleaved_chans_tuple)
    #thanks to https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    return deinterleaved_out_data


def ndarray_to_wave_bytes(in_data, out_data_format):
    """ in_data - ndarray with dimensions (nr_of_channels, len_chan_data)"""
    nr_of_channels = in_data.shape[0]
    chans_tuple = tuple([in_data[i] for i in range(nr_of_channels)])
    interleaved_ndarray = np.vstack(chans_tuple).reshape((-1,), order='F') # not sure why order F has to be used
    #thanks to https://www.schlameel.com/2017/06/09/interleaving-and-de-interleaving-data-with-python/
    interleaved_out_data = interleaved_ndarray.astype(out_data_format).tobytes()
    return interleaved_out_data


def nearest_sample_rate(clock_base, sample_rate):

    if sample_rate < 1:
        raise ValueError('Sample rate < 1 can not be supported by this algorithm ({} given).'.format(sample_rate))
    if clock_base < sample_rate:
        raise ValueError('Sample rate ({}Hz) can not be faster than the clock base ({}Hz).'.format(sample_rate, clock_base))

    clock_divider = clock_base / sample_rate
    nearest_dividers = {'min': int(clock_divider),
                        'max': int(clock_divider)}
    operators = (1, -1)
    i = 0

    for key in nearest_dividers:
        while clock_base % nearest_dividers[key] != 0:
            nearest_dividers[key] += operators[i]
        i += 1

    if (nearest_dividers['max'] - clock_divider) < (clock_divider - nearest_dividers['min']):
        return clock_base / nearest_dividers['min']
    else:
        return clock_base / nearest_dividers['max']


def nearest_supported_sample_rates(clock_base, sample_rates, duplicates=True):

    nearest_sample_rates = []
    for rate in sample_rates:
        try:
            nearest_sample_rates.append(nearest_sample_rate(clock_base, rate))
        except ValueError as exc:
            print(exc)
    if duplicates is False:
        return sorted(list(set(nearest_sample_rates)), reverse=True)
    else:
        return nearest_sample_rates

class SampleTimer(Thread):
    """ A timer thread that attempts to execute the callback function every fixed time interval with maximum accuracy.
    Example applications: data plotting refresh, measurement sequence trigger etc.

    The timer will not issue another trigger unless the previous callback function execution returns. If any delay is
    caused by current callback function execution overrunning the trigger the timer will try to recover from delay by
    shortening time till following callback triggers until its timing returns back on track wrt. the prescribed
    trigger rate.

    NOTE: If the called function permanently takes longer than the rate given SampleTimer can cause visible
    delays in execution compared to the intent.

    NOTE2: Because SampleTimer is based on the Thread object it can not be reused after stop() is issued.

    User can verify the timer accuracy by uncommenting self.actual_trig_times and self.actual_completion_times to help
    debug any performance issues in the callback function execution.
    """
    def __init__(self, rate, callback_function, *args):  # rate = time in seconds between subsequent triggers
        """
        :param <float> rate: target time interval between callback function triggers
        :param <object> callback_function: function to be triggered after every time interval
        :param <tuple> optional parameters to be passed inside the callback function
        """
        Thread.__init__(self)
        self.stop_event = Event()
        self.rate = rate
        self.callback_function = callback_function
        self.callback_args = args  # if done this way arguments will be passed by value instead by reference even on mutable types
        self.start_time = None
        self.actual_trig_times = []  # writing to this property is suppressed by default to prevent memory leak
        self.actual_completion_times = []
        self.iterations = 0

    def start(self):
        """
        Calls the original implementation of the start() method except it saves the reference time at which it is
        called.
        """

        self.start_time = now()
        super().start()

    # def stop(self):
    #     """
    #     Calls the original implementation of the stop() method except it re-init's the object to make it appear
    #     reusable. This does not work however, for some reason...
    #     """
    #     super().stop()
    #     self.__init__(self.rate, self.callback_function, self.callback_args)

    def elapsed_time(self):
        """
        :return <float>: time in seconds that passed from the moment timer start was issued
        """
        return now()-self.start_time

    def time_left_to_trig(self):
        """
        :return <float>: time in seconds left till next callback function trigger should occur
        If the previous callback function execution overrun return value is set to 0
        """
        return max(0, self.iterations*self.rate - self.elapsed_time())

    def run(self):
        """
        Triggers the callback function while keeping track of the trigger time and time of function completion.
        """
        while not self.stop_event.wait(self.time_left_to_trig()):
            # here stop_event.wait works as a simple blocking delay for time_left_to_trig() seconds,
            # by means of waiting for time out. On timeout, wait returns with FALSE causing execution of below code.
            # However as soon as stop_event.set()is issued the stop_event.wait() will immediately return TRUE
            # causing the while loop to exit.
            # self.actual_trig_times.append(self.elapsed_time())  # only uncomment for debugging
            self.callback_function(*self.callback_args)  # its ok to handle without checking args, if not present argument is = None
            # self.actual_completion_times.append(self.elapsed_time())  # only uncomment for debugging
            self.iterations += 1

    def stop_after_sec_blocking(self, timer_duration_sec):
        """
        Blocks further execution of the thread its called from until the indicated time had passed and timer had been
        stopped. You can use this method instead issuing a join() statement to block further execution of the main
        thread till finished.

        :param <float> timer_duration_sec: time in seconds after which the thread will be stopped.
        """

        self.stop_event.wait(timer_duration_sec+0.001)
        # the added time is to make "sure" last event does not get clipped by stop()(yes it can happen)
        self.stop()

    def stop_after_sec_non_blocking(self, timer_duration_sec):
        """
        Prescribes the timer to be stopped after the given time. This method does not block further code execution.
        :param <float> timer_duration_sec: time in seconds after which the thread will be stopped.
        """
        t = Timer(timer_duration_sec+0.001, self.stop)
        # the added time is to make "sure" last event does not get clipped by stop() (yes it can happen)
        t.start()

    def stop(self):
        """
        Stops the timer by setting the event flag.
        """
        self.stop_event.set()


class ListTimer(Thread):
    """ A timer thread that attempts to execute a callback function at times predetermined by a list of trig_times
    with maximum accuracy. Example applications: sporadic triggering of measurement equipment.

    The timer will not issue another trigger unless the previous function execution is complete. If function
    execution overruns the trigger it then continues to try to recover from the delay by shortening time till
    following function triggers until its timing returns back on track wrt input list.

    NOTE: If the called function permanently takes longer than the trig_times prescribed the callback times
    will be permanently delayed compared to trig_times.

    NOTE2: Because SampleTimer is based on the Thread object it can not be reused after stop() is issued.

    User can verify the timer accuracy by checking self.actual_trig_times and self.actual_completion_times to help
    debug any performance issues in the callback function, measurement equipment or transmissions.
    """

    def __init__(self, trig_times, callback_function, *args):
        """
        :param <list> trig_times: times in seconds from thread start at which the callback function should be called.
        :param <object> callback_function:
        """
        Thread.__init__(self)
        self.stop_event = Event()
        self.iteration = 0
        self.trig_times = trig_times
        self.callback_function = callback_function
        self.callback_args = args  # if done this way arguments will be passed by value instead by reference even on mutable types
        self.actual_trig_times = []
        self.actual_completion_times = []
        self.start_time = None

    def start(self):
        """
        Calls the original implementation of the start() method except it saves the reference time at which it is
        called.
        """
        self.start_time = now()
        super().start()

    def elapsed_time(self):
        """
        :return <float>: time in seconds that passed from the moment timer start was issued
        """
        return now()-self.start_time

    def time_left_to_trig(self):
        """
        :return <float>: time in seconds left till next callback function trigger should occur
        If the previous callback function execution overrun return value is set to 0
        """
        return max(0, self.trig_times[self.iteration] - self.elapsed_time())
        # in case callback function overrun last trigger time_left_to_trig will be set to 0

    def run(self):
        """
        Triggers the callback function while keeping track of the trigger time and time of function completion.
        """
        while not self.stop_event.wait(self.time_left_to_trig()):
            # here stop_event.wait works as a simple blocking delay for time_left_to_trig() seconds,
            # by means of waiting for time out. On timeout, wait returns with FALSE causing execution of below code.
            # However as soon as stop_event.set()is issued the stop_event.wait() will immediately return TRUE
            # causing the while loop to exit.
            self.actual_trig_times.append(self.elapsed_time())
            self.callback_function()
            self.actual_completion_times.append(self.elapsed_time())
            self.iteration += 1
            if self.iteration+1 > len(self.trig_times):
                # Could FOR loop be used with same performance instead of while?
                break

    def stop(self):
        """
        Stops the timer by setting the event flag.
        The purpose is to allow user to stop the timer before the list runs out.
        """
        self.stop_event.set()


class FunctionGenerator:
    """
    FunctionGenerator has the functionality similar to a physical device. Its purpose is to generate n identical test
    signals where n is the number of channels. FunctionGenerator allows you to set signal type and parameters and
    manipulate them with set_ methods while generating. Every time generate() is called a new output_frame is created
    with subsequent portion of the signal according to latest property settings. The generator attempts to prevent
    signal discontinuity / pop / click during generation and manipulation of the signals. NOTE: For consistency,
    output_frame has the nr_of_chan dimensions [shape is (nr_of_chan, frame_len)]. In case user requires 1-D array
    [shape equal (frame_len,)], in case of default, shingle channel operation, user must run np.squeeze(output_frame).
    For repeatable, single shot operation user should run reset_phase() after every generate().
    """
    def __init__(self, output_frame_len=1024, sample_rate=44100, freq1=441, freq0=None, amplitude=1, nr_of_chan=1, function_type='sine'):
        """
        :param <int> output_frame_len: length of the output frame in samples
        :param <int> sample_rate: sampling rate of the signals
        :param <float> freq1: frequency of the static signals or end frequency of swept signals
        :param <float> freq0: starting frequency of swept signals
        :param <float> amplitude: amplitude of the output signal
        :param <int> nr_of_chan: number of output channels governing dimensions of the output signal
        :param <str> function_type: indicates which function will be generated.
        Valid keys are 'sine', 'square', 'sweep', 'ess', 'white noise', 'impulse'.
        """
        self.AA_FILTER_ORDER = 4 # these are default values unless user calls the corresponding set method
        self.AA_FILTER_NYQUIST_COEF = 0.7
        self.nr_of_chan = nr_of_chan
        self.amplitude = amplitude
        self.freq0 = freq0
        self.freq1 = freq1
        self.sample_rate = sample_rate
        self._output_frame_len = output_frame_len
        self.__frame_len_sec = None
        self.__set_frame_len_sec()
        self.time_base = None
        self.__set_timebase()
        self.phase_reminder = None
        self.angles_rad = None
        self.reset_phase()
        self.output_frame = None
        self.output_is_exp_sweep_sine = False
        self.exp_sweep_sine = None  # this is jus cache for performance and consistency between subsequent ess frames
        self.sweep_rate = None  # this is jus cache for performance and consistency between subsequent ess frames
        self.check_freqs()
        self.property_lock = Lock()
        self.generate = None
        self.current_function = None
        self.set_function(function_type)
        self.filter_coefs = None
        self.__set_filter_coefs()
        self.filter_current_delays_zf = None
        self.__reset_filter_initial_condition()

        self.chunk_len = 1024
        self.chunk_start = 0


    def __set_timebase(self):
        """
        Updates <ndarray> of time base values in seconds spanning the length of the frame in steps according to sample
        rate. time_base has to have the dimensions according to nr_of_chan because it casts these dimensions on all
        subsequent calculations. time_base can also be used as horizontal values for signal plotting.
        """
        time_series = np.arange(self._output_frame_len, dtype=np.float64) / self.sample_rate
        self.time_base = np.tile(time_series, (self.nr_of_chan, 1))

    def __set_angles_rad(self):
        """
        Updates <ndarray> of radian angles spanning the length of the frame used for periodic signals.
        The array is shifted to the beginning of last frame before signal frequency or type change in order to
        keep signal phase continuous and prevent clicks even during changes made to the signal at runtime.
        """
        self.angles_rad = 2 * np.pi * self.freq1 * self.time_base + self.angles_rad.take(0)
        # last element is to start from last phase to minimize click-pop when changing frequency

    def __set_phase_reminder(self):
        """
        Updates <np.float> scalar value in radians by which every next signal frame must be shifted in order to keep
        the signal phase continuous and prevent pop. Note that this value has to be recalculated only once at the time
        the corresponding signal property is changing (e.g. frequency, sampling rate or frame length).
        """
        samples_per_period = self.sample_rate / self.freq1
        reminder_samples_per_buffer = self._output_frame_len % samples_per_period
        self.phase_reminder = 2 * np.pi * reminder_samples_per_buffer / samples_per_period

    def __set_frame_len_sec(self):
        """
        Updates <float> fm_meas_time_sec property.
        """
        self.__frame_len_sec = self._output_frame_len / self.sample_rate

    def __set_filter_coefs(self):
        #implemet just a nyquist freqyency filter
        self.filter_coefs = sig.bessel(self.AA_FILTER_ORDER, self.AA_FILTER_NYQUIST_COEF, btype='lowpass', analog=False, output='ba')
        #bessel filter is used to have minimum ringing and risk of overshooting the analog output range of the hardware
        pass

    def __reset_filter_initial_condition(self):
        self.filter_current_delays_zf = [[0 for _ in range(self.AA_FILTER_ORDER)]]

    def apply_aa_filter_on_output_signal(self):
        self.output_frame, self.filter_current_delays_zf = sig.lfilter(self.filter_coefs[0],
                                                                       self.filter_coefs[1],
                                                                       self.output_frame,
                                                                       zi=self.filter_current_delays_zf)

    def generate_zeros_output_frame(self):
        """
        NOTE: all generate_ methods should be as concise and performing as possible
        """
        self.output_frame = np.zeros((self.nr_of_chan,self._output_frame_len))

    def generate_sine_output_frame(self):
        """
        NOTE: all generate_ methods should be as concise and performing as possible
        """
        self.output_frame = np.sin(self.angles_rad)*self.amplitude
        self.angles_rad += self.phase_reminder

    def generate_square_output_frame(self):
        self.output_frame = sig.square(self.angles_rad)*self.amplitude*0.7
        self.apply_aa_filter_on_output_signal()
        self.angles_rad += self.phase_reminder

    def generate_sweep_output_frame(self):
        self.output_frame = self.amplitude*sig.chirp(self.time_base, f0=self.freq0, f1=self.freq1,
                                                     t1=self.__frame_len_sec, method='linear', phi=90)

    def generate_random_output_frame(self):
        """
        Generates a signal consisting of random values.
        Note that unless filtered this signal can not be considered white noise!
        """
        self.output_frame = 2*self.amplitude*(np.random.rand(self.nr_of_chan, self._output_frame_len) - 0.5)

    def generate_white_noise_output_frame(self):
        """
        Generates true white noise by band-limiting the random values
        """
        self.output_frame = 1.8*self.amplitude * (np.random.rand(self.nr_of_chan, self._output_frame_len) - 0.5)
        self.apply_aa_filter_on_output_signal()

    def generate_impulse_output_frame(self):
        self.output_frame = sig.unit_impulse((self.nr_of_chan, self._output_frame_len),
                                             idx=(np.arange(self.nr_of_chan), self._output_frame_len // 2))
        self.apply_aa_filter_on_output_signal()
        self.output_frame = rescale(self.output_frame, self.amplitude)

    def generate_ess_output_frame(self):
        """
        Subsequently generates exponential sweep sine and inverse decay sweep signals. After acquisition the two
        signals can be convoluted to generate impulse response and frequency response of the system. Reference:
        https://theaudioprogrammer.com/signal-analysis-ii-linear-vs-logarithmic-sine-sweep/ NOTE: Change to signal
        length or sampling rate in this mode causes ess method to reset to prevent generating two frames of unmatched
        sample size. For meaningful application you must call this method twice without any intermittent changes
        to the signal properties.
        """
        if self.freq0 is None or self.freq1 == self.freq0:
            raise FrequencyRangeIncorrect('ESS method requires a correct frequency range to be set. '
                                          'freq0 is {}Hz and freq1 is {}Hz'.format(self.freq0, self.freq1))

        if not self.output_is_exp_sweep_sine:
            self.sweep_rate = np.log(self.freq1 / self.freq0)
            self.exp_sweep_sine = self.amplitude * np.sin(
                (2 * np.pi * self.freq0 * self.__frame_len_sec / self.sweep_rate) * (
                            np.exp(self.time_base * self.sweep_rate / self.__frame_len_sec) - 1))
            self.output_frame = self.exp_sweep_sine
            self.output_is_exp_sweep_sine = True
        else:
            k = np.exp(self.time_base * self.sweep_rate / self.__frame_len_sec)
            inverse_decay_sweep = np.fliplr(self.exp_sweep_sine) / k
            self.output_frame = inverse_decay_sweep
            self.output_is_exp_sweep_sine = False

    def set_aa_filter(self, wn=None, order=None):
        """
        :param <float> wn: coefficient determining the cutoff of the AA pre-equalization filter. Set to 1 equals fs/2 cutoff.
        :param <int> order: defines filter order
        """
        if wn is not None:
            self.AA_FILTER_NYQUIST_COEF = wn
            self.__set_filter_coefs()
        if order is not None:
            self.AA_FILTER_ORDER = order
            self.__set_filter_coefs()
            self.__reset_filter_initial_condition()

    def set_amplitude(self, amplitude):
        """
        :param <float> amplitude:
        """
        with self.property_lock:
            self.amplitude = amplitude
            #self.output_is_exp_sweep_sine = False

    def set_frequency(self, frequency):
        """
        :param <float> frequency:
        """
        with self.property_lock:
            self.freq1 = frequency
            self.__set_angles_rad()
            self.__set_phase_reminder()
            self.check_freqs() # TODO Eh? why is this last?
            #self.output_is_exp_sweep_sine = False # TODO code repetition

    def set_start_frequency(self, frequency):
        """
        :param <float> frequency:
        """
        with self.property_lock:
            self.freq0 = frequency
            self.__set_angles_rad()
            self.__set_phase_reminder()
            self.check_freqs()
            #self.output_is_exp_sweep_sine = False # TODO code repetition

    def set_sample_rate(self, sample_rate):
        """
        :param <int> sample_rate: new sample rate in Hz
        """
        with self.property_lock:
            self.sample_rate = sample_rate
            self.__set_filter_coefs()
            self.__set_timebase()
            self.__set_angles_rad()  # TODO code repetition
            self.__set_phase_reminder()  # TODO code repetition
            self.__set_frame_len_sec()  # TODO code repetition
            self.output_is_exp_sweep_sine = False  # TODO code repetition

    def set_output_frame_len(self, frame_len):
        """
        :param <int> frame_len: new frame length in samples
        """
        with self.property_lock:
            self._output_frame_len = frame_len
            self.__set_timebase()  # TODO code repetition
            self.__set_angles_rad()  # TODO code repetition
            self.__set_phase_reminder()  # TODO code repetition
            self.__set_frame_len_sec()  # TODO code repetition
            self.output_is_exp_sweep_sine = False # TODO code repetition

    def set_function(self, fcn_name):
        """
        :param <str> fcn_name: name of the required function
        """

        if fcn_name == 'zeros':
            self.generate = self.generate_zeros_output_frame
        elif fcn_name == 'sine':
            self.generate = self.generate_sine_output_frame
        elif fcn_name == 'square':
            self.generate = self.generate_square_output_frame
        elif fcn_name == 'random':
            self.generate = self.generate_random_output_frame
        elif fcn_name == 'white noise':
            self.generate = self.generate_white_noise_output_frame
        elif fcn_name == 'sweep':
            self.generate = self.generate_sweep_output_frame
        elif fcn_name == 'ess':
            self.generate = self.generate_ess_output_frame
        elif fcn_name == 'impulse':
            self.generate = self.generate_impulse_output_frame
        else:
            print('Requested type of function not recognized.')  # TODO perhaps better raise a dedicated error here
            return  # keep previous signal type to prevent disrupting current generation

        self.current_function = fcn_name

    def get_function(self):
        return self.current_function

    def reset_phase(self):
        """
        Resets generator function argument back to 0. This method should be called between generate() methods in
        order to repeat the first frame for single shot operation.
        """
        self.phase_reminder = np.float(0)
        self.__set_phase_reminder()
        self.angles_rad = np.zeros((self.nr_of_chan, self._output_frame_len), dtype=np.float64)
        self.__set_angles_rad()

    # TODO Think about set_nr_of_chanels method NOTE: it will have to retrun all private methods starting from __new_timebase to broadcast correct shape of data

    def check_freqs(self):
        nyquist_freq = self.sample_rate/2
        if self.freq0 is not None and self.freq0 > nyquist_freq:
            raise NyquistFrequencyExceeded('Selected frequency freq0: {}Hz exceeds'
                                           ' the Nyquist frequency at '
                                           '{} Hz sampling rate.'.format(self.freq0, self.sample_rate))
        if self.freq1 > nyquist_freq:
            raise NyquistFrequencyExceeded('Selected frequency freq1: {}Hz exceeds'
                                           ' the Nyquist frequency at '
                                           '{} Hz sampling rate.'.format(self.freq1, self.sample_rate))

    def next_chunk(self):
        # This is to support non blocking write on pyAudio for finite measurement
        # this "emulates" wave file reading
        frame_complete = False
        chunk_end = self.chunk_start + self.chunk_len

        if chunk_end < self._output_frame_len:
            chunk = self.output_frame[:, self.chunk_start:chunk_end]
        elif chunk_end >= self._output_frame_len:
            chunk = self.output_frame[:, self.chunk_start:]
            frame_complete = True # this to allow communicating pyaudio.paComplete with the last chunk

        if frame_complete:
            self.chunk_start = 0
        else:
            self.chunk_start += self.chunk_len

        return chunk, frame_complete

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    """
    example use of the signal generator
    """

    # TODO: Develop into nice set of examples using matplotlib and timer to update the plots slowly

    fig, ax = plt.subplots(1, 5)

    mygen = FunctionGenerator(output_frame_len=2333, sample_rate=44000, freq1=133, freq0=100, nr_of_chan=2)

    mygen.generate_sine_output_frame()
    #ax[0].plot(mygen.time_base.T, mygen.output_frame.T, marker=11)

    complete = False
    i = 0
    while not complete:
        chunk, complete = mygen.next_chunk()
        print(chunk.shape)
        ax[i].plot(chunk.T, marker=11)
        print(complete)
        i += 1

    plt.show()

    #ax[1].plot(mygen.time_base.T, mygen.output_frame.T, marker=11)

    #pass

    #ax[2].plot(mygen.time_base.T, mygen.output_frame.T, marker=11)




