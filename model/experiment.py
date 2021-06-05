from scipy import signal as sig
import numpy as np

from model import config
from model.pftltools import optimal_noverlap, rescale


class Experiment:
    def __init__(self, streaming_device):

        self.streaming_device = streaming_device
        self.streaming_device.function_gen.set_frequency(config.default_frequency)
        self.streaming_device.function_gen.set_start_frequency(config.default_start_frequency)
        self.streaming_device.function_gen.set_function(config.default_function_type)

        # cm experiment properties:
        self._cm_win = None
        self._cm_db_ref = None
        self.cm_freq_base = None
        self.cm_sp_window_type = None
        self.cm_sp_window_size = None
        self.cm_ph_window_size = None
        self.set_cm_sp_window(win_type=config.default_cm_sp_window_type, win_size=config.default_cm_sp_window_size)
        self.set_cm_ph_window(win_size=config.default_cm_ph_window_size)

        # fm experiment properties:
        self.optimal_noverlap = None
        self.fm_sp_window_type = None
        self.fm_sp_window_size = None
        self.set_fm_sp_window(win_type=config.default_fm_sp_window_type, win_size=config.default_fm_sp_window_size)

        self.fm_spg_chan = None
        self.set_fm_spg_chan(config.default_fm_spg_chan)
        self.fm_spg_window_type = None
        self.fm_spg_window_size = None
        self.set_fm_spg_window(win_type=config.default_fm_spg_window_type, win_size=config.default_fm_spg_window_size)

        # arrays for ess measurement data storage
        self.exp_sweep_sine_response = None
        self.inverse_decay_filter_response = None
        self.ess_time_base = None

        self.fm_result_x = None
        self.fm_result_y = None

        self.rot_matrix_45 = self.calculate_rotation_matrix(45)

    def set_cm_sp_window(self, win_type=None, win_size=None):
        if win_type is not None:
            self.cm_sp_window_type = win_type
        if win_size is not None:
            if win_size < self.streaming_device.input_frame_len:
                raise ValueError('>> EXCEPTION << Win size < input frame len.')
            self.cm_sp_window_size = win_size
            self.streaming_device.set_monitor(win_size)
            self.set_cm_freq_base()

        self._cm_win = sig.get_window(self.cm_sp_window_type, self.cm_sp_window_size)
        self._cm_db_ref = np.sum(self._cm_win) * self.streaming_device.ai_max_val

    def set_cm_ph_window(self, win_size=None):
        self.cm_ph_window_size = win_size
        self.streaming_device.set_monitor(win_size)

    def set_cm_freq_base(self):
        self.cm_freq_base = np.fft.rfftfreq(self.cm_sp_window_size,
                                            d=1 / self.streaming_device.ai_fs)

    def set_fm_sp_window(self, win_type=None, win_size=None):
        if win_type is not None:
            self.fm_sp_window_type = win_type
        if win_size is not None:
            self.fm_sp_window_size = win_size

        self.optimal_noverlap = optimal_noverlap(self.fm_sp_window_type, self.fm_sp_window_size)

    def set_fm_spg_chan(self, chan):
        self.fm_spg_chan = chan

    def set_fm_spg_window(self, win_type=None, win_size=None):
        if win_type is not None:
            self.fm_spg_window_type = win_type
        if win_size is not None:
            self.fm_spg_window_size = win_size

    # def set_mode_to_finite(self, meas_time_sec):
    #     self.streaming_device.set_mode_to_finite(meas_time_sec)
    #     self.start_experiment = self.start_fm_experiment
    #
    # def set_mode_to_continuous(self):
    #     self.streaming_device.set_mode_to_continuous()
    #     self.start_experiment = self.start_cm_experiment

    # def start_cm_experiment(self):
    #     if self.streaming_device.get_mode() != 'continuous':
    #         self.streaming_device.set_mode_to_continuous()
    #         print('issue #109 track here')
    #
    #     self.streaming_device.io_start()

    def start_fm_experiment(self):
        if self.streaming_device.get_mode() != 'finite':
            print('\nOverriding streaming device mode to finite.\n')
            self.streaming_device.set_mode_to_finite(self.streaming_device.finite_frame_len_sec)

        self.streaming_device.io_start()
        self.fm_result_x = self.streaming_device.input_time_base[0] #TODO so what was the point of making this into a 2xN matrix
        self.fm_result_y = self.streaming_device.input_frame

    def start_fm_ess_experiment(self):
        if self.streaming_device.get_mode() != 'finite':
            print('\nOverriding streaming device mode to finite.\n')
            self.streaming_device.set_mode_to_finite(self.streaming_device.finite_frame_len_sec)

        if self.streaming_device.function_gen.get_function() != 'ess':
            self.streaming_device.function_gen.set_function('ess')

        self.streaming_device.io_start()
        self.exp_sweep_sine_response = self.streaming_device.input_frame
        self.streaming_device.io_start()
        self.inverse_decay_filter_response = self.streaming_device.input_frame
        self.fm_result_x = np.arange(self.streaming_device.input_frame_len * 2,
                                     dtype=np.float64) / self.streaming_device.ai_fs
        self.fm_result_y = np.concatenate((self.exp_sweep_sine_response,
                                           self.inverse_decay_filter_response), axis=1)

    def calculate_cm_fft(self):
        input_frame = self.streaming_device.get_monitor() * self._cm_win
        cm_fft = np.fft.rfft(input_frame)
        cm_fft = np.abs(cm_fft) * 2 / self._cm_db_ref
        cm_db_fft = 20 * np.log10(cm_fft)

        return cm_db_fft

    def calculate_rotation_matrix(self, angle_deg):
        beta_rad = np.pi * angle_deg/180
        rot_matrix = np.array([[np.cos(beta_rad), -np.sin(beta_rad)],
                               [np.sin(beta_rad),  np.cos(beta_rad)]])

        return rot_matrix

    def calculate_cm_phase(self):
        input_frame = self.streaming_device.get_monitor()
        phase_frame = np.dot(self.rot_matrix_45, np.flipud(input_frame))
        return phase_frame.T

    def calculate_fm_fft(self, input_frame=None):
        if input_frame is None:
            input_frame = self.fm_result_y

        if self.fm_sp_window_size > max(self.fm_result_y.shape):
            raise ValueError(f'>> EXCEPTION << : FFT window {self.fm_sp_window_size}'
                             f' pts > input frame length {self.fm_result_y.shape} pts.')

        fm_freq_base, fm_fft_segment_time, fm_fft = sig.stft(input_frame,
                                                             fs=self.streaming_device.ai_fs,
                                                             window=self.fm_sp_window_type,
                                                             nperseg=self.fm_sp_window_size,
                                                             noverlap=self.optimal_noverlap,
                                                             padded=None, boundary=None)

        fm_db_fft = np.average(2 * np.abs(fm_fft), axis=2)
        fm_db_fft = 20 * np.log10(fm_db_fft / self.streaming_device.ai_max_val)

        return fm_freq_base, fm_fft_segment_time, fm_db_fft

    def calculate_ir_from_fm_ess(self):
        ess_impulse_response = sig.fftconvolve(self.exp_sweep_sine_response,
                                               self.inverse_decay_filter_response,
                                               mode='same',
                                               axes=1)
        ess_impulse_response = rescale(ess_impulse_response,
                                       self.streaming_device.ai_max_val)  # TODO On this line relative scaling between measurements is different. We should be rescaling by some calculated constant, not to HW limit)
        delta_f = self.streaming_device.function_gen.freq1 - self.streaming_device.function_gen.freq0
        db_ref = np.sqrt(self.streaming_device.finite_frame_len_sec / (2 * delta_f))  # TODO this is not exact
        ess_impulse_response /= db_ref
        return ess_impulse_response

    def calculate_fm_spectrogram(self):

        if self.fm_spg_window_size > max(self.fm_result_y.shape):
            raise ValueError(f'>> EXCEPTION << : FFT window {self.fm_sp_window_size}'
                             f' pts > input frame length {max(self.fm_result_y.shape)} pts.')

        f, t, spg = sig.spectrogram(np.array(self.fm_result_y[self.fm_spg_chan]),
                                    fs=self.streaming_device.ai_fs,
                                    window=self.fm_spg_window_type, nperseg=self.fm_spg_window_size,
                                    noverlap=self.fm_spg_window_size-config.spg_window_overlap_skip_pts, scaling='spectrum', mode='magnitude',  #128
                                    detrend=False)

        return f, t, spg


if __name__ == "__main__":
    from model.iostreamingdevice import io_streaming_device_discovery
    from time import sleep

    devices_name_to_model_dict = io_streaming_device_discovery()
    if len(devices_name_to_model_dict) == 0:
        print('No I/O streaming devices were found')
    else:
        device_names = [name for name in devices_name_to_model_dict.keys()]
        for i, device_name in enumerate(device_names):
            print('<' + str(i) + '> ' + device_name + '\n')
        user_dev_index = input("<Select> streaming device from the list.>>")
        user_device_name = device_names[int(user_dev_index)]
        daq_model = devices_name_to_model_dict[user_device_name]
        daq = daq_model(user_device_name)

        print('Instantiating Experiment')
        exp = Experiment(daq)

        exp.streaming_device.set_mode_to_finite(3)
        exp.start_fm_ess_experiment()
        exp.calculate_ir_from_fm_ess()

        exp.streaming_device.set_mode_to_continuous()
        exp.streaming_device.function_gen.set_function('sine')
        exp.streaming_device.io_start()
        sleep(2)
        exp.streaming_device.io_stop()
