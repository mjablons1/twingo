default_start_frequency = 20
default_frequency = 1000
default_function_type = 'sine'

default_monitor_len = 2**15
max_monitor_frame_len = 2**16
default_finite_frame_len_sec = 1

default_cm_sp_window_type = 'blackmanharris'
default_cm_sp_window_size = default_monitor_len

default_fm_sp_window_type = 'blackmanharris'
default_fm_sp_window_size = 2**15

default_cm_ph_window_size = 2**12

default_fm_spg_chan = 0
default_fm_spg_window_type = 'flattop'
default_fm_spg_window_size = 2**11

spg_window_overlap_skip_pts = 128

default_pyAudio_paFloat32_level_range = [1.0, -1.0]

pyaudio_read_offset_msec = 35  # Allows to manually adjust the input and output start synchronization for pyAudio dev.