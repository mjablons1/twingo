import sys
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

from model.experiment import Experiment
from model.iostreamingdevice import io_streaming_device_discovery
from model.pftltools import limit

from view import config
from view.customtheme import apply_dark_theme
from view.GUI_twingo import Ui_MainWindow


class Error(Exception):
    """Base class for other exceptions"""
    pass


class NoInputData(Error):
    """Exception raised when ui triggers analysis before measurement data is available"""
    pass


class LimitExceeded(Error):
    """Exception raised when input value exceeds expected limits"""
    pass


class TwingoExec:

    def __init__(self):

        app = QtWidgets.QApplication(sys.argv)
        apply_dark_theme(app)

        MainWindow = QtWidgets.QMainWindow()

        self.mutex = QtCore.QMutex()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)

        self.SETTINGS_TAB_INDEX = 0
        self.CONTINUOUS_MEAS_TAB_INDEX = 1
        self.CM_TIMESERIES_TAB_INDEX = 0
        self.CM_SPECTRUM_TAB_INDEX = 1
        self.CM_PHASE_TAB_INDEX = 2

        self.FINITE_MEAS_TAB_INDEX = 2
        self.FM_TIMESERIES_TAB_INDEX = 0
        self.FM_SPECTRUM_TAB_INDEX = 1
        self.FM_SPECTROGRAM_TAB_INDEX = 2

        self.ui.tabWidget_main.setCurrentIndex(self.SETTINGS_TAB_INDEX)
        self.ui.tabWidget_fm.setCurrentIndex(self.FM_TIMESERIES_TAB_INDEX)
        self.ui.tabWidget_cm.setCurrentIndex(self.CM_TIMESERIES_TAB_INDEX)

        self.ui.tabWidget_main.setTabEnabled(self.FINITE_MEAS_TAB_INDEX, False)
        self.ui.tabWidget_main.setTabEnabled(self.CONTINUOUS_MEAS_TAB_INDEX, False)

        if config.NR_OF_CHANNELS_TO_PLOT < 2:
            self.ui.tabWidget_cm.setTabEnabled(self.CM_PHASE_TAB_INDEX, False)

        # plot widgets:
        self.cm_tm_plot_widget = None
        self.cm_sp_plot_widget = None
        self.cm_ph_plot_widget = None
        self.fm_tm_plot_widget = None
        self.fm_sp_plot_widget = None
        self.fm_spg_widget = None
        self.graphics_plot = None
        self.img = None
        self.hist = None

        # plot hold functionality
        self.fm_tm_plot_data_items = []
        self.fm_sp_plot_data_items = []

        self.fm_tm_plot_hold_data_items = []
        self.fm_sp_plot_hold_data_items = []

        self.cm_tm_plot_data_items = []
        self.cm_sp_plot_data_items = []
        self.cm_ph_plot_data_item = None
        self.cm_ph_plot_box_data_item = []

        self.cm_sp_plot_hold_data_items = []
        self.cm_tm_plot_hold_data_items = []
        self.cm_ph_plot_hold_data_items = []

        self.pen_case = \
            [pg.mkPen(color=chan_index + config.PLOT_COLOR_MODIFIER_INT) for chan_index in
             range(config.NR_OF_CHANNELS_TO_PLOT)]

        self.hold_pen_case = \
            [pg.mkPen(color=chan_index + config.PLOT_COLOR_MODIFIER_INT, style=QtCore.Qt.DotLine) for chan_index in
             range(config.NR_OF_CHANNELS_TO_PLOT)]

        self.place_fm_tm_graph()
        self.place_fm_sp_graph()
        self.place_fm_spg_graphics()
        self.place_cm_tm_graph()
        self.place_cm_sp_graph()
        self.place_cm_ph_graph()
        self.connect_gui_signals()

        # paint the background of hold items with the same color as pen assigned to hold plots:
        hold_checkbox_style_sheet_strings = \
            [f'color: rgb(0, 0, 0); background-color:rgb{self.pen_case[chan_index].color().getRgb()}' for
             chan_index in range(config.NR_OF_CHANNELS_TO_PLOT)]

        try:
            self.ui.checkBox_fm_tm_hold_a.setStyleSheet(hold_checkbox_style_sheet_strings[0])
            self.ui.checkBox_fm_sp_hold_a.setStyleSheet(hold_checkbox_style_sheet_strings[0])
            self.ui.checkBox_cm_tm_hold_a.setStyleSheet(hold_checkbox_style_sheet_strings[0])
            self.ui.checkBox_cm_sp_hold_a.setStyleSheet(hold_checkbox_style_sheet_strings[0])
            self.ui.checkBox_fm_tm_hold_b.setStyleSheet(hold_checkbox_style_sheet_strings[1])
            self.ui.checkBox_fm_sp_hold_b.setStyleSheet(hold_checkbox_style_sheet_strings[1])
            self.ui.checkBox_cm_tm_hold_b.setStyleSheet(hold_checkbox_style_sheet_strings[1])
            self.ui.checkBox_cm_sp_hold_b.setStyleSheet(hold_checkbox_style_sheet_strings[1])
        except IndexError as exc:
            print(exc)

        self.buffer_indicator_timer = QtCore.QTimer()
        self.buffer_indicator_timer.timeout.connect(self.update_output_buffer_indicator)

        # self.timeseries_timer = QtCore.QTimer()
        # self.timeseries_timer.timeout.connect(self.update_cm_timeseries_plot)

        self.devices_name_to_model = None
        self.e = None

        MainWindow.show()
        sys.exit(app.exec_())

    def connect_gui_signals(self):
        self.ui.tabWidget_main.currentChanged.connect(self.on_tabWidget_main_changed)

        # settings tab:
        self.ui.pushButton_detectDAQ.clicked.connect(self.on_pushButton_detectDAQ_clicked)
        self.ui.comboBox_dev_list.activated.connect(self.on_comboBox_dev_list_activated)
        self.ui.comboBox_ai_a.activated.connect(self.on_comboBox_ai_ab_activated)
        self.ui.comboBox_ai_b.activated.connect(self.on_comboBox_ai_ab_activated)
        self.ui.comboBox_ao_a.activated.connect(self.on_comboBox_ao_ab_activated)
        self.ui.comboBox_ao_b.activated.connect(self.on_comboBox_ao_ab_activated)
        self.ui.comboBox_ai_terminal_cfg.activated.connect(self.on_comboBox_ai_terminal_cfg_activated)
        self.ui.comboBox_ai_max.activated.connect(self.on_comboBox_ai_min_max_activated)
        self.ui.comboBox_ai_min.activated.connect(self.on_comboBox_ai_min_max_activated)
        self.ui.comboBox_ao_max.activated.connect(self.on_comboBox_ao_min_max_activated)
        self.ui.comboBox_ao_min.activated.connect(self.on_comboBox_ao_min_max_activated)
        self.ui.comboBox_ai_fs.activated.connect(self.on_comboBox_ai_fs_activated)
        self.ui.comboBox_ao_fs.activated.connect(self.on_comboBox_ao_fs_activated)
        self.ui.comboBox_ao_max.activated.connect(self.on_comboBox_ao_max_activated)

        # common widgets
        self.ui.comboBox_sig_len.activated.connect(self.on_comboBox_sig_len_activated)
        self.ui.comboBox_out_sig_type.activated.connect(self.on_comboBox_out_sig_activated)
        self.ui.lineEdit_min_f.editingFinished.connect(self.on_lineEdit_min_f_editingFinished)
        self.ui.lineEdit_max_f.editingFinished.connect(self.on_lineEdit_max_f_editingFinished)
        self.ui.pushButton_start.clicked.connect(self.on_pushButton_start_clicked)
        self.ui.pushButton_stop.clicked.connect(self.on_pushButton_stop_clicked)

        # finite measurement widgets
        self.ui.tabWidget_fm.currentChanged.connect(self.update_current_fm_plot)
        # timeseries
        self.ui.checkBox_fm_tm_hold_a.toggled.connect(self.on_checkBox_fm_tm_hold_a_toggled)
        self.ui.checkBox_fm_tm_hold_b.toggled.connect(self.on_checkBox_fm_tm_hold_b_toggled)
        # spectrum
        self.ui.comboBox_fm_window_size.activated.connect(self.on_comboBox_fm_window_size_activated)
        self.ui.comboBox_fm_window_type.activated.connect(self.on_comboBox_fm_window_type_activated)
        self.ui.checkBox_fm_sp_hold_a.toggled.connect(self.on_checkBox_fm_sp_hold_a_toggled)
        self.ui.checkBox_fm_sp_hold_b.toggled.connect(self.on_checkBox_fm_sp_hold_b_toggled)
        # spectrogram
        self.ui.checkBox_fm_spg_log_amp.toggled.connect(self.update_fm_spg)
        self.ui.comboBox_fm_spg_window_size.activated.connect(self.on_comboBox_fm_spg_window_size_activated)
        self.ui.comboBox_fm_spg_chan.activated.connect(self.on_comboBox_fm_spg_chan_activated)
        self.ui.comboBox_fm_spg_window_type.activated.connect(self.on_comboBox_fm_spg_window_type_activated)

        # continuous measurement widgets
        self.ui.tabWidget_cm.currentChanged.connect(self.on_tabWidget_cm_changed)
        # timeseries
        self.ui.checkBox_cm_tm_hold_a.toggled.connect(self.on_checkBox_cm_tm_hold_a_toggled)
        self.ui.checkBox_cm_tm_hold_b.toggled.connect(self.on_checkBox_cm_tm_hold_b_toggled)
        # spectrum
        self.ui.comboBox_cm_window_type.activated.connect(self.on_comboBox_cm_window_type_activated)
        self.ui.comboBox_cm_window_size.activated.connect(self.on_comboBox_cm_window_size_activated)
        self.ui.checkBox_cm_sp_hold_a.toggled.connect(self.on_checkBox_cm_sp_hold_a_toggled)
        self.ui.checkBox_cm_sp_hold_b.toggled.connect(self.on_checkBox_cm_sp_hold_b_toggled)
        # phase
        self.ui.comboBox_cm_ph_window_size.activated.connect(self.on_comboBox_cm_ph_window_size_activated)
        self.ui.checkBox_cm_ph_hold_1.toggled.connect(self.on_checkBox_cm_ph_hold_1_toggled)
        self.ui.checkBox_cm_ph_hold_2.toggled.connect(self.on_checkBox_cm_ph_hold_2_toggled)

        # common
        self.ui.dial_cm_vfine_freq.valueChanged.connect(self.on_dial_value_changed)
        self.ui.dial_cm_fine_freq.valueChanged.connect(self.on_dial_value_changed)
        self.ui.dial_cm_coarse_freq.valueChanged.connect(self.on_dial_value_changed)

    def get_dials_value(self):
        return self.ui.dial_cm_vfine_freq.value() + \
               self.ui.dial_cm_fine_freq.value() + \
               self.ui.dial_cm_coarse_freq.value()

    def on_comboBox_sig_len_activated(self):
        self.e.streaming_device.set_mode_to_finite(float(self.ui.comboBox_sig_len.currentText()))

    def on_dial_value_changed(self):
        new_value = self.get_dials_value()
        self.e.streaming_device.function_gen.set_frequency(new_value)
        self.ui.lineEdit_max_f.setText(str(new_value))
        self.ui.lcdNumber.display(new_value)

    def on_comboBox_fm_window_type_activated(self):
        self.e.set_fm_sp_window(win_type=self.ui.comboBox_fm_window_type.currentText())
        self.update_current_fm_plot()

    def on_comboBox_fm_window_size_activated(self):
        self.e.set_fm_sp_window(win_size=int(self.ui.comboBox_fm_window_size.currentText()))
        self.update_current_fm_plot()

    def on_comboBox_cm_window_type_activated(self):
        self.e.set_cm_sp_window(win_type=self.ui.comboBox_cm_window_type.currentText())

    def on_comboBox_cm_window_size_activated(self):
        self.e.set_cm_sp_window(win_size=int(self.ui.comboBox_cm_window_size.currentText()))

    def on_comboBox_cm_ph_window_size_activated(self):
        self.e.set_cm_ph_window(win_size=int(self.ui.comboBox_cm_ph_window_size.currentText()))

    def on_comboBox_fm_spg_chan_activated(self):
        self.e.set_fm_spg_chan(self.ui.comboBox_fm_spg_chan.currentIndex())
        self.update_fm_spg()

    def on_comboBox_fm_spg_window_type_activated(self):
        self.e.set_fm_spg_window(win_type=self.ui.comboBox_fm_spg_window_type.currentText())
        self.update_fm_spg()

    def on_comboBox_fm_spg_window_size_activated(self):
        self.e.set_fm_spg_window(win_size=int(self.ui.comboBox_fm_spg_window_size.currentText()))
        self.update_fm_spg()

    def on_comboBox_ai_ab_activated(self):
        # TODO some callbacks are coded in the uic generated file already and they can theoretically create race
        #  condition with the checks below. Its probably a good idea to remove the callbacks from Designer and hard
        #  code them here by creating separate, dedicated  "on_comboBox_index_changed" callbacks for each  that
        #  contain both functionalities without allowing race.
        self.e.streaming_device.ai_a_name = self.ui.comboBox_ai_a.currentText()
        self.e.streaming_device.ai_b_name = self.ui.comboBox_ai_b.currentText()
        self.print_qt(f'Input channels : A:{self.e.streaming_device.ai_a_name} B:{self.e.streaming_device.ai_b_name}')

    def on_comboBox_ao_ab_activated(self):
        self.e.streaming_device.ao_a_name = self.ui.comboBox_ao_a.currentText()
        self.e.streaming_device.ao_b_name = self.ui.comboBox_ao_b.currentText()
        self.print_qt(f'Output channels : A:{self.e.streaming_device.ao_a_name} B:{self.e.streaming_device.ao_b_name}')

    def on_comboBox_ai_min_max_activated(self):
        self.e.streaming_device.ai_min_val = float(self.ui.comboBox_ai_min.currentText())
        self.e.streaming_device.ai_max_val = float(self.ui.comboBox_ai_max.currentText())
        self.print_qt(f'Input voltage limit min:{self.e.streaming_device.ai_min_val}V,'
                      f' max:{self.e.streaming_device.ai_max_val}V')
        self.set_plot_limits()

    def on_comboBox_ao_min_max_activated(self):
        self.e.streaming_device.ao_min_val = float(self.ui.comboBox_ao_min.currentText())
        self.e.streaming_device.ao_max_val = float(self.ui.comboBox_ao_max.currentText())
        self.print_qt(f'Output voltage limit min:{self.e.streaming_device.ao_min_val}V,'
                      f' max:{self.e.streaming_device.ao_max_val}V')

    def on_comboBox_ai_terminal_cfg_activated(self):
        self.e.streaming_device.ai_terminal_config = self.ui.comboBox_ai_terminal_cfg.currentText()
        self.print_qt(f'Ai terminal config: {self.e.streaming_device.ai_terminal_config}')

    def print_qt(self, message, suppress_console=True):
        if suppress_console is False:
            print(message)
        self.ui.statusbar.showMessage(message.__str__(), config.STATUS_BAR_MESSAGE_TIME_MS)

    def check_fm_data_present(self):
        if self.e.fm_result_y is None:
            raise NoInputData('No finite input data is available. Press START to run a measurement.')

    def toggle_plot_hold(self, hold_check_box, plot_data_item, plot_hold_data_item):
        """
        Generic plot hold toggle functionality. If hold_check_box state isChechked() the plot_data_item X and Y data
        will be set to plot_hold_data_item, else plot_hold_data_item will be cleared.

        :param hold_check_box: checkBox qt object whose state is to be checked.
        :param plot_data_item: pyqtgraph data_item object with the plot to be held
        :param plot_hold_data_item: pyqtgraph data_item object which hold plot data is to be set or cleared
        :return: None
        """
        if plot_data_item.xData is None or plot_data_item.yData is None:
            self.print_qt("No data to hold")
            return

        if hold_check_box.isChecked():
            plot_hold_data_item.setData(plot_data_item.xData, plot_data_item.yData)
        else:
            plot_hold_data_item.clear()

    def on_checkBox_cm_tm_hold_a_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_cm_tm_hold_a,
                              self.cm_tm_plot_data_items[0],
                              self.cm_tm_plot_hold_data_items[0])

    def on_checkBox_cm_tm_hold_b_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_cm_tm_hold_b,
                              self.cm_tm_plot_data_items[1],
                              self.cm_tm_plot_hold_data_items[1])

    def on_checkBox_cm_sp_hold_a_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_cm_sp_hold_a,
                              self.cm_sp_plot_data_items[0],
                              self.cm_sp_plot_hold_data_items[0])

    def on_checkBox_cm_sp_hold_b_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_cm_sp_hold_b,
                              self.cm_sp_plot_data_items[1],
                              self.cm_sp_plot_hold_data_items[1])

    def on_checkBox_cm_ph_hold_1_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_cm_ph_hold_1,
                              self.cm_ph_plot_data_item,
                              self.cm_ph_plot_hold_data_items[0])

    def on_checkBox_cm_ph_hold_2_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_cm_ph_hold_2,
                              self.cm_ph_plot_data_item,
                              self.cm_ph_plot_hold_data_items[1])

    def on_checkBox_fm_tm_hold_a_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_fm_tm_hold_a,
                              self.fm_tm_plot_data_items[0],
                              self.fm_tm_plot_hold_data_items[0])

    def on_checkBox_fm_tm_hold_b_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_fm_tm_hold_b,
                              self.fm_tm_plot_data_items[1],
                              self.fm_tm_plot_hold_data_items[1])

    def on_checkBox_fm_sp_hold_a_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_fm_sp_hold_a,
                              self.fm_sp_plot_data_items[0],
                              self.fm_sp_plot_hold_data_items[0])

    def on_checkBox_fm_sp_hold_b_toggled(self):
        self.toggle_plot_hold(self.ui.checkBox_fm_sp_hold_b,
                              self.fm_sp_plot_data_items[1],
                              self.fm_sp_plot_hold_data_items[1])

    def check_set_line_edit_number(self, line_edit_caller, casting_class, limits=None, num_on_exc=None):
        """
        Checks, and, if necessary, corrects user QLineEdit.text() entry.

        :param line_edit_caller: QLineEdit object we wish to check/correct .text() of.
        :param casting_class: One of the numeric built-in classes (int, float ...)
        that will be used to test if text can be casted to a specific numeric type.
        :param limits: (tuple) (min, max) limits for user input. If exceeded user input will be clipped
        to either limit respectively.
        :param num_on_exc: Value which will overwrite user input in case it can not be
        casted to a specified numeric type.
        :return: numeric value corresponding to accepted/corrected QlineEdit.text().
        """
        # override the number to return on exception in case its supplied out of the limits required.
        # It can happen if the previous value is used equal to the value in lineEdit from before callback while
        # meantime a property determining the limits was changed (e.g. fs was reduced so max allowed freq is now
        # less than what is entered into freq0 field). In this case defaulting to the previous value on
        # exception is not ok. Therefore num_on_exc has got to be fixed to the closest limit...
        if limits is not None:
            num_on_exc = limit(num_on_exc, limits)

        try:
            num = casting_class(line_edit_caller.text())

            if limits is not None:  # TODO ...but it leads to code repetition which makes me wanna cry
                num_within_limit = limit(num, limits)
                if num_within_limit != num:
                    raise LimitExceeded(f'Value {num} exceeded limits: {limits[0]}-{limits[1]}.')

        except ValueError as err:
            print(err)
            self.print_qt(f'Input must be numeric {casting_class}.')
            line_edit_caller.setText(str(num_on_exc))
            return num_on_exc

        except LimitExceeded as err:
            self.print_qt(err)
            line_edit_caller.setText(str(num_within_limit))
            return num_within_limit

        return num

    def on_comboBox_ao_fs_activated(self):
        self.e.streaming_device.set_ao_fs(float(self.ui.comboBox_ao_fs.currentText()))
        self.print_qt(f'Output sampling rate changed: {self.e.streaming_device.ao_fs}Hz.')
        # recalculate new max for coarse dial and realign all dials
        new_max_dial_rng = self.e.streaming_device.ao_fs // 2 - self.ui.dial_cm_fine_freq.maximum() - \
                           self.ui.dial_cm_vfine_freq.maximum()
        new_max_dial_rng = limit(new_max_dial_rng, (0, config.MAX_COARSE_FREQ_DIAL_LIMIT))
        self.ui.dial_cm_coarse_freq.setRange(0, new_max_dial_rng)
        self.align_dials_position()

    def on_comboBox_ai_fs_activated(self):
        self.e.streaming_device.set_ai_fs(float(self.ui.comboBox_ai_fs.currentText()))
        self.print_qt(f'Input sampling rate changed: {self.e.streaming_device.ai_fs}Hz.')
        self.e.set_cm_freq_base()

    def on_lineEdit_min_f_editingFinished(self):
        prev_value = self.e.streaming_device.function_gen.freq0
        min_value = config.MIN_OUTPUT_FREQUENCY_ALLOWED
        max_value = self.e.streaming_device.ao_fs / 2
        new_value = self.check_set_line_edit_number(self.ui.lineEdit_min_f, float,
                                                    limits=(min_value, max_value),
                                                    num_on_exc=prev_value)
        if new_value == prev_value:
            return
        self.e.streaming_device.function_gen.set_start_frequency(new_value)
        self.print_qt(f'Start frequency changed: {self.e.streaming_device.function_gen.freq0}Hz.')

    def on_lineEdit_max_f_editingFinished(self):
        prev_value = self.e.streaming_device.function_gen.freq1
        min_value = config.MIN_OUTPUT_FREQUENCY_ALLOWED
        max_value = self.e.streaming_device.ao_fs / 2
        new_value = self.check_set_line_edit_number(self.ui.lineEdit_max_f, float,
                                                    limits=(min_value, max_value),
                                                    num_on_exc=prev_value)
        if new_value == prev_value:
            return
        self.e.streaming_device.function_gen.set_frequency(new_value)
        self.print_qt(f'Set/End frequency changed: {self.e.streaming_device.function_gen.freq1}Hz.')
        self.ui.lcdNumber.display(new_value)
        # ensure dials are in consistent position with the current set freq. and allow exploration in its vicinity
        self.align_dials_position()

    def align_dials_position(self):
        new_value = self.e.streaming_device.function_gen.freq1
        self.ui.dial_cm_vfine_freq.blockSignals(True)
        self.ui.dial_cm_fine_freq.blockSignals(True)
        self.ui.dial_cm_coarse_freq.blockSignals(True)
        if new_value <= self.ui.dial_cm_vfine_freq.maximum() // 2:
            self.ui.dial_cm_vfine_freq.setValue(new_value)
            self.ui.dial_cm_fine_freq.setValue(0)
            self.ui.dial_cm_coarse_freq.setValue(0)
        elif new_value <= self.ui.dial_cm_fine_freq.maximum() // 2:
            self.ui.dial_cm_vfine_freq.setValue(self.ui.dial_cm_vfine_freq.maximum() // 2)
            self.ui.dial_cm_fine_freq.setValue(new_value - self.ui.dial_cm_vfine_freq.value())
            self.ui.dial_cm_coarse_freq.setValue(0)
        elif new_value <= self.ui.dial_cm_coarse_freq.maximum():
            self.ui.dial_cm_vfine_freq.setValue(self.ui.dial_cm_vfine_freq.maximum() // 2)
            self.ui.dial_cm_fine_freq.setValue(self.ui.dial_cm_fine_freq.maximum() // 2)
            self.ui.dial_cm_coarse_freq.setValue(new_value - self.ui.dial_cm_vfine_freq.value()
                                                 - self.ui.dial_cm_fine_freq.value())
        elif new_value > self.ui.dial_cm_coarse_freq.maximum():
            self.ui.dial_cm_vfine_freq.setValue(self.ui.dial_cm_vfine_freq.maximum() // 2)
            self.ui.dial_cm_fine_freq.setValue(self.ui.dial_cm_fine_freq.maximum() // 2)
            self.ui.dial_cm_coarse_freq.setValue(self.ui.dial_cm_coarse_freq.maximum())

        self.ui.dial_cm_vfine_freq.blockSignals(False)
        self.ui.dial_cm_fine_freq.blockSignals(False)
        self.ui.dial_cm_coarse_freq.blockSignals(False)

    def on_comboBox_ao_max_activated(self):
        self.e.streaming_device.function_gen.set_amplitude(float(self.ui.comboBox_ao_max.currentText()))
        print(f'Output signal amplitude changed to {self.e.streaming_device.function_gen.amplitude}V.')

    def on_comboBox_out_sig_activated(self):
        self.e.streaming_device.function_gen.set_function(self.ui.comboBox_out_sig_type.currentText())

    def on_tabWidget_main_changed(self):
        if self.ui.tabWidget_main.currentIndex() == self.FINITE_MEAS_TAB_INDEX:
            self.e.streaming_device.set_mode_to_finite(float(self.ui.comboBox_sig_len.currentText()))
            self.ui.comboBox_sig_len.setEnabled(True)
        elif self.ui.tabWidget_main.currentIndex() == self.CONTINUOUS_MEAS_TAB_INDEX:
            self.e.streaming_device.set_mode_to_continuous()
            self.ui.comboBox_sig_len.setEnabled(False)
        elif self.ui.tabWidget_main.currentIndex() == self.SETTINGS_TAB_INDEX:
            pass
        else:
            # reminder for future expansion of tabs
            self.print_qt('WARNING: This tab does not define a specific streaming device mode')

    def update_current_fm_plot(self):
        try:
            if self.ui.tabWidget_fm.currentIndex() == self.FM_TIMESERIES_TAB_INDEX:
                self.update_fm_tm_plot()
            elif self.ui.tabWidget_fm.currentIndex() == self.FM_SPECTRUM_TAB_INDEX:
                self.update_fm_sp_plot()
            elif self.ui.tabWidget_fm.currentIndex() == self.FM_SPECTROGRAM_TAB_INDEX:
                self.update_fm_spg()
            else:
                pass
        except NoInputData as exc:
            self.print_qt(exc)

    def on_pushButton_detectDAQ_clicked(self):
        self.devices_name_to_model = io_streaming_device_discovery()
        self.ui.comboBox_dev_list.clear()
        for name in self.devices_name_to_model:
            self.ui.comboBox_dev_list.addItem(name)

        self.ui.comboBox_dev_list.showPopup()

        if self.ui.comboBox_dev_list.count() == 1:  # if only one device found use it by default
            self.on_comboBox_dev_list_activated()

        if self.ui.comboBox_dev_list.count() == 0:
            self.print_qt("No DAQ devices found and/or supporting packages (pyaudio / nidaqmx) are not installed.",
                          suppress_console=False)

    def on_comboBox_dev_list_activated(self):
        device_name = self.ui.comboBox_dev_list.currentText()
        self.print_qt(f'Device selected: {device_name}')
        streaming_device_model = self.devices_name_to_model[device_name]
        self.e = Experiment(streaming_device_model(device_name))
        self.configure_gui()

    def set_plot_limits(self):

        x_min = self.e.streaming_device.ai_min_val
        x_max = self.e.streaming_device.ai_max_val
        y_min = self.e.streaming_device.ai_min_val
        y_max = self.e.streaming_device.ai_max_val

        plot_items = [self.fm_tm_plot_widget.getPlotItem(),
                      self.cm_tm_plot_widget.getPlotItem()]

        for plot_item in plot_items:
            plot_item.vb.setLimits(yMin=y_min * 1.1, yMax=y_max * 1.1)
            plot_item.setRange(yRange=(y_min * 1.1, y_max * 1.1), disableAutoRange=True)

        phase_box_x = np.array([x_min, x_max, x_max, x_min, x_min]) * config.PHASE_PLOT_OPTIMAL_WIDTH
        phase_box_y = np.array([y_min, y_min, y_max, y_max, y_min])
        self.cm_ph_plot_box_data_item.setData(phase_box_x, phase_box_y)

        cm_ph_plot_item = self.cm_ph_plot_widget.getPlotItem()
        cm_ph_plot_item.vb.setLimits(xMin=x_min * 1.1, xMax=x_max * 1.1, yMin=y_min * 1.1, yMax=y_max * 1.1)
        cm_ph_plot_item.setRange(xRange=(x_min * 1.1, x_max * 1.1), yRange=(y_min * 1.1, y_max * 1.1),
                                 disableAutoRange=True)

        cm_ph_text_m = pg.TextItem(text='M', anchor=(0.5, 0.5))  # , color=(256, 256, 256))
        cm_ph_text_a = pg.TextItem(text='A', anchor=(0.5, 0.5))
        cm_ph_text_b = pg.TextItem(text='B', anchor=(0.5, 0.5))

        cm_ph_text_m.setPos(0, y_max * 0.9)
        cm_ph_text_a.setPos(-x_max * 0.5, y_max * 0.5)
        cm_ph_text_b.setPos(x_max * 0.5, y_max * 0.5)

        cm_ph_plot_item.addItem(cm_ph_text_m)
        cm_ph_plot_item.addItem(cm_ph_text_a)
        cm_ph_plot_item.addItem(cm_ph_text_b)

        #  TODO add limits for the spectral graphs and disable auto range

    def configure_gui(self):
        self.set_settings_page()
        self.ui.lineEdit_min_f.setText(str(self.e.streaming_device.function_gen.freq0))
        self.ui.lineEdit_max_f.setText(str(self.e.streaming_device.function_gen.freq1))

        self.ui.comboBox_out_sig_type.setCurrentText(self.e.streaming_device.function_gen.current_function)
        self.ui.comboBox_sig_len.setCurrentText(str(self.e.streaming_device.finite_frame_len_sec))

        self.ui.comboBox_cm_window_type.setCurrentText(self.e.cm_sp_window_type)  # TODO Rename to cm_sp
        self.ui.comboBox_cm_window_size.setCurrentText(str(self.e.cm_sp_window_size))  # TODO Rename to cm_sp

        self.ui.comboBox_cm_ph_window_size.setCurrentText(str(self.e.cm_ph_window_size))

        self.ui.comboBox_fm_window_type.setCurrentText(self.e.fm_sp_window_type)
        self.ui.comboBox_fm_window_size.setCurrentText(str(self.e.fm_sp_window_size))

        self.ui.comboBox_fm_spg_window_type.setCurrentText(self.e.fm_spg_window_type)
        self.ui.comboBox_fm_spg_window_type.setCurrentText(str(self.e.fm_spg_window_size))
        self.ui.comboBox_fm_spg_chan.setCurrentText(str(self.e.fm_spg_chan))

        self.align_dials_position()
        self.set_plot_limits()
        self.e.streaming_device.input_frame_ready_signal.connect(self.update_cm_tm_plot)

        self.ui.pushButton_start.setEnabled(True)
        self.ui.tabWidget_main.setTabEnabled(self.FINITE_MEAS_TAB_INDEX, True)
        self.ui.tabWidget_main.setTabEnabled(self.CONTINUOUS_MEAS_TAB_INDEX, True)

    def set_settings_page(self):
        self.set_comboBoxes_ai()
        self.set_comboBoxes_ao()
        self.set_comboBoxes_ai_range()
        self.set_comboBoxes_ao_range()
        self.set_comboBox_ai_config()
        self.set_comboBox_ai_fs()
        self.set_comboBox_ao_fs()
        self.set_comboBoxes_win_size()

    def set_comboBox_ai_fs(self):
        self.ui.comboBox_ai_fs.clear()
        self.ui.comboBox_ai_fs.addItems([str(item) for item in self.e.streaming_device.limits.supported_input_rates])

    def set_comboBox_ao_fs(self):
        self.ui.comboBox_ao_fs.clear()
        self.ui.comboBox_ao_fs.addItems([str(item) for item in self.e.streaming_device.limits.supported_output_rates])

    def set_comboBoxes_win_size(self):
        self.ui.comboBox_cm_window_size.clear()  # TODO RENAME to cm_sp
        self.ui.comboBox_cm_ph_window_size.clear()
        self.ui.comboBox_fm_window_size.clear()
        self.ui.comboBox_fm_spg_window_size.clear()
        list_of_win_sizes = [str(item) for item in self.e.streaming_device.limits.supported_monitor_frame_lengths]
        self.ui.comboBox_cm_window_size.addItems(list_of_win_sizes)  # TODO RENAME to cm_sp
        self.ui.comboBox_cm_ph_window_size.addItems(list_of_win_sizes)
        self.ui.comboBox_fm_window_size.addItems(list_of_win_sizes)
        self.ui.comboBox_fm_spg_window_size.addItems(list_of_win_sizes)

    def set_comboBoxes_ai(self):
        self.ui.comboBox_ai_a.clear()
        self.ui.comboBox_ai_b.clear()

        counter = 0
        for channel in self.e.streaming_device.limits.ai_physical_chans:
            if counter % 2 == 0:
                self.ui.comboBox_ai_a.addItem(channel)
            else:
                self.ui.comboBox_ai_b.addItem(channel)
            counter += 1

    def set_comboBoxes_ao(self):
        self.ui.comboBox_ao_a.clear()
        self.ui.comboBox_ao_b.clear()

        counter = 0
        for channel in self.e.streaming_device.limits.ao_physical_chans:
            if counter % 2 == 0:
                self.ui.comboBox_ao_a.addItem(channel)
            else:
                self.ui.comboBox_ao_b.addItem(channel)
            counter += 1

    def set_comboBoxes_ai_range(self):
        self.ui.comboBox_ai_max.clear()
        self.ui.comboBox_ai_min.clear()

        counter = 0
        # count backwards since ranges are listed in increasing order in the DAQ object
        # while we want to default to the largest input range that is the safest bet.
        for voltage in self.e.streaming_device.limits.ai_voltage_rngs:
            if counter % 2 != 1:
                self.ui.comboBox_ai_max.addItem(str(voltage))
            else:
                self.ui.comboBox_ai_min.addItem(str(voltage))
            counter += 1

    def set_comboBoxes_ao_range(self):
        self.ui.comboBox_ao_max.clear()
        self.ui.comboBox_ao_min.clear()

        counter = 0
        for voltage in self.e.streaming_device.limits.ao_voltage_rngs:
            if counter % 2 != 1:
                self.ui.comboBox_ao_max.addItem(str(voltage))
            else:
                self.ui.comboBox_ao_min.addItem(str(voltage))
            counter += 1

    def set_comboBox_ai_config(self):
        self.ui.comboBox_ai_terminal_cfg.clear()
        for item in self.e.streaming_device.limits.terminal_configs:
            self.ui.comboBox_ai_terminal_cfg.addItem(item)
            # its possible to use the comboBox string entries later as keywords to the enumerated type that
            # corresponds to them. i.e. nidaqmx.constants.TerminalConfiguration['RSE'] returns the enum same as
            # nidaqmx.constants.TerminalConfiguration.RSE. More documentation here:
            # https://docs.python.org/3/library/enum.html

    def on_pushButton_start_clicked(self):
        if self.ui.tabWidget_main.currentIndex() == self.FINITE_MEAS_TAB_INDEX:

            if self.ui.comboBox_out_sig_type.currentText() != 'ess':
                self.print_qt('>>> Finite measurement running...')
                self.e.start_fm_experiment()
            elif self.ui.comboBox_out_sig_type.currentText() == 'ess':
                self.print_qt('>>> Finite ESS measurement running...')
                self.e.start_fm_ess_experiment()
            self.update_current_fm_plot()

        elif self.ui.tabWidget_main.currentIndex() == self.CONTINUOUS_MEAS_TAB_INDEX:

            self.ui.pushButton_start.setEnabled(False)
            self.ui.tabWidget_main.setTabEnabled(self.SETTINGS_TAB_INDEX, False)
            self.ui.tabWidget_main.setTabEnabled(self.FINITE_MEAS_TAB_INDEX, False)
            self.e.streaming_device.io_start()
            self.print_qt('>>> Continuous measurement started.')
            self.ui.pushButton_stop.setEnabled(True)
            self.buffer_indicator_timer.start(config.IO_BUFFER_INDICATOR_REFRESH_MSEC)
            self.on_tabWidget_cm_changed()
        else:
            self.print_qt('Move to one of the measurement tabs to run a measurement.')

    def on_pushButton_stop_clicked(self):
        self.buffer_indicator_timer.stop()
        # self.timeseries_timer.stop()
        self.e.streaming_device.io_stop()
        self.ui.pushButton_stop.setEnabled(False)
        self.ui.pushButton_start.setEnabled(True)
        self.ui.tabWidget_main.setTabEnabled(self.SETTINGS_TAB_INDEX, True)
        self.ui.tabWidget_main.setTabEnabled(self.FINITE_MEAS_TAB_INDEX, True)
        self.print_qt('>>> Continuous measurement stopped.')

    def on_tabWidget_cm_changed(self):
        if self.e.streaming_device.cm_measurement_is_running:  # TODO why is this needed?
            self.disconnect_all_drawing()
            if self.ui.tabWidget_cm.currentIndex() == self.CM_TIMESERIES_TAB_INDEX:
                self.e.streaming_device.input_frame_ready_signal.connect(self.update_cm_tm_plot)
            elif self.ui.tabWidget_cm.currentIndex() == self.CM_SPECTRUM_TAB_INDEX:
                self.e.streaming_device.set_monitor(self.e.cm_sp_window_size)
                self.e.streaming_device.monitor_ready_signal.connect(self.update_cm_sp_plot)
            elif self.ui.tabWidget_cm.currentIndex() == self.CM_PHASE_TAB_INDEX:
                self.e.streaming_device.set_monitor(self.e.cm_ph_window_size)
                self.e.streaming_device.monitor_ready_signal.connect(self.update_cm_ph_plot)
            else:
                # reminder for future expansion of tabs
                self.print_qt('WARNING: This tab does not define a specific streaming device configuration')

    def disconnect_all_drawing(
            self):  # TODO FIND A BETTER WAY TO DO THIS BY LEARNING WHICH SIGNAL IS ACTUALLY ASSIGNED TO A GIVEN SLOT
        all_input_frame_draw_methods = [self.update_cm_tm_plot]
        all_monitor_draw_methods = [self.update_cm_sp_plot, self.update_cm_ph_plot]

        for method in all_input_frame_draw_methods:
            try:
                self.e.streaming_device.input_frame_ready_signal.disconnect(method)
            except TypeError:  # This is thrown when we try to disconnect a method that isn't connected
                pass

        for method in all_monitor_draw_methods:
            try:
                self.e.streaming_device.monitor_ready_signal.disconnect(method)
            except TypeError:
                pass

    def place_fm_tm_graph(self):
        self.fm_tm_plot_widget = pg.PlotWidget(name='Timeseries')
        self.ui.tm_verticalLayout.addWidget(self.fm_tm_plot_widget)
        fm_tm_plot_item = self.fm_tm_plot_widget.getPlotItem()
        fm_tm_plot_item.showGrid(True, True, alpha=1)
        fm_tm_plot_item.setLabel('left', 'Lvl', units='<b>V</b>')
        # fm_tm_plot_item.setLabel('bottom', 'time', units='s')

        for chan_index in range(config.NR_OF_CHANNELS_TO_PLOT):
            this_plot_data_item = fm_tm_plot_item.plot(pen=self.pen_case[chan_index])
            self.fm_tm_plot_data_items.append(this_plot_data_item)
            this_plot_data_item = fm_tm_plot_item.plot(pen=self.hold_pen_case[chan_index])
            self.fm_tm_plot_hold_data_items.append(this_plot_data_item)

        self.fm_tm_plot_widget.show()
        # fm_tm_plot_item.vb.setLimits(yMin=self.streaming_device.ai_min_val, yMax=self.streaming_device.ai_max_val)

    def place_fm_sp_graph(self):
        self.fm_sp_plot_widget = pg.PlotWidget(name='Spectral Analysis')
        self.ui.sp_verticalLayout.addWidget(self.fm_sp_plot_widget)
        fm_sp_plot_item = self.fm_sp_plot_widget.getPlotItem()
        fm_sp_plot_item.setLogMode(True, False)
        fm_sp_plot_item.showGrid(True, True, alpha=1)
        fm_sp_plot_item.vb.setLimits(yMin=config.FM_SP_GRAPH_MIN_dB_LIMIT, yMax=config.FM_SP_GRAPH_MAX_dB_LIMIT)
        fm_sp_plot_item.setRange(yRange=(config.FM_SP_GRAPH_MIN_dB_LIMIT, config.FM_SP_GRAPH_MAX_dB_LIMIT),
                                 disableAutoRange=True)
        fm_sp_plot_item.setLabel('left', 'Lvl', units='<b>dBFS</b>')

        for chan_index in range(config.NR_OF_CHANNELS_TO_PLOT):
            this_plot_data_item = fm_sp_plot_item.plot(pen=self.pen_case[chan_index])
            self.fm_sp_plot_data_items.append(this_plot_data_item)
            this_plot_data_item = fm_sp_plot_item.plot(pen=self.hold_pen_case[chan_index])
            self.fm_sp_plot_hold_data_items.append(this_plot_data_item)

        self.fm_sp_plot_widget.show()

    def place_fm_spg_graphics(self):
        self.fm_spg_widget = pg.GraphicsLayoutWidget()
        self.ui.spg_verticalLayout.addWidget(self.fm_spg_widget)

        # view = self.ui.graphics_widget.addViewBox()
        self.graphics_plot = self.fm_spg_widget.addPlot()  # will generate the view and axes automatically
        # Add labels to the axis
        # self.graphics_plot.setLabel('bottom', 'Time', units='s') # this way lots of space is wasted
        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in this case kHz)
        self.graphics_plot.setLabel('left', "Frequency", units='<b>Hz</b>')

        self.img = pg.ImageItem(border='w')
        self.graphics_plot.addItem(self.img)

        # self.ui.graphics_plot.setLogMode(True, False)

        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.fm_spg_widget.addItem(self.hist)

        self.hist.gradient.restoreState(config.default_spg_hist_gradient)

    def place_cm_tm_graph(self):
        self.cm_tm_plot_widget = pg.PlotWidget(name='Continuous Timeseries')
        self.ui.cm_tm_verticalLayout.addWidget(self.cm_tm_plot_widget)
        cm_tm_plot_item = self.cm_tm_plot_widget.getPlotItem()
        # cm_tm_plot_item.vb.disableAutoRange()
        # cm_tm_plot_item.vb.setLimits(ymin=self.streaming_device.ai_min_val,ymax=self.streaming_device.ai_max_val)
        cm_tm_plot_item.showGrid(True, True, alpha=1)
        cm_tm_plot_item.setLabel('left', 'Lvl', units='<b>V</b>')

        for chan_index in range(config.NR_OF_CHANNELS_TO_PLOT):
            this_plot_data_item = cm_tm_plot_item.plot(pen=self.pen_case[chan_index])
            self.cm_tm_plot_data_items.append(this_plot_data_item)
            this_plot_data_item = cm_tm_plot_item.plot(pen=self.hold_pen_case[chan_index])
            self.cm_tm_plot_hold_data_items.append(this_plot_data_item)

    def place_cm_sp_graph(self):
        self.cm_sp_plot_widget = pg.PlotWidget(name='Continuous Spectral Analysis')
        self.ui.cm_sp_verticalLayout.addWidget(self.cm_sp_plot_widget)
        cm_sp_plot_item = self.cm_sp_plot_widget.getPlotItem()

        # cm_sp_plot_item.disableAutoRange()
        cm_sp_plot_item.setLogMode(True, False)
        cm_sp_plot_item.showGrid(True, True, alpha=1)
        cm_sp_plot_item.setLabel('left', 'Lvl', units='<b>dBFS</b>')

        for chan_index in range(config.NR_OF_CHANNELS_TO_PLOT):
            this_plot_data_item = cm_sp_plot_item.plot(pen=self.pen_case[chan_index])
            self.cm_sp_plot_data_items.append(this_plot_data_item)
            this_plot_data_item = cm_sp_plot_item.plot(pen=self.hold_pen_case[chan_index])
            self.cm_sp_plot_hold_data_items.append(this_plot_data_item)

        x_min = np.log10(config.MIN_OUTPUT_FREQUENCY_ALLOWED)
        x_max = np.log10(config.MAX_COARSE_FREQ_DIAL_LIMIT)
        y_min = config.CM_SP_GRAPH_MIN_dB_LIMIT
        y_max = config.CM_SP_GRAPH_MAX_dB_LIMIT

        cm_sp_plot_item.vb.setLimits(xMin=x_min, xMax=x_max,
                                     yMin=y_min, yMax=y_max)

        cm_sp_plot_item.setRange(xRange=(x_min, x_max),
                                 yRange=(y_min, y_max),
                                 disableAutoRange=True)

    def place_cm_ph_graph(self):
        self.cm_ph_plot_widget = pg.PlotWidget(name='Phase Scope')
        self.ui.cm_ph_horizontalLayout.addWidget(self.cm_ph_plot_widget)
        cm_ph_plot_item = self.cm_ph_plot_widget.getPlotItem()
        cm_ph_plot_item.showGrid(True, True, alpha=1)
        # cm_ph_plot_item.setLabel('left', 'Lvl', units='<b>V</b>')

        self.cm_ph_plot_data_item = cm_ph_plot_item.plot(pen=config.PHASE_PLOT_COLOR_MODIFIER)
        plot_hold_data_item1 = cm_ph_plot_item.plot(pen=config.PHASE_PLOT_HOLD_COLOR_MODIFIER)
        self.cm_ph_plot_hold_data_items.append(plot_hold_data_item1)
        plot_hold_data_item2 = cm_ph_plot_item.plot(pen=config.PHASE_PLOT_HOLD_COLOR_MODIFIER + 1)
        self.cm_ph_plot_hold_data_items.append(plot_hold_data_item2)
        self.cm_ph_plot_box_data_item = cm_ph_plot_item.plot(pen=config.PHASE_PLOT_LIMIT_COLOR_MODIFIER)

        # TODO GENERATE THE ROTATION MATRIX FOR 45*

    def update_output_buffer_indicator(self):
        ai_buffer_level = self.e.streaming_device.get_ai_buffer_level_prc()
        ao_buffer_level = self.e.streaming_device.get_ao_buffer_level_prc()

        self.ui.buffer_indicator_label.setText(f'ai:{ai_buffer_level}/ao:{ao_buffer_level}')

    def update_cm_tm_plot(self):
        self.e.streaming_device.read_lock.acquire()  # TODO this probably isn't working
        input_frame = self.e.streaming_device.input_frame
        input_frame_timebase = self.e.streaming_device.input_time_base
        self.e.streaming_device.read_lock.release()

        for chan in range(config.NR_OF_CHANNELS_TO_PLOT):
            self.cm_tm_plot_data_items[chan].setData(input_frame_timebase[chan], input_frame[chan])

    def update_cm_sp_plot(self):
        for chan in range(config.NR_OF_CHANNELS_TO_PLOT):
            # plot except for DC component to prevent warning with X logscale
            self.cm_sp_plot_data_items[chan].setData(self.e.cm_freq_base[1:], self.e.calculate_cm_fft()[chan][1:])

    def update_cm_ph_plot(self):
        phase_frame = self.e.calculate_cm_phase()
        self.cm_ph_plot_data_item.setData(phase_frame)

    def update_fm_tm_plot(self):
        self.check_fm_data_present()
        self.print_qt('>>> Drawing...')

        for chan in range(config.NR_OF_CHANNELS_TO_PLOT):
            self.fm_tm_plot_data_items[chan].setData(self.e.fm_result_x, self.e.fm_result_y[chan])

    def update_fm_sp_plot(self):
        self.check_fm_data_present()
        self.print_qt('>>> Calculating...')

        try:
            if self.e.streaming_device.function_gen.current_function != 'ess':
                fm_freq_base, fm_fft_segment_time, fm_db_fft = self.e.calculate_fm_fft()
            else:
                impulse_response = self.e.calculate_ir_from_fm_ess()
                fm_freq_base, fm_fft_segment_time, fm_db_fft = self.e.calculate_fm_fft(impulse_response)
        except ValueError as err:
            self.print_qt(err)
            return

        self.print_qt(f'{len(fm_fft_segment_time)} FFT segments average.')

        for chan in range(config.NR_OF_CHANNELS_TO_PLOT):
            self.fm_sp_plot_data_items[chan].setData(fm_freq_base[1:], fm_db_fft[chan][1:])

    def update_fm_spg(self):
        self.check_fm_data_present()
        self.print_qt('>>> Calculating...')

        try:
            f, t, spg = self.e.calculate_fm_spectrogram()
            # axis 1 is the time axis
        except ValueError as err:
            self.print_qt(err)
            return
        except MemoryError as err:
            print(err)
            self.print_qt('>> EXCEPTION << : Out of memory. Try analyzing less data.')
            return

        # convert amplitudes to log scale if required:
        if self.ui.checkBox_fm_spg_log_amp.isChecked():
            spg = np.log10(spg)

        # limit the frequency range of the analysis
        spg = spg[:][:int(np.size(spg, axis=0) * config.SPG_FREQUENCY_LIMIT / (
                self.e.streaming_device.ai_fs / 2))]

        # Clip the freq axis accordingly
        f = f[:int(np.size(spg, axis=0))]

        self.hist.setLevels(np.min(spg), np.max(spg))

        self.img.resetTransform()  # Ensures that the previous transformation does not interfere with the current one
        self.img.scale(t[-1] / np.size(spg, axis=1), f[-1] / np.size(spg, axis=0))
        self.img.setImage(spg.T)

        # Limit panning/zooming to the spectrogram
        self.graphics_plot.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])


if __name__ == "__main__":
    TwingoExec()
