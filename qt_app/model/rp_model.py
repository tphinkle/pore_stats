import sys

import PyQt4.QtCore as QtCore

PORE_STATS_DIR = '/home/preston/Desktop/Science/Research/pore_stats/'
sys.path.append(PORE_STATS_DIR)
import resistive_pulse as rp
import rp_file
import time_series as ts
import time
import time_series_loader

class RPModel(QtCore.QObject):
    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        self._path_name = file_name
        self.announce_update_file_path()

    @property
    def display_data(self):
        return self._display_data

    @display_data.setter
    def display_data(self, display_data):
        self._display_data = display_data
        self.announce_update_display_data()

    @property
    def display_baseline_data(self):
        return self._display_baseline_data

    @display_baseline_data.setter
    def display_baseline_data(self, display_baseline_data):
        self._display_baseline_data = display_baseline_data
        self.announce_update_display_baseline_data()

    @property
    def processes_enabled(self):
        return self._processes_enabled

    @processes_enabled.setter
    def processes_enabled(self, processes_enabled):
        self._processes_enabled = processes_enabled
        self.announce_update_processes_enabled(self._processes_enabled)
        return

    @property
    def raw_events(self):
        return self._raw_events

    @raw_events.setter
    def raw_events(self, raw_events):
        self._raw_events = raw_events
        self.announce_update_raw_events()
        return

    @property
    def targeted_event(self):
        return self._targeted_event

    @targeted_event.setter
    def targeted_event(self, targeted_event):
        self._targeted_event = targeted_event
        self.announce_update_targeted_event()

    display_decimation_threshold = 100000
    decimation_factor = 2




    def __init__(self, parent_controller):
        super(RPModel, self).__init__()

        self._parent_controller = controller

        self._status = {}

        self._active_file = None

        self._main_ts = None

        self._baseline_ts = None

        self._raw_events = []

        self._targeted_event = None

    def set_active_file(self, file_path):
        self._active_file = rp_file.RPFile(file_path)
        return

    def load_main_ts(self):

        self._main_ts = ts.TimeSeries(self.display_decimation_threshold,\
                                      self.decimation_factor,\
                                      file_path=self._active_file._file_path)

        self._time_series_loader = time_series_loader.TimeSeriesLoader(self._main_ts)
        self.connect(self._time_series_loader, QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'), self._main_ts.add_decimated_data_tier)
        self._time_series_loader.start()

        return

    def load_baseline_ts(self):
        baseline_data = ts.decimate_data(self._main_ts._decimation_data_tiers[0],
                                         self._baseline_avg_length)

        self._baseline_ts = ts.TimeSeries(rp.get_full_baseline(self.display_decimation_threshold,\
                                                               self.decimation_factor,\
                                                               full_data = )

    def set_baseline_avg_length(self, baseline_avg_length):
        self._baseline_avg_length = baseline_avg_length
        return

    def set_trigger_sigma_threshold(self, trigger_sigma_threshold):
        self._trigger_sigma_threshold = trigger_sigma_threshold
        return

    def set_max_search_length(self, max_search_length):
        self._max_search_length = max_search_length
        return

    def change_t_range(self, t_range):

        self.display_data = self._main_ts.return_data(t_range[0], t_range[1])

        self.display_baseline_data = self._baseline_data.get_decimated_data(t_range[0], t_range[1])

        return

    def find_events(self):
        self._events =\
          rp.find_events_data(self._data._data, raw_data = None,
                                baseline_avg_length = self._baseline_avg_length,
                                trigger_sigma_threshold = self._trigger_sigma_threshold,
                                max_search_length = self._max_search_length)


        if self._targeted_event == None and len(self._events) > 0:
            self.targeted_event = self._events[0]

    def increment_targeted_event(self):
        new_index = (self._targeted_event._index + 1)%len(self._events)
