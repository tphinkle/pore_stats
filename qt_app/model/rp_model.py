import sys

import PyQt4.QtCore as QtCore

PORE_STATS_DIR = '/home/preston/Desktop/Science/Research/pore_stats/'
sys.path.append(PORE_STATS_DIR)
import resistive_pulse as rp
import rp_file
import time_series as ts
import time
import time_series_loader
import event_finder
import copy

class RPModel(QtCore.QObject):

    display_decimation_threshold = 100000
    decimation_factor = 2
    busy = QtCore.pyqtSignal(bool)
    event_added = QtCore.pyqtSignal('PyQt_PyObject')
    targeted_event_changed = QtCore.pyqtSignal('PyQt_PyObject')
    events_cleared = QtCore.pyqtSignal()

    def __init__(self, parent_controller):
        super(RPModel, self).__init__()

        self._parent_controller = parent_controller



        self._active_file = None

        self._main_ts = None

        self._baseline_ts = None

        self._raw_events = []

        self._targeted_event = None

    def set_busy(self):
        self._busy = True
        self.emit(QtCore.SIGNAL('busy(bool)'), self._busy)

        return

    def set_not_busy(self):
        self._busy = False
        self.emit(QtCore.SIGNAL('busy(bool)'), self._busy)

        return

    def set_active_file(self, file_path):
        self._active_file = rp_file.RPFile(file_path)
        return

    def load_main_ts(self):

        self._main_ts = ts.TimeSeries(self.display_decimation_threshold,\
                                      self.decimation_factor,\
                                      file_path=self._active_file._file_path)

        self._time_series_loader = time_series_loader.TimeSeriesLoader(self._main_ts)
        self.connect(self._time_series_loader, QtCore.SIGNAL('started()'), self.set_busy)
        self.connect(self._time_series_loader, QtCore.SIGNAL('finished()'), self.set_not_busy)
        self.connect(self._time_series_loader, QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'),\
         self._main_ts.add_decimated_data_tier)
        self._time_series_loader.start()

        return




    def load_baseline_ts(self):
        baseline_data = ts.decimate_data(self._main_ts._decimated_data_list[0],
                                         self._baseline_avg_length)


        self._baseline_ts = ts.TimeSeries(self.display_decimation_threshold,\
                                          self.decimation_factor,\
                                          full_data = baseline_data)

        self._time_series_loader = time_series_loader.TimeSeriesLoader(self._baseline_ts)
        self.connect(self._time_series_loader, QtCore.SIGNAL('started()'), self.set_busy)
        self.connect(self._time_series_loader, QtCore.SIGNAL('finished()'), self.set_not_busy)
        self.connect(self._time_series_loader, QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'),\
         self._baseline_ts.add_decimated_data_tier)
        self._time_series_loader.start()

        return

    def set_baseline_avg_length(self, baseline_avg_length):
        self._baseline_avg_length = baseline_avg_length
        return

    def set_trigger_sigma_threshold(self, trigger_sigma_threshold):
        self._trigger_sigma_threshold = trigger_sigma_threshold
        return

    def set_max_search_length(self, max_search_length):
        self._max_search_length = max_search_length

        return

    def get_main_display_data(self, t_range):
        try:
            main_display_data = self._main_ts.return_data(t_range[0], t_range[1])
        except:
            main_display_data = None


        return main_display_data

    def get_baseline_display_data(self, t_range):
        try:
            baseline_display_data = self._baseline_ts.return_data(t_range[0], t_range[1])
        except:
            baseline_display_data = None

        return baseline_display_data

    def find_events(self):
        #self.clear_events()
        #data = copy.deepcopy(self._main_ts._decimated_data_list[0])
        self._event_finder = event_finder.EventFinder(self._main_ts._decimated_data_list[0][:,:], \
            self._baseline_avg_length, self._trigger_sigma_threshold, self._max_search_length)

        self.connect(self._event_finder, QtCore.SIGNAL('started()'), self.set_busy)
        self.connect(self._event_finder, QtCore.SIGNAL('finished()'), self.set_not_busy)
        self.connect(self._event_finder, QtCore.SIGNAL('event_found(PyQt_PyObject)'), self.add_event)

        self._event_finder.start()

        return

    def clear_events(self):
        self._raw_events = []
        self._targeted_event = None
        self.emit(QtCore.SIGNAL('events_cleared()'))

    def add_event(self, event):
        self._raw_events.append(event)
        if self._targeted_event == None:
            self.set_targeted_event(event)
        self.emit(QtCore.SIGNAL('event_added(PyQt_PyObject)'), event)

        return

    def set_targeted_event(self, event):
        self._targeted_event = event
        self.emit(QtCore.SIGNAL('targeted_event_changed(PyQt_PyObject)'), event)

        return

    def increment_targeted_event(self):
        new_index = (self._targeted_event._index + 1)%len(self._events)
