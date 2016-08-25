import sys
import PyQt4.QtCore as QtCore
import resistive_pulse as rp
import rp_file
import time_series as ts
import time
import time_series_loader
import event_finder
import copy
import numpy as np
import scipy.signal
import rp_event_manager

PORE_STATS_DIR = '/home/preston/Desktop/Science/Research/pore_stats/'
sys.path.append(PORE_STATS_DIR)

class RPModel(QtCore.QObject):

    # Class variables
    display_decimation_threshold = 100000
    decimation_factor = 2

    # Signals
    busy = QtCore.pyqtSignal(bool)
    #event_added = QtCore.pyqtSignal('PyQt_PyObject')
    event_added = QtCore.pyqtSignal()
    targeted_event_changed = QtCore.pyqtSignal('PyQt_PyObject')
    events_cleared = QtCore.pyqtSignal()

    def __init__(self, parent_controller):
        super(RPModel, self).__init__()

        self._parent_controller = parent_controller

        self._time_series_loaders = []

        self._active_file = None

        self._main_ts = ts.TimeSeries(key_parameters = [])

        self._baseline_ts = ts.TimeSeries(key_parameters = ['baseline_avg_length'])

        self._pos_thresh_ts = ts.TimeSeries(key_parameters = ['baseline_avg_length', 'trigger_sigma_threshold'])

        self._neg_thresh_ts = ts.TimeSeries(key_parameters = ['baseline_avg_length', 'trigger_sigma_threshold'])

        self._filtered_ts = ts.TimeSeries(key_parameters = ['filter_frequency'])

        self._event_manager = rp_event_manager.RPEventManager()

    def load_ts(self, ts):
        new_ts_loader = time_series_loader.TimeSeriesLoader(ts)
        self._time_series_loaders.append(new_ts_loader)
        self.connect(new_ts_loader, QtCore.SIGNAL('started()'), self.set_busy)
        self.connect(new_ts_loader, QtCore.SIGNAL('finished()'), self.set_not_busy)
        self.connect(new_ts_loader, QtCore.SIGNAL('finished()'),\
                     lambda: self.remove_ts_loader(new_ts_loader._id))

        self.connect(new_ts_loader, QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'),\
         ts.add_decimated_data_tier)

        new_ts_loader.start()

    def load_main_ts(self):

        self._main_ts.initialize(file_path = self._active_file._file_path,
                                 full_data = None,
                                 max_pts_returned = self.display_decimation_threshold,
                                 decimation_factor = self.decimation_factor)

        self.load_ts(self._main_ts)

        return

    def save_events(self):
        file_path = self._active_file._file_path
        file_path = file_path.split('.')[0]
        file_path = file_path + '_events'

        self._event_manager.save_events(file_path)

        return


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




    def remove_ts_loader(self, id):
        self._time_series_loaders = [ts for ts in self._time_series_loaders if ts._id != id]
        return






    def load_baseline_ts(self):

        if ((self._baseline_ts._key_parameters['baseline_avg_length'] != self._baseline_avg_length) or
            (self._pos_thresh_ts._key_parameters['trigger_sigma_threshold'] != self._trigger_sigma_threshold)):

            baseline_data = np.empty((int(self._main_ts._decimated_data_list[0].shape[0]/self._baseline_avg_length),4))

            for i in xrange(baseline_data.shape[0]):
                baseline_data[i,:] = rp.get_baseline(self._main_ts._decimated_data_list[0], \
                    i*self._baseline_avg_length, self._baseline_avg_length, self._trigger_sigma_threshold)

            # Average
            self._baseline_ts.initialize(file_path = None,
                                         full_data = baseline_data[:,[0,1]],
                                         max_pts_returned = self.display_decimation_threshold,
                                         decimation_factor = self.decimation_factor, \
                                         key_parameters = {'baseline_avg_length': self._baseline_avg_length})

            self.load_ts(self._baseline_ts)

            # Pos thresh
            self._pos_thresh_ts.initialize(file_path = None,
                                         full_data = baseline_data[:,[0,2]],
                                         max_pts_returned = self.display_decimation_threshold,
                                         decimation_factor = self.decimation_factor,
                                         key_parameters = {'baseline_avg_length': self._baseline_avg_length,
                                                           'trigger_sigma_threshold': self._trigger_sigma_threshold})
            self.load_ts(self._pos_thresh_ts)

            # Neg thresh
            self._neg_thresh_ts.initialize(file_path = None,
                                         full_data = baseline_data[:,[0,3]],
                                         max_pts_returned = self.display_decimation_threshold,
                                         decimation_factor = self.decimation_factor,
                                         key_parameters = {'baseline_avg_length': self._baseline_avg_length,
                                                           'trigger_sigma_threshold': self._trigger_sigma_threshold})
            self.load_ts(self._neg_thresh_ts)

        return

    def load_filtered_ts(self):
        if self._filtered_ts._key_parameters['filter_frequency'] != self._filter_frequency:
            cutoff = self._filter_frequency
            nyquist = self._main_ts._sampling_frequency/2.
            Wn = cutoff/nyquist
            filt_b, filt_a = scipy.signal.butter(N=5, Wn = Wn, btype = 'low', analog = False)

            filtered_data = scipy.signal.lfilter(filt_b, filt_a, self._main_ts._decimated_data_list[0][:,1])

            filtered_data = np.hstack((self._main_ts._decimated_data_list[0][:,0].reshape(-1,1), filtered_data.reshape(-1,1)))[1000:-1000,:]


            self._filtered_ts.initialize(file_path = None,
                                         full_data = filtered_data,
                                         max_pts_returned = self.display_decimation_threshold,
                                         decimation_factor = self.decimation_factor,
                                         key_parameters = {'filter_frequency': self._filter_frequency})

            self.load_ts(self._filtered_ts)


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

    def set_filter_frequency(self, filter_frequency):
        self._filter_frequency = filter_frequency

    def get_main_display_data(self, t_range):
        main_display_data = None

        if self._main_ts._display_ready == True:
            main_display_data = self._main_ts.return_data(t_range[0], t_range[1])


        return main_display_data

    def get_baseline_display_data(self, t_range):
        baseline_display_data = None


        if self._baseline_ts._display_ready == True:
            baseline_display_data = self._baseline_ts.return_data(t_range[0], t_range[1])

        return baseline_display_data

    def get_pos_thresh_display_data(self, t_range):
        pos_thresh_display_data = None

        if self._pos_thresh_ts._display_ready == True:
            pos_thresh_display_data = self._pos_thresh_ts.return_data(t_range[0], t_range[1])

        return pos_thresh_display_data

    def get_neg_thresh_display_data(self, t_range):
        neg_thresh_display_data = None

        if self._neg_thresh_ts._display_ready == True:
            neg_thresh_display_data = self._neg_thresh_ts.return_data(t_range[0], t_range[1])

        return neg_thresh_display_data

    def get_filtered_display_data(self, t_range):
        filtered_display_data = None

        if self._filtered_ts._display_ready == True:
            filtered_display_data = self._filtered_ts.return_data(t_range[0], t_range[1])

        return filtered_display_data


    def find_events(self, ti = -1, tf = -1, filter = False):

        self._event_manager.clear_events()
        parameters = []
        parameters.append(('file_path', self._active_file._file_path))
        parameters.append(('baseline_avg_length', str(self._baseline_avg_length)))
        parameters.append(('trigger_sigma_threshold', str(self._trigger_sigma_threshold)))
        parameters.append(('max_search_length', str(self._max_search_length)))

        self._event_manager._parameters = parameters

        self._event_thread = QtCore.QThread()

        if filter == False:
            filtered_data = None
        else:
            filtered_data = self._filtered_ts._decimated_data_list[0]

        self._event_finder = event_finder.EventFinder(copy.copy(self._main_ts._decimated_data_list[0]), ti = ti, tf = tf, \
            baseline_avg_length = self._baseline_avg_length, trigger_sigma_threshold = self._trigger_sigma_threshold, \
            max_search_length =  self._max_search_length, filtered_data = filtered_data, go_past_length = 0)


        self._event_finder.moveToThread(self._event_thread)
        self.connect(self._event_thread, QtCore.SIGNAL('started()'), self.set_busy)
        self.connect(self._event_thread, QtCore.SIGNAL('started()'), self._event_finder.find_events)

        # Process all event detections at conclusion of search
        self.connect(self._event_finder, QtCore.SIGNAL('events_found(PyQt_PyObject)'),\
                     self._event_manager.add_events)
        self.connect(self._event_finder, QtCore.SIGNAL('finished()'), self.event_added)
        self.connect(self._event_finder, QtCore.SIGNAL('finished()'), self.set_not_busy)

        self._event_thread.start()


        return
