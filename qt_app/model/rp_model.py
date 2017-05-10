# Imports

# Standard library
import sys
import os
import copy
import time

# Scipy
import numpy as np
import scipy.signal

# Qt
import PyQt4.QtCore

# Program specific
PORE_STATS_BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib/rp/')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib/oi/')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/conts')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/model')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/threads')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/views')
import rp_event_manager
import event_finder
import resistive_pulse as rp
import rp_file
import time_series as ts
import time_series_loader






class RPModel(PyQt4.QtCore.QObject):
    """
    - Model for all functionality regarding ResistivePulse in the qt_app.
    - Keeps track of the data, the resistive pulse events, and etc.
    """

    # Class variables
    display_decimation_threshold = 100000
    decimation_factor = 2

    # Signals
    busy = PyQt4.QtCore.pyqtSignal(bool)
    event_added = PyQt4.QtCore.pyqtSignal()
    targeted_event_changed = PyQt4.QtCore.pyqtSignal('PyQt_PyObject')
    events_cleared = PyQt4.QtCore.pyqtSignal()

    def __init__(self, parent_controller):
        super(RPModel, self).__init__()

        self._parent_controller = parent_controller

        self._time_series_loaders = []

        self._active_file = None

        # Empty/non-initialized TimeSeries objects
        self._main_ts = ts.TimeSeries(key_parameters = [])
        self._baseline_ts = ts.TimeSeries(key_parameters = ['baseline_avg_length'])
        self._pos_thresh_ts = ts.TimeSeries(key_parameters = ['baseline_avg_length', 'trigger_sigma_threshold'])
        self._neg_thresh_ts = ts.TimeSeries(key_parameters = ['baseline_avg_length', 'trigger_sigma_threshold'])
        self._filtered_ts = ts.TimeSeries(key_parameters = ['filter_frequency'])

        self._event_manager = rp_event_manager.RPEventManager()

    def load_ts(self, ts):
        """
        * Description:
            - Loads a time series by creating a new time_series_loader, the class that
            manages loading the time series and emits signals when its data are ready
            - Connects the proper signals for the time_series_loader
            - Starts the time_series_loader thread
        * Return: None
        * Arguments:
            - ts: The time-series to be loaded.
        """
        new_ts_loader = time_series_loader.TimeSeriesLoader(ts)
        self._time_series_loaders.append(new_ts_loader)
        self.connect(new_ts_loader, PyQt4.QtCore.SIGNAL('started()'), self.set_busy)
        self.connect(new_ts_loader, PyQt4.QtCore.SIGNAL('finished()'), self.set_not_busy)
        self.connect(new_ts_loader, PyQt4.QtCore.SIGNAL('finished()'),\
                     lambda: self.remove_ts_loader(new_ts_loader._id))

        self.connect(new_ts_loader, PyQt4.QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'),\
         ts.add_decimated_data_tier)

        new_ts_loader.start()

    def load_main_ts(self):
        """
        * Description: Initializes the main time series, then calls the load_ts function.
        * Return:
        * Arguments:
        """

        self._main_ts.initialize(file_path = self._active_file._file_path,
                                 full_data = None,
                                 max_pts_returned = self.display_decimation_threshold,
                                 decimation_factor = self.decimation_factor)

        self.load_ts(self._main_ts)

        return

    def save_events(self):
        """
        * Description:
            - Instructs the _event_manager to save the events to the file
            file_path.
        * Return:
        * Arguments:
        """
        file_path = self._active_file._file_path
        file_path = file_path.split('.')[0]
        file_path = file_path + '_events.json'

        self._event_manager.save_events_json(file_path)

        return


    def set_busy(self):
        """
        * Description:
            - Sets internal status to busy.
            - Informs the controller about change to busy state.
        * Return:
        * Arguments:
        """
        self._busy = True
        self.emit(PyQt4.QtCore.SIGNAL('busy(bool)'), self._busy)

        return

    def set_not_busy(self):
        """
        * Description:
            - Sets internal status to not busy.
            - Informs the controller about change to not busy state.
        * Return:
        * Arguments:
        """
        self._busy = False
        self.emit(PyQt4.QtCore.SIGNAL('busy(bool)'), self._busy)

        return

    def set_active_file(self, file_path):
        """
        * Description:
            - Creates a new rp_file class object with file_path
        * Return:
        * Arguments:
            - file_path: Location of the RP file.
        """
        self._active_file = rp_file.RPFile(str(file_path))
        return




    def remove_ts_loader(self, id):
        """
        * Description:
            - Removes the ts_loader with id ID from the list of ts_loaders
        * Return:
        * Arguments:
            -
        """
        self._time_series_loaders = [ts for ts in self._time_series_loaders if ts._id != id]
        return






    def load_baseline_ts(self):
        """
        * Description:
            - Loads 3 time series:
                - pos threshold
                - average
                - negative threshold
            - Checks to make sure that the parameters set in the UI agree with those
            set in the baseline TimeSeries classes; if not, this means that those options
            have been changed by the user and they will have to be recalculated.
        * Return:
        * Arguments:
        """

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
        """
        * Description:
            - Loads the filtered TimeSeries.
            - First checks to see if the loaded filtered TimeSeries parameters differ from
            those set in the GUI; if they do differ, the TimeSeries will be loaded again.
        * Return:
        * Arguments:
        """
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
        """
        * Description: Setter for _baseline_avg_length, teh number of points that are
        averaged in calculating the baseline.
        * Return:
        * Arguments:
        """
        self._baseline_avg_length = baseline_avg_length
        return

    def set_trigger_sigma_threshold(self, trigger_sigma_threshold):
        """
        * Description: Setter for _trigger_sigma_threshold, the number of std. dev. factors
        away from the baseline that that signal must venture before the event start/stop
        is triggered.
        * Return:
        * Arguments:
        """
        self._trigger_sigma_threshold = trigger_sigma_threshold
        return

    def set_max_search_length(self, max_search_length):
        """
        * Description: Setter for _max_search_length, the maximum length that the search
        algorithm will search for the end of an event after the start of an event is
        detected.
        * Return:
        * Arguments:
        """
        self._max_search_length = max_search_length

        return

    def set_filter_frequency(self, filter_frequency):
        """
        * Description: Setter for _filter_frequency, the frequency at which the data is
        filtered for the _filtered_ts TimeSeries object.
        * Return:
        * Arguments:
        """
        self._filter_frequency = filter_frequency

    def get_main_display_data(self, t_range):
        """
        * Description:
            - Returns the data that will be displayed by calling the _main_ts
            TimeSeries' return_data() function. The end result is that a decimated array is
            returned for display purposes.
        * Return:
            - main_display_data: The (decimated) array that is to be displayed.
        * Arguments:
            - t_range: tuple that contains ti, tf; the start and stop times of the interval
            that the GUI wishes to display.
        """
        main_display_data = None

        if self._main_ts._display_ready == True:
            main_display_data = self._main_ts.return_data(t_range[0], t_range[1])


        return main_display_data

    def get_baseline_display_data(self, t_range):
        """
        * Description:
            - Returns the baseline data that will be displayed by calling the _baseline_ts
            TimeSeries' return_data() function. The end result is that a decimated array is
            returned for display purposes.
        * Return:
            - baseline_display_data: The (decimated) array that is to be displayed.
        * Arguments:
            - t_range: tuple that contains ti, tf; the start and stop times of the interval
            that the GUI wishes to display.
        """
        baseline_display_data = None


        if self._baseline_ts._display_ready == True:
            baseline_display_data = self._baseline_ts.return_data(t_range[0], t_range[1])

        return baseline_display_data

    def get_pos_thresh_display_data(self, t_range):
        """
        * Description:
            - Returns the positive threshold data that will be displayed by calling
            the _pos_thresh_ts TimeSeries' return_data() function. The end result is that
            a decimated array is returned for display purposes.
        * Return:
            - pos_thresh_display_data: The (decimated) array that is to be displayed.
        * Arguments:
            - t_range: tuple that contains ti, tf; the start and stop times of the interval
            that the GUI wishes to display.
        """
        pos_thresh_display_data = None

        if self._pos_thresh_ts._display_ready == True:
            pos_thresh_display_data = self._pos_thresh_ts.return_data(t_range[0], t_range[1])

        return pos_thresh_display_data

    def get_neg_thresh_display_data(self, t_range):
        """
        * Description:
            - Returns the negative threshold data that will be displayed by calling
            the _neg_thresh_ts TimeSeries' return_data() function. The end result is that
            a decimated array is returned for display purposes.
        * Return:
            - neg_thresh_display_data: The (decimated) array that is to be displayed.
        * Arguments:
            - t_range: tuple that contains ti, tf; the start and stop times of the interval
            that the GUI wishes to display.
        """
        neg_thresh_display_data = None

        if self._neg_thresh_ts._display_ready == True:
            neg_thresh_display_data = self._neg_thresh_ts.return_data(t_range[0], t_range[1])

        return neg_thresh_display_data

    def get_filtered_display_data(self, t_range):
        """
        * Description:
            - Returns the filtered data that will be displayed by calling the
            filtered_ts TimeSeries' return_data() function. The end result is that
            a decimated array is returned for display purposes.
        * Return:
            - filtered_display_data: The (decimated) array that is to be displayed.
        * Arguments:
            - t_range: tuple that contains ti, tf; the start and stop times of the interval
            that the GUI wishes to display.
        """
        filtered_display_data = None

        if self._filtered_ts._display_ready == True:
            filtered_display_data = self._filtered_ts.return_data(t_range[0], t_range[1])

        return filtered_display_data


    def find_events(self, ti = -1, tf = -1, filter = False):
        """
        * Description:
            - Function that begins the event search.
        * Return:
        * Arguments:
            -
        """

        # First clear all the events that have already been found
        self._event_manager.clear_events()

        # Set the correct search parameters for the event_manager
        parameters = []
        parameters.append(('file_path', self._active_file._file_path))
        parameters.append(('baseline_avg_length', str(self._baseline_avg_length)))
        parameters.append(('trigger_sigma_threshold', str(self._trigger_sigma_threshold)))
        parameters.append(('max_search_length', str(self._max_search_length)))

        self._event_manager._parameters = parameters

        if filter == False:
            filtered_data = None
        else:
            filtered_data = self._filtered_ts._decimated_data_list[0]

        self._event_finder = event_finder.EventFinder(copy.copy(self._main_ts._decimated_data_list[0]), ti = ti, tf = tf, \
            baseline_avg_length = self._baseline_avg_length, trigger_sigma_threshold = self._trigger_sigma_threshold, \
            max_search_length =  self._max_search_length, filtered_data = filtered_data, go_past_length = 0)

        self.connect(self._event_finder, PyQt4.QtCore.SIGNAL('events_found(PyQt_PyObject)'),\
                     self._event_manager.add_events)
        self.connect(self._event_finder, PyQt4.QtCore.SIGNAL('finished()'), self.event_added)
        self.connect(self._event_finder, PyQt4.QtCore.SIGNAL('finished()'), self.set_not_busy)

        self._event_finder.start()

        return
