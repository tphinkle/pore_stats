# Python standard library
import os
import sys
import math

# Qt
import PyQt4.QtCore as QtCore

# Program specific
PORE_STATS_BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__)).replace('/qt_app/model', '')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib/rp/')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib/oi/')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/conts')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/model')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/threads')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/views')
import time_series as ts
import resistive_pulse as rp
import rp_file


class TimeSeriesLoader(QtCore.QThread):
    """
    - QThread object that loads a time_series.
    - Starts with a file name and hten loads all of the data from that file into the
    undecimated list.
    - After loading the undecimated data, begins to load hte decimated data arrays.
    - Every time a data array is finished loading (dec. and non-dec.), it emits a signal
    that sends the data array to its associated TimeSeries class.
    - This connection is established when the TimeSeriesLoader is created in rp_model.
    """

    loaders = 0
    def __init__(self, time_series):
        super(TimeSeriesLoader, self).__init__()
        self._max_pts_returned = time_series._max_pts_returned
        self._decimation_factor = time_series._decimation_factor
        self._decimation_tiers = time_series._decimation_tiers

        if time_series._full_data != None:
            self._full_data = time_series._full_data
        else:
            self._full_data = None

        if time_series._file_path:
            self._file_path = time_series._file_path

        self.loaders+=1
        self._id = self.loaders


    def __del__(self):
        pass


    def run(self):
        """
        * Description:
            - Triggered when this class' start() function is called
            (PyQt4.QtCore.QThread.start())
            - Emits PyQt4.QtCore.QThread.finished() when return is called
        * Return:
        * Arguments:
        """

        if self._full_data == None:
            self._full_data = rp_file.get_data(self._file_path)
            print self._full_data
            self.emit(QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'), self._full_data, 0)

        for i in xrange(self._decimation_tiers-1, 0, -1):
            current_decimation_factor = self._decimation_factor**i
            decimated_data = ts.decimate_data(self._full_data, current_decimation_factor)
            self.emit(QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'), decimated_data, i)

        return
