import PyQt4.QtCore as QtCore
import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/qt_app/')
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/')
import time_series as ts
import resistive_pulse as rp
import rp_file
import math

class LoadTimeSeriesThread(QtCore.QThread):
    def __init__(self, file_name, max_pts_returned, decimation_factor):
        super(LoadDataThread,self).__init__()
        self._max_pts_returned = max_pts_returned
        self._decimation_factor = decimation_factor



    def __del__(self):
        pass


    def run(self):
        """
        * Description:
        * Return:
        * Arguments:
            -
        """

        print 'started loading data...'

        full_data = rp_file.get_data(self._file_name)

        self._max_decimation = full_data.shape[0]/self._max_pts_returned
        self._decimation_tiers = int(math.ceil(math.log(1.*full_data.shape[0]/self._max_pts_returned, self._decimation_factor)))+1

        parameters = [self._max_pts_returned, self._decimation_factor, self._max_decimation, self._decimation_tiers, ts.get_sampling_frequency(full_data)]


        self.emit(QtCore.SIGNAL('set_parameters(PyQt_PyObject)'), parameters)

        self.emit(QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'), full_data, 0)


        for i in xrange(self._decimation_tiers-1, 0, -1):
            print 'loaded tier', i,' out of ', self._decimation_tiers
            current_decimation_factor = self._decimation_factor**i
            decimated_data=ts.decimate_data(full_data, current_decimation_factor)
            self.emit(QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'), decimated_data, i)

        return
