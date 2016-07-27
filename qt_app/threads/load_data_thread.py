import PyQt4.QtCore as QtCore
import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/qt_app/')
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/')
import time_series as ts

class LoadDataThread(QtCore.QThread):
    def __init__(self, data, decimation_tiers, decimation_factor):
        QtCore.QThread.__init__(self)
        self._data = data
        self._decimation_tiers = decimation_tiers
        self._decimation_factor = decimation_factor


    def __del__(self):
        self.wait()

    def run(self):
        """
        * Description:
        * Return:
        * Arguments:
            -
        """
        decimated_data_list = []
        for i in xrange(self._decimation_tiers, 0, -1):
            current_decimation_factor = self._decimation_factor**i
            decimated_data=ts.decimate_data(self._data, current_decimation_factor)
            self.emit(QtCore.SIGNAL('add_tier(PyQt_PyObject, int)'), decimated_data, i)
        return
