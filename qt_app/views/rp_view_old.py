import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/qt_app/pyqtgraph-0.9.10')
import pyqtgraph as pg

import PyQt4.QtCore
import PyQt4.QtGui

class RPView(QObject):
    def __init__(self, parent_widget = None, parent_model = None):
        super(RPView, self).__init__(parent = parent_widget)

        # Plot
        self._baseline_plot = None

        self._parent_model = parent_model
        self._parent_model.add_subscriber_data(self)
        self._parent_model.add_subscriber_file_name(self)



        self.setup_baseline_plot()
        print 'SIG RANGE!!!!', self._baseline_plot.viewRect()

    def RangeChanged(self, x, y):
        self.range.emit(self.range)





    def setup_baseline_plot(self):
        self._baseline_plot = self.addPlot(title = 'Baseline')
        self._baseline_plot.plot(range(10), range(10))


        return



    def receive_update_file_name(self):
        pass

    def receive_update_data(self):
        new_data = self._parent_model._data
        self._baseline_plot.plot(new_data[:,0], new_data[:,1])

        return
