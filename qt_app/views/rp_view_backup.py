import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/qt_app/pyqtgraph-0.9.10')
import pyqtgraph as pg

import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui

class RPView(QtGui.QWidget):
    def __init__(self, parent_widget = None, parent_model = None):
        self._parent_widget = parent_widget
        self._parent_model = parent_model
        super(RPView, self).__init__(parent = self._parent_widget)

        #self._parent_widget._mdi.addSubWindow(self)

        self._subwindow = None
        self._main_plot = None



        #new_subwindow.show()


        #self._graphics_layout = pg.GraphicsLayoutWidget(parent = self._parent_widget)
        #self._baseline_plot = self._graphics_layout.addPlot()
        #self._parent_widget.layout().addWidget(self._graphics_layout)

        self.setup_subwindow()
        self.setup_main_plot()

        self.setup_event_plot()
        self.setup_controls()

        self.setup_layout()

        self.show()


    def setup_subwindow(self):
        self._subwindow = self._parent_widget._mdi.addSubWindow(self)
        self._subwindow.setWindowTitle('Resistive pulse')

        return

    def setup_main_plot(self):
        self._main_plot = pg.PlotWidget(parent = self)
        main_plot_policy = QtGui.QSizePolicy()
        main_plot_policy.setVerticalStretch(3)
        main_plot_policy.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
        main_plot_policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self._main_plot.setSizePolicy(main_plot_policy)

        return

    def setup_bottom_pane(self):
        return


    def setup_event_plot(self):



        return

    def setup_controls(self):




        self._controls_pane = QtGui.QWidget(parent = self)



        controls_pane_policy = QtGui.QSizePolicy()
        controls_pane_policy.setVerticalStretch(1)
        controls_pane_policy.setHorizontalStretch(4)
        controls_pane_policy.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
        controls_pane_policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self._controls_pane.setSizePolicy(controls_pane_policy)



        self._event_plot = pg.PlotWidget(parent = self)
        event_plot_size_policy = QtGui.QSizePolicy()
        event_plot_size_policy.setVerticalStretch(1)
        event_plot_size_policy.setHorizontalStretch(1)
        event_plot_size_policy.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
        #event_plot_size_policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self._event_plot.setSizePolicy(event_plot_size_policy)





        return



    def setup_layout(self):
        self._layout_0 = QtGui.QGridLayout()
        self._layout_0.addWidget(self._main_plot, 0, 0, 1, 2)
        self._layout_0.addWidget(self._event_plot, 1, 0)
        self._layout_0.addWidget(self._controls_pane, 1, 1)
        self.setLayout(self._layout_0)



        return

    def show(self):
        self._subwindow.show()

        return


    def receive_update_file_name(self):
        return

    def receive_update_data(self):
        self._main_plot.plot(self._parent_model.data)
