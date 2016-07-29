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

        self._subwindow = None
        self._main_plot = None

        self._search_data_toggle = None

        self.setup_subwindow()
        self.setup_main_plot()

        self.setup_event_plot()
        self.setup_controls()

        self.setup_layout()

        self._event_plot_items = []



        self.show()


    def setup_subwindow(self):
        self._subwindow = self._parent_widget._mdi.addSubWindow(self)
        self._subwindow.setWindowTitle('Resistive pulse')

        return

    def setup_main_plot(self):

        pen_1 = QtGui.QPen(QtGui.QColor(200,200,200))
        pen_2 = QtGui.QPen(QtGui.QColor(50,200,50))
        pen_2.setWidth(.015)

        self._main_plot = pg.PlotWidget(parent = self)

        self._main_plot_item = pg.PlotDataItem()
        self._main_plot_item.setPen(pen_1)
        self._main_plot.addItem(self._main_plot_item)

        self._baseline_plot_item = pg.PlotDataItem()
        self._baseline_plot_item.setPen(pen_2)
        self._main_plot.addItem(self._baseline_plot_item)

        self._thresh_high_plot_item = pg.PlotDataItem()
        self._thresh_high_plot_item.setPen(pen_2)
        self._main_plot.addItem(self._thresh_high_plot_item)

        self._thresh_low_plot_item = pg.PlotDataItem()
        self._thresh_low_plot_item.setPen(pen_2)
        self._main_plot.addItem(self._thresh_low_plot_item)

        main_plot_policy = QtGui.QSizePolicy()
        main_plot_policy.setVerticalStretch(5)
        main_plot_policy.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
        main_plot_policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self._main_plot.setSizePolicy(main_plot_policy)

        return


    def setup_event_plot(self):
        self._event_plot = pg.PlotWidget(parent = self)
        event_plot_size_policy = QtGui.QSizePolicy()
        event_plot_size_policy.setVerticalStretch(2)
        event_plot_size_policy.setHorizontalStretch(1)
        event_plot_size_policy.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
        event_plot_size_policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self._event_plot.setSizePolicy(event_plot_size_policy)

        self._event_plot_item = pg.PlotDataItem()
        self._event_plot.addItem(self._event_plot_item)

        return

    def setup_controls(self):
        # Set up pane
        self._controls_pane = QtGui.QWidget(parent = self)
        controls_pane_policy = QtGui.QSizePolicy()
        controls_pane_policy.setVerticalStretch(2)
        controls_pane_policy.setHorizontalStretch(4)
        controls_pane_policy.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
        controls_pane_policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self._controls_pane.setSizePolicy(controls_pane_policy)


        # Set up buttons
        self._show_baseline_button = QtGui.QPushButton('Show baseline', parent = self._controls_pane)
        self._show_baseline_button.setGeometry(0,0,100,100)

        self._find_events_button = QtGui.QPushButton('Find events', parent = self._controls_pane)
        self._find_events_button.setGeometry(100,0,100,100)

        # Set up fields
        self._baseline_avg_length_field = QtGui.QLineEdit(parent = self._controls_pane)
        self._baseline_avg_length_field.setGeometry(0,100,100,100)


        self._trigger_sigma_threshold_field = QtGui.QLineEdit(parent = self._controls_pane)
        self._trigger_sigma_threshold_field.setGeometry(100,100,100,100)


        self._max_search_length_field = QtGui.QLineEdit(parent = self._controls_pane)
        self._max_search_length_field.setGeometry(200,100,100,100)

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
        self._main_plot.show()
        self._event_plot.show()

        return

    def enable_ui(self, enable):
        self._show_baseline_button.setEnabled(enable)
        self._find_events_button.setEnabled(enable)
        self._baseline_avg_length_field.setEnabled(enable)
        self._trigger_sigma_threshold_field.setEnabled(enable)
        self._max_search_length_field.setEnabled(enable)
        return
