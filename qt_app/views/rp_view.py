import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/qt_app/pyqtgraph-0.9.10')
import pyqtgraph as pg

import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui



class RPView(QtGui.QWidget):

    key_pressed = QtCore.pyqtSignal('PyQt_PyObject')

    # Gray pen
    pen_0 = QtGui.QPen(QtGui.QColor(200,200,200))

    # Green pen
    pen_1 = QtGui.QPen(QtGui.QColor(50,200,50))

    # Blue pen
    pen_2 = QtGui.QPen(QtGui.QColor(50,50,200))
    brush_2 = QtGui.QBrush(QtGui.QColor(50, 50, 200))

    # Dark gray pen
    pen_3 = QtGui.QPen(QtGui.QColor(120,120,120))

    # Red pen
    pen_4 = QtGui.QPen(QtGui.QColor(200, 50, 50))
    brush_4 = QtGui.QBrush(QtGui.QColor(200, 50, 50, 0))

    # Green color
    main_plot_text_color = QtGui.QColor(50, 205, 50)

    def __init__(self, parent_widget = None, parent_model = None):
        self._parent_widget = parent_widget
        self._parent_model = parent_model
        super(RPView, self).__init__(parent = self._parent_widget)

        self._subwindow = None
        self._main_plot = None

        self._search_data_toggle = None

        self.setup_subwindow()
        self.setup_main_plot()

        self.setup_targeted_event_plot()
        self.setup_stats_plot()
        self.setup_controls()

        self.setup_layout()

        self._event_plot_items = []





        self.show()

    def keyPressEvent(self, key):
        self.emit(QtCore.SIGNAL('key_pressed(PyQt_PyObject)'), key)


    def setup_subwindow(self):
        self._subwindow = self._parent_widget._mdi.addSubWindow(self)
        self._subwindow.setWindowTitle('Resistive pulse')

        return

    def setup_main_plot(self):

        self._main_plot = pg.PlotWidget(parent = self)
        self._main_plot.setLabel('left', text = 'Current (uA)')
        self._main_plot.setLabel('bottom', text = 'Time (s)')

        self._main_plot_item = pg.PlotDataItem()
        self._main_plot_item.setZValue(0)
        self._main_plot_item.setPen(self.pen_0)
        self._main_plot.addItem(self._main_plot_item)

        self._baseline_plot_item = pg.PlotDataItem()
        self._baseline_plot_item.setZValue(3)
        self._baseline_plot_item.setPen(self.pen_2)
        self._main_plot.addItem(self._baseline_plot_item)

        self._pos_thresh_plot_item = pg.PlotDataItem()
        self._pos_thresh_plot_item.setZValue(3)
        self._pos_thresh_plot_item.setPen(self.pen_2)
        self._main_plot.addItem(self._pos_thresh_plot_item)

        self._neg_thresh_plot_item = pg.PlotDataItem()
        self._neg_thresh_plot_item.setZValue(3)
        self._neg_thresh_plot_item.setPen(self.pen_2)
        self._main_plot.addItem(self._neg_thresh_plot_item)

        self._filtered_plot_item = pg.PlotDataItem()
        self._filtered_plot_item.setZValue(2)
        self._filtered_plot_item.setPen(self.pen_3)
        self._main_plot.addItem(self._filtered_plot_item)

        self._targeted_event_plot_item = pg.PlotDataItem()
        self._targeted_event_plot_item.setZValue(10)
        self._targeted_event_plot_item.setPen(self.pen_4)
        self._main_plot.addItem(self._targeted_event_plot_item)

        self._targeted_event_marker_scatter_item = pg.ScatterPlotItem()
        self._targeted_event_marker_scatter_item.setZValue(10)
        self._targeted_event_marker_scatter_item.setPen(self.pen_4)
        self._targeted_event_marker_scatter_item.setSymbol('x')
        self._main_plot.addItem(self._targeted_event_marker_scatter_item)

        self._event_plot_items = []

        self._main_plot_text_item = QtGui.QTextEdit(parent = self._main_plot)
        self._main_plot_text_item.setReadOnly(True)
        self._main_plot_text_item.setAlignment(QtCore.Qt.AlignLeft)
        self._main_plot_text_item.viewport().setAutoFillBackground(False)
        self._main_plot_text_item.setFrameStyle(QtGui.QFrame.NoFrame)
        self._main_plot_text_item.setTextColor(RPView.main_plot_text_color)
        self._main_plot_text_item.setText('')
        self._main_plot_text_item.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self._main_plot_text_item.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents);



        self._main_plot_layout = QtGui.QGridLayout(self._main_plot)
        self._main_plot_layout.addWidget(self._main_plot_text_item, 0, 1, QtCore.Qt.AlignRight)
        self._main_plot_layout.setColumnStretch(0,1)


        main_plot_policy = QtGui.QSizePolicy()
        main_plot_policy.setVerticalStretch(4)
        main_plot_policy.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
        main_plot_policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self._main_plot.setSizePolicy(main_plot_policy)

        return

    def plot_new_event(self, data, dur, amp, selected = True):
        if selected:
            pen = self.pen_4
        else:
            pen = self.pen_2

        self._event_plot_items.append(\
            self._main_plot.plot(data, pen = pen, zValue = 1))

        self._stats_plot_item.addPoints([dur], [amp], pen = pen)

        return


    def setup_targeted_event_plot(self):
        self._targeted_event_plot = pg.PlotWidget(parent = self)

        self._targeted_event_plot.setLabel('left', text = 'Current (uA)')
        self._targeted_event_plot.setLabel('bottom', text = 'Time (s)')

        targeted_event_plot_size_policy = QtGui.QSizePolicy()
        targeted_event_plot_size_policy.setVerticalStretch(1)
        targeted_event_plot_size_policy.setHorizontalStretch(2)
        targeted_event_plot_size_policy.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
        targeted_event_plot_size_policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self._targeted_event_plot.setSizePolicy(targeted_event_plot_size_policy)

        self._targeted_event_plot_item = pg.PlotDataItem()
        self._targeted_event_plot.addItem(self._targeted_event_plot_item)


        parent_geometry = self._targeted_event_plot.geometry()


        fixed_size_policy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        #fixed_size_policy.setHeightForWidth(True)




        self._previous_event_button = QtGui.QPushButton('>', parent = self._targeted_event_plot)
        self._previous_event_button.setFixedSize(QtCore.QSize(50, 50))
        self._previous_event_button.setSizePolicy(fixed_size_policy)
        self._previous_event_button.setStyleSheet('background-color: rgba(255,255,255,0)')

        self._next_event_button = QtGui.QPushButton('<', parent = self._targeted_event_plot)
        self._next_event_button.setFixedSize(QtCore.QSize(50, 50))
        self._next_event_button.setSizePolicy(fixed_size_policy)
        self._next_event_button.setStyleSheet('background-color: rgba(255,255,255,0)')


        self._main_plot_layout = QtGui.QGridLayout(self._targeted_event_plot)
        self._main_plot_layout.addWidget(self._previous_event_button, 0, 0, QtCore.Qt.AlignLeft)
        self._main_plot_layout.addWidget(self._next_event_button, 0, 2, QtCore.Qt.AlignRight)
        self._main_plot_layout.setColumnStretch(0,2)


        return

    def setup_stats_plot(self):
        self._stats_plot = pg.PlotWidget(parent = self)

        self._stats_plot.setLabel('left', text = 'Current (uA)')
        self._stats_plot.setLabel('bottom', text = 'Time (s)')

        stats_plot_size_policy = QtGui.QSizePolicy()
        stats_plot_size_policy.setVerticalStretch(1)
        stats_plot_size_policy.setHorizontalStretch(2)
        stats_plot_size_policy.setVerticalPolicy(QtGui.QSizePolicy.Expanding)
        stats_plot_size_policy.setHorizontalPolicy(QtGui.QSizePolicy.Expanding)
        self._stats_plot.setSizePolicy(stats_plot_size_policy)

        self._stats_plot_item = pg.ScatterPlotItem()
        #self._stats_plot_item.setPen(RPView.pen_2)
        #self._stats_plot_item.setSymbol('o')
        self._stats_plot.addItem(self._stats_plot_item)

        parent_geometry = self._stats_plot.geometry()






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
        self._show_main_data_button = QtGui.QPushButton('Hide\nraw', parent = self._controls_pane)
        self._show_main_data_button.setGeometry(0,0,100,100)

        self._show_baseline_button = QtGui.QPushButton('Show\nbaseline', parent = self._controls_pane)
        self._show_baseline_button.setGeometry(100,0,100,100)

        self._filter_data_button = QtGui.QPushButton('Show\nfiltered data', parent = self._controls_pane)
        self._filter_data_button.setGeometry(200,0,100,100)

        self._find_events_button = QtGui.QPushButton('Find\nevents', parent = self._controls_pane)
        self._find_events_button.setGeometry(300,0,100,100)

        self._save_events_button = QtGui.QPushButton('Save\nevents', parent = self._controls_pane)
        self._save_events_button.setGeometry(400,0,100,100)

        self._predict_button = QtGui.QPushButton('Predict\nevents', parent = self._controls_pane)
        self._predict_button.setGeometry(500,0,100,100)

        # Set up checkboxes
        self._use_main_checkbox = QtGui.QCheckBox('Use raw', parent = self._controls_pane)
        self._use_main_checkbox.setCheckState(QtCore.Qt.Checked)
        self._use_main_checkbox.setGeometry(300,100,100,25)

        self._use_filtered_checkbox = QtGui.QCheckBox('Use filtered', parent = self._controls_pane)
        self._use_filtered_checkbox.setCheckState(QtCore.Qt.Unchecked)
        self._use_filtered_checkbox.setGeometry(300,125,100,25)

        # Set up labels

        self._baseline_avg_length_label = QtGui.QLabel('Baseline avg length', parent = self._controls_pane, wordWrap = True)
        self._baseline_avg_length_label.setGeometry(0,100,100,50)

        self._trigger_sigma_threshold_label = QtGui.QLabel('Sigma thresh', parent = self._controls_pane, wordWrap = True)
        self._trigger_sigma_threshold_label.setGeometry(0,150,100,50)

        self._max_search_length_label = QtGui.QLabel('Max search length', parent = self._controls_pane, wordWrap = True)
        self._max_search_length_label.setGeometry(0,200,100,50)

        self._filter_frequency_label = QtGui.QLabel('Filter frequency', parent = self._controls_pane, wordWrap = True)
        self._filter_frequency_label.setGeometry(300,200,100,50)

        # Set up fields
        self._baseline_avg_length_field = QtGui.QLineEdit(parent = self._controls_pane)
        self._baseline_avg_length_field.setGeometry(100,100,100,50)


        self._trigger_sigma_threshold_field = QtGui.QLineEdit(parent = self._controls_pane)
        self._trigger_sigma_threshold_field.setGeometry(100,150,100,50)


        self._max_search_length_field = QtGui.QLineEdit(parent = self._controls_pane)
        self._max_search_length_field.setGeometry(100,200,100,50)

        self._filter_frequency_field = QtGui.QLineEdit(parent = self._controls_pane)
        self._filter_frequency_field.setGeometry(400,200,100,50)

        return



    def setup_layout(self):
        self._layout_0 = QtGui.QGridLayout()
        self._layout_0.addWidget(self._main_plot, 0, 0, 1, 3)
        self._layout_0.addWidget(self._targeted_event_plot, 1, 0)
        self._layout_0.addWidget(self._stats_plot, 1, 1)
        self._layout_0.addWidget(self._controls_pane, 1, 2)

        self.setLayout(self._layout_0)



        return

    def show(self):
        self._subwindow.show()
        self._main_plot.show()
        self._targeted_event_plot.show()

        return


    def enable_ui(self, enable):
        self._show_main_data_button.setEnabled(enable)
        self._show_baseline_button.setEnabled(enable)
        self._find_events_button.setEnabled(enable)
        self._filter_data_button.setEnabled(enable)
        self._find_events_button.setEnabled(enable)
        self._save_events_button.setEnabled(enable)
        self._predict_button.setEnabled(enable)

        self._baseline_avg_length_field.setEnabled(enable)
        self._trigger_sigma_threshold_field.setEnabled(enable)
        self._max_search_length_field.setEnabled(enable)
        self._filter_frequency_field.setEnabled(enable)

        self._use_main_checkbox.setEnabled(enable)
        self._use_filtered_checkbox.setEnabled(enable)


        return
