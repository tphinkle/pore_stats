# /qt_app/conts/rp_controller.py
"""
* Contains RPController class.

* Sections:
    1. Imports
    2. Classes
        - RPController(QtCore.QObject)

"""

"""
Imports
"""
import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/')
import resistive_pulse as rp
from rp_model import RPModel
import rp_predictor
import pyqtgraph as pg

import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import time
import csv
import numpy as np
import pyqtgraph.functions as fn
import time

"""
Classes
"""

class RPController(QtCore.QObject):

    """
    This is the Controller that handles all resistive pulse functionality. The user interacts
    with the GUI via signals such as clicks and key presses. RPController makes signal/slot
    connections relevant to resistive pulse analysis in its constructor. User interactions are
    routed to RPController via these signals. RPController then interprets the signals and
    asks the resistive pulse Model to update its state. RPController directly updates the View
    based on the Model state change.
    """

    # KEY MAPPINGS
    KEY_1 = 49
    KEY_2 = 50
    KEY_3 = 51
    KEY_4 = 52
    KEY_LSHIFT = 16777248
    KEY_LARROW = 16777234
    KEY_RARROW = 16777236
    KEY_LCTRL = 16777249
    KEY_DEL = 16777223
    KEY_F1 = 16777264
    KEY_F2 = 16777265

    # Temp placement for training data path
    ML_FILE_PATH = '/home/preston/Desktop/Science/Research/pore_stats/qt_app/ML/event_predictions_train'
    default_rpfile_directory = '/home/preston/Desktop/Science/Research/cancer_cells/data/'

    def __init__(self, main_model, main_view):
        self._main_model = main_model
        self._main_view = main_view
        super(RPController, self).__init__()

        # Get file_path
        file_path = QtGui.QFileDialog.getOpenFileName(parent = self._main_view,\
            directory = RPController.default_rpfile_directory)


        # Create new RP Model and set file
        new_rp_model = self._main_model.create_rp_model(self)
        new_rp_model.set_active_file(file_path)

        # Create new RP View, subscribe View to Model
        new_rp_view = self._main_view.create_rp_view()#parent_model = new_rp_model)

        # Connect signals to slots
        self.add_rp_slots(new_rp_model, new_rp_view)

        # Set default state for View
        self.set_rp_view_defaults(new_rp_view)

        # Tell the model to load the time series data
        new_rp_model.load_main_ts()

        # Make sure the main time series is set to visible when the data is ready
        self.toggle_show_main_ts(new_rp_model, new_rp_view)

        new_rp_view.setWindowTitle('Resistive Pulse--'+file_path)


        # Couldn't open the file; most likely due to unrecognizable file type.
        #file_type = file_path.split('.')[-1]
        #raise TypeError('Could not open file of type ' + str(file_type))

        return


    def add_rp_slots(self, rp_model, rp_view):
        """
        * Description: Establishes all signal/slot connections relevant to this instance
                       of RP analysis, including signals that come from the associated
                       RP Model AND View.
        * Return:
        * Arguments:
            - rp_model: The Model associated with this instance of RP analysis.
            - rp_view: The view associated with this instance of RP analysis. Origin of
              GUI based signals.
        """

        # Model -> View
        rp_model.busy.connect(lambda busy:\
         self.enable_ui(rp_model, rp_view, not busy))

        rp_model.event_added.connect(lambda:\
            self.plot_all_events(rp_model, rp_view))
        rp_model.targeted_event_changed.connect(lambda targeted_event:\
            self.plot_targeted_event(rp_model, rp_view, targeted_event))
        rp_model.events_cleared.connect(lambda:\
            self.clear_events(rp_model, rp_view))

        """
        UI -> Model
        """

        # Buttons
        rp_view._show_main_data_button.clicked.connect(lambda: \
            self.toggle_show_main_ts(rp_model, rp_view))

        rp_view._show_baseline_button.clicked.connect(lambda: \
            self.toggle_show_baseline_ts(rp_model, rp_view))

        rp_view._find_events_button.clicked.connect(lambda: \
            self.find_events(rp_model, rp_view))

        rp_view._filter_data_button.clicked.connect(lambda: \
            self.toggle_show_filtered_ts(rp_model, rp_view))

        rp_view._save_events_button.clicked.connect(lambda: \
            self.save_events(rp_model, rp_view))

        rp_view._next_event_button.clicked.connect(lambda: \
            self.target_next_event(rp_model, rp_view))

        rp_view._previous_event_button.clicked.connect(lambda: \
            self.target_previous_event(rp_model, rp_view))

        rp_view._predict_button.clicked.connect(lambda: \
            self.predict_select_unselect(rp_model, rp_view))

        rp_view._select_all_button.clicked.connect(lambda: \
            self.select_all_events(rp_model, rp_view))

        rp_view._unselect_all_button.clicked.connect(lambda: \
            self.unselect_all_events(rp_model, rp_view))


        # Check boxes
        rp_view._use_main_checkbox.stateChanged.connect(lambda: \
            self.validate_use_main_checkstate(rp_model, rp_view))

        rp_view._use_filtered_checkbox.stateChanged.connect(lambda: \
            self.validate_use_filtered_checkstate(rp_model, rp_view))


        # Fields

        rp_view._trigger_sigma_threshold_field.textChanged.connect(lambda text: \
            self.change_trigger_sigma_threshold(rp_model, rp_view, text))

        rp_view._baseline_avg_length_field.textChanged.connect(lambda text: \
            self.change_baseline_avg_length(rp_model, rp_view, text))

        rp_view._max_search_length_field.textChanged.connect(lambda text: \
            self.change_max_search_length(rp_model, rp_view, text))

        rp_view._filter_frequency_field.textChanged.connect(lambda text: \
            self.change_filter_frequency(rp_model, rp_view, text))

        # Plot interactions
        rp_view._main_plot.sigRangeChanged.connect(lambda rng: \
            self.update_main_plot_range(rp_model, rp_view, rng))

        rp_view._main_plot.scene().sigMouseClicked.connect(lambda click: \
            self.main_plot_clicked(rp_model, rp_view, click))

        rp_view._targeted_event_plot.scene().sigMouseClicked.connect(lambda click: \
            self.targeted_event_plot_clicked(rp_model, rp_view, click))

        rp_view._stats_plot_item.sigClicked.connect(lambda scatter_plot_item, points: \
            self.handle_stats_plot_click(rp_model, rp_view, scatter_plot_item, points))

        #rp_view._stats_plot.sigMouseHover(lambda: \
            #self.handle_stats_plot_hover(rp_model, rp_view))

        rp_view._lcursor.sigPositionChanged.connect(lambda: \
            self.update_main_plot_text(rp_model, rp_view))

        rp_view._rcursor.sigPositionChanged.connect(lambda: \
            self.update_main_plot_text(rp_model, rp_view))

        # Keyboard events
        rp_view.key_pressed.connect(lambda key: \
            self.catch_key_press(rp_model, rp_view, key))

        return

    def handle_stats_plot_hover(self, rp_model, rp_view):
        print 'hovering!\n', time.time()
        return

    def catch_key_press(self, rp_model, rp_view, key):
        """
        * Description: Handles all keyboard stroke press redirects caught by this Controller's
          View.
        * Return:
        * Arguments:
            - key: The integer ID of the key pressed. Integer ID's are matched to idiomatic
              class variables such as 'KEY_RARROW' (the right arrow key).
        """
        #print 'key_press:', key.key()
        key_press = key.key()
        print key_press


        if key_press == self.KEY_1:
            self.select_event(rp_model, rp_view, rp_model._event_manager._targeted_event)

        elif key_press == self.KEY_2:
            self.unselect_event(rp_model, rp_view, rp_model._event_manager._targeted_event)

        elif key_press == self.KEY_LARROW:
            self.target_previous_event(rp_model, rp_view)

        elif key_press == self.KEY_RARROW:
            self.target_next_event(rp_model, rp_view)

        elif key_press == self.KEY_F1:
            self.toggle_cursor_mode(rp_model, rp_view)

        elif key_press == self.KEY_F2:
            self.toggle_stats_roi_mode(rp_model, rp_view)

        elif key_press == self.KEY_3:
            self.select_events_roi(rp_model, rp_view)

        elif key_press == self.KEY_4:
            self.unselect_events_roi(rp_model, rp_view)

        return


    def toggle_cursor_mode(self, rp_model, rp_view):
        """
        * Description: Sets cursor mode for rp_view, which when on allows users to drop a
          vertical line onto the screen indicating the event search range.
        * Return:
        * Arguments:
            - on: Boolean state (turn on vs. turn off)
        """
        rp_view._lcursor.setVisible(not rp_view._lcursor.isVisible())
        rp_view._rcursor.setVisible(not rp_view._rcursor.isVisible())
        self.update_main_plot_text(rp_model, rp_view)

        return

    def toggle_stats_roi_mode(self, rp_model, rp_view):
        """
        * Description: Toggles the visibility of the ROI in the stats plot.
        * Return:
        * Arguments:
            - on: Boolean state (turn on vs. turn off)
        """
        rp_view._stats_plot_roi.setVisible(not rp_view._stats_plot_roi.isVisible())

        return




    def set_rp_view_defaults(self, rp_view):
        """
        * Description: Sets the widgets in the View associated with this Controller to
          show their default values. Example widgets include search parameter QLineEdits.
        * Return:
        * Arguments:
        """
        rp_view._baseline_avg_length_field.setText('250')
        rp_view._trigger_sigma_threshold_field.setText('3')
        rp_view._max_search_length_field.setText('10000')
        rp_view._filter_frequency_field.setText('1000')

        return

    def enable_ui(self, rp_model, rp_view, enable):
        """
        * Description: Issues a command to the View associated with this Controller to
          enable/disable all of its command UI elements. For example, UI elements may
          need to be disabled if the Model is performing heavy computation (e.g. finding
          events)
        * Return:
        * Arguments:
            - enable: Boolean value corresponding to whether to enable (True) or
              disable (False).
        """
        rp_view.enable_ui(enable)
        return

    def update_main_plot_range(self, rp_model, rp_view, rng):
        """
        * Description: Signal received when the user changes either axis in the main plot.
          This signal routes to methods telling the main, baseline, and filtered
          data to plot.
        * Return:
        * Arguments:
          - rng: The PyQtGraph.ViewBox() that contains all information about the new
            data range in the plot.
        """
        view_range = rng.viewRange()

        ti = view_range[0][0]
        tf = view_range[0][1]

        t_range = (ti,tf)

        if rp_model._main_ts._visible == True:
            self.plot_main_data(rp_model, rp_view, t_range)

        if rp_model._baseline_ts._visible == True:
            self.plot_baseline_data(rp_model, rp_view, t_range)

        if rp_model._filtered_ts._visible == True:
            self.plot_filtered_data(rp_model, rp_view, t_range)

        self.update_main_plot_text(rp_model, rp_view)

        return

    def toggle_show_main_ts(self, rp_model, rp_view):
        """
        * Description: Signal received when the 'Show/Hide raw' button is clicked. First
          checks to see if the main time series is visible or not. e.g. if not visible,
          routes to method that plots the main time series data. After updating the state
          of the plot and displaying the data, the function ends by toggling the state of
          the button.
        * Return:
        * Arguments:
        """
        if rp_model._main_ts._visible == False:

            viewRange = rp_view._main_plot.viewRange()
            ti = viewRange[0][0]
            tf = viewRange[0][1]

            t_range = (ti,tf)

            rp_model._main_ts._visible = True

            self.plot_main_data(rp_model, rp_view, t_range)

            rp_view._show_main_data_button.setText('Hide\nraw')

        else:
            rp_model._main_ts._visible = False
            rp_view._main_plot_item.setData([0,0])
            rp_view._show_main_data_button.setText('Show\nraw')


    def plot_main_data(self, rp_model, rp_view, t_range):
        """
        * Description: First checks to see if the RPModel._main_ts is ready to display.
          If so, requests data from the RPModel._main_ts and sets the View's
          rp_view._main_plot_item with that data. If the data is not ready to display,
          harmlessly passes.
        * Arguments:
            - t_range: The range of data requested from the Model's time-series.
        """
        if rp_model._main_ts._display_ready == True:
            main_data = rp_model.get_main_display_data(t_range)
            rp_view._main_plot_item.setData(main_data[:,0], main_data[:,1])

        else:
            pass

        return


    def toggle_show_baseline_ts(self, rp_model, rp_view):
        """
        * Description: Signal received when the 'Show/Hide raw' button is clicked. Toggles
          the visibility status of the rp_model._baseline_ts. E.g., if the data is visible
          at the time of the click, it will be made invisible. If the data is toggled
          to be visible, routes to the method that plots the data. After updating the
          state of the plot and displaying the data, the function ends by toggling the
          state of the button.
        * Return:
        * Arguments:
        """

        if rp_model._baseline_ts._visible == False:
            view_range = rp_view._main_plot.viewRange()

            ti = view_range[0][0]
            tf = view_range[0][1]

            t_range = (ti,tf)

            rp_model.load_baseline_ts()

            rp_model._baseline_ts._visible = True
            rp_model._pos_thresh_ts._visible = True
            rp_model._neg_thresh_ts._visible = True

            self.plot_baseline_data(rp_model, rp_view, t_range)

            rp_view._show_baseline_button.setText('Hide\nbaseline')

        else:
            rp_model._baseline_ts._visible = False
            rp_view._baseline_plot_item.setData([0,0])

            rp_model._pos_thresh_ts._visible = False
            rp_view._pos_thresh_plot_item.setData([0,0])

            rp_model._neg_thresh_ts._visible = False
            rp_view._neg_thresh_plot_item.setData([0,0])

            rp_view._show_baseline_button.setText('Show\nbaseline')

        return

    def plot_baseline_data(self, rp_model, rp_view, t_range):
        """
        * Description: First checks to see if the rp_model._baseline_ts is ready to display.
          If so, requests data from the rpmodel._baseline_ts and sets the View's
          rp_view._baseline_plot_item with that data. If the data is not ready to display,
          harmlessly passes.
        * Arguments:
            - t_range: The range of data requested from the Model's time-series.
        """

        baseline_data = rp_model.get_baseline_display_data(t_range)

        rp_view._baseline_plot_item.setData(baseline_data)



        pos_thresh_data = rp_model.get_pos_thresh_display_data(t_range)
        rp_view._pos_thresh_plot_item.setData(pos_thresh_data)


        neg_thresh_data = rp_model.get_neg_thresh_display_data(t_range)
        rp_view._neg_thresh_plot_item.setData(neg_thresh_data)

        return


    def toggle_show_filtered_ts(self, rp_model, rp_view):
        """
        * Description: Signal received when the 'Show/Hide raw' button is clicked. Toggles
          the visibility status of the rp_model._filtered_ts. E.g., if the data is visible
          at the time of the click, it will be made invisible. If the data is toggled
          to be visible, routes to the method that plots the data. After updating the
          state of the plot and displaying the data, the function ends by toggling the
          state of the button.
        * Return:
        * Arguments:
        """
        if rp_model._filtered_ts._visible == False:

            view_range = rp_view._main_plot.viewRange()

            t_i = view_range[0][0]
            t_f = view_range[0][1]

            t_range = (t_i,t_f)


            rp_model.load_filtered_ts()

            rp_view._filter_data_button.setText('Hide\nfiltered data')
            rp_model._filtered_ts._visible = True
            self.plot_filtered_data(rp_model, rp_view, t_range)


        else:
            rp_model._filtered_ts._visible = False
            rp_view._filtered_plot_item.setData([0,0])
            rp_view._filter_data_button.setText('Show\nfiltered data')

            return


    def plot_filtered_data(self, rp_model, rp_view, t_range):
        """
        * Description: First checks to see if the RPModel._filtered_ts is ready to display.
          If so, requests data from the RPModel._filtered_ts and sets the View's
          rp_view._main_plot_item with that data. If the data is not ready to display,
          harmlessly passes.
        * Arguments:
            - t_range: The range of data requested from the Model's time-series.
        """
        if rp_model._filtered_ts._display_ready == True:
            filtered_data = rp_model.get_filtered_display_data(t_range)
            rp_view._filtered_plot_item.setData(filtered_data)

        return




    def find_events(self, rp_model, rp_view):
        """
        * Description: All Controller actions associated with finding events in an
          RPModel time series.
        * Return:
        * Arguments:
        """

        # Have to remove the current events before new ones added
        self.clear_events(rp_model, rp_view)

        # Check whether to use the filtered data or not
        if rp_view._use_main_checkbox.checkState() == QtCore.Qt.Checked:
            filter = False
        else:
            filter = True

        # Check to see if the user wants to search over an interval or the entire segment
        if rp_view._lcursor.isVisible() == True:
            ti = rp_view._lcursor.value()
            tf = rp_view._rcursor.value()

            # In case the left cursor is to the right of the right cursor...
            if ti > tf:
                temp = ti
                ti = tf
                tf = temp


        else:
            ti = -1
            tf = float('inf')

        rp_model.find_events(ti = ti, tf = tf, filter = filter)

        return

    def clear_events(self, rp_model, rp_view):
        """
        * Description: Tells the View and Model to get rid of any data associated with
          currently identified RP events.
        * Return:
        * Arguments:
        """
        # Clear rp_view._main_plot
        for event_plot_item in rp_view._event_plot_items:
            event_plot_item.clear()
        rp_view._event_plot_items = []
        rp_view._targeted_event_marker_scatter_item.clear()

        # Clear rp_view._stats_plot
        rp_view._stats_plot_item.clear()

        # Clear rp_view._targeted_event_plot
        rp_view._targeted_event_plot_item.clear()

        # Clear main plot text
        self.update_main_plot_text(rp_model, rp_view)

        # Clear Model
        rp_model._event_manager.clear_events()



        return


    def save_events(self, rp_model, rp_view):
        """
        * Description: Tells the RP Model to save the events to file.
          *Temporary*: Tells the ML module to save the training data associated with the
          detected events.
        * Return:
        * Arguments:
        """
        rp_model.save_events()

        self.save_events_ML(rp_model, rp_view)


        return

    def predict_select_unselect(self, rp_model, rp_view):
        """
        * Description: All-in-one button that makes predictions about whether present
          events will be accepted or rejected. In order for this to work, there must be a
          pickled, *trained* model in the './ML/params/' folder. First, an RPPredictor
          class is instantiated and the model is loaded from the parameter file. The
          RPPredictor then calculates the features from the list of events, and finally
          makes predictions about acceptance/rejectance. Events have their
          selected/unselected status changed.
        * Return:
        * Arguments:
        """
        predictor = rp_predictor.RPPredictor()
        predictor.load_model('/home/preston/Desktop/Science/Research/pore_stats/qt_app/ML/params/interp_100_0.pkl')
        predictor._test_features = predictor.get_features(rp_model._event_manager._events)
        predictions = predictor.make_predictions_proba(predictor._test_features)

        for i, event in enumerate(rp_model._event_manager._events):
            if predictions[i][1] >= 0.5:
                self.select_event(rp_model, rp_view, event)

            else:
                self.unselect_event(rp_model, rp_view, event)



        return

    def update_stats_plot(self, rp_model, rp_view):
        """
        * Description: Updates the stats plot markers to reflect status of selected,
          unselected, and targeted events.
        * Return:
        * Arguments:
            -
        """

        pen_types = [rp_view.pen_4, rp_view.pen_2]
        brush_types = [rp_view.brush_4, rp_view.brush_2]
        symbol_types = ['o', 'x']
        pens = [pen_types[int(rp_model._event_manager.is_selected(e))] for e in rp_model._event_manager._events]
        brushes = [brush_types[int(rp_model._event_manager.is_selected(e))] for e in rp_model._event_manager._events]
        symbols = [symbol_types[int(rp_model._event_manager.is_selected(e))] for e in rp_model._event_manager._events]
        symbols[rp_model._event_manager.get_targeted_index()] = 's'

        rp_view._stats_plot_item.setPen(pens)
        rp_view._stats_plot_item.setBrush(brushes)
        rp_view._stats_plot_item.setSymbol(symbols)

        return

    def change_targeted_event(self, rp_model, rp_view, targeted_event):
        """
        * Description: Commands the rp_model._event_manager to change its targeted event,
          and commands the View to update its rp_view._targeted_event_plot_item data and
          the marker used in the rp_view._stats_plot
        * Return:
        * Arguments:
            -
        """
        rp_model._event_manager._targeted_event = targeted_event

        self.plot_targeted_event(rp_model, rp_view, targeted_event)
        self.update_main_plot_text(rp_model, rp_view)
        self.update_stats_plot(rp_model, rp_view)

        return


    def target_previous_event(self, rp_model, rp_view):
        """
        * Description: Sets the targeted event to the event nearest the currently targeted
          event on the left. First, tells the rp_model to set the targeted event, then
          calls the relevant method to change the status of the plots reflecting the new
          targeted event.
        * Return:
        * Arguments:
        """
        # Check if alt held
        alt_held = (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier)
        ctrl_held = (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier)

        if alt_held == False and ctrl_held == False:
            # Move to next event
            targeted_event = rp_model._event_manager.get_previous_targeted_event()
        elif alt_held == True and ctrl_held == False:
            # Move to next selected event
            targeted_event = rp_model._event_manager.get_previous_targeted_selected_event()

        elif alt_held == False and ctrl_held == True:
            # Move to next unselected event
            targeted_event = rp_model._event_manager.get_previous_targeted_unselected_event()

        else:
            # Both are held, don't do anything
            return

        self.change_targeted_event(rp_model, rp_view, targeted_event)

        return

    def target_next_event(self, rp_model, rp_view):
        """
        * Description: Sets the targeted event to the event nearest the currently targeted
          event on the right. First, tells the rp_model to set the targeted event, then
          calls the relevant method to change the status of the plots reflecting the new
          targeted event.
        * Return:
        * Arguments:
        """

        # Check if alt held
        alt_held = (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier)
        ctrl_held = (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier)

        if alt_held == False and ctrl_held == False:
            # Move to next event
            targeted_event = rp_model._event_manager.get_next_targeted_event()
        elif alt_held == True and ctrl_held == False:
            # Move to next selected event
            targeted_event = rp_model._event_manager.get_next_targeted_selected_event()

        elif alt_held == False and ctrl_held == True:
            # Move to next unselected event
            targeted_event = rp_model._event_manager.get_next_targeted_unselected_event()

        else:
            # Both are held, don't do anything
            return

        self.change_targeted_event(rp_model, rp_view, targeted_event)

        return



    def validate_use_main_checkstate(self, rp_model, rp_view):
        """
        * Description: Slot connected to 'Use ____' checkbox in GUI, which allows
          user to choose whether they use the main (raw) or filtered time series when
          looking for events. Ensures that exactly one option can be checked at a time.
        * Return:
        * Arguments:
        """
        if rp_view._use_main_checkbox.checkState() == QtCore.Qt.Checked:
            rp_view._use_filtered_checkbox.setCheckState(QtCore.Qt.Unchecked)
        else:
            rp_view._use_filtered_checkbox.setCheckState(QtCore.Qt.Checked)

        return

    def validate_use_filtered_checkstate(self, rp_model, rp_view):
        """
        * Description: Slot connected to 'Use ____' checkbox in GUI, which allows
          user to choose whether they use the main (raw) or filtered time series when
          looking for events. Ensures that exactly one option can be checked at a time.
        * Return:
        * Arguments:
        """
        if rp_view._use_filtered_checkbox.checkState() == QtCore.Qt.Checked:
            rp_view._use_main_checkbox.setCheckState(QtCore.Qt.Unchecked)
        else:
            rp_view._use_main_checkbox.setCheckState(QtCore.Qt.Checked)

        return

    def change_baseline_avg_length(self, rp_model, rp_view, text):
        """
        * Description: Validates the user's entry for event search parameter
          'baseline_avg_length' and changes the parameter in the Model if accepted.
        * Return:
        * Arguments:
            - text: The proposed value for 'baseline_avg_length'
        """
        rp_model.set_baseline_avg_length(int(text))
        return

    def change_trigger_sigma_threshold(self, rp_model, rp_view, text):
        """
        * Description: Validates the user's entry for event search parameter
          'trigger_sigma_threshold' and changes the parameter in the Model if accepted.
        * Return:
        * Arguments:
            - text: The proposed value for 'trigger_sigma_threshold'.
        """
        rp_model.set_trigger_sigma_threshold(float(text))
        return

    def change_max_search_length(self, rp_model, rp_view, text):
        """
        * Description: Validates the user's entry for event search parameter
          'max_search_length' and changes the parameter in the Model if accepted.
        * Return:
        * Arguments:
            - text: The proposed value for 'max_search_length'.
        """
        rp_model.set_max_search_length(int(text))
        return

    def change_filter_frequency(self, rp_model, rp_view, text):
        """
        * Description: Validates the user's entry for event search parameter
          'filter_frequency' and changes the parameter in the Model if accepted.
        * Return:
        * Arguments:
            - text: The proposed value for 'filter_frequency'
        """
        rp_model.set_filter_frequency(int(text))
        return

    def plot_all_events(self, rp_model, rp_view):
        """
        * Description: Issues a command to rp_view to add a new event_plot_item.
        * Return:
        * Arguments:
            - event: The event to be plotted.
        """

        events = rp_model._event_manager._events

        # Tell rp view to plot the event
        for event in events:
            rp_view.plot_new_event(event._data, event._duration, event._amplitude, \
                                   rp_model._event_manager.is_selected(event))


        # Select all events
        self.select_all_events(rp_model, rp_view)

        # Set targeted event
        self.change_targeted_event(rp_model, rp_view, rp_model._event_manager._targeted_event)

        # Set stats roi
        durs = [event._duration for event in rp_model._event_manager._events]
        amps = [event._amplitude for event in rp_model._event_manager._events]
        p0 = (min(durs), min(amps))
        #p1 = (min(durs), max(amps))
        size = (max(durs), max(amps))
        #p2 = (max(durs), max(amps))
        #p3 = (max(durs), min(amps))
        handles = rp_view._stats_plot_roi.getHandles()

        rp_view._stats_plot_roi.movePoint(handles[0], pg.Point(QtCore.QPointF(p0[0], p0[1])))
        rp_view._stats_plot_roi.setSize(size)
        #rp_view._stats_plot_roi.movePoint(handles[1], pg.Point(QtCore.QPointF(p1[0], p1[1])))
        #rp_view._stats_plot_roi.movePoint(handles[2], pg.Point(QtCore.QPointF(p2[0], p2[1])))
        #rp_view._stats_plot_roi.movePoint(handles[3], pg.Point(QtCore.QPointF(p3[0], p3[1])))


        print type(handles[0])

        return

    def plot_targeted_event(self, rp_model, rp_view, targeted_event):
        """
        * Description: Commands the rp_view to update the
          rp_view._targeted_event_plot_item with the newly targeted event, to plot an x
          over the new targeted event in the main plot, and to update the
          rp_view._main_plot_text.
        * Return:
        * Arguments:
            - targeted_event: The RPEvent that is targeted.
        """
        pen = rp_view.pen_0
        if rp_model._event_manager.is_selected(targeted_event):
            pen = rp_view.pen_2
        else:
            pen = rp_view.pen_4
        rp_view._targeted_event_plot_item.setPen(pen)
        rp_view._targeted_event_plot_item.setData(targeted_event._data)

        self.update_targeted_event_marker_scatter_item(rp_model, rp_view, targeted_event)

        QtCore.QCoreApplication.processEvents()

        return

    def update_targeted_event_marker_scatter_item(self, rp_model, rp_view, targeted_event):
        """
        * Description: Updates the indicator in the main plot to show where the targeted
          event is.
        * Return:
        * Arguments:
            - targeted_event: The RPEvent that is targeted.
        """
        # Get coordinates of marker
        x = float((np.max(targeted_event._data[:,0])+np.min(targeted_event._data[:,0]))/2.)
        offset = 1000
        y = float(np.max(targeted_event._data[:,1])) + offset

        rp_view._targeted_event_marker_scatter_item.setData([x], [y])

        return


    def main_plot_clicked(self, rp_model, rp_view, mouse_click):
        """
        * Description: Slot connected to signal when the rp_view._main_plot is clicked.
          Checks coordinates of click and whether it is single or double. If coordinates
          are over an event in the plot, will respond by targeting the event if the click
          was single, or toggle select/unselect if the click was double.
        * Return:
        * Arguments:
            - mouse_click: The mouseclick instance created by the user.
        """


        click_coords = rp_view._main_plot.getPlotItem().getViewBox().mapSceneToView(mouse_click.scenePos())
        x_pos = click_coords.x()

        clicked_event = None
        for event in rp_model._event_manager._events:
            if event._data[0,0] <= x_pos and event._data[-1,0] >= x_pos:
                clicked_event = event
                break

        clicked_event_plot_item = None
        for event_plot_item in rp_view._event_plot_items:
            if event_plot_item.xData[0] <= x_pos and event_plot_item.xData[-1] >= x_pos:
                clicked_event_plot_item = event_plot_item

        if clicked_event != None and clicked_event_plot_item != None:
            self.change_targeted_event(rp_model, rp_view, clicked_event)

            if mouse_click.double():
                self.toggle_select_event(rp_model, rp_view, clicked_event)

        return





    def toggle_select_event(self, rp_model, rp_view, event):
        """
        * Description: Checks whether the event is selected or not, and calls the
          appropriate method to change the selected status to the toggled value.
        * Return:
        * Arguments:
            - event: The event to be toggled.
        """
        if not rp_model._event_manager.is_selected(event):
            self.select_event(rp_model, rp_view, event)
        else:
            self.unselect_event(rp_model, rp_view, event)

        return



    def select_event(self, rp_model, rp_view, event):
        """
        * Description: Sets the event status to selected, affecting both the state
          of the rp_model._event_manager and the plot in the rp_view.
        * Return:
        * Arguments:
            -
        """

        # Update model
        rp_model._event_manager.select_event(event)

        # Update main plot
        event_plot_item = self.get_event_plot_item(rp_model, rp_view, event)
        event_plot_item.setPen(rp_view.pen_2)

        # Update targeted event plot if the event is targeted
        if rp_model._event_manager.is_targeted(event):
            rp_view._targeted_event_plot_item.setPen(rp_view.pen_2)

        # Update the stats plot

        self.update_stats_plot(rp_model, rp_view)




        # Update main plot text
        self.update_main_plot_text(rp_model, rp_view)



        return

    def unselect_event(self, rp_model, rp_view, event):
        """
        * Description: Sets the event status to unselected, affecting both the state
          of the rp_model._event_manager and the plot in the rp_view.
        * Return:
        * Arguments:
            -
        """

        # Update model
        rp_model._event_manager.unselect_event(event)

        # Update main plot
        event_plot_item = self.get_event_plot_item(rp_model, rp_view, event)
        event_plot_item.setPen(rp_view.pen_4)

        # Update targeted event plot
        if rp_model._event_manager.is_targeted(event):
            rp_view._targeted_event_plot_item.setPen(rp_view.pen_4)

        self.update_stats_plot(rp_model, rp_view)


        # Update main plot text
        self.update_main_plot_text(rp_model, rp_view)

        return

    def get_event_plot_item(self, rp_model, rp_view, event):
        """
        * Description: Connects the RPEvent instance to its associated PlotDataItem
          in rp_view. Somewhat hacky and may need to be reworked.
        * Return: Returns the event_plot_item.
        * Arguments:
            - event: The RPEvent whose plot_data_item we are looking for.
        """
        event_plot_item = None
        epsilon = 1./rp_model._main_ts._sampling_frequency
        for plot_item in rp_view._event_plot_items:

            if abs(plot_item.dataBounds(0)[0] - event._data[0,0]) <= epsilon:
                event_plot_item = plot_item
                break

        return event_plot_item

    def get_event_stats_index(self, rp_model, rp_view, event):
        """
        * Description: Connects the RPEvent instance to its associated PlotScatterItem
          in rp_view. Somewhat hacky and may need to be reworked.
        * Return:
        * Arguments:
        """
        scatter_points = rp_view._stats_plot_item.data
        for i in range(scatter_points.shape[0]):
            if ((scatter_points[i][0] == event._duration) and\
                (scatter_points[i][1] == event._amplitude)):
                index = i
                break
            else:
                pass

        return index






    def save_events_ML(self, rp_model, rp_view):
        """
        * Description: When events are saved, this function is also called so that the
          data can be stored as training data to update the model.
        * Return:
        * Arguments:
        """

        directory = '/home/preston/Desktop/Science/Research/pore_stats/qt_app/ML/'

        date_suffix = time.strftime('%d-%m-%y')
        time_suffix = time.strftime('%I-%M-%S')

        file_path = directory + 'event_predictions_train_'+date_suffix+'_'+time_suffix

        events = rp_model._event_manager._events


        with open(file_path , 'w') as f:
            writer = csv.writer(f, delimiter = '\t')
            header_row = ['file = ', rp_model._active_file._file_path]
            writer.writerow(['file = ', rp_model._active_file._file_path])
            writer.writerow(['events = ', len(events), 'selected = ', len(rp_model._event_manager._selected_events),
            'unselected = ', len(events)-len(rp_model._event_manager._selected_events)])

            for i, event in enumerate(events):
                selected = int(rp_model._event_manager.is_selected(event))
                header_row = ['event#', i, 'length', event._data.shape[0],\
                    'baseline', event._baseline[0], event._baseline[1], event._baseline[2], event._baseline[3], 'selected = ', selected]

                writer.writerow(header_row)

                for row in event._data:
                    writer.writerow([row[0], row[1]])


        return


    def update_main_plot_text(self, rp_model, rp_view):
        """
        * Description: Updates the text in the top right of the rp_view._main_plot that
          contains info about the current decimation value, total events found, etc.
        * Return:
        * Arguments:
        """

        # Decimation line
        line_1 = 'Decimation factor: ' + str(rp_model._main_ts._current_decimation_factor) + \
        'x'

        # Event lines
        if len(rp_model._event_manager._events) == 0:
            line_2 = 'Events total: --'
            line_3 = 'Events selected: --'
            line_4 = 'Event targeted: --'

        else:
            line_2 = 'Events total: ' + str(len(rp_model._event_manager._events))
            line_3 = 'Events selected: ' + str(len(rp_model._event_manager._selected_events))
            line_4 = 'Event targeted: ' + str(rp_model._event_manager._targeted_event._id)

        # Cursor lines
        line_5 = 'Left cursor: '
        line_6 = 'Right cursor: '
        line_7 = 'Delta: '
        if rp_view._lcursor.isVisible() == True:
            lcursor_x = round(rp_view._lcursor.value(), 6)
            rcursor_x = round(rp_view._rcursor.value(), 6)
            delta_x = round(rcursor_x - lcursor_x, 6)
            if delta_x > 0:
                line_5 += str(lcursor_x) + ' s'
                line_6 += str(rcursor_x) + ' s'
            else:
                line_5 += str(rcursor_x) + ' s'
                line_6 += str(lcursor_x) + ' s'
            line_7 += str(abs(delta_x)) + ' s'

        else:
            line_5 += '--'
            line_6 += '--'
            line_7 += '--'



        rp_view._main_plot_text_item.setText(line_1 + '\n' + line_2 + '\n' + line_3 + \
        '\n' + line_4 + '\n' + line_5 + '\n' + line_6 + '\n' + line_7)

        return

    def handle_stats_plot_click(self, rp_model, rp_view, scatter_plot_item, point):
        """
        * Description:
        * Return:
        * Arguments:
            -
        """

        epsilon = 1.*10.**(-5.)

        dur = point[0].pos()[0]
        amp = point[0].pos()[1]

        for event in rp_model._event_manager._events:
            if ((abs(amp - event._amplitude) < epsilon) and (abs(dur - event._duration) < epsilon)):
                self.change_targeted_event(rp_model, rp_view, event)
                #rp_model._event_manager.set_targeted_event(event)
                #self.plot_targeted_event(rp_model, rp_view, event)

                break

        return


    def select_all_events(self, rp_model, rp_view):
        for i, event in enumerate(rp_model._event_manager._events):
            self.select_event(rp_model, rp_view, event)
        return

    def unselect_all_events(self, rp_model, rp_view):
        for i, event in enumerate(rp_model._event_manager._events):
            self.unselect_event(rp_model, rp_view, event)
        return

    def select_events_roi(self, rp_model, rp_view):
        roi_shape = rp_view._stats_plot_roi.mapToItem(rp_view._stats_plot_item, rp_view._stats_plot_roi.shape())

        points = rp_view._stats_plot_item.getData()

        #print points.shape

        points_qt = [QtCore.QPointF(pt[0], pt[1]) for pt in zip(points[0][:].tolist(), points[1][:].tolist())]


        targeted_points = [i for i in range(len(points_qt)) if roi_shape.contains(points_qt[i])]

        for i, event in enumerate([rp_model._event_manager._events[i] for i in targeted_points]):
            self.select_event(rp_model, rp_view, event)

        return

    def unselect_events_roi(self, rp_model, rp_view):
        roi_shape = rp_view._stats_plot_roi.mapToItem(rp_view._stats_plot_item, rp_view._stats_plot_roi.shape())

        points = rp_view._stats_plot_item.getData()

        #print points.shape

        points_qt = [QtCore.QPointF(pt[0], pt[1]) for pt in zip(points[0][:].tolist(), points[1][:].tolist())]


        targeted_points = [i for i in range(len(points_qt)) if roi_shape.contains(points_qt[i])]

        for i, event in enumerate([rp_model._event_manager._events[i] for i in targeted_points]):
            self.unselect_event(rp_model, rp_view, event)

        return
