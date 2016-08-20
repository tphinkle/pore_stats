import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/')
import resistive_pulse as rp
from rp_model import RPModel
import rp_predictor

import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import time
import csv
import numpy as np

class RPController(QtCore.QObject):

    # KEY MAPPINGS
    KEY_1 = 49
    KEY_2 = 50
    KEY_LARROW = 16777234
    KEY_RARROW = 16777236

    ML_FILE_PATH = '/home/preston/Desktop/Science/Research/pore_stats/qt_app/ML/event_predictions_train'


    rp_max_data_points = 100000000

    def __init__(self, main_model, main_view):
        self._main_model = main_model
        self._main_view = main_view

        super(RPController, self).__init__()

    def add_rp(self):
        """
        * Description: Creates a resistive pulse UI group and model.
        * Return: None.
        * Arguments:
        """
        # Get file name to load
        file_path = QtGui.QFileDialog.getOpenFileName(parent = self._main_view,\
            directory = '/home/preston/Desktop/Science/Research/cancer_cells/data/')

        if file_path:

            # Create new RP model and set file
            new_rp_model = self._main_model.create_rp_model(self)

            new_rp_model.set_active_file(file_path)


            # Create new RP view, subscribe view to model
            new_rp_view = self._main_view.create_rp_view(parent_model = new_rp_model)

            # Connect signals to slots
            self.add_rp_slots(new_rp_model, new_rp_view)

            # Set defaults
            self.set_rp_view_defaults(new_rp_view)

            new_rp_model.load_main_ts()

            new_rp_model._main_ts._visible = True

            self.plot_main_data(new_rp_model, new_rp_view, (0,1))

            new_rp_view.setWindowTitle('Resistive Pulse--'+file_path)

        return


    def add_rp_slots(self, rp_model, rp_view):

        # Model -> View
        rp_model.busy.connect(lambda busy:\
         self.enable_ui(rp_model, rp_view, not busy))

        rp_model.event_added.connect(lambda event:\
            self.plot_new_event(rp_model, rp_view, event))
        rp_model.targeted_event_changed.connect(lambda targeted_event:\
            self.plot_targeted_event(rp_model, rp_view, targeted_event))
        rp_model.events_cleared.connect(lambda:\
            self.clear_events(rp_model, rp_view))

        """
        UI -> Model
        """

        # Buttons
        rp_view._show_main_data_button.clicked.connect(lambda clicked: \
            self.show_main_button_clicked(rp_model, rp_view, clicked))

        rp_view._show_baseline_button.clicked.connect(lambda clicked: \
            self.show_baseline_button_clicked(rp_model, rp_view, clicked))

        rp_view._find_events_button.clicked.connect(lambda clicked: \
            self.find_events_button_clicked(rp_model, rp_view, clicked))

        rp_view._filter_data_button.clicked.connect(lambda clicked: \
            self.filter_data_button_clicked(rp_model, rp_view, clicked))

        rp_view._save_events_button.clicked.connect(lambda clicked: \
            self.save_events_button_clicked(rp_model, rp_view, clicked))

        rp_view._next_event_button.clicked.connect(lambda clicked: \
            self.target_next_event(rp_model, rp_view))

        rp_view._previous_event_button.clicked.connect(lambda clicked: \
            self.target_previous_event(rp_model, rp_view))

        rp_view._predict_button.clicked.connect(lambda clicked: \
            self.predict_select_unselect(rp_model, rp_view))


        # Check boxes
        rp_view._use_main_checkbox.stateChanged.connect(lambda state: \
            self.use_main_checkbox_stateChanged(rp_model, rp_view, state))

        rp_view._use_filtered_checkbox.stateChanged.connect(lambda state: \
            self.use_filtered_checkbox_stateChanged(rp_model, rp_view, state))


        # Fields

        rp_view._trigger_sigma_threshold_field.textChanged.connect(lambda text: \
            self.trigger_sigma_threshold_field_changed(rp_model, rp_view, text))

        rp_view._baseline_avg_length_field.textChanged.connect(lambda text: \
            self.baseline_avg_length_field_changed(rp_model, rp_view, text))

        rp_view._max_search_length_field.textChanged.connect(lambda text: \
            self.max_search_length_field_changed(rp_model, rp_view, text))

        rp_view._filter_frequency_field.textChanged.connect(lambda text: \
            self.filter_frequency_field_changed(rp_model, rp_view, text))

        # Plot interactions
        rp_view._main_plot.sigRangeChanged.connect(lambda rng: \
            self.main_plot_range_changed(rp_model, rp_view, rng))

        rp_view._main_plot.scene().sigMouseClicked.connect(lambda click: \
            self.main_plot_clicked(rp_model, rp_view, click))

        rp_view._targeted_event_plot.scene().sigMouseClicked.connect(lambda click: \
            self.targeted_event_plot_clicked(rp_model, rp_view, click))

        # Keyboard events
        rp_view.key_pressed.connect(lambda key: \
            self.catch_key_press(rp_model, rp_view, key))

        return

    def catch_key_press(self, rp_model, rp_view, key):
        #print 'key_press:', key.key()
        key_press = key.key()

        if key_press == self.KEY_1:
            self.select_event(rp_model, rp_view, rp_model._event_manager._targeted_event)

        elif key_press == self.KEY_2:
            self.unselect_event(rp_model, rp_view, rp_model._event_manager._targeted_event)

        elif key_press == self.KEY_LARROW:
            self.target_previous_event(rp_model, rp_view)

        elif key_press == self.KEY_RARROW:
            self.target_next_event(rp_model, rp_view)

        return




    def set_rp_view_defaults(self, rp_view):
        rp_view._baseline_avg_length_field.setText('250')
        rp_view._trigger_sigma_threshold_field.setText('3')
        rp_view._max_search_length_field.setText('10000')
        rp_view._filter_frequency_field.setText('1000')

        return

    def enable_ui(self, rp_model, rp_view, enable):
        rp_view.enable_ui(enable)
        return

    def main_plot_range_changed(self, rp_model, rp_view, rng):
        viewRange = rng.viewRange()

        ti = viewRange[0][0]
        tf = viewRange[0][1]

        t_range = (ti,tf)

        if rp_model._main_ts._visible == True:
            self.plot_main_data(rp_model, rp_view, t_range)

        if rp_model._baseline_ts._visible == True:
            self.plot_baseline_data(rp_model, rp_view, t_range)

        if rp_model._filtered_ts._visible == True:
            self.plot_filtered_data(rp_model, rp_view, t_range)

        return




    def show_main_button_clicked(self, rp_model, rp_view, clicked):
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
        if rp_model._main_ts._display_ready == True:
            main_data = rp_model.get_main_display_data(t_range)
            rp_view._main_plot_item.setData(main_data[:,0], main_data[:,1])

        return


    def show_baseline_button_clicked(self, rp_model, rp_view, clicked):

        if rp_model._baseline_ts._visible == False:
            viewRange = rp_view._main_plot.viewRange()

            ti = viewRange[0][0]
            tf = viewRange[0][1]

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

        baseline_data = rp_model.get_baseline_display_data(t_range)

        rp_view._baseline_plot_item.setData(baseline_data)



        pos_thresh_data = rp_model.get_pos_thresh_display_data(t_range)
        rp_view._pos_thresh_plot_item.setData(pos_thresh_data)


        neg_thresh_data = rp_model.get_neg_thresh_display_data(t_range)
        rp_view._neg_thresh_plot_item.setData(neg_thresh_data)

        return


    def filter_data_button_clicked(self, rp_model, rp_view, clicked):

        if rp_model._filtered_ts._visible == False:

            viewRange = rp_view._main_plot.viewRange()

            t_i = viewRange[0][0]
            t_f = viewRange[0][1]

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
        if rp_model._filtered_ts._display_ready == True:
            filtered_data = rp_model.get_filtered_display_data(t_range)
            rp_view._filtered_plot_item.setData(filtered_data)

        return




    def find_events_button_clicked(self, rp_model, rp_view, clicked):
        self.clear_events(rp_model, rp_view)

        if rp_view._use_main_checkbox.checkState() == QtCore.Qt.Checked:
            rp_model.find_events(filter = False)

        else:
            rp_model.find_events(filter = True)
        return

    def clear_events(self, rp_model, rp_view):
        # Clear view
        for event_plot_item in rp_view._event_plot_items:
            event_plot_item.clear()
        rp_view._event_plot_items = []

        # Clear model
        rp_model._event_manager.clear_events()

        return








    def save_events_button_clicked(self, rp_model, rp_view, clicked):

        rp_model.save_events()

        self.save_events_ML(rp_model, rp_view)


        return

    def predict_select_unselect(self, rp_model, rp_view):
        predictor = rp_predictor.RPPredictor()
        predictor.load_model('/home/preston/Desktop/Science/Research/pore_stats/qt_app/ML/params/interp_100_0.pkl')
        predictor._test_features = predictor.get_features(rp_model._event_manager._events)
        predictions = predictor.make_predictions_proba(predictor._test_features)

        for i, event in enumerate(rp_model._event_manager._events):
            print 'prediction = ', predictions[i][1]
            if predictions[i][1] >= 0.5:
                self.select_event(rp_model, rp_view, event)

            else:
                self.unselect_event(rp_model, rp_view, event)



        return



    def target_previous_event(self, rp_model, rp_view):
        rp_model._event_manager.decrement_targeted_event()
        targeted_event = rp_model._event_manager._targeted_event

        self.plot_targeted_event(rp_model, rp_view, targeted_event)
        return

    def target_next_event(self, rp_model, rp_view):
        rp_model._event_manager.increment_targeted_event()
        targeted_event = rp_model._event_manager._targeted_event

        self.plot_targeted_event(rp_model, rp_view, targeted_event)
        return


    def use_main_checkbox_stateChanged(self, rp_model, rp_view, state):
        if rp_view._use_main_checkbox.checkState() == QtCore.Qt.Checked:
            rp_view._use_filtered_checkbox.setCheckState(QtCore.Qt.Unchecked)
        else:
            rp_view._use_filtered_checkbox.setCheckState(QtCore.Qt.Checked)

        return

    def use_filtered_checkbox_stateChanged(self, rp_model, rp_view, state):
        if rp_view._use_filtered_checkbox.checkState() == QtCore.Qt.Checked:
            rp_view._use_main_checkbox.setCheckState(QtCore.Qt.Unchecked)
        else:
            rp_view._use_main_checkbox.setCheckState(QtCore.Qt.Checked)

        return

    def baseline_avg_length_field_changed(self, rp_model, rp_view, text):
        rp_model.set_baseline_avg_length(int(text))
        return

    def trigger_sigma_threshold_field_changed(self, rp_model, rp_view, text):
        rp_model.set_trigger_sigma_threshold(float(text))
        return

    def max_search_length_field_changed(self, rp_model, rp_view, text):
        rp_model.set_max_search_length(int(text))
        return

    def filter_frequency_field_changed(self, rp_model, rp_view, text):
        rp_model.set_filter_frequency(int(text))
        return

    def plot_new_event(self, rp_model, rp_view, event):
        rp_view.plot_new_event(event._data, rp_model._event_manager.is_selected(event))

        return

    def plot_targeted_event(self, rp_model, rp_view, targeted_event):
        pen = rp_view.pen_0
        if rp_model._event_manager.is_selected(targeted_event):
            pen = rp_view.pen_2
        else:
            pen = rp_view.pen_4
        rp_view._targeted_event_plot_item.setPen(pen)
        rp_view._targeted_event_plot_item.setData(targeted_event._data)

        x = float((np.max(targeted_event._data[:,0])+np.min(targeted_event._data[:,0]))/2.)
        y = float(np.max(targeted_event._data[:,1]))


        rp_view._targeted_event_marker_scatter_item.setData([x], [y])

        return

    def main_plot_clicked(self, rp_model, rp_view, mouse_click):
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
            rp_model._event_manager.set_targeted_event(clicked_event)




            if mouse_click.double():
                self.toggle_event(rp_model, rp_view, clicked_event)

            self.plot_targeted_event(rp_model, rp_view, clicked_event)

        return

    def toggle_event(self, rp_model, rp_view, event):
        if not rp_model._event_manager.is_selected(event):
            self.select_event(rp_model, rp_view, event)
        else:
            self.unselect_event(rp_model, rp_view, event)

    def get_event_plot_item(self, rp_model, rp_view, event):
        event_plot_item = None
        epsilon = 1./rp_model._main_ts._sampling_frequency
        for plot_item in rp_view._event_plot_items:

            if abs(plot_item.dataBounds(0)[0] - event._data[0,0]) <= epsilon:
                event_plot_item = plot_item
                break

        return event_plot_item

    def select_event(self, rp_model, rp_view, event):
        print 'select!'
        rp_model._event_manager.select_event(event)
        event_plot_item = self.get_event_plot_item(rp_model, rp_view, event)
        event_plot_item.setPen(rp_view.pen_2)
        if rp_model._event_manager.is_targeted(event):
            rp_view._targeted_event_plot_item.setPen(rp_view.pen_2)
        return

    def unselect_event(self, rp_model, rp_view, event):
        print 'unselect!'
        rp_model._event_manager.unselect_event(event)
        event_plot_item = self.get_event_plot_item(rp_model, rp_view, event)
        event_plot_item.setPen(rp_view.pen_4)
        if rp_model._event_manager.is_targeted(event):
            rp_view._targeted_event_plot_item.setPen(rp_view.pen_4)
        return



    def targeted_event_plot_clicked(self, rp_model, rp_view, mouse_click):
        pass
        #if mouse_click.double():
            #rp_model._event_manager.toggle_selected(rp_model._event_manager._targeted_event)




    def save_events_ML(self, rp_model, rp_view):

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
