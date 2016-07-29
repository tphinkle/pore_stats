import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/')
import resistive_pulse as rp
from rp_model import RPModel

import PyQt4.QtCore as QtCore
from PyQt4.QtGui import *
import time
import load_data_thread

class RPController(QtCore.QObject):

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
        file_path = QFileDialog.getOpenFileName(parent = self._main_view)

        if file_path:

            # Create new RP model and set file
            new_rp_model = self._main_model.create_rp_model(self)

            new_rp_model.set_active_file(file_path)


            # Create new RP view, subscribe view to model
            new_rp_view = self._main_view.create_rp_view(parent_model = new_rp_model)

            # Connect signals to slots
            self.add_rp_slots(new_rp_view, new_rp_model)

            # Set defaults
            self.set_rp_view_defaults(new_rp_view)

            new_rp_model.load_main_ts()

        return


    def add_rp_slots(self, rp_view, rp_model):
        rp_model.busy.connect(lambda busy: self.enable_ui(not busy, rp_model, rp_view))
        rp_model.event_added.connect(lambda event:\
            self.plot_event(event, rp_model, rp_view))
        rp_model.targeted_event_changed.connect(lambda targeted_event:\
            self.plot_targeted_event(targeted_event, rp_model, rp_view))
        rp_model.events_cleared.connect(lambda:\
            self.clear_events(rp_model, rp_view))

        rp_view._main_plot.sigRangeChanged.connect(lambda rng: \
            self.main_plot_range_changed(rng, rp_model, rp_view))

        rp_view._show_baseline_button.clicked.connect(lambda clicked: \
            self.show_baseline_button_clicked(clicked, rp_model, rp_view))

        rp_view._find_events_button.clicked.connect(lambda clicked: \
            self.find_events_button_clicked(clicked, rp_model, rp_view))

        rp_view._trigger_sigma_threshold_field.textChanged.connect(lambda text: \
            self.trigger_sigma_threshold_field_changed(text, rp_model, rp_view))

        rp_view._baseline_avg_length_field.textChanged.connect(lambda text: \
            self.baseline_avg_length_field_changed(text, rp_model, rp_view))

        rp_view._max_search_length_field.textChanged.connect(lambda text: \
            self.max_search_length_field_changed(text, rp_model, rp_view))

        return



    def set_rp_view_defaults(self, rp_view):
        rp_view._baseline_avg_length_field.setText('1000')
        rp_view._trigger_sigma_threshold_field.setText('3')
        rp_view._max_search_length_field.setText('10000')

        return


    def enable_ui(self, enable, rp_model, rp_view):
        rp_view.enable_ui(enable)
        return

    def plot_event(self, event, rp_model, rp_view):
        pen = QPen(QColor(50,200,50))

        rp_view._event_plot_items.append(rp_view._main_plot.plot(event._data, pen = pen))

        return

    def plot_targeted_event(self, targeted_event, rp_model, rp_view):
        rp_view._event_plot_item.setData(targeted_event._data)


    def main_plot_range_changed(self, rng, rp_model, rp_view):
        viewRange = rng.viewRange()

        ti = viewRange[0][0]
        tf = viewRange[0][1]

        t_range = (ti,tf)


        self.plot_main_data(rp_view, rp_model, t_range)

        self.plot_baseline_data(rp_view, rp_model, t_range)

        return

    def show_baseline_button_clicked(self, clicked, rp_model, rp_view):
        viewRange = rp_view._main_plot.viewRange()

        ti = viewRange[0][0]
        tf = viewRange[0][1]

        t_range = (ti,tf)

        rp_model.load_baseline_ts()

        self.plot_baseline_data(rp_view, rp_model, t_range)

        return

    def plot_main_data(self, rp_view, rp_model, t_range):
        if rp_model._main_ts._display_ready == True:
            main_data = rp_model.get_main_display_data(t_range)
            if main_data != None:
                rp_view._main_plot_item.setData(main_data[:,0], main_data[:,1])

        return

    def plot_baseline_data(self, rp_view, rp_model, t_range):
        if rp_model._baseline_ts._display_ready == True:
            baseline_data = rp_model.get_baseline_display_data(t_range)

            if baseline_data != None:
                rp_view._baseline_plot_item.setData(baseline_data[:,0], baseline_data[:,1])

        return

    def clear_events(self, rp_model, rp_view):
        for event_plot_item in rp_view._event_plot_items:
            event_plot_item.clear()
        rp_view._event_plot_items = []

        return




    def find_events_button_clicked(self, clicked, rp_model, rp_view):
        rp_model.find_events()
        return

    def baseline_avg_length_field_changed(self, text, rp_model, rp_view):
        rp_model.set_baseline_avg_length(int(text))
        return

    def trigger_sigma_threshold_field_changed(self, text, rp_model, rp_view):
        rp_model.set_trigger_sigma_threshold(float(text))
        return

    def max_search_length_field_changed(self, text, rp_model, rp_view):
        rp_model.set_max_search_length(int(text))
        return
