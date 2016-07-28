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
            new_rp_model = self._main_model.create_rp_model(file_path)

            new_rp_model.set_active_file(file_path)


            # Create new RP view, subscribe view to model
            new_rp_view = self._main_view.create_rp_view(parent_model = new_rp_model)



            new_rp_model.add_subscriber(new_rp_view)

            # Connect signals to slots
            self.add_rp_slots(new_rp_view, new_rp_model)

            # Set defaults
            self.set_rp_view_defaults(new_rp_view)



            #new_rp_model._loader = load_data_thread.LoadDataThread(file_path, new_rp_model.display_decimation_threshold, new_rp_model.decimation_factor)

            new_rp_model._loader.start()

        return


    def add_rp_slots(self, rp_view, rp_model):
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

    def data_toggle_clicked(self, clicked, rp_model, rp_view):
        checked=rp_view._data_toggle.isChecked()
        rp_model.data_active=checked
        pass

    def main_plot_range_changed(self, rng, rp_model, rp_view):
        viewRange = rng.viewRange()

        ti = viewRange[0][0]
        tf = viewRange[0][1]
        rp_model.set_t_range((ti, tf))

        return

    def show_baseline_button_clicked(self, clicked, rp_model, rp_view):
        rp_model.calculate_baseline()
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
