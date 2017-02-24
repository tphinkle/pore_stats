# Imports

# Standard library
import sys

# Program specific
sys.path.append('/home/prestonh/Desktop/Research/pore_stats/lib/rp/')
import resistive_pulse as rp


sys.path.append('/home/prestonh/Desktop/Research/pore_stats/qt_app/model/')
import rp_model






class MainModel(object):
    """
    Creates new RP Models so that more than one may be open and operated independently at
    any given time.
    """

    def __init__(self):
        self._main_controller = None
        self._rp_models = []

    def set_main_controller(self, main_controller):
        self._main_controller = main_controller
        return

    def create_rp_model(self, file_name):
        new_rp_model = rp_model.RPModel(file_name)
        self._rp_models.append(new_rp_model)
        return new_rp_model
