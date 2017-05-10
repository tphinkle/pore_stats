# Imports

# Standard library
import sys
import os
import copy

# Program specific
PORE_STATS_BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib/rp/')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/lib/oi/')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/conts')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/model')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/threads')
sys.path.append(PORE_STATS_BASE_DIRECTORY + '/qt_app/views')
import resistive_pulse as rp
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
