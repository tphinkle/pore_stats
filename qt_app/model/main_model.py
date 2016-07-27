import sys

PORE_STATS_DIR = '/home/preston/Desktop/Science/Research/pore_stats/'

sys.path.append(PORE_STATS_DIR)
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/qt_app/model/')
import resistive_pulse as rp
import rp_model



class MainModel(object):

    def __init__(self):
        self._main_controller = None
        self._rp_models = []

    def set_main_controller(self, main_controller):
        self._main_controller = main_controller
        return

    def create_rp_model(self):
        new_rp_model = rp_model.RPModel()
        self._rp_models.append(new_rp_model)
        return new_rp_model

    
