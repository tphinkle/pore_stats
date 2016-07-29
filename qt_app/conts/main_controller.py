import sys
sys.path.append('/home/preston/Desktop/Science/Research/pore_stats/')
import resistive_pulse as rp
from rp_model import RPModel
import rp_controller
import PyQt4.QtCore as QtCore
from PyQt4.QtGui import *


class MainController(QtCore.QObject):

    def __init__(self, main_model, main_view):
        super(MainController, self).__init__()
        self._main_model = main_model
        self._main_view = main_view

        self._rp_controller = rp_controller.RPController(self._main_model, self._main_view)

        self.setup_connections()


    def setup_connections(self):
        """
        * Description: Call function for setting up all connections.
        * Return:None
        * Arguments:
        """
        self.setup_bar_connections()

        return

    def setup_bar_connections(self):
        """
        * Description: Connections for UI MenuBar signals.
        * Return: None
        * Arguments:
        """
        self._main_view._rp_load_file_action.triggered.connect(self._rp_controller.add_rp)

        return
