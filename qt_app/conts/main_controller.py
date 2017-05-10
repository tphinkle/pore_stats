# /qt_app/conts/main_controller.py

"""
* Contains MainController class.

* Sections:
    1. Imports
    2. Classes
        - MainController(QtCore.QObject)

"""

"""
Imports
"""

# Python specific
import sys
import os

# Qt
import PyQt4.QtCore as QtCore
from PyQt4.QtGui import *

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
from rp_model import RPModel
import rp_controller


"""
Classes
"""

class MainController(QtCore.QObject):
    """
    The main controller in the model, view, controller paradigm for gui programming. The
    two functions of the class are to hold and instantiate sub-controllers (e.g.
    RPController) and to set up signals and slots at the QMainWindow level. Most program
    logic is handled by specialized sub-controllers.
    """

    def __init__(self, main_model, main_view):
        super(MainController, self).__init__()
        self._main_model = main_model
        self._main_view = main_view

        self._controllers = []

        self.setup_connections()


    def setup_connections(self):
        """
        * Description: Call function for setting up all connections.
        * Return:
        * Arguments:
        """
        self.setup_bar_connections()

        return

    def setup_bar_connections(self):
        """
        * Description: Connections for UI MenuBar signals.
        * Return:
        * Arguments:
        """
        self._main_view._rp_load_file_action.triggered.connect(self.create_rp_controller)
        self._main_view._show_controls_action.triggered.connect(self.create_controls_window)

        return

    def create_rp_controller(self):
        """
        * Description: Slot for QMenuBar 'ResistivePulse' button. Creates a RPController,
          which then creates associated RPModel and RPView.
        * Return: None
        * Arguments:
        """
        self._controllers.append(rp_controller.RPController(self._main_model,\
         self._main_view))

        return

    def create_controls_window(self):
        self._main_view.create_controls_window()

        return
