import sys
sys.path.append('./../pyqtgraph')
import rp_view

from PyQt4.QtCore import *
from PyQt4.QtGui import *


class MainView(QMainWindow):
    def __init__(self, parent = None):
        super(MainView, self).__init__(None)

        # Bar
        self._bar = None
        self._rp_menu = None
        self._rp_load_file_action = None
        self._oi_menu = None
        self._oi_load_file_action = None

        # MDI
        self._mdi = None

        self.setup_ui()

    def set_main_controller(self, main_controller):
        self._main_controller = main_controller
        return


    def setup_ui(self):
        self.showMaximized()
        self.setWindowTitle("pore_stats")

        self.setup_bar()

        self.setup_subwindows()

        return

    def setup_bar(self):
        # Create bar
        self._bar = self.menuBar()

        # Add rp  menu
        self._rp_menu = self._bar.addMenu('Resistive pulse')
        self._rp_load_file_action = self._rp_menu.addAction('Load file')


        # Add oi menu
        self._oi_menu = self._bar.addMenu('Optical imaging')
        self._oi_load_file_action = self._oi_menu.addAction('Load file')

        return

    def setup_subwindows(self):
        # Add subwindows
        self._mdi = QMdiArea()
        self.setCentralWidget(self._mdi)

        return

    def create_rp_view(self, parent_model = None):
        view = rp_view.RPView(parent_widget = self, parent_model = parent_model)
        self._mdi.tileSubWindows()
        #new_subwindow.#
        return view
