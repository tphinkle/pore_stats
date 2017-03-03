import sys
sys.path.append('./../pyqtgraph')
import rp_view

import PyQt4.QtCore
import PyQt4.QtGui


class MainView(PyQt4.QtGui.QMainWindow):
    def __init__(self, parent = None):
        super(MainView, self).__init__(None)

        # Bar
        self._bar = None
        self._rp_menu = None
        self._rp_load_file_action = None

        self._controls_menu = None
        self._show_controls_action = None

        # Controls window
        self._controls_window = None


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

        self.setup_controls_window()

        return

    def setup_bar(self):
        # Create bar
        self._bar = self.menuBar()

        # Add rp  menu
        self._rp_menu = self._bar.addMenu('Resistive pulse')
        self._rp_load_file_action = self._rp_menu.addAction('Load file')

        # Add controls menu
        self._controls_menu = self._bar.addMenu('Controls')
        self._show_controls_action = self._controls_menu.addAction('Show controls')


        return

    def setup_controls_window(self):
        self._controls_window = PyQt4.QtGui.QDialog(self)
        self._controls_window.setWindowTitle("Controls")


        self._controls_layout = PyQt4.QtGui.QVBoxLayout()


        self._controls_table = PyQt4.QtGui.QTableWidget()
        self._controls_table.setColumnCount(2)
        self._controls_table.setRowCount(10)
        self._controls_table.verticalHeader().setVisible(False)



        self._controls_table.setHorizontalHeaderLabels(PyQt4.QtCore.QStringList(["Command", "Key"]))

        self._controls_table.setItem(0, 0, PyQt4.QtGui.QTableWidgetItem("Show/hide cursor"))
        self._controls_table.setItem(0, 1, PyQt4.QtGui.QTableWidgetItem("F1"))

        self._controls_table.setItem(1, 0, PyQt4.QtGui.QTableWidgetItem("Show/hide ROI"))
        self._controls_table.setItem(1, 1, PyQt4.QtGui.QTableWidgetItem("F2"))

        self._controls_table.setItem(2, 0, PyQt4.QtGui.QTableWidgetItem("Accept event"))
        self._controls_table.setItem(2, 1, PyQt4.QtGui.QTableWidgetItem("1"))

        self._controls_table.setItem(3, 0, PyQt4.QtGui.QTableWidgetItem("Reject event"))
        self._controls_table.setItem(3, 1, PyQt4.QtGui.QTableWidgetItem("2"))

        self._controls_table.setItem(4, 0, PyQt4.QtGui.QTableWidgetItem("Accept event ROI"))
        self._controls_table.setItem(4, 1, PyQt4.QtGui.QTableWidgetItem("3"))

        self._controls_table.setItem(5, 0, PyQt4.QtGui.QTableWidgetItem("Reject event ROI"))
        self._controls_table.setItem(5, 1, PyQt4.QtGui.QTableWidgetItem("4"))

        self._controls_table.setItem(6, 0, PyQt4.QtGui.QTableWidgetItem("Next event"))
        self._controls_table.setItem(6, 1, PyQt4.QtGui.QTableWidgetItem("Right arrow"))

        self._controls_table.setItem(7, 0, PyQt4.QtGui.QTableWidgetItem("Previous event"))
        self._controls_table.setItem(7, 1, PyQt4.QtGui.QTableWidgetItem("Left arrow"))

        self._controls_table.setItem(8, 0, PyQt4.QtGui.QTableWidgetItem("Next/previous selected event"))
        self._controls_table.setItem(8, 1, PyQt4.QtGui.QTableWidgetItem("Alt + L/R arrow"))

        self._controls_table.setItem(9, 0, PyQt4.QtGui.QTableWidgetItem("Next/previous unselected event"))
        self._controls_table.setItem(9, 1, PyQt4.QtGui.QTableWidgetItem("Ctrl + L/R arrow"))

        self._controls_table.resizeRowsToContents()
        self._controls_table.resizeColumnsToContents()

        self._controls_layout.addWidget(self._controls_table)

        self._controls_window.setLayout(self._controls_layout)

        self._controls_window.adjustSize()


        '''
        self._controls_layout = PyQt4.QtGui.QVBoxLayout()



        self._controls_cursor = PyQt4.QtGui.QTextEdit("Show/hide cursor: F1")
        self._controls_roi = PyQt4.QtGui.QTextEdit("Show/hide ROI: F2")

        self._controls_accept = PyQt4.QtGui.QTextEdit("Accept event: 1")
        self._controls_reject = PyQt4.QtGui.QTextEdit("Accept event: 2")

        self._controls_accept_roi = PyQt4.QtGui.QTextEdit("Accept event roi: 3")
        self._controls_reject_roi = PyQt4.QtGui.QTextEdit("Accept event roi: 4")

        for widget in [self._controls_cursor, self._controls_roi, self._controls_accept,\
        self._controls_reject, self._controls_accept_roi, self._controls_reject_roi]:
            widget.setReadOnly(True)

        self._controls_layout.addWidget(self._controls_cursor)
        self._controls_layout.addWidget(self._controls_roi)
        self._controls_layout.addWidget(self._controls_accept)
        self._controls_layout.addWidget(self._controls_reject)
        self._controls_layout.addWidget(self._controls_accept_roi)
        self._controls_layout.addWidget(self._controls_reject_roi)

        self._controls_window.setLayout(self._controls_layout)
        '''


        return

    def setup_subwindows(self):
        # Add subwindows
        self._mdi = PyQt4.QtGui.QMdiArea()
        self.setCentralWidget(self._mdi)

        return

    def create_rp_view(self, parent_model = None):
        view = rp_view.RPView(parent_widget = self, parent_model = parent_model)
        self._mdi.tileSubWindows()

        return view

    def create_controls_window(self):
        if self._controls_window.isVisible():
            self._controls_window.hide()
        else:
            self._controls_window.show()
            self._controls_window.adjustSize()
        return
