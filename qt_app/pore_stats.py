import sys

sys.path.append('./model/')
from main_model import MainModel

sys.path.append('./views/')
from main_view import MainView

sys.path.append('./conts/')
from main_controller import MainController

import PyQt4.QtCore
import PyQt4.QtGui

class App(PyQt4.QtGui.QApplication):
    def __init__(self, sys_argv):
        super(App, self).__init__(sys_argv)
        self._main_model = MainModel()
        self._main_view = MainView()
        self._main_controller = MainController(self._main_model, self._main_view)

        self._main_model.set_main_controller(self._main_controller)
        self._main_view.set_main_controller(self._main_controller)



        self._main_view.show()

def main():
    app = App(sys.argv)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
