import sys

from PyQt5.QtWidgets import QApplication

from gui.MainGUI import MainGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainGUI()
    win.show()
    sys.exit(app.exec_())
