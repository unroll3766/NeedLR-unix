import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def main():
    # Must be set before any QApplication exists; matters on high-DPI
    # (retina) displays, e.g. on macOS. The gui import below instantiates a
    # QApplication (via mayavi), so it is deferred until after this.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    from gui.window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
