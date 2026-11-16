from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QStyle,
    QScrollArea,
)
from PyQt5 import QtGui
from gui.pointcloud_view import PointcloudView
from gui.controls import ControlPanel
import os
import webbrowser

import config.settings as settings


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeedLR")
        self.setWindowIcon(
            QtGui.QIcon(os.path.join(settings.BASE_DIR, "ReadME_Assets", "window_icon.png"))
        )
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.pointcloud_view = PointcloudView()
        self.controls = ControlPanel(self, self.pointcloud_view)

        self.github_button = QPushButton("Visit Project's Github", self)
        self.github_button.clicked.connect(self.open_github)
        self.github_button.setIcon(
            self.style().standardIcon(QStyle.SP_MessageBoxInformation)
        )  # Set path to your icon

        # self.layout.addWidget(self.pointcloud_view)
        # Scroll area keeps the whole panel reachable on short screens
        # (taskbar/dock), where the full layout would overflow the window.
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.controls)
        self.layout.addWidget(self.scroll_area)

        self.layout.addWidget(self.github_button)

        # Fit the window to the available screen geometry (excludes menu
        # bar / taskbar) instead of letting the layout dictate the height.
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(480, available.width()), min(available.height() - 40, 900)
            )

    def closeEvent(self, event):
        # Save settings on close
        settings.save_settings(settings.current_settings)
        super().closeEvent(event)
        quit()

    def open_github(self):
        webbrowser.open("https://github.com/GunnerStone/NeedLR")
