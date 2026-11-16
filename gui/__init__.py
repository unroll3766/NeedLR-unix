"""GUI package init.

Pins the Qt backend before mayavi/pyface are imported. pyface selects a
binding by first checking sys.modules (PySide6 before PyQt5) and only then
the QT_API env var, so an explicit pin is required to guarantee it binds
PyQt5 even if a Qt6 binding (pyside6/pyqt6) happens to be installed.
Loading both libQt5Core and libQt6Core in one process produces objc
duplicate-class warnings on macOS and can crash. All mayavi imports in
this project happen through gui submodules, so this runs first.
"""

import os

os.environ.setdefault("QT_API", "pyqt5")
