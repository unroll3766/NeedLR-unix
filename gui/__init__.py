"""GUI package init.

Pins the Qt backend before mayavi/pyface are imported. The conda env
contains both Qt5 (pyqt=5, used by the app) and Qt6 (pulled in by
matplotlib's pyside6 dependency); if pyface/traitsui auto-select pyside6,
both libQt5Core and libQt6Core end up in the process, which on macOS
produces objc duplicate-class warnings and can cause crashes. Forcing
QT_API=pyqt5 makes pyface bind to PyQt5 only. All mayavi imports in this
project happen through gui submodules, so this runs first.
"""

import os

os.environ.setdefault("QT_API", "pyqt5")
