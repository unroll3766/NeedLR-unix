"""Check which Qt Python bindings end up imported by the app's gui stack.

Run with the NeedLR env:
    python tests/check_qt_backend.py            # uses whatever QT_API is set (default pin)
    QT_API= python tests/check_qt_backend.py    # unset -> observe default behavior

Exit 0 when only PyQt5 modules are loaded (no pyside6 => libQt6Core never
loads => no objc duplicate-class warnings on macOS). Exit 1 if any
pyside* module got imported.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gui.window  # noqa: F401  (imports mayavi/traitsui/pyface chain)

qt_mods = sorted(m for m in sys.modules if m.lower().startswith(("pyside", "pyqt")))
pyside_mods = [m for m in qt_mods if m.lower().startswith("pyside")]
print("QT_API =", os.environ.get("QT_API"))
print("Qt binding modules loaded:", qt_mods)
if pyside_mods:
    print("FAIL: pyside6 was imported -> libQt6Core loaded alongside libQt5Core:", pyside_mods)
    sys.exit(1)
print("OK: only PyQt5 bindings loaded; libQt6Core is not in the process")
