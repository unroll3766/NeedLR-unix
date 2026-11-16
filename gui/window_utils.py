# gui/window_utils.py
"""Cross-platform helpers for positioning Mayavi/VTK render windows.

The primary path uses VTK-native calls (SetPosition/SetSize on the render
window), which work on Linux and with recent VTK versions on macOS. On
Windows, win32gui is used as a fallback when it is installed. All failures
are non-fatal: the window simply stays at its default position.
"""

import sys


def move_vtk_window(figure, x, y, width, height):
    """Move and resize the render window of a Mayavi figure.

    Args:
        figure: A mayavi figure (or None, in which case this is a no-op).
        x, y: Target window position in screen coordinates.
        width, height: Target window size in pixels.
    """
    if figure is None or figure.scene is None:
        return
    try:
        render_window = figure.scene.render_window
        render_window.SetPosition(x, y)
        render_window.SetSize(width, height)
        figure.scene.interactor.ReInitialize()
        return
    except Exception:
        pass
    if sys.platform == "win32":
        _move_window_win32(figure, x, y, width, height)


def _move_window_win32(figure, x, y, width, height):
    """Windows-only window move via pywin32, used only if VTK calls fail."""
    try:
        import win32gui
    except ImportError:
        return
    title = figure.name
    found = {"hwnd": None}

    def enum_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == title:
            found["hwnd"] = hwnd

    try:
        win32gui.EnumWindows(enum_callback, None)
        if found["hwnd"]:
            win32gui.MoveWindow(found["hwnd"], x, y, width, height, True)
    except Exception:
        pass
