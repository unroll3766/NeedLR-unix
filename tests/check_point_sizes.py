"""Checks for the separate background/cluster point-size option
(notes.md "Thoughts" 1): a "Cluster Size" spinbox next to "Point Size",
with `cluster_point_size` used for the red highlighted cluster in both
panes and `point_size` (at 0.6x) for the background points. Also covers
the settings-defaults merge so older app_settings.json files (without the
new key) keep working.

Run with the NeedLR env:
    QT_QPA_PLATFORM=offscreen python tests/check_point_sizes.py

No GL needed: mayavi plotting calls are faked and record their arguments.
"""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("/tmp")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
app = QApplication([])

import numpy as np

import config.settings as settings
from gui.window import MainWindow
from mayavi import mlab

# Tests must never write the repo's app_settings.json (MainWindow.closeEvent
# saves settings when the window is destroyed at exit), even though this
# test mutates settings.current_settings.
settings.save_settings = lambda *a, **k: None

# --- Settings: defaults merge over the file, new key always present --------
assert "cluster_point_size" in settings.current_settings, (
    "cluster_point_size must be present after loading app_settings.json"
)

tmpd = tempfile.mkdtemp()
old_file = settings.settings_file
settings.settings_file = os.path.join(tmpd, "old_settings.json")
with open(settings.settings_file, "w") as f:
    json.dump({"point_size": 0.7}, f)  # old file: no cluster_point_size key
try:
    loaded = settings.load_settings()
finally:
    settings.settings_file = old_file
assert loaded["point_size"] == 0.7, "file values must win over defaults"
assert loaded["cluster_point_size"] == settings.default_settings["cluster_point_size"], (
    "missing key must fall back to the default"
)
assert loaded["night_mode"] == settings.default_settings["night_mode"]
print("settings merge OK: file values win, missing keys get defaults")

# --- GUI: both spinboxes exist with the settings values ---------------------
win = MainWindow()
win.show()
app.processEvents()
c = win.controls
assert c.point_size_selector.value() == settings.current_settings["point_size"]
assert (
    c.cluster_point_size_selector.value()
    == settings.current_settings["cluster_point_size"]
)
print("widgets OK: Point Size and Cluster Size spinboxes present, values from settings")

# --- Eco mode disables both selectors (repo settings ship eco_mode=true) -----
assert not c.point_size_selector.isEnabled() or not settings.current_settings["eco_mode"]
c.eco_mode_checkbox.setChecked(True)
assert not c.point_size_selector.isEnabled() and not c.cluster_point_size_selector.isEnabled()
c.eco_mode_checkbox.setChecked(False)
assert c.point_size_selector.isEnabled() and c.cluster_point_size_selector.isEnabled()
print("eco mode OK: both point-size selectors disabled in eco mode, re-enabled after")

# --- Each selector updates only its own setting ------------------------------
c.pointcloud_view.current_folder_path = None  # no re-render while changing values
bg_before = settings.current_settings["point_size"]
cl_before = settings.current_settings["cluster_point_size"]
c.point_size_selector.setValue(bg_before + 0.3)
assert settings.current_settings["point_size"] == bg_before + 0.3
assert settings.current_settings["cluster_point_size"] == cl_before
c.cluster_point_size_selector.setValue(cl_before + 1.1)
assert settings.current_settings["cluster_point_size"] == cl_before + 1.1
assert settings.current_settings["point_size"] == bg_before + 0.3
print("selectors OK: each spinbox writes only its own setting")

# --- Render mapping: background uses point_size*0.6, cluster uses ----------
# --- cluster_point_size, in both panes --------------------------------------
settings.current_settings["point_size"] = 0.8
settings.current_settings["cluster_point_size"] = 2.0

calls = []


def fake_figure(name=None, **kwargs):
    cam = types.SimpleNamespace(
        position=(0, 0, 0),
        focal_point=(0, 0, 0),
        view_angle=30.0,
        view_up=(0, 0, 1),
        clipping_range=(0.0, 1.0),
        zoom=lambda factor: None,
        compute_view_plane_normal=lambda: None,
    )
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(camera=cam, render=lambda: None), name=name, children=[]
    )


def fake_points3d(x, y, z, **kw):
    fig = kw.get("figure")
    calls.append((getattr(fig, "name", None), kw.get("color"), kw.get("scale_factor"), len(x)))


def fake_quiver3d(x, y, z, u, v, w, **kw):
    fig = kw.get("figure")
    calls.append((getattr(fig, "name", None), "quiver", kw.get("scale_factor"), None))


orig = {n: getattr(mlab, n) for n in ("figure", "clf", "points3d", "quiver3d")}
mlab.figure = fake_figure
mlab.clf = lambda *a, **k: None
mlab.points3d = fake_points3d
mlab.quiver3d = fake_quiver3d
try:
    view = c.pointcloud_view
    view.points = np.arange(30, dtype=float).reshape(10, 3)
    view.clusters = [np.arange(0, 4), np.arange(4, 10)]
    view.cluster_labels = {0: -1, 1: -1}
    view.current_cluster_index = 0
    view.fig_full = None
    view.fig_zoom = None
    view.current_folder_path = "/tmp"  # handlers re-render only when a folder is loaded

    calls.clear()
    view.render_pointcloud(show_full_view=True, show_cluster_view=True)
    full_bg = [c for c in calls if c[0] == "Full Model" and c[1] not in ((1, 0, 0), "quiver")]
    full_red = [c for c in calls if c[0] == "Full Model" and c[1] == (1, 0, 0)]
    zoom_red = [c for c in calls if c[0] == "Cluster Zoom" and c[1] == (1, 0, 0)]
    assert full_bg and all(c[2] == 0.8 * 0.6 for c in full_bg), (
        f"background points must use point_size*0.6, got {full_bg}"
    )
    assert len(full_red) == 1 and full_red[0][2] == 2.0, f"full-pane red must use cluster_point_size: {full_red}"
    assert len(zoom_red) == 1 and zoom_red[0][2] == 2.0, f"zoom-pane red must use cluster_point_size: {zoom_red}"
    print("render mapping OK: background=point_size*0.6, red cluster=cluster_point_size in both panes")

    # --- The Cluster Size handler itself re-renders with the new value ------
    view.fig_full = None
    view.fig_zoom = None
    calls.clear()
    c.cluster_point_size_selector.setValue(3.0)
    red_after = [c for c in calls if c[1] == (1, 0, 0)]
    assert red_after and all(c[2] == 3.0 for c in red_after), (
        f"after changing Cluster Size the red points must render at 3.0, got {red_after}"
    )
    assert settings.current_settings["cluster_point_size"] == 3.0
    print("handler OK: changing Cluster Size updates the setting and re-renders at the new size")
finally:
    for n, v in orig.items():
        setattr(mlab, n, v)

print("POINT SIZE OPTIONS: ALL PASSED")
