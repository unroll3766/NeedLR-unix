"""Offscreen smoke checks for the GUI fixes (notes.md issues 2-5).

Run from anywhere with the NeedLR env:
    QT_QPA_PLATFORM=offscreen python tests/smoke_gui.py

Exits 0 when all checks pass. Requires no display; Mayavi figures are not
created (both views disabled), so no GL context is needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("/tmp")

from PyQt5.QtWidgets import QApplication, QScrollArea
from PyQt5.QtCore import Qt

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
app = QApplication([])

import numpy as np
import laspy

from gui.window import MainWindow

win = MainWindow()
win.show()
app.processEvents()

# --- Issue 4: scroll area + window fits available screen ---
assert win.findChild(QScrollArea) is not None, "controls must be inside a QScrollArea"
screen = app.primaryScreen().availableGeometry()
assert win.height() <= screen.height(), (win.height(), screen.height())
print(
    f"issue4 OK: QScrollArea present; window {win.width()}x{win.height()} "
    f"fits available screen {screen.width()}x{screen.height()}"
)

# --- Issue 2: empty folder must not leave 'Loading...' ---
win.controls.pointcloud_files = []
win.controls.pointcloud_view.current_folder_path = "/tmp"
win.controls.file_name_display.setText("Loading...")
win.controls.display_current_file()
label = win.controls.file_name_display.text()
assert label == "No point cloud files found", label
print(f"issue2 OK: label after empty folder = {label!r}")

# --- Issue 5: .laz listed and loadable through the app path ---
os.makedirs("/tmp/pc_test", exist_ok=True)
pts = np.random.default_rng(2).random((120, 3)) * 10.0
hdr = laspy.LasHeader(point_format=3, version="1.2")
hdr.offsets = pts.min(axis=0)
hdr.scales = np.array([0.001, 0.001, 0.001])
las = laspy.LasData(hdr)
las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
las.write("/tmp/pc_test/a.laz")
with open("/tmp/pc_test/b.txt", "w") as f:
    for p in pts[:50]:
        f.write(f"{p[0]} {p[1]} {p[2]}\n")

win.controls.load_pointcloud_files("/tmp/pc_test")
assert sorted(win.controls.pointcloud_files) == ["a.laz", "b.txt"], (
    win.controls.pointcloud_files
)
print("issue5a OK: folder listing includes .laz:", win.controls.pointcloud_files)

view = win.controls.pointcloud_view
view.load_and_display_pointcloud(
    "/tmp/pc_test/a.laz",
    downsampling_method="random",
    target_count=64,
    show_full_view=False,
    show_cluster_view=False,
    superpoint_graph_method="knn",
    k_value=4,
    clustering_algorithm="label_propagation",
)
assert view.points is not None and view.points.shape == (64, 3), getattr(view, "points", None)
assert view.clusters and sum(len(c) for c in view.clusters) == 64
print(
    f"issue5b OK: .laz loaded, downsampled to {view.points.shape}, "
    f"{len(view.clusters)} clusters"
)

# .txt path still works
view.load_and_display_pointcloud(
    "/tmp/pc_test/b.txt",
    downsampling_method="random",
    target_count=64,
    show_full_view=False,
    show_cluster_view=False,
    superpoint_graph_method="knn",
    k_value=4,
    clustering_algorithm="label_propagation",
)
assert view.points is not None and view.points.shape == (50, 3)
print(f"issue5c OK: .txt still loads, shape {view.points.shape}")

# --- Issue 3: arbitrary string labels must not break color mapping ---
m = view.label_color_map(["tree", -1, "rock-01", 7])
assert -1 not in m, "unlabeled sentinel must be excluded"
assert len({m["tree"], m["rock-01"], m[7]}) == 3, "distinct labels need distinct colors"
assert m == view.label_color_map([7, "rock-01", -1, "tree"]), "mapping must be stable"
print(f"issue3 OK: label_color_map -> {m}")

# matplotlib-free plasma gradient (issue 1 follow-up): valid, distinct RGB colors
colors = view.get_dynamic_colors(4)
assert len(colors) == 4
assert all(len(c) == 3 and all(0.0 <= v <= 1.0 for v in c) for c in colors)
assert len(set(colors)) == 4, "gradient colors must be distinct"
assert len(view.get_dynamic_colors(1)) == 1
print("colormap OK:", [tuple(round(v, 3) for v in c) for c in colors])

# --- Issue 6: label_map persistence in the annotation JSON ---
import tempfile
from algorithms.label_export import load_annotation_file

tmpd = tempfile.mkdtemp(prefix="needlr_json_test_")
view2 = win.controls.pointcloud_view
view2.destination_folder = tmpd
view2.current_file = "test.las"
view2.points = pts[:20]
view2.clusters = [np.arange(0, 10), np.arange(10, 20)]
view2.cluster_labels = {}
view2.label_map = {}
view2.initialize_labels_file()  # creates the JSON in object schema
jpath = os.path.join(tmpd, "test.las.json")
clusters_j, map_j = load_annotation_file(jpath)
assert len(clusters_j) == 2 and map_j == {}, (len(clusters_j), map_j)
view2.label_current_cluster("tree")
view2.save_labels_to_file()
clusters_j, map_j = load_annotation_file(jpath)
assert map_j == {"tree": 1}, map_j
assert clusters_j[0]["label"] == "tree"
print(f"issue6 OK: label_map persisted and reloaded: {map_j}, cluster0 label={clusters_j[0]['label']!r}")

# gap-safe id assignment: next id must not collide with existing ids
view2.label_map = {"x": 1, "y": 5}
view2.label_current_cluster("z")
assert view2.label_map["z"] == 6, view2.label_map
print(f"issue6b OK: gap-safe label id assignment: {view2.label_map}")

# --- Issue 6 UI regression: export widgets exist; select_dest_folder must not crash ---
from PyQt5.QtWidgets import QFileDialog

assert hasattr(win.controls, "btn_export_labels"), "btn_export_labels missing from initUI"
assert hasattr(win.controls, "export_header_name"), "export_header_name missing from initUI"
assert win.controls.btn_export_labels.isEnabled() is False
assert win.controls.export_header_name.text() == "semantic"
tmpd2 = tempfile.mkdtemp(prefix="needlr_dest_test_")
view3 = win.controls.pointcloud_view
view3.current_file = "x.las"
view3.clusters = [np.arange(0, 3)]
view3.points = pts[:3]
view3.cluster_labels = {}
view3.label_map = {}
_orig_dialog = QFileDialog.getExistingDirectory
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: tmpd2)
try:
    win.controls.select_dest_folder()  # previously raised AttributeError
finally:
    QFileDialog.getExistingDirectory = _orig_dialog
assert win.controls.btn_export_labels.isEnabled() is True
assert win.controls.dest_folder_path == tmpd2
print("issue6-ui OK: export widgets present; select_dest_folder enables button, no crash")

# --- Cluster-view crash regression: stale figure after closing the Cluster
# --- Zoom window, and rendering with clusters=None must not crash ---
import types
from mayavi import mlab
from gui.pointcloud_view import _figure_attached

assert _figure_attached(None) is False
dead = types.SimpleNamespace(scene=None)
assert _figure_attached(dead) is False

view4 = win.controls.pointcloud_view
view4.points = None
view4.clusters = None
view4.fig_full = dead  # stale figure, as after manually closing the window
view4.fig_zoom = dead
made = []


def fake_figure(*a, **k):
    cam = types.SimpleNamespace(
        position=(0, 0, 0),
        focal_point=(0, 0, 0),
        view_angle=30.0,
        view_up=(0, 0, 1),
        clipping_range=(0.0, 1.0),
        zoom=lambda factor: None,
    )
    scene = types.SimpleNamespace(
        camera=cam, render=lambda: None, render_window=None, interactor=None
    )
    f = types.SimpleNamespace(scene=scene, name="fake")
    made.append(f)
    return f


orig_figure, orig_clf = mlab.figure, mlab.clf
mlab.figure = fake_figure
mlab.clf = lambda *a, **k: None
try:
    view4.render_pointcloud(show_full_view=True, show_cluster_view=True)
    assert view4.clusters == [], "clusters must be normalized to [] not None"
    assert view4.fig_full in made and view4.fig_zoom in made, "stale figures must be recreated"
    view4.render_pointcloud(show_full_view=True, show_cluster_view=False)  # toggle-off path
finally:
    mlab.figure, mlab.clf = orig_figure, orig_clf
# close_figure on a stale figure must be a safe no-op
view4.close_figure("fig_zoom")
assert view4.fig_zoom is None
print("cluster-view OK: stale figures replaced, clusters=None safe, close_figure safe")

print("ISSUES 2-5 VERIFICATION: ALL PASSED")
