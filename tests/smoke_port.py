"""Offscreen smoke test for the cross-platform port.

Run with the NeedLR env, from any directory:
    QT_QPA_PLATFORM=offscreen python tests/smoke_port.py

Exits 0 when all checks pass. Launches from a temp CWD to prove the app no
longer depends on the working directory.
"""
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(tempfile.mkdtemp(prefix="needlr_cwd_test_"))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
app = QApplication([])

import config.settings as settings

assert settings.settings_file == os.path.join(REPO_ROOT, "app_settings.json"), (
    settings.settings_file
)
print("settings_file anchored to repo root:", settings.settings_file)

from gui.window import MainWindow
from gui import window_utils

win = MainWindow()
win.show()
window_utils.move_vtk_window(None, 0, 0, 100, 100)  # no-op path must not crash
QTimer.singleShot(500, app.quit)
app.exec_()
print("GUI constructed and shown (offscreen) OK")

import numpy as np
from algorithms import downsampling as ds

rng = np.random.default_rng(0)
pts = rng.random((5000, 3)) * 100.0
cases = [
    ("random", lambda p: ds.random_downsample(p, 512)),
    ("vanilla_fps", lambda p: ds.vanilla_fps(p, 512)),
    ("fps_npdu", lambda p: ds.fps_npdu(p, 512)),
    ("fps_npdu_kdtree", lambda p: ds.fps_npdu_kdtree(p, 512)),
    ("bucket_fps_kdline", lambda p: ds.bucket_fps_kdline(p, 512, h=7)),
]
for name, fn in cases:
    out = fn(pts)
    assert out.shape == (512, 3), (name, out.shape)
    assert np.isfinite(out).all(), name
    print(f"{name}: ok, shape={out.shape}")

from algorithms import superpoint_graph, community_detection as cd

graph = superpoint_graph.create_knn_graph(out, 8)
clusters = cd.label_propagation(graph)
assert sum(len(c) for c in clusters) == graph.number_of_nodes()
print(f"knn graph: {graph.number_of_nodes()} nodes, {len(clusters)} clusters")

mst = superpoint_graph.create_mst_graph(out)
print(f"mst graph: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges")

print("PORT SMOKE TEST: ALL PASSED")
