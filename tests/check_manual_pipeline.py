"""Checks for manual pipeline control: a "Manual Pipeline" mode where selector
changes only store settings and explicit buttons run each stage (Load File,
Downsample, Build Graph, Cluster). In auto mode (default) the pipeline still
runs on every selector change.

Run with the NeedLR env:
    QT_QPA_PLATFORM=offscreen python tests/check_manual_pipeline.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("/tmp")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
app = QApplication([])

import numpy as np

import config.settings as settings

# Tests must never write the repo's app_settings.json.
settings.save_settings = lambda *a, **k: None

from gui.pointcloud_view import PointcloudView
from gui.window import MainWindow

rng = np.random.default_rng(11)

assert "manual_pipeline" in settings.current_settings
assert settings.current_settings["manual_pipeline"] is False
print("settings OK: manual_pipeline present, default False")

tmpd = tempfile.mkdtemp()
txt_path = os.path.join(tmpd, "cloud.txt")
np.savetxt(txt_path, rng.random((60, 5)))

# --- View stages run independently and in order -------------------------------
v = PointcloudView()
assert v.load_file(txt_path) is True
assert v.features.shape == (60, 5) and v.points.shape == (60, 3)
assert v.graph is None and v.clusters is None and v.sampled_features is None
v.apply_downsampling("random", 40)
assert v.points.shape == (40, 3) and v.sampled_features.shape == (40, 5)
assert v.graph is None and v.clusters is None, "downsampling must invalidate graph/clusters"
v.build_graph("knn", 3)
assert v.graph is not None and v.graph.number_of_nodes() == 40
assert v.clusters is None, "building a new graph must invalidate clusters"
v.apply_clustering("label_propagation")
assert v.clusters and v.current_cluster_index == 0
print("stages OK: load -> downsample -> graph -> cluster, each invalidating downstream state")

# unsupported extension leaves state untouched
assert v.load_file(os.path.join(tmpd, "nope.bin")) is False
assert v.points.shape == (40, 3)
print("load_file OK: unsupported extension returns False, state untouched")

# --- Auto path still runs the full pipeline in one call ------------------------
v2 = PointcloudView()
v2.load_and_display_pointcloud(
    txt_path,
    downsampling_method="random",
    target_count=40,
    show_full_view=False,
    show_cluster_view=False,
    superpoint_graph_method="knn",
    k_value=3,
    clustering_algorithm="label_propagation",
)
assert v2.points.shape == (40, 3) and v2.graph is not None and v2.clusters
print("auto path OK: load_and_display_pointcloud still runs the whole pipeline")

# --- GUI: buttons + mode switching ---------------------------------------------
win = MainWindow()
win.show()
app.processEvents()
c = win.controls
view = c.pointcloud_view

for b in (c.btn_load_file, c.btn_downsample, c.btn_build_graph, c.btn_cluster):
    assert not b.isEnabled(), "pipeline buttons must be disabled in auto mode"

c.manual_pipeline_checkbox.setChecked(True)
assert settings.current_settings["manual_pipeline"] is True
for b in (c.btn_load_file, c.btn_downsample, c.btn_build_graph, c.btn_cluster):
    assert b.isEnabled(), "pipeline buttons must be enabled in manual mode"
print("mode UI OK: buttons enabled only in manual mode, setting persisted")

# manual mode: selector changes store settings but do not run the pipeline
recorded = []
orig_load = view.load_and_display_pointcloud
view.load_and_display_pointcloud = lambda *a, **k: recorded.append((a, k))
c.confirm_deletion = lambda: True  # avoid modal dialogs
try:
    c.on_downsampling_changed()
    c.on_community_detection_changed()
    assert recorded == [], f"manual mode must not auto-run the pipeline: {recorded}"
    view.feature_names = ["x", "y", "z", "col3", "col4"]
    c.populate_cluster_dims()
    c.cluster_dim_rows["col4"]["checkbox"].setChecked(True)
    assert recorded == [] and view.cluster_dims == ["x", "y", "z", "col4"]
    print("manual mode OK: selector/dims changes do not run the pipeline, settings updated")

    # buttons run exactly their stage
    stages = []
    view.load_file = lambda *a, **k: stages.append("load_file")
    view.apply_downsampling = lambda *a, **k: stages.append(("downsample", a, k))
    view.build_graph = lambda *a, **k: stages.append(("graph", a, k))
    view.apply_clustering = lambda *a, **k: stages.append(("cluster", a, k))
    view.render_pointcloud = lambda *a, **k: stages.append("render")
    view.current_file = txt_path
    view.current_folder_path = tmpd
    c.pointcloud_files = ["cloud.txt"]
    c.current_file_index = 0
    c.dest_folder_path = None  # no annotation file -> no prompt
    c.run_load_file()
    c.run_downsampling()
    c.run_build_graph()
    c.run_cluster()
    kinds = [s if isinstance(s, str) else s[0] for s in stages]
    assert kinds == ["load_file", "render", "downsample", "render", "graph", "render", "cluster", "render"], kinds
    assert stages[2][1][0] == c.downsampling_algorithm_selector.currentData()
    assert stages[4][1][0] == c.superpoint_graph_algorithm_selector.currentData()
    assert stages[6][1][0] == c.community_detection_selector.currentData()
    print("buttons OK: Load File/Downsample/Build Graph/Cluster run their stage + re-render")
finally:
    view.load_and_display_pointcloud = orig_load

# auto mode: selector change runs the full pipeline again (regression guard)
c.manual_pipeline_checkbox.setChecked(False)
assert settings.current_settings["manual_pipeline"] is False
for b in (c.btn_load_file, c.btn_downsample, c.btn_build_graph, c.btn_cluster):
    assert not b.isEnabled()
recorded.clear()
view.load_and_display_pointcloud = lambda *a, **k: recorded.append((a, k))
try:
    c.on_downsampling_changed()
    assert len(recorded) == 1, f"auto mode must run the pipeline on change: {recorded}"
    print("auto mode OK: selector change runs the full pipeline (regression guard)")

    # display_current_file: manual loads only, auto runs full pipeline
    c.pointcloud_files = ["cloud.txt"]
    c.current_file_index = 0
    view.current_folder_path = tmpd
    view.current_file = txt_path
    loaded = []
    view.load_file = lambda *a, **k: loaded.append(a)
    view.render_pointcloud = lambda *a, **k: None
    c.manual_pipeline_checkbox.setChecked(True)
    recorded.clear()
    c.display_current_file()
    assert loaded and not recorded, "manual display_current_file must load only, not run the pipeline"
    c.manual_pipeline_checkbox.setChecked(False)
    loaded.clear()
    c.display_current_file()
    assert not loaded and len(recorded) == 1, "auto display_current_file must run the full pipeline"
    print("file display OK: manual mode loads on file select, auto mode runs pipeline")
finally:
    view.load_and_display_pointcloud = orig_load

print("MANUAL PIPELINE: ALL PASSED")
