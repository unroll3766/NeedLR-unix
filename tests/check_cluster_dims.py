"""Checks for selectable clustering dimensions: the user picks which
point-cloud columns feed the graph/cluster distance (e.g. reflectance for
spectral clustering), any number of columns but at least one.

Covers: downsampling `return_indices` consistency, feature-matrix loading
(.txt numbered columns, .laz extra dims), graph construction on the selected
columns only, and the GUI list (populated per file, min-1 enforcement,
reload on change).

Run with the NeedLR env:
    QT_QPA_PLATFORM=offscreen python tests/check_cluster_dims.py
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
import laspy

import config.settings as settings

# Tests must never write the repo's app_settings.json.
settings.save_settings = lambda *a, **k: None

from algorithms import downsampling as ds
from algorithms import superpoint_graph as spg
from gui.pointcloud_view import PointcloudView
from gui.window import MainWindow

rng = np.random.default_rng(7)

# --- cluster_dims setting present (defaults merge covers old files) ---------
assert "cluster_dims" in settings.current_settings
assert settings.current_settings["cluster_dims"] == ["x", "y", "z"]
print("settings OK: cluster_dims present, default [x, y, z]")

# --- downsampling return_indices ---------------------------------------------
pts = rng.random((200, 3))
for fn, kw in [
    (ds.vanilla_fps, {}),
    (ds.fps_npdu, {}),
    (ds.fps_npdu_kdtree, {}),
    (ds.kdtree_fps, {}),
    (ds.bucket_fps_kdline, {"h": 7}),
]:
    idx = fn(pts, 50, return_indices=True, **kw)
    arr = fn(pts, 50, **kw)
    assert idx.shape == (50,) and len(set(idx.tolist())) == 50, fn.__name__
    assert np.array_equal(arr, pts[idx]), f"{fn.__name__}: default return must equal points[indices]"
idx = ds.random_downsample(pts, 50, return_indices=True)
assert idx.shape == (50,) and len(set(idx.tolist())) == 50
assert ds.random_downsample(pts, 50).shape == (50, 3)
# target >= n returns all indices
assert np.array_equal(ds.vanilla_fps(pts, 500, return_indices=True), np.arange(200))
print("downsampling OK: return_indices matches default return for all methods")

# --- .txt loading: numbered columns ------------------------------------------
tmpd = tempfile.mkdtemp()
txt_path = os.path.join(tmpd, "cloud.txt")
data5 = rng.random((60, 5))
np.savetxt(txt_path, data5)

view = PointcloudView()
view.load_and_display_pointcloud(
    txt_path,
    downsampling_method="random",
    target_count=40,
    show_full_view=False,
    show_cluster_view=False,
    superpoint_graph_method="knn",
    k_value=3,
    clustering_algorithm="label_propagation",
)
assert view.feature_names == ["x", "y", "z", "col3", "col4"], view.feature_names
assert view.features.shape == (60, 5)
assert view.points.shape == (40, 3)  # rendering cloud stays xyz
assert view.sampled_features.shape == (40, 5)
assert np.array_equal(view.points, view.sampled_features[:, :3])
print("txt loading OK: 5 columns exposed as x,y,z,col3,col4; points stay Nx3")

# --- empty .txt: degenerate but must not produce a (1, 0) matrix -------------
empty_path = os.path.join(tmpd, "empty.txt")
open(empty_path, "w").close()
view_e = PointcloudView()
view_e.load_and_display_pointcloud(
    empty_path,
    downsampling_method="random",
    target_count=40,
    show_full_view=False,
    show_cluster_view=False,
    superpoint_graph_method="knn",
    k_value=3,
    clustering_algorithm="label_propagation",
)
assert view_e.features.shape == (0, 3), view_e.features.shape
assert view_e.points.shape == (0, 3), view_e.points.shape
assert view_e.feature_names == ["x", "y", "z"]
print("empty txt OK: features/points are (0, 3), names [x, y, z]")

# --- .laz loading: extra dims exposed ----------------------------------------
laz_path = os.path.join(tmpd, "cloud.laz")
hdr = laspy.LasHeader(point_format=3, version="1.2")
xyz = rng.random((80, 3)) * 10.0
hdr.offsets = xyz.min(axis=0)
hdr.scales = np.array([0.001, 0.001, 0.001])
las = laspy.LasData(hdr)
las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
las.add_extra_dim(laspy.ExtraBytesParams(name="reflectance", type=np.float32))
las.reflectance = rng.random(80).astype(np.float32)
las.write(laz_path)

view2 = PointcloudView()
view2.load_and_display_pointcloud(
    laz_path,
    downsampling_method="random",
    target_count=40,
    show_full_view=False,
    show_cluster_view=False,
    superpoint_graph_method="knn",
    k_value=3,
    clustering_algorithm="label_propagation",
)
assert "reflectance" in view2.feature_names, view2.feature_names
assert view2.feature_names[:3] == ["x", "y", "z"]
assert view2.features.shape == (80, len(view2.feature_names))
assert view2.points.shape == (40, 3)
assert np.array_equal(view2.points, view2.sampled_features[:, :3])
print(f"laz loading OK: {len(view2.feature_names)} columns incl. reflectance; points stay Nx3")

# --- graph is built on the selected columns only ------------------------------
captured = {}
orig_knn = spg.create_knn_graph


def recording_knn(points, k):
    captured["points"] = np.asarray(points).copy()
    captured["k"] = k
    return orig_knn(points, k)


spg.create_knn_graph = recording_knn
try:
    view.cluster_dims = ["col4"]
    view.load_and_display_pointcloud(
        txt_path,
        downsampling_method="random",
        target_count=40,
        show_full_view=False,
        show_cluster_view=False,
        superpoint_graph_method="knn",
        k_value=3,
        clustering_algorithm="label_propagation",
    )
    assert captured["points"].shape == (40, 1), captured["points"].shape
    assert np.array_equal(captured["points"][:, 0], view.sampled_features[:, 4])
    print("graph dims OK: cluster_dims=['col4'] -> graph built on col4 only")

    view.cluster_dims = ["x", "y", "z"]
    view.load_and_display_pointcloud(
        txt_path,
        downsampling_method="random",
        target_count=40,
        show_full_view=False,
        show_cluster_view=False,
        superpoint_graph_method="knn",
        k_value=3,
        clustering_algorithm="label_propagation",
    )
    assert captured["points"].shape == (40, 3)
    assert np.array_equal(captured["points"], view.sampled_features[:, :3])
    print("graph dims OK: default cluster_dims=[x,y,z] -> graph built on xyz")

    # unknown names fall back to xyz
    view.cluster_dims = ["nonexistent"]
    m = view._clustering_matrix()
    assert m.shape == (40, 3) and np.array_equal(m, view.sampled_features[:, :3])
    print("graph dims OK: unknown dim names fall back to xyz")
finally:
    spg.create_knn_graph = orig_knn

# --- GUI: dimension list -------------------------------------------------------
win = MainWindow()
win.show()
app.processEvents()
c = win.controls
view3 = c.pointcloud_view
view3.feature_names = ["x", "y", "z", "intensity", "reflectance"]
settings.current_settings["cluster_dims"] = ["x", "y", "z"]
c.populate_cluster_dims()
assert list(c.cluster_dim_rows) == ["x", "y", "z", "intensity", "reflectance"]
checked = [n for n, row in c.cluster_dim_rows.items() if row["checkbox"].isChecked()]
assert checked == ["x", "y", "z"], checked
assert view3.cluster_dims == ["x", "y", "z"]
print("dims UI OK: rows populated from file columns, saved selection applied")

# unchecking everything is not allowed: the last toggle is reverted
for n in ("x", "y", "z"):  # uncheck one by one
    c.cluster_dim_rows[n]["checkbox"].setChecked(False)
assert c.selected_cluster_dims() == ["z"], c.selected_cluster_dims()
assert settings.current_settings["cluster_dims"] == ["z"]
print("dims UI OK: empty selection reverted (min 1 enforced), settings updated")

# changing the selection triggers a pipeline reload with the new dims
view3.current_folder_path = "/tmp"
recorded = []
orig_load = view3.load_and_display_pointcloud
view3.load_and_display_pointcloud = lambda *a, **k: recorded.append((a, k))
try:
    c.cluster_dim_rows["intensity"]["checkbox"].setChecked(True)  # add "intensity"
    assert recorded, "changing clustering dims must trigger a reload"
    assert view3.cluster_dims == ["z", "intensity"], view3.cluster_dims
finally:
    view3.load_and_display_pointcloud = orig_load
print("dims UI OK: selection change reloads the pipeline with the new dims")

print("CLUSTERING DIMENSIONS: ALL PASSED")
