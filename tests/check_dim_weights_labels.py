"""Checks for the three user-requested QOL changes:

1. LAS feature names must not duplicate the coordinate triple: raw scaled
   ints X/Y/Z are dropped (x/y/z float coords are kept, first).
2. Navigating clusters with prev/next populates the label entry box with the
   current cluster's saved label (cleared when unlabeled).
3. Per-dimension clustering weights: spinbox per row in the dims list,
   default 1.0, clamped at 0, persisted in cluster_dim_weights, applied as
   column scaling in _clustering_matrix (graph/cluster distance only).

Also covers case-insensitive dim resolution (a saved "X" maps to "x").

Run with the NeedLR env:
    QT_QPA_PLATFORM=offscreen python tests/check_dim_weights_labels.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("/tmp")

from PyQt5.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox
from PyQt5.QtCore import Qt

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
app = QApplication([])

import numpy as np
import laspy

import config.settings as settings

# Tests must never write the repo's app_settings.json.
settings.save_settings = lambda *a, **k: None

from gui.pointcloud_view import PointcloudView
from gui.window import MainWindow

rng = np.random.default_rng(11)

# --- 1. LAS feature names: no X/Y/Z duplicates --------------------------------
hdr = laspy.LasHeader(point_format=3, version="1.2")
hdr.offsets = np.zeros(3)
hdr.scales = np.array([0.001, 0.001, 0.001])
las = laspy.LasData(hdr)
xyz = rng.random((50, 3)) * 10.0
las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
las.add_extra_dim(laspy.ExtraBytesParams(name="reflectance", type=np.float32))
las.reflectance = rng.random(50).astype(np.float32)
# laspy exposes the raw scaled ints as X/Y/Z in dimension_names (the float
# x/y/z are computed properties, not listed) -- this is the duplication source
assert {"X", "Y", "Z"} <= set(las.point_format.dimension_names)
assert not {"x", "y", "z"} & set(las.point_format.dimension_names)

view = PointcloudView()
view._load_las_features(las)
names = view.feature_names
assert names[:3] == ["x", "y", "z"], names
assert not any(n.lower() in ("x", "y", "z") for n in names[3:]), names
assert "intensity" in names and "reflectance" in names, names
assert view.features.shape == (50, len(names))
print(f"las names OK: {names} -- X/Y/Z dropped, x/y/z kept first")

# --- 1b. case-insensitive dim resolution --------------------------------------
view.points = rng.random((20, 3))
view.sampled_features = None
view.feature_names = ["x", "y", "z", "intensity"]
view.cluster_dims = ["X", "Y", "Z"]  # as an old settings file might store
m = view._clustering_matrix()
assert m.shape == (20, 3) and np.array_equal(m, view.points), "saved X/Y/Z must map to x/y/z"
print("case-insensitive dims OK: ['X','Y','Z'] resolves to the x/y/z columns")

# --- 3. weights in _clustering_matrix ------------------------------------------
view.sampled_features = rng.random((20, 4))
view.cluster_dims = ["x", "y"]
view.cluster_dim_weights = {}
m = view._clustering_matrix()
assert np.array_equal(m, view.sampled_features[:, :2]), "default weight 1.0 = unscaled"

view.cluster_dim_weights = {"x": 2.0}
m = view._clustering_matrix()
assert np.allclose(m[:, 0], view.sampled_features[:, 0] * 2.0)
assert np.allclose(m[:, 1], view.sampled_features[:, 1] * 1.0)
print("weights OK: x scaled by 2.0, y by default 1.0")

view.cluster_dim_weights = {"x": 0.0, "y": -5.0}  # 0 mutes; negatives clamp to 0
m = view._clustering_matrix()
assert np.allclose(m[:, 0], 0.0) and np.allclose(m[:, 1], 0.0)
print("weights OK: 0 mutes a column, negative clamps to 0")

# weight keyed by the raw saved name ("X") still applies via canonical name
view.cluster_dims = ["X"]
view.cluster_dim_weights = {"X": 3.0}
m = view._clustering_matrix()
assert np.allclose(m[:, 0], view.sampled_features[:, 0] * 3.0)
print("weights OK: weight keyed by raw dim name resolves through the canonical name")

# unknown dims still fall back to unweighted xyz
view.cluster_dims = ["nonexistent"]
view.cluster_dim_weights = {}
m = view._clustering_matrix()
assert m.shape == (20, 3) and np.array_equal(m, view.sampled_features[:, :3])
print("fallback OK: unknown dims -> unweighted xyz")

# --- GUI: dims list with per-row weight spinboxes ------------------------------
win = MainWindow()
win.show()
app.processEvents()
c = win.controls
view3 = c.pointcloud_view

view3.feature_names = ["x", "y", "z", "intensity", "reflectance"]
settings.current_settings["cluster_dims"] = ["X", "Y", "Z"]  # case-insensitive save
settings.current_settings["cluster_dim_weights"] = {"reflectance": 2.5}
c.populate_cluster_dims()
assert c.cluster_dims_list.count() == 5
checked = [
    c.cluster_dims_list.item(i).text()
    for i in range(c.cluster_dims_list.count())
    if c.cluster_dims_list.item(i).checkState() == Qt.Checked
]
assert checked == ["x", "y", "z"], checked  # saved X/Y/Z mapped onto x/y/z
assert view3.cluster_dims == ["x", "y", "z"]
rows = [
    c.cluster_dims_list.itemWidget(c.cluster_dims_list.item(i))
    for i in range(c.cluster_dims_list.count())
]
spins = [r.findChild(QDoubleSpinBox) for r in rows]
cbs = [r.findChild(QCheckBox) for r in rows]
assert all(r is not None and s is not None and cb is not None for r, s, cb in zip(rows, spins, cbs)), (
    "every row must be a widget with a name checkbox and a weight spinbox"
)
# the item widget covers the item text, so the row checkbox must show the name
for i in range(c.cluster_dims_list.count()):
    item = c.cluster_dims_list.item(i)
    assert cbs[i].text() == item.text(), f"row {i}: name label missing ({cbs[i].text()!r} != {item.text()!r})"
    assert cbs[i].isChecked() == (item.checkState() == Qt.Checked)
vals = [s.value() for s in spins]
assert vals == [1.0, 1.0, 1.0, 1.0, 2.5], vals  # default 1.0, saved 2.5 applied
assert all(s.minimum() == 0.0 for s in spins), "weights clamped at 0"
assert view3.cluster_dim_weights == {"reflectance": 2.5}
print("dims UI OK: name visible per row (checkbox label), spinbox per row, defaults 1.0, saved weight applied, min 0")

# clicking the row checkbox toggles the item check state (min-1 still enforced)
cbs[0].setChecked(False)
assert c.selected_cluster_dims() == ["y", "z"], c.selected_cluster_dims()
c.cluster_dims_list.item(1).setCheckState(Qt.Unchecked)
c.cluster_dims_list.item(2).setCheckState(Qt.Unchecked)
# last one reverted, and the row checkbox re-syncs
assert c.selected_cluster_dims() == ["z"]
assert cbs[2].isChecked(), "reverted row must re-check its checkbox"
settings.current_settings["cluster_dims"] = ["x", "y", "z"]
print("dims UI OK: row checkbox drives selection, min-1 revert re-syncs the row")

# changing a weight persists it and reloads the pipeline (auto mode)
view3.current_folder_path = "/tmp"
recorded = []
orig_load = view3.load_and_display_pointcloud
view3.load_and_display_pointcloud = lambda *a, **k: recorded.append((a, k))
try:
    settings.current_settings["manual_pipeline"] = False
    spins[0].setValue(4.0)  # weight of x
    assert settings.current_settings["cluster_dim_weights"]["x"] == 4.0
    assert view3.cluster_dim_weights["x"] == 4.0
    assert recorded, "weight change must reload the pipeline in auto mode"
    # manual mode: persist only, no reload
    recorded.clear()
    settings.current_settings["manual_pipeline"] = True
    spins[1].setValue(0.0)
    assert settings.current_settings["cluster_dim_weights"]["y"] == 0.0
    assert not recorded, "manual mode must not auto-run the pipeline"
finally:
    view3.load_and_display_pointcloud = orig_load
    settings.current_settings["manual_pipeline"] = False
    settings.current_settings["cluster_dim_weights"] = {}
print("weight UI OK: change persists + auto reload; manual mode stores only")

# --- 2. label entry box follows cluster navigation -----------------------------
view3.points = np.arange(12, dtype=float).reshape(4, 3)
view3.clusters = [np.array([0, 1]), np.array([2, 3])]
view3.cluster_labels = {0: "tree", 1: -1}
view3.current_cluster_index = 0
c.update_cluster_label()
assert c.label_entry_textbox.text() == "tree", c.label_entry_textbox.text()
view3.current_cluster_index = 1
c.update_cluster_label()
assert c.label_entry_textbox.text() == "", "unlabeled cluster must clear the box"
view3.clusters = []
c.update_cluster_label()
assert c.cluster_label.text() == "No clusters"
assert c.label_entry_textbox.text() == ""
print("label box OK: coded cluster populates the box, unlabeled clears it")

# save_label_and_next: after Enter, the box shows the NEXT cluster's label
dest = tempfile.mkdtemp()
view3.destination_folder = dest
view3.current_file = "cloud.las"
view3.clusters = [np.array([0, 1]), np.array([2, 3])]
view3.cluster_labels = {1: "pine"}
view3.current_cluster_index = 0
c.label_entry_textbox.setText("oak")
c.save_label_and_next()
assert view3.cluster_labels[0] == "oak", view3.cluster_labels
assert view3.current_cluster_index == 1
assert c.label_entry_textbox.text() == "pine", c.label_entry_textbox.text()
print("save+next OK: box repopulated with the newly current cluster's label")

print("DIM WEIGHTS + LABEL BOX + LAS NAMES: ALL PASSED")
