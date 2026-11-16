"""Tests for algorithms/label_export.py (notes.md issue 6).

Run with the NeedLR env:
    python tests/test_label_export.py
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laspy
import numpy as np
from scipy.spatial import cKDTree

from algorithms.label_export import (
    LabelExportError,
    assign_labels_nearest,
    derive_label_map,
    export_labels_to_original,
    load_annotation_file,
)

WORK = tempfile.mkdtemp(prefix="needlr_export_test_")
rng = np.random.default_rng(7)
N = 200
pts = rng.random((N, 3)) * 100.0


def write_las(path, points, extra_dims=()):
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.offsets = points.min(axis=0)
    hdr.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    for name, dtype, values in extra_dims:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        setattr(las, name, values)
    las.write(path)
    return las


def make_json_list(path, subset, labels):
    clusters = []
    half = len(subset) // 2
    for i, label in enumerate(labels):
        chunk = subset[i * half : (i + 1) * half] if i < len(labels) - 1 else subset[i * half :]
        clusters.append(
            {
                "cluster": i,
                "label": label,
                "points": [[float(c) for c in p] for p in chunk],
            }
        )
    with open(path, "w") as f:
        json.dump(clusters, f)
    return clusters


# --- setup: original clouds + list-format annotation JSON ---
orig_las = os.path.join(WORK, "orig.las")
orig_laz = os.path.join(WORK, "orig.laz")
orig_txt = os.path.join(WORK, "orig.txt")
write_las(orig_las, pts, extra_dims=[("foo", np.int32, np.arange(N, dtype=np.int32))])
write_las(orig_laz, pts)
np.savetxt(orig_txt, pts, fmt="%.10f")

subset = pts[::4]  # 50 annotated points, exact original coordinates
ann_json = os.path.join(WORK, "orig.las.json")
make_json_list(ann_json, subset, ["tree", "rock"])

# --- 1. LAS export: extra dim added, existing extra dim preserved, NN labels ---
stats = export_labels_to_original(orig_las, ann_json, WORK, "semantic")
out = stats["out_path"]
assert out == os.path.join(WORK, "orig_labeled.las"), out
back = laspy.read(out)
dims = {d.name: str(d.dtype) for d in back.point_format.extra_dimensions}
assert dims == {"foo": "int32", "semantic": "int8"}, dims
assert np.array_equal(np.asarray(back.foo), np.arange(N, dtype=np.int32)), "foo changed!"
assert np.allclose(np.vstack((back.x, back.y, back.z)).T, pts, atol=1e-3)

tree = cKDTree(subset)
_, nn = tree.query(pts, k=1)
expected = np.where(nn < len(subset) // 2, 2, 1).astype(np.int8)  # tree=2, rock=1 (sorted map)
got = np.asarray(back.semantic)
assert np.array_equal(got, expected), "labels do not match nearest-neighbor ground truth"
assert stats["label_counts"] == {2: int((expected == 2).sum()), 1: int((expected == 1).sum())}
print("1 OK: LAS export, extra dims preserved, NN labels correct; counts:", stats["label_counts"])

# --- 2. name conflict -> error, nothing written ---
conflict_las = os.path.join(WORK, "conflict.las")
write_las(conflict_las, pts, extra_dims=[("semantic", np.int8, np.zeros(N, dtype=np.int8))])
conflict_json = os.path.join(WORK, "conflict.las.json")
make_json_list(conflict_json, subset, ["tree", "rock"])
try:
    export_labels_to_original(conflict_las, conflict_json, WORK, "semantic")
    raise AssertionError("expected LabelExportError")
except LabelExportError as e:
    assert "already exists" in str(e), str(e)
assert not os.path.exists(os.path.join(WORK, "conflict_labeled.las"))
print("2 OK: name conflict raises LabelExportError, no output written")

# --- 3. object-schema JSON with stored label_map is honored ---
obj_json = os.path.join(WORK, "orig2.las.json")
clusters = make_json_list(os.path.join(WORK, "tmp.json"), subset, ["tree", "rock"])
with open(obj_json, "w") as f:
    json.dump({"label_map": {"tree": 7, "rock": 9}, "clusters": clusters}, f)
orig2 = os.path.join(WORK, "orig2.las")
write_las(orig2, pts)
stats3 = export_labels_to_original(orig2, obj_json, WORK, "semantic")
back3 = laspy.read(stats3["out_path"])
expected3 = np.where(nn < len(subset) // 2, 7, 9).astype(np.int8)
assert np.array_equal(np.asarray(back3.semantic), expected3)
print("3 OK: stored label_map used (tree=7, rock=9)")

# --- 4. too many classes for int8 ---
try:
    derive_label_map([f"class_{i}" for i in range(130)])
    raise AssertionError("expected LabelExportError")
except LabelExportError as e:
    assert "int8" in str(e)
print("4 OK: >127 classes raises LabelExportError")

# --- 5. .txt export: 4th column ---
txt_json = os.path.join(WORK, "orig.txt.json")
make_json_list(txt_json, subset, ["tree", "rock"])
stats5 = export_labels_to_original(orig_txt, txt_json, WORK, "semantic")
out5 = np.loadtxt(stats5["out_path"])
assert out5.shape == (N, 4), out5.shape
assert np.array_equal(out5[:, 3].astype(np.int8), expected)
assert np.allclose(out5[:, :3], pts, atol=1e-6)
print("5 OK: .txt export writes x y z label, labels correct")

# --- 6. .laz input -> .laz output with extra dim ---
laz_json = os.path.join(WORK, "orig.laz.json")
make_json_list(laz_json, subset, ["tree", "rock"])
stats6 = export_labels_to_original(orig_laz, laz_json, WORK, "semantic")
assert stats6["out_path"].endswith(".laz")
back6 = laspy.read(stats6["out_path"])
assert "semantic" in {d.name for d in back6.point_format.extra_dimensions}
assert np.array_equal(np.asarray(back6.semantic), expected)
print("6 OK: .laz export round-trips with semantic dim")

# --- 7. backward compat: list-format JSON loads ---
clusters7, map7 = load_annotation_file(ann_json)
assert map7 == {"rock": 1, "tree": 2}, map7  # sorted by string: rock < tree
assert len(clusters7) == 2
print("7 OK: list-format JSON loads, label_map derived:", map7)

# --- 8. missing annotation file -> error ---
try:
    export_labels_to_original(orig_las, os.path.join(WORK, "nope.json"), WORK)
    raise AssertionError("expected LabelExportError")
except LabelExportError as e:
    assert "annotation" in str(e).lower()
print("8 OK: missing JSON raises LabelExportError")

shutil.rmtree(WORK)
print("LABEL EXPORT TESTS: ALL PASSED")
