"""Diagnose the "all points render red" report: is the red highlight leaking
beyond the current cluster (regression), or is the current cluster simply
huge (data/sampling-dependent)?

Run with the NeedLR env:
    QT_QPA_PLATFORM=offscreen python tests/check_render_highlight.py

Uses fake mayavi figures and records every mlab.points3d call, so no GL
context is needed.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication

app = QApplication([])

import numpy as np
from mayavi import mlab

from gui.pointcloud_view import PointcloudView


def make_fake_figure(name):
    cam = types.SimpleNamespace(
        position=(0, 0, 0),
        focal_point=(0, 0, 0),
        view_angle=30.0,
        view_up=(0, 0, 1),
        clipping_range=(0.0, 1.0),
        zoom=lambda factor: None,
        compute_view_plane_normal=lambda: None,
    )
    scene = types.SimpleNamespace(
        camera=cam, render=lambda: None, render_window=None, interactor=None
    )
    return types.SimpleNamespace(scene=scene, name=name)


def render_and_collect(view):
    calls = []
    orig = (mlab.figure, mlab.clf, mlab.points3d, mlab.quiver3d)
    mlab.figure = lambda *a, **k: make_fake_figure(a[0] if a else "f")
    mlab.clf = lambda *a, **k: None
    mlab.quiver3d = lambda *a, **k: None

    def recording_points3d(x, y, z, **kw):
        calls.append((kw.get("color"), len(x)))

    mlab.points3d = recording_points3d
    try:
        view.render_pointcloud(show_full_view=True, show_cluster_view=False)
    finally:
        mlab.figure, mlab.clf, mlab.points3d, mlab.quiver3d = orig
    return calls


# --- Case 1: two clusters, current = cluster 0 -> red must cover only cluster 0 ---
view = PointcloudView()
view.points = np.arange(30, dtype=float).reshape(10, 3)
view.clusters = [np.arange(0, 4), np.arange(4, 10)]
view.cluster_labels = {0: -1, 1: -1}
view.current_cluster_index = 0
view.fig_full = view.fig_zoom = None

calls = render_and_collect(view)
red = [c for c in calls if c[0] == (1, 0, 0)]
others = [c for c in calls if c[0] != (1, 0, 0)]
assert len(red) == 1 and red[0][1] == 4, f"red highlight leaked: {red}"
assert sum(c[1] for c in others) == 10, others
print(f"case1 OK: 2 clusters -> red covers exactly the current cluster ({red[0][1]}/10 points)")

# --- Case 2: a single cluster -> red covers everything, BY DESIGN ---
view2 = PointcloudView()
view2.points = view.points
view2.clusters = [np.arange(0, 10)]
view2.cluster_labels = {0: -1}
view2.current_cluster_index = 0
view2.fig_full = view2.fig_zoom = None
calls2 = render_and_collect(view2)
red2 = [c for c in calls2 if c[0] == (1, 0, 0)]
assert len(red2) == 1 and red2[0][1] == 10
print("case2 OK: 1 cluster -> red covers all 10 points (expected, not a leak)")

# --- Case 3: how many clusters does the default pipeline produce on a
# --- synthetic cloud, per sampler? (shows the count is sampling-dependent) ---
from algorithms import downsampling as ds, superpoint_graph as spg, community_detection as cd

rng = np.random.default_rng(3)
blob = rng.normal(0, 1, (4000, 3))
for method in ["random", "bucket_fps_kdline_medium", "vanilla_fps"]:
    sampled = (
        ds.random_downsample(blob, 2048)
        if method == "random"
        else ds.bucket_fps_kdline(blob, 2048, h=7)
        if method == "bucket_fps_kdline_medium"
        else ds.vanilla_fps(blob, 2048)
    )
    graph = spg.create_mst_graph(sampled)
    clusters = cd.label_propagation(graph)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"sampler={method:24s} -> {len(clusters)} clusters, largest={sizes[0]}/{len(sampled)}")

print("RENDER HIGHLIGHT DIAGNOSTIC: ALL PASSED")
