"""Diagnostic + regression test for "Full Model pane goes black when
navigating clusters" (next/prev cluster buttons).

Hypothesis under test: the tail of render_pointcloud calls
zoom_to_cluster(self.fig_full, 2) on EVERY render, and vtkCamera.zoom(2)
halves the perspective view angle each call (distance unchanged -- verified
against vtk 9.7.2). self.full_camera is assigned the LIVE camera object, so
the "restore" lines in set_camera_focal_point (camera.view_angle =
original_camera.view_angle, etc.) are self-assignments and nothing ever
restores the zoom. The full pane's frustum therefore collapses
(30 -> 15 -> 7.5 -> ... deg), clipping out the whole cloud => black pane.
The Cluster Zoom pane never gets zoom_to_cluster, so it stays fine.

This script drives render_pointcloud repeatedly (simulating next/prev
clicks) with fake mayavi figures whose scene.camera is a REAL vtk.vtkCamera
(no GL context needed) and asserts the CORRECT behavior: the full pane's
view angle is stable across renders, both panes' focal points track the
current cluster centroid, and full_camera is a snapshot rather than the
live camera. On the buggy code it fails and prints the observed halving.

Run with the NeedLR env:
    QT_QPA_PLATFORM=offscreen python tests/check_full_pane_camera.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication

app = QApplication([])

import numpy as np
from tvtk.api import tvtk  # mayavi's wrapped VTK classes: real behavior, snake_case API
from mayavi import mlab
import mayavi.tools.engine_manager as em_module
from gui.pointcloud_view import PointcloudView, _figure_attached

INITIAL_VIEW_ANGLE = 30.0  # vtk perspective default


# --- Fake mayavi engine + mlab, with a REAL vtkCamera per figure -------------
class FakeEngine:
    def __init__(self):
        self.scenes = []
        self.current_scene = None


class FakeEngineManager:
    def __init__(self, engine):
        self._engine = engine

    def find_figure_engine(self, fig):
        if any(s is fig for s in self._engine.scenes):
            return self._engine
        raise TypeError("Figure not attached to a mayavi engine.")


class FakeFigure:
    """Mimics a mayavi figure, but scene.camera is a real (tvtk-wrapped)
    VTK camera so zoom()/attribute assignments in set_camera_focal_point
    behave exactly as in production (pure VTK, no GL)."""

    def __init__(self, name):
        cam = tvtk.Camera()
        cam.position = (0, 0, 12)
        cam.focal_point = (0, 0, 0)
        cam.view_angle = INITIAL_VIEW_ANGLE
        cam.clipping_range = (0.01, 1000.0)
        self.scene = types.SimpleNamespace(camera=cam, render=lambda: None)
        self.name = name
        self.children = []


engine = FakeEngine()
orig_em = em_module.engine_manager
em_module.engine_manager = FakeEngineManager(engine)


def fake_figure(name=None, **kwargs):
    for s in engine.scenes:
        if s.name == name:
            engine.current_scene = s
            return s
    f = FakeFigure(name)
    engine.scenes.append(f)
    engine.current_scene = f
    return f


def fake_gcf():
    return engine.current_scene


def fake_clf(figure=None):
    fig = figure if figure is not None else fake_gcf()
    fig.children[:] = []


def fake_points3d(x, y, z, **kw):
    fig = kw.get("figure") or fake_gcf()
    mod = ("points3d", kw.get("color"), len(x))
    fig.children.append(mod)
    return mod


def fake_quiver3d(x, y, z, u, v, w, **kw):
    fig = kw.get("figure") or fake_gcf()
    mod = ("quiver3d", kw.get("color"))
    fig.children.append(mod)
    return mod


orig = {n: getattr(mlab, n) for n in ("figure", "clf", "points3d", "quiver3d")}
mlab.figure = fake_figure
mlab.clf = fake_clf
mlab.points3d = fake_points3d
mlab.quiver3d = fake_quiver3d

try:
    # --- End-to-end: repeated renders (next/prev clicks) --------------------
    view = PointcloudView()
    view.points = np.arange(30, dtype=float).reshape(10, 3)
    view.clusters = [np.arange(0, 4), np.arange(4, 7), np.arange(7, 10)]
    view.cluster_labels = {0: -1, 1: -1, 2: -1}
    view.fig_full = None
    view.fig_zoom = None

    seq = []
    focal_ok = True
    for idx in range(4):
        view.current_cluster_index = idx % len(view.clusters)
        view.render_pointcloud(show_full_view=True, show_cluster_view=True)
        va_full = float(view.fig_full.scene.camera.view_angle)
        va_zoom = float(view.fig_zoom.scene.camera.view_angle)
        seq.append((va_full, va_zoom))
        centroid = view.points[view.clusters[view.current_cluster_index], :].mean(axis=0)
        fp_full = np.asarray(view.fig_full.scene.camera.focal_point)
        fp_zoom = np.asarray(view.fig_zoom.scene.camera.focal_point)
        if not (np.allclose(fp_full, centroid) and np.allclose(fp_zoom, centroid)):
            focal_ok = False
        print(
            f"render {idx}: full view_angle={va_full:9.4f}  zoom view_angle={va_zoom:9.4f}  "
            f"full focal={tuple(np.round(fp_full, 3))} zoom focal={tuple(np.round(fp_zoom, 3))}"
        )
    print("observed full/zoom view-angle sequence:",
          [f"{f:.3f}/{z:.3f}" for f, z in seq])

    # Root-cause diagnostics: is full_camera the live camera, and does
    # set_camera_focal_point actually restore a collapsed view angle?
    cam_full = view.fig_full.scene.camera
    print("full_camera is the live camera object:", view.full_camera is cam_full)
    cam_full.view_angle = 3.75  # simulate the accumulated zoom
    view.set_camera_focal_point(np.zeros(3), view.fig_full, view.full_camera)
    print("view_angle after set_camera_focal_point 'restore':",
          float(cam_full.view_angle))

    assert all(abs(f - INITIAL_VIEW_ANGLE) < 1e-6 for f, _ in seq), (
        f"BUG PRESENT: full-pane view angle collapses across renders {[round(f, 4) for f, _ in seq]} "
        f"(expected stable {INITIAL_VIEW_ANGLE}) -- zoom_to_cluster(fig_full, 2) halves it every "
        "render and set_camera_focal_point cannot restore it because full_camera is the live camera; "
        "the collapsed frustum clips out the whole cloud => black Full Model pane"
    )
    assert all(abs(z - INITIAL_VIEW_ANGLE) < 1e-6 for _, z in seq), (
        f"zoom pane view angle changed: {[round(z, 4) for _, z in seq]} (it should never be zoomed)"
    )
    assert focal_ok, "focal point must track the current cluster centroid in both panes"
    print("OK: view angles stable across renders; focal points track the current cluster")

    # --- Root-cause unit checks ----------------------------------------------
    assert view.full_camera is not view.fig_full.scene.camera, (
        "full_camera is the LIVE camera object, so set_camera_focal_point's restore lines are "
        "self-assignments -- nothing can undo the accumulated zoom"
    )

    # Simulate the accumulated zoom, then check that set_camera_focal_point
    # restores the original view angle from the snapshot.
    view.fig_full.scene.camera.view_angle = 3.75
    view.set_camera_focal_point(np.zeros(3), view.fig_full, view.full_camera)
    restored = float(view.fig_full.scene.camera.view_angle)
    assert abs(restored - INITIAL_VIEW_ANGLE) < 1e-6, (
        f"set_camera_focal_point did not restore the view angle (got {restored}, expected "
        f"{INITIAL_VIEW_ANGLE}) -- the original_camera reference is the live camera"
    )
    print("OK: full_camera is a snapshot and set_camera_focal_point restores the view angle")
finally:
    for n, v in orig.items():
        setattr(mlab, n, v)
    em_module.engine_manager = orig_em

print("FULL PANE CAMERA REGRESSION: ALL PASSED")
