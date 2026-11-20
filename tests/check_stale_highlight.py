"""Regression test for the "stale cluster highlight" bug: navigating to another
cluster must replace the red points + green arrow in both panes, not stack
them. Prior clusters must not remain labelled.

Root cause (commit 7401c63): PointcloudView._figure_attached imported
`mayavi.core.engine_manager`, which does not exist in mayavi 4.9.0 (the module
lives at mayavi.tools.engine_manager), so it returned False for EVERY figure.
render_pointcloud then dropped both figures to None on every call, skipped
mlab.clf, and mlab.figure("Full Model"/"Cluster Zoom") re-attached to the same
named scenes with their old children intact -- so every render drew another
red cluster and arrow on top of the previous ones.

Run with the NeedLR env:
    QT_QPA_PLATFORM=offscreen python tests/check_stale_highlight.py

Uses a faithful fake mlab (name-based figure lookup, engine current_scene,
clf clearing children, plotting calls landing on `figure` or gcf()), so no GL
context is needed. The real _figure_attached is exercised against a fake
engine registered on mayavi.tools.engine_manager.engine_manager.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication

app = QApplication([])

import numpy as np
from mayavi import mlab
import mayavi.tools.engine_manager as em_module
from gui.pointcloud_view import PointcloudView, _figure_attached


# --- Faithful fake mayavi engine + mlab -------------------------------------
class FakeEngine:
    """Mimics mayavi.core.engine.Engine: scenes list + current_scene."""

    def __init__(self):
        self.scenes = []
        self.current_scene = None


class FakeEngineManager:
    """Mimics EngineManager.find_figure_engine (identity membership, like the
    real engine.scenes list of HasTraits objects)."""

    def __init__(self, engine):
        self._engine = engine

    def find_figure_engine(self, fig):
        if any(s is fig for s in self._engine.scenes):
            return self._engine
        raise TypeError("Figure not attached to a mayavi engine.")


class FakeFigure:
    """Mimics a mayavi figure: .scene, .name, .children (identity equality,
    like the real mayavi Scene which does not override __eq__)."""

    def __init__(self, name):
        cam = types.SimpleNamespace(
            position=(0, 0, 0),
            focal_point=(0, 0, 0),
            view_angle=30.0,
            view_up=(0, 0, 1),
            clipping_range=(0.0, 1.0),
            zoom=lambda factor: None,
            compute_view_plane_normal=lambda: None,
        )
        # no render_window/interactor attrs: window_utils.move_vtk_window
        # then hits its try/except and is a no-op (non-win32)
        self.scene = types.SimpleNamespace(camera=cam, render=lambda: None)
        self.name = name
        self.children = []


engine = FakeEngine()
orig_em = em_module.engine_manager
em_module.engine_manager = FakeEngineManager(engine)


def fake_figure(name=None, **kwargs):
    # mayavi's mlab.figure finds existing scenes by name and reuses them
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
    # mayavi's clf: clears the passed figure's (or gcf's) children
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
    # --- Unit: _figure_attached must recognize live figures -----------------
    # (with the buggy mayavi.core.engine_manager import this is always False)
    live = FakeFigure("live")
    engine.scenes.append(live)
    assert _figure_attached(live) is True, (
        "_figure_attached returned False for a live figure -- the engine_manager "
        "import path is broken, so render_pointcloud drops figures every call "
        "and skips mlab.clf (stale highlights accumulate)"
    )
    detached = FakeFigure("detached")  # has .scene but not in any engine
    assert _figure_attached(detached) is False
    assert _figure_attached(None) is False
    assert _figure_attached(types.SimpleNamespace(scene=None)) is False
    print("unit OK: _figure_attached True for live, False for detached/None")

    # --- End-to-end: navigating clusters must not accumulate highlights ------
    view = PointcloudView()
    view.points = np.arange(30, dtype=float).reshape(10, 3)
    view.clusters = [np.arange(0, 4), np.arange(4, 7), np.arange(7, 10)]
    view.cluster_labels = {0: -1, 1: -1, 2: -1}
    view.fig_full = None
    view.fig_zoom = None

    def reds(fig):
        return [c for c in fig.children if c[0] == "points3d" and c[1] == (1, 0, 0)]

    def arrows(fig):
        return [c for c in fig.children if c[0] == "quiver3d"]

    for idx in range(len(view.clusters)):
        view.current_cluster_index = idx
        view.render_pointcloud(show_full_view=True, show_cluster_view=True)
        fr, zr = reds(view.fig_full), reds(view.fig_zoom)
        fa, za = arrows(view.fig_full), arrows(view.fig_zoom)
        assert len(fr) == 1, f"after render {idx}: {len(fr)} red point sets in full pane (stale highlights accumulate)"
        assert len(zr) == 1, f"after render {idx}: {len(zr)} red point sets in cluster pane (stale highlights accumulate)"
        assert fr[0][2] == len(view.clusters[idx]) and zr[0][2] == len(view.clusters[idx]), (
            f"after render {idx}: red set covers {fr[0][2]}/{zr[0][2]} points, expected {len(view.clusters[idx])}"
        )
        assert len(fa) == 1 and len(za) == 0, (
            f"after render {idx}: arrows full={len(fa)} zoom={len(za)} -- the green arrow must "
            "be drawn exactly once, in the full pane"
        )
        print(f"render {idx} OK: 1 red set ({fr[0][2]} pts) + 1 arrow in full pane, 1 red set in cluster pane")
finally:
    for n, v in orig.items():
        setattr(mlab, n, v)
    em_module.engine_manager = orig_em

print("STALE HIGHLIGHT REGRESSION: ALL PASSED")
