# CHECKLOG — NeedLR cross-platform port verification

Environment used for all checks: conda env `NeedLR` (Python 3.11.18, PyQt5 5.15.15,
mayavi 4.9.0, vtk 9.7.2, numpy 2.4.6, networkx 3.6.1, laspy 2.7.0), created from the
new `environment.yml` on linux-64. No display server on this machine; GUI checks use
`QT_QPA_PLATFORM=offscreen`.

- 2026-11-16 — `conda env create -f environment.yml` solves and installs cleanly from
  conda-forge on linux-64 (dry-run + real create, exit 0).
- 2026-11-16 — `python -m py_compile` passes on main.py, gui/*.py, config/*.py,
  algorithms/*.py.
- 2026-11-16 — Import + offscreen GUI smoke test (`QT_QPA_PLATFORM=offscreen`, run from a
  temp CWD): `config.settings.settings_file` resolves to the repo-root
  `app_settings.json` regardless of CWD; `MainWindow` constructs and shows; all five
  downsampling methods return the requested point count on a 5000-point cloud (numpy
  FPS fallbacks active, one-time warning on stderr); KNN graph (512 nodes, 40 clusters
  via label propagation) and MST graph (512 nodes, 511 edges) build correctly.
  Script: tests/smoke_port.py, exit 0.
- 2026-11-16 — `python main.py` under `QT_QPA_PLATFORM=offscreen` starts and stays in the
  Qt event loop until killed by timeout (exit 124), no high-DPI warning after deferring
  the gui import until after `AA_EnableHighDpiScaling` is set.
- 2026-11-16 — Entry point is CWD-independent: `python /home/theo/Documents/NeedLR-unix/main.py`
  launched from /tmp under `QT_QPA_PLATFORM=offscreen` runs in the Qt event loop until
  killed by timeout (exit 124), no warnings besides the offscreen plugin notice.
- 2026-11-16 — Numpy FPS fallbacks match a brute-force FPS reference exactly on a
  300-point cloud (both `_fps_numpy` and `_fps_numpy_kdtree`, 25 samples, identical
  index sequence, no duplicates); edge cases num_samples >= n and num_samples == 1 OK;
  2048 samples from a 200k-point cloud via the kdtree fallback in ~7 s.
  Script: tests/check_fps_fallbacks.py, exit 0.
- 2026-11-16 — notes.md issues 2-5 verified offscreen via `tests/smoke_gui.py`
  (QT_QPA_PLATFORM=offscreen, exit 0): (2) empty folder now shows "No point cloud files
  found" instead of stuck "Loading..."; (3) `label_color_map` accepts string/int/mixed
  labels, stable ordering, unlabeled -1 excluded; (4) controls wrapped in QScrollArea,
  window 480x560 fits 800x600 available screen; (5) `.laz` listed by the folder loader
  and loaded+downsampled+clustered through `load_and_display_pointcloud` (120 pts ->
  64 samples, 10 clusters), `.txt` path unchanged. laspy 2.7 + lazrs-python 0.8.2
  round-trip a .laz file correctly.
- 2026-11-16 — Review pass: all verification scripts promoted from /tmp into tests/
  (smoke_port.py, check_fps_fallbacks.py, smoke_gui.py) and re-run from the repo with
  the NeedLR env; all exit 0. `py_compile` clean on main.py, gui/, config/, algorithms/,
  tests/.
- 2026-11-16 — Issue 1 fix (Qt5/Qt6 objc duplicate-class warnings): root cause is
  matplotlib's conda-forge build hard-depending on pyside6 -> qt6-main, so Qt6 libs sit
  in the env; on the reporter's Mac pyface then bound pyside6 and libQt6Core loaded
  next to libQt5Core. Fix: new gui/__init__.py pins QT_API=pyqt5 (setdefault) before any
  mayavi/pyface import, since all mayavi imports go through gui submodules.
  tests/check_qt_backend.py verifies that after `import gui.window` only PyQt5 modules
  are in sys.modules (libQt6Core never enters the process), exit 0; all smoke suites
  still pass. Caveat: the Qt6 load could not be reproduced on this Linux container
  (pyside6 import fails here even when forced), so the disappearance of the macOS objc
  warnings still needs one confirmation on a Mac.
- 2026-11-16 — Issue 1 follow-up (Mac retest showed the objc warnings persist):
  conda-meta inspection showed full `matplotlib` depended on `pyside6` and `vtk-base`
  on `qt6-main`; pyside6 (auto-selected by pyface/matplotlib when QT_API was unset)
  was the libQt6Core loader. VTK's own Qt libs (libvtkRenderingQt/GUISupportQt) are
  never imported by mayavi. Fix: matplotlib removed from environment.yml (it was only
  used for `plt.get_cmap("plasma")`; pointcloud_view.py now has a built-in plasma
  gradient), QT_API=pyqt5 pin kept in gui/__init__.py. Local env updated with
  `conda env update --prune`: pyside6/matplotlib gone; qt6-main still installed via
  vtk-base but nothing loads it. tests/check_qt_backend.py exit 0 (only PyQt5 in
  sys.modules); all smoke suites pass, incl. new get_dynamic_colors assertions.
  Mac action: pull + recreate the env; with no pyside6 installed and the pin set,
  libQt6Core cannot enter the process. Confirm warnings are gone there.
- 2026-11-16 — Issue 1 root-caused for real (Mac warnings persisted after the first fix):
  LD_DEBUG=libs showed `import gui.window` loads libQt6Core via
  vtkmodules/vtkRenderingQt -> libvtkRenderingQt -> libQt6Core (conda-forge vtk links
  Qt6 since 9.3; vtk-base 9.2.6 is the last Qt-free conda build but mayavi needs
  vtk>=9.4). Fix: vtk + mayavi now come from PyPI (Qt-free wheels; macosx arm64 wheel
  verified to exist), conda keeps only python/pyqt/numpy/scipy/sklearn/networkx/laspy/
  lazrs-python. After rebuilding the env: `conda list` has no qt6-main/pyside6/vtk-base;
  readelf shows no libQt* NEEDED in pip vtk .so files; LD_DEBUG on both
  `import gui.window` and a full `python main.py` launch shows 0 libQt6Core loads and
  exactly 1 libQt5Core load — the objc duplicate-class warnings are now impossible.
  All smoke suites + qt-backend check pass on the rebuilt env.
- 2026-11-16 — Issue 6 implemented and verified (tests/test_label_export.py, exit 0, plus a
  persistence check in smoke_gui.py): nearest-neighbor label assignment via cKDTree onto the
  original cloud; LAS/LAZ output gains an int8 extra dim (default "semantic") while existing
  extra dims (tested with a pre-existing "foo" int32 dim) and xyz are preserved; name
  conflict with an existing extra dim raises LabelExportError and writes nothing; >127
  classes raises; .txt gets a 4th column; label_map persisted in the JSON (object schema,
  old list-format files still load with a derived sorted map; stored maps are honored).
  UI: "Export Labels to Original Cloud" button + header-name field in the Output groupbox.
- 2026-11-16 — Review pass on issue 6; three edge cases found and fixed: (a) label-id
  assignment used len(map)+1 which could collide with gappy stored maps — now
  max(values)+1 (tested: {"x":1,"y":5} + "z" -> 6); (b) label_map was not reset when
  switching files — now reset in display_current_file; (c) a cluster with an empty
  points list crashed cKDTree with a raw IndexError — points are reshaped to (-1, 3)
  and an empty annotation raises LabelExportError. The view's JSON load also tolerates
  >127-label old files (map re-derived at export, where the proper error is raised).
  All 5 test suites exit 0 after the fixes; test_label_export.py now has 9 checks.
- 2026-11-16 — Bugfix: `select_dest_folder` referenced `self.btn_export_labels` but the
  widget-creation block in initUI had been lost (never committed), so selecting a
  destination folder raised AttributeError (crash on macOS). Re-added the block
  (Header field + Export button in the Output groupbox). Regression test in
  smoke_gui.py: widgets exist after initUI, button starts disabled, and a monkeypatched
  select_dest_folder() enables it without error. All 5 suites exit 0.
- 2026-11-16 — Cluster-view crash fix (user-reported): closing the Cluster Zoom window
  manually detaches the figure from the mayavi engine, but fig_zoom stayed set, so the
  next render raised TypeError("Figure not attached to a mayavi engine"); toggling the
  view off with clusters=None raised TypeError('NoneType' object is not iterable), and
  the tail used a possibly-unbound `centroid`. Fixes in pointcloud_view.py:
  _figure_attached() check drops stale figures at the top of render_pointcloud so they
  are recreated; clusters normalized to [] when None; centroid initialized and the final
  set_camera_focal_point guarded; new PointcloudView.close_figure() makes the view
  toggles safe on stale figures. Regression test in smoke_gui.py uses fake figures
  (mlab.figure/clf monkeypatched, no GL needed): stale figures replaced, clusters=None
  render safe, close_figure no-op on stale. All 5 suites exit 0.
- 2026-11-16 — Review pass on the cluster-view fix: verified all intended changes are in
  commit 7401c63 and the working tree (grep-checked), removed the now-unused
  `from mayavi import mlab` import in controls.py (the toggles were its only users),
  re-ran all 5 suites (exit 0, unpiped) + py_compile + offscreen `python main.py`
  launch (exit 124 = event loop OK).
- 2026-11-16 — "All points render red" report diagnosed via tests/check_render_highlight.py
  (exit 0): the red highlight pass provably covers only clusters[current_cluster_index]
  (2-cluster case: red call got exactly 4/10 points; single-cluster case: all 10, by
  design). Not a regression — the label-color pass covers all points, the red pass only the
  current cluster. "All red" happens when community detection yields one giant cluster for
  the sampled cloud (sampling-dependent; the numpy FPS fallback samples differently than
  the CUDA fpsample did, so cluster structure on a given cloud can differ from before).
  On a uniform synthetic cloud the default pipeline gives ~630 clusters, largest ~11/2048.
- 2026-11-16 — Stale-cluster-highlight regression (user-reported: navigating clusters
  left prior clusters' red dots + green arrow stacked in both panes) root-caused to
  commit 7401c63: `_figure_attached` did `from mayavi.core.engine_manager import
  engine_manager`, a module that does not exist in mayavi 4.9.0 (it lives at
  `mayavi.tools.engine_manager`), so the import always raised and the function
  returned False for every figure. render_pointcloud then nulled both figures each
  call, skipped `mlab.clf`, and `mlab.figure("Full Model"/"Cluster Zoom")` re-attached
  to the same named scenes with old children intact, stacking a new red highlight +
  arrow per render. Fix in pointcloud_view.py: import from `mayavi.tools.engine_manager`
  with a `mayavi.core.engine_manager` fallback for older mayavi; also pass
  `figure=self.fig_full` to the centroid `mlab.quiver3d` so the arrow lands in the
  full pane deterministically instead of on whatever figure `gcf()` points to.
  Regression test tests/check_stale_highlight.py (fake mlab with name-based figure
  lookup, gcf/current_scene, children-clearing clf; real `_figure_attached` against a
  fake engine): fails on pre-fix code (`_figure_attached(live)` is False; red sets
  stack 1->2->3 across renders), passes after the fix (exactly 1 red set per pane and
  1 arrow in the full pane after every render, red set covering only the current
  cluster). All 7 test scripts exit 0, py_compile clean, offscreen `python main.py`
  launch OK (exit 124).
- 2026-11-20 — Separate background/cluster point sizes (notes.md "Thoughts" 1)
  implemented and verified: new `cluster_point_size` setting (default 1.0) in
  config/settings.py; `load_settings()` now merges the JSON file over
  `default_settings` so keys missing from older app_settings.json files fall back
  to defaults (previously the file was returned verbatim, so a new key would
  KeyError); key added to app_settings.json. render_pointcloud uses
  `point_size * 0.6` for background points (unchanged) and `cluster_point_size`
  for the red highlighted cluster in both panes. UI: new "Cluster Size" spinbox
  next to "Point Size" (controls.py) with its own handler that re-renders live;
  eco mode disables both selectors, and the eco checkbox now connects before
  setChecked so an initial eco_mode=true disables them at startup. Test
  tests/check_point_sizes.py (exit 0): defaults-merge with an old settings file,
  both spinboxes present with settings values, each writes only its own key,
  render mapping verified via recorded mlab calls (background 0.8*0.6, red 2.0
  in both panes), handler re-renders at the new cluster size, eco mode toggling.
  All 8 test scripts exit 0, py_compile clean, offscreen `python main.py` OK (124).
- 2026-11-20 — Review pass on the point-size commit found a test side effect:
  check_point_sizes.py mutates settings.current_settings and MainWindow.closeEvent
  saves settings when the window is destroyed at interpreter exit, so a test run
  overwrote the repo's app_settings.json with test values (caught via git status;
  file restored). Fixed: all tests that construct MainWindow (smoke_port.py,
  smoke_gui.py, check_point_sizes.py) now stub settings.save_settings to a no-op,
  and git status stays clean after running them. Also hardened load_settings() to
  fall back to defaults when the JSON file is valid but not an object. Re-ran all
  8 scripts (exit 0) + py_compile; app_settings.json untouched by the test runs.
- 2026-11-20 — Selectable clustering dimensions implemented and verified: the user
  picks which point-cloud columns feed the graph/cluster distance (e.g. reflectance
  for spectral clustering); any number of columns, at least one enforced. View now
  loads a full feature matrix (`self.features`, x/y/z first) alongside the xyz
  `self.points` (rendering/annotation-JSON/label-export unchanged, still Nx3):
  LAS/LAZ expose all point-format dimensions incl. extra dims via
  `las.point_format.dimension_names`; .txt exposes all columns as x,y,z,col3,col4...
  Downsampling stays spatial (xyz) and all public downsamplers gained a backward-
  compatible `return_indices=True` kwarg so the view slices features with the same
  indices. Graph construction uses `_clustering_matrix()` = sampled features
  restricted to `cluster_dims` (default ["x","y","z"], persisted in settings;
  unknown names fall back to xyz). UI: checkable "Clustering Dimensions (min 1)"
  list in the Graph Construction groupbox, populated per file from feature_names,
  saved selection applied, unchecking the last item is reverted, any change persists
  cluster_dims and reloads the pipeline. Test tests/check_cluster_dims.py (exit 0):
  return_indices == points[indices] for all 6 downsamplers + target>=n edge;
  5-col .txt and .laz-with-reflectance loading (points stay Nx3); knn graph captured
  receives exactly the selected columns; UI populate/min-1/reload checks. All 9
  test scripts exit 0, py_compile clean, offscreen `python main.py` OK (124),
  app_settings.json untouched by test runs.
- 2026-11-20 — Review pass on the clustering-dims commit: verified the full diff
  (downsampling return_indices on all 6 functions, controls list + handlers, view
  feature loading/graph matrix), confirmed load_xyz_file fully removed, re-ran all
  9 scripts (exit 0). Hardened one edge found in review: an empty .txt file now
  loads as a (0, 3) matrix instead of (1, 0) (test added to check_cluster_dims.py).
  Known limitation, accepted: on the prev/next-file path that loads an existing
  annotation JSON, the dims list keeps the previous file's columns (no
  re-clustering happens on that path, so cluster_dims is unused there).
- 2026-11-20 — Manual pipeline control implemented and verified: new
  `manual_pipeline` setting (default false, so existing behavior is unchanged).
  pointcloud_view.py's load_and_display_pointcloud is refactored into stage
  methods -- load_file (read only), apply_downsampling (spatial, invalidates
  graph/clusters), build_graph (on _clustering_matrix(), invalidates clusters),
  apply_clustering -- and load_and_display_pointcloud composes them, so the auto
  path is unchanged. controls.py gains a "Pipeline" groupbox with a "Manual
  Pipeline" checkbox (persisted, connect-before-setChecked) and four buttons
  (Load File, Downsample, Build Graph, Cluster) enabled only in manual mode;
  each button runs its stage and re-renders. In manual mode the downsampling/
  graph/community/dims handlers only store settings, and display_current_file
  just loads the file + renders (file auto-loads on selection, Load File
  re-reads). Stage buttons ask before invalidating annotations only when an
  annotation JSON exists for the current file. Test tests/check_manual_pipeline.py
  (exit 0): stages run independently with correct invalidation, auto path
  regression, button enable/disable per mode, manual mode does not auto-run on
  selector changes, buttons call exactly their stage, display_current_file
  branches per mode. All 10 test scripts exit 0, py_compile clean, offscreen
  `python main.py` OK (124), app_settings.json untouched by test runs.
- 2026-11-20 — Review pass on the manual-pipeline commit: read the full diff;
  confirmed the stage split preserves the auto path (guard + stages + render),
  each stage self-guards on empty state (buttons are safe no-ops with no file),
  and all five manual-mode branches sit in the right handlers
  (display_current_file, on_cluster_dims_changed, on_downsampling_changed,
  on_community_detection_changed, create_superpoint_graph). Re-ran all 10
  scripts from the committed tree (exit 0) + py_compile + offscreen main.py
  (124); app_settings.json untouched by test runs.
- Not verified here: actual on-screen Mayavi/VTK rendering and window positioning on
  macOS/Linux (needs a display; VTK SetPosition/SetSize path is used, win32gui fallback
  remains Windows-only). Worth a manual check on a Mac.
- 2026-11-20 — Full-model black-pane bug (user-reported: next/prev cluster click blacks
  the Full Model pane, clicking it stays black, wheel zoom revives it; Cluster Zoom
  unaffected) root-caused, proven, and fixed. Diagnosis proven by the new
  tests/check_full_pane_camera.py run against pre-fix code with a real tvtk camera on
  fake figures: render_pointcloud's tail called zoom_to_cluster(fig_full, 2) on every
  render and vtkCamera.zoom(2) halves the perspective view_angle (30 -> 15 -> 7.5 ->
  3.75 -> 1.875 across 4 renders) while the zoom pane stayed at 30; self.full_camera
  held the LIVE camera, so set_camera_focal_point's restore lines were self-assignments
  (manually setting view_angle=3.75 then calling it left it at 3.75). The collapsed
  frustum clips out the whole cloud => black pane; wheel zoom (Dolly toward the focal
  centroid) revives it. Fix in pointcloud_view.py: new _camera_snapshot() stores
  parameter copies instead of the live camera; full_camera/zoom_camera are reset in the
  clf block (previously dead locals), when a stale figure is dropped, and in
  close_figure, so a fresh snapshot is taken at each draw; the tail now always calls
  set_camera_focal_point, which restores position/view_angle/view_up/clipping_range
  from the snapshot and overrides only the focal point when a centroid exists. Test
  passes after the fix: view angles stable at 30/30 across renders, focal points track
  the current cluster, full_camera is a snapshot, restore works (3.75 -> 30). Bug
  predates the port (identical tail in original c61e50f). smoke_gui fake camera gained
  compute_view_plane_normal to match real cameras. All 11 test scripts exit 0,
  py_compile clean, offscreen main.py OK (124), app_settings.json untouched.
