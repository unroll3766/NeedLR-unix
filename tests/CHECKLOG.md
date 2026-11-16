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
- Not verified here: actual on-screen Mayavi/VTK rendering and window positioning on
  macOS/Linux (needs a display; VTK SetPosition/SetSize path is used, win32gui fallback
  remains Windows-only). Worth a manual check on a Mac.
