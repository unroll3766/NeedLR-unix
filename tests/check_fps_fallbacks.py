"""Correctness checks for the numpy FPS fallbacks vs a brute-force reference.

Run with the NeedLR env:
    python tests/check_fps_fallbacks.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from algorithms.downsampling import _fps_numpy, _fps_numpy_kdtree


def brute_force_fps(points, num_samples):
    """Reference FPS: recompute min distance to the selected set every step."""
    n = len(points)
    if n <= num_samples:
        return np.arange(n)
    selected = [0]
    min_dist = np.sum((points - points[0]) ** 2, axis=1)
    while len(selected) < num_samples:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        d = np.sum((points - points[nxt]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
    return np.array(selected)


rng = np.random.default_rng(42)
pts = rng.random((300, 3)) * 50.0
k = 25

ref = brute_force_fps(pts, k)
got_plain = _fps_numpy(pts, k)
got_tree = _fps_numpy_kdtree(pts, k)

assert np.array_equal(ref, got_plain), "numpy FPS differs from brute-force reference"
assert np.array_equal(ref, got_tree), "kdtree FPS differs from brute-force reference"
assert len(set(got_plain.tolist())) == k, "duplicate indices in FPS result"

# Edge cases
assert np.array_equal(_fps_numpy(pts, len(pts)), np.arange(len(pts)))
assert np.array_equal(_fps_numpy(pts, len(pts) + 100), np.arange(len(pts)))
assert _fps_numpy(pts, 1).tolist() == [0]
assert _fps_numpy_kdtree(pts, 1).tolist() == [0]

# Larger cloud: count + validity + runtime sanity
big = rng.random((200_000, 3)) * 100.0
t0 = time.time()
out = _fps_numpy_kdtree(big, 2048)
dt = time.time() - t0
assert out.shape == (2048,), out.shape
assert out.min() >= 0 and out.max() < len(big)
assert len(set(out.tolist())) == 2048
print(f"200k-point cloud -> 2048 samples via kdtree fallback in {dt:.2f}s")

print("FPS FALLBACK CORRECTNESS: ALL PASSED")
