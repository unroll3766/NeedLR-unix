# algorithms/downsampling.py
import sys

import numpy as np

_FPSAMPLE = None
_FPSAMPLE_CHECKED = False
_FALLBACK_WARNED = False


def _load_fpsample():
    """Return the optional CUDA 'fpsample' module, or None if not installed.

    The result is cached so the import is only attempted once per process.
    """
    global _FPSAMPLE, _FPSAMPLE_CHECKED
    if not _FPSAMPLE_CHECKED:
        try:
            import fpsample

            _FPSAMPLE = fpsample
        except ImportError:
            _FPSAMPLE = None
        _FPSAMPLE_CHECKED = True
    return _FPSAMPLE


def _warn_fallback(method_name: str) -> None:
    """Print a one-time notice that the numpy fallback is being used."""
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        print(
            "fpsample (CUDA) is not installed; using the slower pure-numpy "
            "FPS fallback. Install fpsample on a CUDA-capable machine for "
            "faster downsampling.",
            file=sys.stderr,
        )
        _FALLBACK_WARNED = True
    print(f"numpy FPS fallback used for: {method_name}", file=sys.stderr)


def _fps_numpy(points: np.ndarray, num_samples: int) -> np.ndarray:
    """Farthest point sampling in pure numpy.

    Maintains the squared distance from every point to the nearest already
    selected point, so each step is one vectorized pass over the cloud.
    O(num_samples * n); fine for the target sizes used here, slower than CUDA.

    Returns:
        np.ndarray: Indices of the sampled points.
    """
    n = len(points)
    if n <= num_samples:
        return np.arange(n)
    selected = np.empty(num_samples, dtype=np.int64)
    selected[0] = 0
    min_dist = np.sum((points - points[0]) ** 2, axis=1)
    for i in range(1, num_samples):
        nxt = int(np.argmax(min_dist))
        selected[i] = nxt
        d = np.sum((points - points[nxt]) ** 2, axis=1)
        np.minimum(min_dist, d, out=min_dist)
    return selected


def _fps_numpy_kdtree(points: np.ndarray, num_samples: int) -> np.ndarray:
    """FPS with a scipy cKDTree accelerating the distance update.

    After each selection, only points within sqrt(max remaining distance) of
    the new point can improve their min distance, so the update is local.

    Returns:
        np.ndarray: Indices of the sampled points.
    """
    from scipy.spatial import cKDTree

    n = len(points)
    if n <= num_samples:
        return np.arange(n)
    tree = cKDTree(points)
    selected = np.empty(num_samples, dtype=np.int64)
    selected[0] = 0
    min_dist = np.sum((points - points[0]) ** 2, axis=1)
    for i in range(1, num_samples):
        nxt = int(np.argmax(min_dist))
        selected[i] = nxt
        radius = float(np.sqrt(min_dist.max()))
        if radius > 0:
            idxs = tree.query_ball_point(points[nxt], r=radius)
            if idxs:
                d = np.sum((points[idxs] - points[nxt]) ** 2, axis=1)
                min_dist[idxs] = np.minimum(min_dist[idxs], d)
    return selected


def random_downsample(
    points: np.ndarray, target_count: int, return_indices: bool = False
) -> np.ndarray:
    """
    Randomly downsamples a point cloud to a specified number of points.

    Parameters:
    points (np.ndarray): A numpy array of shape (N, 3), where N is the number of points in the point cloud.
    target_count (int): The desired number of points after downsampling.

    Returns:
    np.ndarray: A downsampled numpy array of shape (target_count, 3).
    If return_indices is True, returns the sampled row indices instead.
    """
    if len(points) <= target_count:
        indices = np.arange(len(points))
        return indices if return_indices else points
    indices = np.random.choice(len(points), target_count, replace=False)
    return indices if return_indices else points[indices]


# Vanilla FPS
def vanilla_fps(
    points: np.ndarray, num_samples: int, return_indices: bool = False
) -> np.ndarray:
    """
    Perform vanilla farthest point sampling on a point cloud.

    Uses the CUDA fpsample package when installed, otherwise a pure-numpy
    fallback.

    Args:
    points (np.ndarray): Point cloud data.
    num_samples (int): Number of points to sample.
    return_indices (bool): Return the sampled row indices instead of the
    sampled points.

    Returns:
    np.ndarray: The sampled points (or indices).
    """
    fpsample = _load_fpsample()
    if fpsample is not None:
        indices = fpsample.fps_sampling(points, num_samples)
    else:
        _warn_fallback("vanilla_fps")
        indices = _fps_numpy(points, num_samples)
    return indices if return_indices else points[indices]


# FPS + NPDU
def fps_npdu(
    points: np.ndarray, num_samples: int, k: int = None, return_indices: bool = False
) -> np.ndarray:
    """
    Perform FPS with NPDU heuristic on a point cloud.
    **Require dimensional locality and give sub-optimal answers.

    Uses the CUDA fpsample package when installed, otherwise a pure-numpy
    incremental-distance fallback.

    Args:
    points (np.ndarray): Point cloud data.
    num_samples (int): Number of points to sample.
    k (int, optional): Window size for NPDU.
    return_indices (bool): Return the sampled row indices instead of the
    sampled points.

    Returns:
    np.ndarray: The sampled points (or indices).
    """
    fpsample = _load_fpsample()
    if fpsample is not None:
        if k is not None:
            indices = fpsample.fps_npdu_sampling(points, num_samples, k=k)
        else:
            indices = fpsample.fps_npdu_sampling(points, num_samples)
    else:
        _warn_fallback("fps_npdu")
        indices = _fps_numpy(points, num_samples)
    return indices if return_indices else points[indices]


# FPS + NPDU + KDTree
def fps_npdu_kdtree(
    points: np.ndarray, num_samples: int, k: int = None, return_indices: bool = False
) -> np.ndarray:
    """
    Perform FPS with NPDU heuristic and KDTree on a point cloud.

    Uses the CUDA fpsample package when installed, otherwise a scipy
    cKDTree-based numpy fallback.

    Args:
    points (np.ndarray): Point cloud data.
    num_samples (int): Number of points to sample.
    k (int, optional): Window size for NPDU.
    return_indices (bool): Return the sampled row indices instead of the
    sampled points.

    Returns:
    np.ndarray: The sampled points (or indices).
    """
    fpsample = _load_fpsample()
    if fpsample is not None:
        if k is not None:
            indices = fpsample.fps_npdu_kdtree_sampling(points, num_samples, k=k)
        else:
            indices = fpsample.fps_npdu_kdtree_sampling(points, num_samples)
    else:
        _warn_fallback("fps_npdu_kdtree")
        indices = _fps_numpy_kdtree(points, num_samples)
    return indices if return_indices else points[indices]


# KDTree-based FPS
# Deprecated
def kdtree_fps(
    points: np.ndarray, num_samples: int, return_indices: bool = False
) -> np.ndarray:
    """
    Perform KDTree-based farthest point sampling on a point cloud.

    Deprecated; kept for compatibility. Uses fpsample when installed,
    otherwise a scipy cKDTree-based numpy fallback.

    Args:
    points (np.ndarray): Point cloud data.
    num_samples (int): Number of points to sample.
    return_indices (bool): Return the sampled row indices instead of the
    sampled points.

    Returns:
    np.ndarray: The sampled points (or indices).
    """
    fpsample = _load_fpsample()
    if fpsample is not None:
        indices = fpsample.bucket_fps_kdtree_sampling(points, num_samples)
    else:
        _warn_fallback("kdtree_fps")
        indices = _fps_numpy_kdtree(points, num_samples)
    return indices if return_indices else points[indices]


# Bucket-based FPS or QuickFPS
def bucket_fps_kdline(
    points: np.ndarray, num_samples: int, h: int = 3, return_indices: bool = False
) -> np.ndarray:
    """
    Perform bucket-based (QuickFPS) farthest point sampling on a point cloud.

    Uses the CUDA fpsample package when installed; otherwise falls back to
    the cKDTree-based numpy FPS (same sampling quality, less bucketing speedup).

    Args:
    points (np.ndarray): Point cloud data.
    num_samples (int): Number of points to sample.
    h (int): Height of the KDTree (3 for small, 7 for medium, 9 for large data).
    return_indices (bool): Return the sampled row indices instead of the
    sampled points.

    Returns:
    np.ndarray: The sampled points (or indices).
    """
    fpsample = _load_fpsample()
    if fpsample is not None:
        indices = fpsample.bucket_fps_kdline_sampling(points, num_samples, h=h)
    else:
        _warn_fallback("bucket_fps_kdline")
        indices = _fps_numpy_kdtree(points, num_samples)
    return indices if return_indices else points[indices]
