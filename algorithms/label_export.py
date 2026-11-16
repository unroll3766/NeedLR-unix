# algorithms/label_export.py
"""Apply cluster labels from a NeedLR annotation JSON back onto the original
(full-resolution) point cloud.

The annotation JSON stores the downsampled points per cluster plus their
labels. Downsampling selects a subset of the original points, so every
original point is assigned the label of its nearest annotated point
(scipy cKDTree). For LAS/LAZ inputs the per-point labels are written into an
extra dimension (default name "semantic", int8); existing extra dimensions are
preserved, and a name conflict with an existing extra dimension is an error.
For .txt inputs a fourth column is appended instead.
"""

import json
import os

import laspy
import numpy as np
from scipy.spatial import cKDTree

UNLABELED = -1
DEFAULT_DIM_NAME = "semantic"
MAX_LABEL_ID = 127  # int8 range: ids 1..127 for classes, UNLABELED (-1) for the rest


class LabelExportError(Exception):
    """Raised when a label export cannot be performed."""


def derive_label_map(labels) -> dict:
    """Build a stable label -> int id map (ids 1..N, unlabeled stays -1).

    Args:
        labels: Iterable of label values (strings or ints); -1 means unlabeled.

    Returns:
        dict: str(label) -> int id, ordered by string representation.

    Raises:
        LabelExportError: If there are more distinct labels than int8 allows.
    """
    unique = sorted({str(label) for label in labels if str(label) != str(UNLABELED)})
    if len(unique) > MAX_LABEL_ID:
        raise LabelExportError(
            f"{len(unique)} distinct labels exceed the int8 extra-dimension limit "
            f"of {MAX_LABEL_ID} classes."
        )
    return {name: i + 1 for i, name in enumerate(unique)}


def load_annotation_file(path: str):
    """Load an annotation JSON, accepting both the old list format and the
    new object format {"label_map": {...}, "clusters": [...]}.

    Returns:
        tuple: (clusters, label_map) where clusters is a list of
        {"cluster", "label", "points"} dicts and label_map is str -> int.
    """
    with open(path, "r") as file:
        data = json.load(file)
    if isinstance(data, dict):
        clusters = data.get("clusters", [])
        stored = {str(k): int(v) for k, v in data.get("label_map", {}).items()}
    else:
        clusters = data
        stored = {}
    labels = [cluster.get("label", UNLABELED) for cluster in clusters]
    derived = derive_label_map(labels)
    labeled_names = {str(l) for l in labels if str(l) != str(UNLABELED)}
    # Use the stored map when it covers every label; otherwise derive a
    # stable sorted map so ids do not depend on cluster order.
    if stored and labeled_names <= set(stored):
        return clusters, stored
    return clusters, derived


def save_annotation_file(path: str, clusters: list, label_map: dict) -> None:
    """Write an annotation JSON in the object format (label_map + clusters)."""
    payload = {
        "label_map": {str(k): int(v) for k, v in label_map.items()},
        "clusters": clusters,
    }
    with open(path, "w") as file:
        json.dump(payload, file, indent=4)


def assign_labels_nearest(
    original_points: np.ndarray, annotated_points: np.ndarray, annotated_ids: np.ndarray
) -> np.ndarray:
    """Assign each original point the id of its nearest annotated point.

    Returns:
        np.ndarray: int8 array of per-point label ids (UNLABELED = -1).
    """
    tree = cKDTree(annotated_points)
    _, nearest = tree.query(original_points, k=1)
    return annotated_ids[nearest].astype(np.int8)


def _read_xyz(path: str) -> np.ndarray:
    with open(path, "r") as file:
        data = [line.strip().split()[:3] for line in file]
    return np.array(data, dtype=float)


def export_labeled_las(
    src_path: str, dst_path: str, labels: np.ndarray, dim_name: str = DEFAULT_DIM_NAME
) -> None:
    """Write a copy of a LAS/LAZ file with the labels in an extra dimension.

    Existing extra dimensions are preserved. Raises LabelExportError if
    dim_name collides with an existing extra dimension name.
    """
    if not dim_name.isidentifier():
        raise LabelExportError(f'Invalid extra-dimension name: "{dim_name}"')
    las = laspy.read(src_path)
    existing = {d.name.lower() for d in las.point_format.extra_dimensions}
    if dim_name.lower() in existing:
        raise LabelExportError(
            f'Extra dimension "{dim_name}" already exists in '
            f"{os.path.basename(src_path)}; choose a different header name."
        )
    if hasattr(las, dim_name):
        raise LabelExportError(
            f'"{dim_name}" collides with a built-in LAS field; choose a different '
            "header name."
        )
    if len(labels) != len(las.points):
        raise LabelExportError(
            f"Label count ({len(labels)}) does not match point count "
            f"({len(las.points)}) of {os.path.basename(src_path)}."
        )
    las.add_extra_dim(laspy.ExtraBytesParams(name=dim_name, type=np.int8))
    setattr(las, dim_name, np.asarray(labels, dtype=np.int8))
    las.write(dst_path)


def export_labels_to_original(
    src_path: str,
    json_path: str,
    out_dir: str,
    dim_name: str = DEFAULT_DIM_NAME,
) -> dict:
    """Export labels from an annotation JSON onto the original point cloud.

    Args:
        src_path: Original point cloud (.las, .laz or .txt).
        json_path: Annotation JSON for that cloud.
        out_dir: Directory for the output file (<stem>_labeled.<ext>).
        dim_name: Extra-dimension name for LAS/LAZ outputs.

    Returns:
        dict: {"out_path", "n_points", "dim_name", "label_map", "label_counts"}.

    Raises:
        LabelExportError: On missing inputs, no annotated points, name
        conflicts, or too many classes for int8.
    """
    if not os.path.exists(src_path):
        raise LabelExportError(f"Original point cloud not found: {src_path}")
    if not os.path.exists(json_path):
        raise LabelExportError(
            f"No annotation file found at {json_path}; label some clusters first."
        )
    clusters, label_map = load_annotation_file(json_path)

    annotated_points = []
    annotated_ids = []
    for cluster in clusters:
        label = cluster.get("label", UNLABELED)
        key = str(label)
        if key == str(UNLABELED):
            label_id = UNLABELED
        elif key in label_map:
            label_id = label_map[key]
        else:
            raise LabelExportError(f'Label "{label}" has no id in the label map.')
        pts = np.asarray(cluster["points"], dtype=float)
        annotated_points.append(pts)
        annotated_ids.append(np.full(len(pts), label_id, dtype=np.int8))
    if not annotated_points:
        raise LabelExportError("The annotation file contains no clusters.")
    annotated_points = np.vstack(annotated_points)
    annotated_ids = np.concatenate(annotated_ids)

    ext = os.path.splitext(src_path)[1].lower()
    stem = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(out_dir, stem + "_labeled" + ext)

    if ext in (".las", ".laz"):
        las = laspy.read(src_path)
        original_points = np.vstack((las.x, las.y, las.z)).transpose()
        labels = assign_labels_nearest(original_points, annotated_points, annotated_ids)
        export_labeled_las(src_path, out_path, labels, dim_name)
    elif ext == ".txt":
        original_points = _read_xyz(src_path)
        labels = assign_labels_nearest(original_points, annotated_points, annotated_ids)
        np.savetxt(
            out_path,
            np.column_stack([original_points, labels]),
            fmt=["%.10f", "%.10f", "%.10f", "%d"],
        )
    else:
        raise LabelExportError(f"Unsupported point cloud format: {ext}")

    values, counts = np.unique(labels, return_counts=True)
    return {
        "out_path": out_path,
        "n_points": int(len(labels)),
        "dim_name": dim_name,
        "label_map": label_map,
        "label_counts": {int(v): int(c) for v, c in zip(values, counts)},
    }
