# settings.py

import json
import os

# Anchor all project paths to the repo root so the app works regardless of
# the current working directory it is launched from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
settings_file = os.path.join(BASE_DIR, "app_settings.json")

default_settings = {
    "night_mode": False,
    "point_size": 1.0,  # Point size of the background (non-highlighted) points
    "cluster_point_size": 1.0,  # Point size of the highlighted cluster
    "cluster_dims": ["x", "y", "z"],  # Point-cloud columns used for clustering distance
    "cluster_dim_weights": {},  # Per-dimension weights for clustering distance (default 1.0)
    "manual_pipeline": False,  # True: user runs each pipeline stage via buttons
    "show_full_view": True,
    "show_cluster_view": True,
    "eco_mode": False,
    "num_classes": 3,
    "superpoint_graph": "mst",
    "KNN_graph": 8,
    "Subsampling": "bucket_fps_kdline_medium",
    "Subsample_size": 2048,
    "Community_detection": "label_propagation",
}


def load_settings():
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                # Merge over the defaults so keys missing from an older
                # settings file (e.g. newly introduced options) still get a
                # value.
                return {**default_settings, **loaded}
        except json.JSONDecodeError:
            pass
    return dict(default_settings)


def save_settings(settings):
    with open(settings_file, "w") as file:
        json.dump(settings, file, indent=4)


# Initialize settings on import
current_settings = load_settings()
