from mayavi import mlab
import laspy
import numpy as np
import os
import config.settings as settings
from algorithms import downsampling, superpoint_graph
from algorithms.label_export import (
    LabelExportError,
    load_annotation_file,
    save_annotation_file,
)
from gui import window_utils
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QGuiApplication
import time
import algorithms.community_detection as cd
import json

# Anchor colors of the matplotlib "plasma" colormap (sampled at fixed
# positions). Used to build the label color gradient without importing
# matplotlib: its conda-forge build depends on pyside6/qt6-main, which
# would put a second Qt major into the env (see gui/__init__.py).
_PLASMA_STOPS = [
    (0.00, (0.050383, 0.029803, 0.527975)),
    (0.25, (0.482001, 0.018646, 0.652278)),
    (0.50, (0.800142, 0.278929, 0.470873)),
    (0.75, (0.940015, 0.585616, 0.250466)),
    (1.00, (0.941176, 0.972549, 0.129412)),
]


def _plasma_color(t: float) -> tuple:
    """Piecewise-linear plasma gradient at t in [0, 1]; returns an RGB tuple."""
    for (t0, c0), (t1, c1) in zip(_PLASMA_STOPS, _PLASMA_STOPS[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(c0[i] + f * (c1[i] - c0[i]) for i in range(3))
    return _PLASMA_STOPS[-1][1]


def _figure_attached(figure) -> bool:
    """True if a mayavi figure is still attached to a running engine.

    Closing a figure's window manually detaches it from the engine; using it
    afterwards raises TypeError("Figure not attached to a mayavi engine").
    """
    if figure is None or getattr(figure, "scene", None) is None:
        return False
    try:
        try:
            from mayavi.tools.engine_manager import engine_manager
        except ImportError:
            # Older mayavi releases keep this module under mayavi.core.
            from mayavi.core.engine_manager import engine_manager

        engine_manager.find_figure_engine(figure)
        return True
    except Exception:
        return False


class PointcloudView:
    def __init__(self, update_filename_callback=None):
        self.update_filename = update_filename_callback
        self.bg_color = (
            (0, 0, 0) if settings.current_settings["night_mode"] else (1, 1, 1)
        )
        self.cluster_label_update_func = None
        self.cluster_labels = {}
        self.label_map = {}  # str(label) -> int id, persisted in the JSON
        self.fig_full = None
        self.fig_zoom = None
        self.current_folder_path = None
        self.downsampling_method = None
        self.old_downsampling_method = None
        self.target_count = None
        self.old_target_count = None
        self.points = None
        self.features = None  # full per-point matrix (x, y, z in the first 3 cols)
        self.feature_names = []
        self.sampled_features = None
        self.cluster_dims = list(
            settings.current_settings.get("cluster_dims", ["x", "y", "z"])
        )
        self.current_file = None
        self.old_file = None
        self.graph = None

        self.superpoint_graph_method = None
        self.old_superpoint_graph_method = None
        self.superpoint_graph_args = None
        self.old_superpoint_graph_args = None

        self.clustering_algorithm = None
        self.old_clustering_algorithm = None
        self.clusters = None
        self.current_cluster_index = 0

        self.full_camera = None
        self.zoom_camera = None

        self.destination_folder = ""

    def set_destination_folder(self, folder_path):
        self.destination_folder = folder_path
        self.initialize_labels_file()

    def initialize_labels_file(self):
        label_file_path = os.path.join(
            self.destination_folder, os.path.basename(self.current_file) + ".json"
        )
        print("looking for save data: ", self.current_file + ".json")
        self.cluster_labels = {}
        if not os.path.exists(label_file_path):
            print("Did not find save data, creating new file")
            label_data = []
            for idx, cluster_indices in enumerate(self.clusters):
                # Retrieve actual points from self.points using indices in cluster
                cluster_points = self.points[cluster_indices]
                # Convert points to a list of lists with native Python data types
                cluster_points_list = (
                    cluster_points.tolist()
                    if isinstance(cluster_points, np.ndarray)
                    else cluster_points
                )
                cluster_points_list = [
                    [float(coord) for coord in point] for point in cluster_points_list
                ]

                label_data.append(
                    {"cluster": int(idx), "label": -1, "points": cluster_points_list}
                )

            save_annotation_file(label_file_path, label_data, self.label_map)
        else:
            print("Found save data, loading from file")
            # Handles both the old list format and {"label_map", "clusters"}
            try:
                save_data, self.label_map = load_annotation_file(label_file_path)
            except LabelExportError:
                # e.g. >127 distinct labels in an old file: keep the clusters,
                # the map is re-derived (and validated) at export time
                with open(label_file_path, "r") as file:
                    raw = json.load(file)
                save_data = raw.get("clusters", []) if isinstance(raw, dict) else raw
                self.label_map = {}

            # Flatten all points into a single array
            all_points = np.concatenate(
                [np.array(data["points"], dtype=float) for data in save_data]
            )
            self.points = all_points

            # Partition points into clusters based on the lengths of each cluster's points
            cluster_lengths = [len(data["points"]) for data in save_data]
            indices = np.cumsum([0] + cluster_lengths[:-1])
            self.clusters = [
                np.arange(start, start + length)
                for start, length in zip(indices, cluster_lengths)
            ]
            self.cluster_labels = {
                idx: data["label"] for idx, data in enumerate(save_data)
            }

            # Render the point cloud with loaded data
            self.render_pointcloud(
                show_full_view=self.fig_full, show_cluster_view=self.fig_zoom
            )

    def save_labels_to_file(self):
        label_file_path = os.path.join(
            self.destination_folder, os.path.basename(self.current_file) + ".json"
        )
        if os.path.exists(label_file_path):
            # Update the label file with current labels
            with open(label_file_path, "r") as file:
                data = json.load(file)
            label_data = data.get("clusters", []) if isinstance(data, dict) else data
            for cluster in label_data:
                cluster_idx = cluster["cluster"]
                if cluster_idx in self.cluster_labels:
                    cluster["label"] = self.cluster_labels[cluster_idx]
            # Ensure every assigned label has an id in the label map
            for label in self.cluster_labels.values():
                key = str(label)
                if key != "-1" and key not in self.label_map:
                    self.label_map[key] = max(self.label_map.values(), default=0) + 1
            save_annotation_file(label_file_path, label_data, self.label_map)

    def label_current_cluster(self, label):
        if self.current_cluster_index < len(self.clusters):
            self.cluster_labels[self.current_cluster_index] = label
            if str(label) != "-1" and str(label) not in self.label_map:
                self.label_map[str(label)] = max(self.label_map.values(), default=0) + 1
            # Optionally update visualization to reflect the new label

    def next_cluster(self):
        if self.current_cluster_index < len(self.clusters) - 1:
            self.current_cluster_index += 1
            self.update_cluster_visualization()

    def open_folder(self, folder_path, downsampling_method=None, target_count=None):
        self.current_folder_path = folder_path
        self.downsampling_method = downsampling_method
        self.target_count = target_count

        self.load_and_display_pointcloud(folder_path, downsampling_method, target_count)

    def position_window(self, figure, x, y):
        # Access the underlying VTK render window
        window = figure.scene.render_window
        interactor = figure.scene.interactor

        # Set the position
        window.SetPosition(x, y)

        # Necessary steps to apply the position change
        interactor.ReInitialize()

    def get_dynamic_colors(self, num_classes: int) -> list:
        return [_plasma_color(i / num_classes) for i in range(num_classes)]

    def label_color_map(self, labels):
        """Map label values to stable color indices.

        Labels may be ints or arbitrary strings (e.g. "tree"); they do not
        need to be numeric. The unlabeled sentinel -1 is excluded and is
        rendered gray by the caller.

        Returns:
            dict: label -> color index, ordered by string representation.
        """
        labeled = sorted((label for label in labels if label != -1), key=str)
        return {label: index for index, label in enumerate(labeled)}

    def render_pointcloud(self, show_full_view=None, show_cluster_view=None):
        # A manually closed figure window detaches the figure from the mayavi
        # engine; drop such stale figures so they are recreated below.
        if self.fig_full is not None and not _figure_attached(self.fig_full):
            self.fig_full = None
        if self.fig_zoom is not None and not _figure_attached(self.fig_zoom):
            self.fig_zoom = None

        # Clear existing point cloud data from the figures
        if self.fig_full:
            mlab.clf(self.fig_full)
            full_camera = None
        if self.fig_zoom:
            mlab.clf(self.fig_zoom)
            zoom_camera = None

        # Rendering before any file loaded (or after a failed load) must not crash
        if self.clusters is None:
            self.clusters = []
        centroid = None

        screen_size = QGuiApplication.primaryScreen().availableGeometry()
        width, height = screen_size.width() // 4, screen_size.height() // 2
        x = screen_size.width() - width

        point_mode = "point" if settings.current_settings["eco_mode"] else "sphere"

        # Create new figures if they don't exist
        if show_full_view is False and show_cluster_view is False:
            return  # Don't render anything if both views are disabled

        if not self.fig_full and show_full_view:
            self.fig_full = mlab.figure(
                "Full Model",
                size=(width, height),
                bgcolor=self.bg_color,
            )

        if not self.fig_zoom and show_cluster_view:
            self.fig_zoom = mlab.figure(
                "Cluster Zoom",
                size=(width, height),
                bgcolor=self.bg_color,
            )

        if self.fig_full:
            # Group clusters by their labels
            clusters_by_label = {}  # key is label, value is list of cluster indices

            for idx, cluster in enumerate(self.clusters):
                # cluster is a list of indices of points in the cluster
                label = self.cluster_labels.get(idx, -1)
                if label not in clusters_by_label:
                    clusters_by_label[label] = []
                clusters_by_label[label].extend(cluster)

            # Define colors for labels; labels may be arbitrary strings,
            # only the unlabeled sentinel -1 is special (rendered gray)
            label_indices = self.label_color_map(clusters_by_label.keys())
            dynamic_colors = self.get_dynamic_colors(max(len(label_indices), 1))
            # Render each group with its corresponding color
            for label, cluster_indices in clusters_by_label.items():
                color = (
                    dynamic_colors[label_indices[label]]
                    if label != -1
                    else (0.7, 0.7, 0.7)
                )  # Gray for unlabeled
                group_points = self.points[cluster_indices, :]

                mlab.points3d(
                    group_points[:, 0],
                    group_points[:, 1],
                    group_points[:, 2],
                    figure=self.fig_full,
                    scale_factor=settings.current_settings["point_size"] * 0.6,
                    color=color,
                    mode=point_mode,
                )

            # Highlight the current cluster in red
            if self.current_cluster_index < len(self.clusters):
                current_cluster = self.clusters[self.current_cluster_index]
                cluster_points = self.points[current_cluster, :]
                mlab.points3d(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    cluster_points[:, 2],
                    figure=self.fig_full,
                    scale_factor=settings.current_settings["cluster_point_size"],
                    color=(1, 0, 0),  # Red color
                    mode="sphere",
                )
                centroid_full = np.mean(self.points, axis=0)
                # Calculate the center of the cluster
                centroid = np.mean(cluster_points, axis=0)

                """ Create arrow that points to cluster centroid """
                # Render the centroid arrow if centroid is provided
                if centroid is not None:
                    scale_factor = 2  # Controls the size of the arrow
                    # Arrow's starting point (you might need to adjust this based on your visualization)
                    start_point = centroid - np.array([0, 0, 2.5])

                    # Arrow's direction (pointing towards the centroid)
                    direction = np.array([0, 0, 1])

                    # Adding the arrow to the visualization
                    mlab.quiver3d(
                        start_point[0],
                        start_point[1],
                        start_point[2],
                        direction[0],
                        direction[1],
                        direction[2],
                        scale_factor=scale_factor,
                        mode="arrow",
                        color=(0, 1, 0),
                        figure=self.fig_full,
                    )  # Green arrow

            if self.full_camera is None and self.fig_full is not None:
                self.full_camera = self.fig_full.scene.camera

        # Highlight the current cluster in red
        if self.fig_zoom:
            if self.clusters and self.current_cluster_index < len(self.clusters):
                current_cluster = self.clusters[self.current_cluster_index]
                cluster_points = self.points[current_cluster, :]

                # Calculate the center of the cluster
                centroid = np.mean(cluster_points, axis=0)

                # Render cluster points in red
                mlab.points3d(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    cluster_points[:, 2],
                    figure=self.fig_zoom,
                    scale_factor=settings.current_settings["cluster_point_size"],
                    color=(1, 0, 0),  # Red color,
                    mode="sphere",
                )
                if self.zoom_camera is None:
                    self.zoom_camera = self.fig_zoom.scene.camera

                # Set the camera's focal point to the centroid of the cluster
                # only do this one time per file
                self.set_camera_focal_point(centroid, self.fig_zoom, self.zoom_camera)

        self.zoom_to_cluster(self.fig_full, 2)
        if centroid is not None:
            self.set_camera_focal_point(centroid, self.fig_full, self.full_camera)

        # Move the windows after rendering
        if show_full_view:
            self.move_window_to_position("Full Model", x, 0, width, height)
        if show_cluster_view:
            self.move_window_to_position("Cluster Zoom", x, height, width, height)
        else:
            self.move_window_to_position("Full Model", x, 0, width, height * 2)

    def load_and_display_pointcloud(
        self,
        full_path,
        downsampling_method=None,
        target_count=None,
        show_full_view=True,
        show_cluster_view=True,
        superpoint_graph_method=None,
        k_value=None,
        clustering_algorithm=None,
    ):
        self.current_file = full_path
        self.superpoint_graph_method = superpoint_graph_method
        self.clustering_algorithm = clustering_algorithm
        # print(self.current_file)
        print("Being ran with clustering algorithm: ", self.clustering_algorithm)
        try:
            self.target_count = target_count
            if (
                target_count != self.old_target_count
                or downsampling_method != self.old_downsampling_method
                or self.current_file != self.old_file
            ):
                if self.current_file.endswith((".las", ".laz")):
                    las = laspy.read(self.current_file)
                    self._load_las_features(las)
                elif self.current_file.endswith(".txt"):
                    self._load_txt_features(self.current_file)
                # self.points stays the xyz cloud (rendering/labels use 3 cols)
                self.points = self.features[:, :3].copy()

            # Apply downsampling (spatial, on xyz only); keep the feature
            # matrix aligned by sampling with the same indices
            # print("Applying {} downsampling".format(downsampling_method))
            curr_time = time.time()
            indices = None
            if downsampling_method == "random":
                indices = downsampling.random_downsample(
                    self.points, target_count, return_indices=True
                )
            elif downsampling_method == "vanilla_fps":
                indices = downsampling.vanilla_fps(
                    self.points, target_count, return_indices=True
                )
            elif downsampling_method == "fps_npdu":
                indices = downsampling.fps_npdu(
                    self.points, target_count, return_indices=True
                )
            elif downsampling_method == "fps_npdu_kdtree":
                indices = downsampling.fps_npdu_kdtree(
                    self.points, target_count, return_indices=True
                )
            elif downsampling_method == "bucket_fps_kdline_small":
                indices = downsampling.bucket_fps_kdline(
                    self.points, target_count, h=4, return_indices=True
                )
            elif downsampling_method == "bucket_fps_kdline_medium":
                indices = downsampling.bucket_fps_kdline(
                    self.points, target_count, h=7, return_indices=True
                )
            elif downsampling_method == "bucket_fps_kdline_large":
                indices = downsampling.bucket_fps_kdline(
                    self.points, target_count, h=9, return_indices=True
                )
            if indices is not None:
                self.points = self.points[indices]
                if self.features is not None:
                    self.sampled_features = self.features[indices]
                else:
                    self.sampled_features = self.points
            print("Downsampling took", time.time() - curr_time, "seconds")

            # Perform superpoint graph construction on the selected
            # clustering dimensions
            # print(f"Performing {superpoint_graph_method} construction")
            graph_points = self._clustering_matrix()
            if superpoint_graph_method == "knn":
                # Call superpoint graph construction method
                curr_time = time.time()
                self.graph = superpoint_graph.create_knn_graph(graph_points, k_value)
            elif superpoint_graph_method == "mst":
                # Call superpoint graph construction method
                curr_time = time.time()
                self.graph = superpoint_graph.create_mst_graph(graph_points)
                # print(f"Graph construction took {time.time() - curr_time} seconds")

            # Apply community detection
            # print("Applying community detection algorithm: ", self.clustering_algorithm)
            curr_time = time.time()
            self.apply_community_detection(self.graph)
            # print("Community detection took", time.time() - curr_time, "seconds")
            self.render_pointcloud(show_full_view, show_cluster_view)

        except Exception as e:
            print(f"Error loading point cloud: {e}")

    def set_camera_focal_point(self, focal_point, figure, original_camera):
        # using original camera as a reference to set the focal point
        if figure is not None and figure.scene is not None:
            camera = figure.scene.camera
            camera.position = original_camera.position
            camera.focal_point = focal_point
            camera.view_angle = original_camera.view_angle
            camera.view_up = original_camera.view_up
            camera.clipping_range = original_camera.clipping_range
            camera.compute_view_plane_normal()
            figure.scene.render()

    def zoom_to_cluster(self, figure, zoom_factor):
        """Zoom into the scene by adjusting the camera distance.

        Args:
            figure (mayavi.core.scene.Scene): The Mayavi figure to adjust.
            zoom_factor (float): The factor by which to zoom. Less than 1 to zoom in, greater than 1 to zoom out.
        """
        if figure is not None and figure.scene is not None:
            # print("Zooming in on figure")
            camera = figure.scene.camera
            camera.zoom(zoom_factor)
            figure.scene.render()

    def apply_community_detection(self, graph):
        if self.clustering_algorithm == "label_propagation":
            self.clusters = cd.label_propagation(graph)
        elif self.clustering_algorithm == "async_lpa":
            self.clusters = cd.asyn_lpa_communities(graph)
        elif self.clustering_algorithm == "louvain":
            self.clusters = cd.louvain(graph)
        elif self.clustering_algorithm == "girvan_newman":
            self.clusters = cd.girvan_newman(graph)
        elif self.clustering_algorithm == "modularity":
            self.clusters = cd.modularity(graph)
        elif self.clustering_algorithm == "kernighan_lin":
            self.clusters = cd.kernighan_lin(graph)

        self.current_cluster_index = 0
        self.update_cluster_label_callback()

    def _load_las_features(self, las):
        """Build the full per-point feature matrix from a laspy LasData:
        all point-format dimensions (standard + extra), x/y/z first."""
        rest = [
            n for n in las.point_format.dimension_names if n not in ("x", "y", "z")
        ]
        names = ["x", "y", "z"] + list(rest)
        cols = [np.asarray(getattr(las, n), dtype=float) for n in names]
        self.features = np.column_stack(cols)
        self.feature_names = names

    def _load_txt_features(self, file_path):
        """Load all columns of a whitespace-separated .txt file into the
        feature matrix; columns past xyz are named col3, col4, ..."""
        with open(file_path, "r") as file:
            data = np.array(
                [line.strip().split() for line in file if line.strip()], dtype=float
            )
        if data.size == 0:
            data = np.empty((0, 3), dtype=float)
        elif data.ndim == 1:
            data = data.reshape(1, -1)
        self.features = data
        k = data.shape[1]
        self.feature_names = ["x", "y", "z"] + [f"col{i}" for i in range(3, k)]

    def _clustering_matrix(self):
        """Feature matrix used for graph/cluster distances: the sampled
        points restricted to the selected cluster_dims columns.

        Falls back to xyz when no features were loaded or none of the
        selected names exist in the current file (at least one column is
        always returned)."""
        base = self.sampled_features if self.sampled_features is not None else self.points
        if base is None:
            return None
        names = self.feature_names if self.feature_names else ["x", "y", "z"]
        idx = [
            names.index(d)
            for d in self.cluster_dims
            if d in names and names.index(d) < base.shape[1]
        ]
        if not idx:
            idx = list(range(min(3, base.shape[1])))
        return base[:, idx]

    def set_point_size(self, size):
        settings.current_settings["point_size"] = size
        if self.current_folder_path:
            self.load_and_display_pointcloud(self.current_folder_path)

    def set_night_mode(self, is_night_mode):
        bg_color = (0, 0, 0) if is_night_mode else (1, 1, 1)
        if self.fig_full and self.fig_zoom:
            # Only update the background color if the figures are initialized
            self.fig_full.scene.background = bg_color
            self.fig_zoom.scene.background = bg_color

    def close_figure(self, attr: str) -> None:
        """Close a mayavi figure if it is still attached to an engine, then
        forget it (set the attribute to None). Safe on stale/closed figures."""
        figure = getattr(self, attr)
        if figure is not None and _figure_attached(figure):
            try:
                mlab.close(figure)
            except Exception:
                pass
        setattr(self, attr, None)

    def move_window_to_position(self, title, x, y, width, height):
        # Window titles are set when the figures are created; map them back
        # to the figure objects and delegate to the cross-platform helper.
        figures = {"Full Model": self.fig_full, "Cluster Zoom": self.fig_zoom}
        window_utils.move_vtk_window(figures.get(title), x, y, width, height)

    # New method to trigger the update of the cluster label in the ControlPanel
    def update_cluster_label_callback(self):
        if self.cluster_label_update_func:
            self.cluster_label_update_func()

    def update_cluster_visualization(self):
        # Clear the existing visualization
        # Depending on how you've set up your visualization, you might need to clear
        # or reset the view. This could be done with mlab.clf() or similar.

        # Now, render the point cloud with the new focus on the current cluster
        self.render_pointcloud(
            show_full_view=self.fig_full, show_cluster_view=self.fig_zoom
        )

        # Update any additional UI elements or views that depend on the current cluster
        # For example, update labels, details, or stats about the current cluster.
        self.update_cluster_label_callback()
