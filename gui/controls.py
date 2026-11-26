import typing
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QCheckBox,
    QDoubleSpinBox,
    QComboBox,
    QSpinBox,
    QListWidget,
    QListWidgetItem,
    QApplication,
    QMessageBox,
    QStyle,
    QGroupBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import os
import config.settings as settings
from algorithms import label_export
from algorithms.label_export import DEFAULT_DIM_NAME


class ControlPanel(QWidget):
    def __init__(self, parent, pointcloud_view):
        super().__init__(parent)
        self.pointcloud_view = pointcloud_view
        self.pointcloud_view.cluster_label_update_func = self.update_cluster_label
        self.pointcloud_files = []  # List of point cloud files
        self.current_file_index = 0  # Index of the currently displayed file
        self.dest_folder_path = None  # Path to the destination folder for labels
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Add graphic banner to top of layout and make it fit the width of the window
        banner = QLabel(self)
        pixmap = QPixmap(os.path.join(settings.BASE_DIR, "ReadME_Assets", "NeedLR.png"))
        banner.setPixmap(pixmap)
        banner.setScaledContents(True)
        # Keep aspect ratio of the image
        banner.setFixedHeight(int(pixmap.height() * 0.3))
        # set size policy to ignore
        banner.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(banner)

        """DATA IMPORT GROUPBOX"""
        input_navigation_groupbox = QGroupBox("Data Import")
        input_navigation_groupbox_layout = QVBoxLayout()

        # Add widgets to the input navigation groupbox

        # Folder selection button and path display
        self.btn_select_folder = QPushButton("Select Pointcloud Folder", self)
        self.btn_select_folder.clicked.connect(self.select_folder)
        input_navigation_groupbox_layout.addWidget(self.btn_select_folder)

        self.folder_path_display = QLineEdit(self)
        self.folder_path_display.setReadOnly(True)
        input_navigation_groupbox_layout.addWidget(self.folder_path_display)

        # File navigation layout
        nav_layout = QHBoxLayout()
        self.btn_prev_file = QPushButton("<", self)
        self.btn_prev_file.clicked.connect(self.prev_file)
        nav_layout.addWidget(self.btn_prev_file)

        self.file_name_display = QLabel("No file selected", self)
        self.file_name_display.mousePressEvent = self.change_file
        self.file_name_display.setFixedWidth(200)  # Set a fixed width for the label
        nav_layout.addWidget(self.file_name_display)

        self.btn_next_file = QPushButton(">", self)
        self.btn_next_file.clicked.connect(self.next_file)
        nav_layout.addWidget(self.btn_next_file)

        input_navigation_groupbox_layout.addLayout(nav_layout)

        # add input navigation groupbox to layout
        input_navigation_groupbox.setLayout(input_navigation_groupbox_layout)
        layout.addWidget(input_navigation_groupbox)

        """ OUTPUT GROUPBOX """
        output_groupbox = QGroupBox("Output Settings")
        output_groupbox_layout = QVBoxLayout()

        # Add widgets to the output groupbox
        """ Destination Folder Selection """
        # Button and line edit for destination folder selection
        self.btn_select_dest_folder = QPushButton("Select Destination Folder", self)
        self.btn_select_dest_folder.setEnabled(False)
        self.btn_select_dest_folder.clicked.connect(self.select_dest_folder)

        self.dest_folder_path_display = QLineEdit(self)
        self.dest_folder_path_display.setReadOnly(True)

        # Layout for destination folder selection
        dest_folder_layout = QHBoxLayout()
        dest_folder_layout.addWidget(self.btn_select_dest_folder)
        dest_folder_layout.addWidget(self.dest_folder_path_display)

        output_groupbox_layout.addLayout(dest_folder_layout)

        """ Label export to the original point cloud """
        export_layout = QHBoxLayout()
        export_layout.addWidget(QLabel("Header:"))
        self.export_header_name = QLineEdit(DEFAULT_DIM_NAME, self)
        self.export_header_name.setMaximumWidth(120)
        self.export_header_name.setToolTip(
            "Extra-dimension name for the labels in the exported LAS/LAZ file"
        )
        self.btn_export_labels = QPushButton("Export Labels to Original Cloud", self)
        self.btn_export_labels.setEnabled(False)
        self.btn_export_labels.setToolTip(
            "Write a copy of the original cloud with per-point labels in an "
            "extra dimension (int8), based on the saved annotation JSON"
        )
        self.btn_export_labels.clicked.connect(self.export_labels)
        export_layout.addWidget(self.export_header_name)
        export_layout.addWidget(self.btn_export_labels)
        output_groupbox_layout.addLayout(export_layout)

        # add output groupbox to layout
        output_groupbox.setLayout(output_groupbox_layout)
        layout.addWidget(output_groupbox)

        """VISUALIZATION SETTINGS GROUPBOX"""
        visualization_settings_groupbox = QGroupBox("Visualization Settings")
        visualization_settings_groupbox_layout = QVBoxLayout()

        # Add widgets to the visualization settings groupbox
        """ Point-size and Nightmode Layout """
        ps_nm_layout = QHBoxLayout()

        # Night mode checkbox
        self.night_mode_checkbox = QCheckBox("Night Mode", self)
        self.night_mode_checkbox.setChecked(settings.current_settings["night_mode"])
        self.night_mode_checkbox.toggled.connect(self.toggle_night_mode)

        # Point size layout
        point_size_layout = QHBoxLayout()
        self.point_size_label = QLabel("Point Size:", self)
        point_size_layout.addWidget(self.point_size_label)

        self.point_size_selector = QDoubleSpinBox(self)
        self.point_size_selector.setSingleStep(0.1)
        self.point_size_selector.setValue(settings.current_settings["point_size"])
        self.point_size_selector.valueChanged.connect(self.on_point_size_changed)
        point_size_layout.addWidget(self.point_size_selector)

        # Cluster point size layout (size of the highlighted cluster only)
        cluster_size_layout = QHBoxLayout()
        self.cluster_point_size_label = QLabel("Cluster Size:", self)
        cluster_size_layout.addWidget(self.cluster_point_size_label)

        self.cluster_point_size_selector = QDoubleSpinBox(self)
        self.cluster_point_size_selector.setSingleStep(0.1)
        self.cluster_point_size_selector.setValue(
            settings.current_settings["cluster_point_size"]
        )
        self.cluster_point_size_selector.valueChanged.connect(
            self.on_cluster_point_size_changed
        )
        cluster_size_layout.addWidget(self.cluster_point_size_selector)
        cluster_size_layout.addStretch(1)

        # Eco mode checkbox (connect before setChecked so the initial
        # eco state disables the point-size selectors right away)
        self.eco_mode_checkbox = QCheckBox("Eco Mode", self)
        self.eco_mode_checkbox.toggled.connect(self.toggle_eco_mode)
        self.eco_mode_checkbox.setChecked(settings.current_settings["eco_mode"])

        # Make the layout tight
        point_size_layout.addStretch(1)

        # add ps and nm layouts to ps_nm_layout
        ps_nm_layout.addLayout(point_size_layout)
        ps_nm_layout.addLayout(cluster_size_layout)
        ps_nm_layout.addWidget(self.night_mode_checkbox)
        ps_nm_layout.addWidget(self.eco_mode_checkbox)

        visualization_settings_groupbox_layout.addLayout(ps_nm_layout)

        # add visualization settings groupbox to layout
        visualization_settings_groupbox.setLayout(
            visualization_settings_groupbox_layout
        )
        layout.addWidget(visualization_settings_groupbox)

        """DOWNSAMPLING GROUPBOX"""
        downsampling_groupbox = QGroupBox("Downsampling Options")
        downsampling_groupbox_layout = QVBoxLayout()
        """ Downsampling Layout """
        downsampling_layout = QHBoxLayout()
        # Dropdown for downsampling algorithm selection
        self.downsampling_algorithm_selector = QComboBox(self)
        self.downsampling_algorithm_selector.addItem("Random Downsampling", "random")
        self.downsampling_algorithm_selector.addItem("Vanilla FPS", "vanilla_fps")
        self.downsampling_algorithm_selector.addItem("FPS + NPDU", "fps_npdu")
        self.downsampling_algorithm_selector.addItem(
            "FPS + NPDU + KDTree", "fps_npdu_kdtree"
        )
        self.downsampling_algorithm_selector.addItem(
            "QuickFPS (Small)", "bucket_fps_kdline_small"
        )
        self.downsampling_algorithm_selector.addItem(
            "QuickFPS (Medium)", "bucket_fps_kdline_medium"
        )
        self.downsampling_algorithm_selector.addItem(
            "QuickFPS (Large)", "bucket_fps_kdline_large"
        )

        # Load the current selection from settings
        current_subsampling = settings.current_settings["Subsampling"]
        print("Current subsampling found from settings: ", current_subsampling)
        index = self.downsampling_algorithm_selector.findData(current_subsampling)
        print("index on subsampling: ", index)
        if index >= 0:
            self.downsampling_algorithm_selector.setCurrentIndex(index)

        self.downsampling_algorithm_selector.currentIndexChanged.connect(
            self.on_downsampling_changed
        )

        # Input field for specifying target number of points
        self.target_point_count = QSpinBox(self)
        self.target_point_count.setRange(1, 1000000)  # Adjust range as needed
        self.target_point_count.setValue(
            settings.current_settings["Subsample_size"]
        )  # Default value
        self.target_point_count.lineEdit().returnPressed.connect(
            self.on_downsampling_changed
        )

        downsampling_layout.addWidget(self.downsampling_algorithm_selector)
        downsampling_layout.addWidget(self.target_point_count)

        # add downsampling groupbox to layout
        downsampling_groupbox_layout.addLayout(downsampling_layout)
        downsampling_groupbox.setLayout(downsampling_groupbox_layout)
        layout.addWidget(downsampling_groupbox)

        """GRAPH CONSTRUCTION GROUPBOX"""
        graph_groupbox = QGroupBox("Graph Construction")
        graph_groupbox_layout = QVBoxLayout()

        """ Superpoint Graph Algorithm selection """
        graph_layout = QHBoxLayout()
        # Dropdown for superpoint graph algorithm selection
        self.superpoint_graph_algorithm_selector = QComboBox(self)
        self.superpoint_graph_algorithm_selector.addItem("KNN Graph", "knn")
        self.superpoint_graph_algorithm_selector.addItem("MST Graph", "mst")

        # Load the current selection from settings
        current_spg = settings.current_settings["superpoint_graph"]
        print("Current spg found from settings: ", current_spg)
        index = self.superpoint_graph_algorithm_selector.findData(current_spg)
        if index >= 0:
            self.superpoint_graph_algorithm_selector.setCurrentIndex(index)

        self.superpoint_graph_algorithm_selector.currentIndexChanged.connect(
            self.create_superpoint_graph
        )

        # Input field for specifying K for KNN
        self.k_value = QSpinBox(self)
        self.k_value.setRange(1, 40)  # Adjust range as needed
        self.k_value.setValue(settings.current_settings["KNN_graph"])  # Default value
        # when k value is changed, refresh the point cloud view
        self.k_value.lineEdit().returnPressed.connect(self.create_superpoint_graph)
        print("index on spg: ", index)
        if index != 0:
            self.k_value.setEnabled(
                False
            )  # disable k selection for anything other than knn

        graph_layout.addWidget(self.superpoint_graph_algorithm_selector)
        graph_layout.addWidget(self.k_value)

        graph_groupbox_layout.addLayout(graph_layout)

        # Which point-cloud columns feed the graph/cluster distance
        graph_groupbox_layout.addWidget(QLabel("Clustering Dimensions (min 1):", self))
        self.cluster_dims_list = QListWidget(self)
        self.cluster_dims_list.setMaximumHeight(120)
        self.cluster_dims_list.itemChanged.connect(self.on_cluster_dims_changed)
        self.cluster_dims_list.itemChanged.connect(self._sync_dim_row)
        graph_groupbox_layout.addWidget(self.cluster_dims_list)
        graph_groupbox.setLayout(graph_groupbox_layout)
        layout.addWidget(graph_groupbox)

        """PIPELINE GROUPBOX"""
        pipeline_groupbox = QGroupBox("Pipeline")
        pipeline_groupbox_layout = QVBoxLayout()

        # Manual mode: selector changes only store settings; the buttons below
        # run each pipeline stage explicitly. Connect before setChecked so the
        # initial state applies to the buttons right away.
        self.manual_pipeline_checkbox = QCheckBox(
            "Manual Pipeline (run stages via buttons)", self
        )
        self.manual_pipeline_checkbox.toggled.connect(self.toggle_manual_pipeline)
        self.manual_pipeline_checkbox.setChecked(
            settings.current_settings["manual_pipeline"]
        )
        pipeline_groupbox_layout.addWidget(self.manual_pipeline_checkbox)

        pipeline_buttons_layout = QHBoxLayout()
        self.btn_load_file = QPushButton("Load File", self)
        self.btn_load_file.clicked.connect(self.run_load_file)
        self.btn_downsample = QPushButton("Downsample", self)
        self.btn_downsample.clicked.connect(self.run_downsampling)
        self.btn_build_graph = QPushButton("Build Graph", self)
        self.btn_build_graph.clicked.connect(self.run_build_graph)
        self.btn_cluster = QPushButton("Cluster", self)
        self.btn_cluster.clicked.connect(self.run_cluster)
        for b in (
            self.btn_load_file,
            self.btn_downsample,
            self.btn_build_graph,
            self.btn_cluster,
        ):
            b.setEnabled(settings.current_settings["manual_pipeline"])
            pipeline_buttons_layout.addWidget(b)
        pipeline_groupbox_layout.addLayout(pipeline_buttons_layout)

        pipeline_groupbox.setLayout(pipeline_groupbox_layout)
        layout.addWidget(pipeline_groupbox)

        """CLUSTERING AND LABELING GROUPBOX"""
        clustering_groupbox = QGroupBox("Clustering and Labeling")
        clustering_groupbox_layout = QVBoxLayout()

        """ Dropdown for community detection algorithm selection """
        self.community_detection_selector = QComboBox(self)
        self.community_detection_selector.addItem(
            "Label Propagation", "label_propagation"
        )
        self.community_detection_selector.addItem("Async LPA", "async_lpa")
        self.community_detection_selector.addItem("Louvain", "louvain")
        # self.community_detection_selector.addItem("Girvan-Newman", "girvan_newman") # Does not work
        self.community_detection_selector.addItem("Modularity", "modularity")
        self.community_detection_selector.addItem("Bipartitions", "kernighan_lin")

        # Load the current selection from settings
        current_cd = settings.current_settings["Community_detection"]
        print("Current cd found from settings: ", current_cd)
        index = self.community_detection_selector.findData(current_cd)
        if index >= 0:
            self.community_detection_selector.setCurrentIndex(index)

        self.community_detection_selector.currentIndexChanged.connect(
            self.on_community_detection_changed
        )

        clustering_groupbox_layout.addWidget(self.community_detection_selector)

        # Cluster navigation layout
        cluster_nav_layout = QHBoxLayout()
        self.btn_prev_cluster = QPushButton("<", self)
        self.btn_prev_cluster.clicked.connect(self.prev_cluster)
        cluster_nav_layout.addWidget(self.btn_prev_cluster)

        self.cluster_label = QLabel("Cluster 0 of 0", self)
        cluster_nav_layout.addWidget(self.cluster_label)

        self.btn_next_cluster = QPushButton(">", self)
        self.btn_next_cluster.clicked.connect(self.next_cluster)
        cluster_nav_layout.addWidget(self.btn_next_cluster)

        clustering_groupbox_layout.addLayout(cluster_nav_layout)

        """ Class Selection """
        # Textbox for label entry
        self.label_entry_textbox = QLineEdit(self)
        self.label_entry_textbox.setPlaceholderText("Enter label and press Enter")
        self.label_entry_textbox.returnPressed.connect(self.save_label_and_next)

        # Layout for label entry
        label_entry_layout = QHBoxLayout()
        label_entry_layout.addWidget(QLabel("Label:"))
        label_entry_layout.addWidget(self.label_entry_textbox)

        clustering_groupbox_layout.addLayout(label_entry_layout)

        clustering_groupbox.setLayout(clustering_groupbox_layout)
        layout.addWidget(clustering_groupbox)

        """ADDITIONAL CONTROLS GROUPBOX """
        additional_controls_groupbox = QGroupBox("Additional Controls")
        additional_controls_groupbox_layout = QVBoxLayout()

        """ Checkboxes for view options """
        view_layout = QHBoxLayout()
        self.checkbox_full_view = QCheckBox("Show Full View", self)
        self.checkbox_full_view.setChecked(settings.current_settings["show_full_view"])
        self.checkbox_full_view.toggled.connect(self.toggle_full_view)

        self.checkbox_cluster_view = QCheckBox("Show Cluster View", self)
        self.checkbox_cluster_view.setChecked(
            settings.current_settings["show_cluster_view"]
        )
        self.checkbox_cluster_view.toggled.connect(self.toggle_cluster_view)

        # Button for deleting save data and reloading
        self.btn_delete_save = QPushButton(self)
        self.btn_delete_save.setText("Reset")
        self.btn_delete_save.setIcon(
            self.style().standardIcon(QStyle.SP_DialogCancelButton)
        )  # Set path to your icon
        self.btn_delete_save.setToolTip("Delete Current Annotation Data")
        self.btn_delete_save.clicked.connect(self.confirm_deletion)

        view_layout.addWidget(self.btn_delete_save)

        view_layout.addWidget(self.checkbox_full_view)
        view_layout.addWidget(self.checkbox_cluster_view)
        additional_controls_groupbox_layout.addLayout(view_layout)

        additional_controls_groupbox.setLayout(additional_controls_groupbox_layout)
        layout.addWidget(additional_controls_groupbox)

        self.setLayout(layout)

    def select_dest_folder(self):
        self.dest_folder_path = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder"
        )
        if self.dest_folder_path:
            self.dest_folder_path_display.setText(self.dest_folder_path)
            self.pointcloud_view.set_destination_folder(self.dest_folder_path)
            self.btn_export_labels.setEnabled(True)
        self.update_cluster_label()

    def export_labels(self):
        if not self.pointcloud_files or not self.dest_folder_path:
            QMessageBox.warning(
                self,
                "Warning",
                "Please select a point cloud folder and a destination folder first.",
            )
            return
        filename = self.pointcloud_files[self.current_file_index]
        src_path = os.path.join(self.folder_path_display.text(), filename)
        if not os.path.exists(src_path):
            QMessageBox.critical(self, "Export Failed", f"File not found: {src_path}")
            return
        json_path = os.path.join(
            self.dest_folder_path, os.path.basename(filename) + ".json"
        )
        # Persist current in-memory labels before reading the JSON back
        self.pointcloud_view.current_file = src_path
        self.pointcloud_view.save_labels_to_file()
        dim_name = self.export_header_name.text().strip() or DEFAULT_DIM_NAME
        try:
            stats = label_export.export_labels_to_original(
                src_path, json_path, self.dest_folder_path, dim_name
            )
        except label_export.LabelExportError as e:
            QMessageBox.critical(self, "Export Failed", str(e))
            return
        lines = [
            f"Wrote {stats['out_path']}",
            f"Points: {stats['n_points']}  (extra dim '{stats['dim_name']}', int8)",
        ]
        for label_id in sorted(stats["label_counts"]):
            count = stats["label_counts"][label_id]
            if label_id == -1:
                lines.append(f"  unlabeled: {count} points")
            else:
                name = next(
                    (k for k, v in stats["label_map"].items() if v == label_id),
                    str(label_id),
                )
                lines.append(f'  "{name}" (id {label_id}): {count} points')
        QMessageBox.information(self, "Export Complete", "\n".join(lines))

    def save_label_and_next(self):
        # Check if a destination folder is set
        if not self.pointcloud_view.destination_folder:
            QMessageBox.warning(
                self, "Warning", "Please select a destination folder for the labels."
            )
            return

        # Proceed with label saving
        label = self.label_entry_textbox.text().strip()
        if label:
            self.pointcloud_view.label_current_cluster(label)
            # next_cluster -> update_cluster_label repopulates the entry box
            # with the newly current cluster's label (or clears it)
            self.pointcloud_view.next_cluster()

            # Save the current state of labels to a file
            self.pointcloud_view.save_labels_to_file()

    def select_folder(self):
        self.btn_select_dest_folder.setEnabled(
            True
        )  # Enable the destination folder selection button
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.show_loading_indicator()
            QApplication.processEvents()  # Process any pending GUI events
            self.folder_path_display.setText(folder_path)
            self.load_pointcloud_files(folder_path)
            self.pointcloud_view.current_folder_path = folder_path
            self.display_current_file()

    def load_pointcloud_files(self, folder_path):
        # Load all point cloud files (.las, .laz and .txt) from the folder
        self.pointcloud_files = [
            f for f in os.listdir(folder_path) if f.endswith((".las", ".laz", ".txt"))
        ]
        self.current_file_index = 0

    def display_current_file(self):
        print("Running function: display_current_file")
        self.pointcloud_view.cluster_labels = {}
        self.pointcloud_view.label_map = {}  # label ids are per-file
        if self.pointcloud_files and self.pointcloud_view.current_folder_path:
            filename = self.pointcloud_files[self.current_file_index]
            self.file_name_display.setText(filename)
            full_path = os.path.join(self.pointcloud_view.current_folder_path, filename)
            self.pointcloud_view.current_file = full_path
            if settings.current_settings["manual_pipeline"]:
                # Manual mode: selecting a file only loads it (no compute);
                # the pipeline buttons run the stages.
                try:
                    self.pointcloud_view.load_file(full_path)
                except Exception as e:
                    print(f"Error loading point cloud: {e}")
                self.pointcloud_view.render_pointcloud(
                    show_full_view=self.checkbox_full_view.isChecked(),
                    show_cluster_view=self.checkbox_cluster_view.isChecked(),
                )
            else:
                self.pointcloud_view.load_and_display_pointcloud(
                    full_path,
                    downsampling_method=self.downsampling_algorithm_selector.currentData(),
                    target_count=self.target_point_count.value(),
                    show_full_view=self.checkbox_full_view.isChecked(),
                    show_cluster_view=self.checkbox_cluster_view.isChecked(),
                    superpoint_graph_method=self.superpoint_graph_algorithm_selector.currentData(),
                    k_value=self.k_value.value(),
                    clustering_algorithm=self.community_detection_selector.currentData(),
                )
        else:
            self.file_name_display.setText("No point cloud files found")
            print("No files to display or folder path is not set.")
        self.populate_cluster_dims()

    def populate_cluster_dims(self):
        """Fill the clustering-dimensions list from the loaded file's columns,
        applying the persisted cluster_dims selection (default x, y, z) and
        per-dimension weights (default 1.0). Dim names are matched
        case-insensitively so saved "X", "Y", "Z" map onto x, y, z."""
        names = self.pointcloud_view.feature_names or ["x", "y", "z"]
        saved = settings.current_settings.get("cluster_dims", ["x", "y", "z"])
        lower = {n.lower(): n for n in names}
        selected = []
        for d in saved:
            canon = d if d in names else lower.get(str(d).lower())
            if canon is not None and canon not in selected:
                selected.append(canon)
        if not selected:
            selected = [n for n in ("x", "y", "z") if n in names] or names[:3]
        weights = settings.current_settings.get("cluster_dim_weights", {})
        self.cluster_dims_list.blockSignals(True)
        self.cluster_dims_list.clear()
        for n in names:
            item = QListWidgetItem(n)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if n in selected else Qt.Unchecked)
            self.cluster_dims_list.addItem(item)
            # The item widget covers the item's own text, so the row widget
            # carries the dimension name itself (as the checkbox label).
            row = QWidget(self.cluster_dims_list)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(4, 0, 4, 0)
            cb = QCheckBox(n, row)
            cb.setChecked(item.checkState() == Qt.Checked)
            spin = QDoubleSpinBox(row)
            spin.setRange(0.0, 1000.0)  # weights are clamped at 0, no negatives
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            spin.blockSignals(True)
            spin.setValue(max(0.0, float(weights.get(n, 1.0))))
            spin.blockSignals(False)
            spin.valueChanged.connect(
                lambda value, name=n: self.on_cluster_dim_weight_changed(name, value)
            )
            cb.toggled.connect(
                lambda checked, it=item: it.setCheckState(
                    Qt.Checked if checked else Qt.Unchecked
                )
            )
            row_layout.addWidget(cb)
            row_layout.addStretch(1)
            row_layout.addWidget(spin)
            self.cluster_dims_list.setItemWidget(item, row)
        self.cluster_dims_list.blockSignals(False)
        self.pointcloud_view.cluster_dims = self.selected_cluster_dims()
        self.pointcloud_view.cluster_dim_weights = dict(weights)

    def selected_cluster_dims(self):
        return [
            self.cluster_dims_list.item(i).text()
            for i in range(self.cluster_dims_list.count())
            if self.cluster_dims_list.item(i).checkState() == Qt.Checked
        ]

    def _sync_dim_row(self, item):
        """Reflect an item's check state in its row widget's QCheckBox
        (signals blocked, so this never re-triggers itemChanged)."""
        row = self.cluster_dims_list.itemWidget(item)
        cb = row.findChild(QCheckBox) if row is not None else None
        if cb is not None:
            cb.blockSignals(True)
            cb.setChecked(item.checkState() == Qt.Checked)
            cb.blockSignals(False)

    def toggle_manual_pipeline(self, checked):
        settings.current_settings["manual_pipeline"] = checked
        for b in (
            self.btn_load_file,
            self.btn_downsample,
            self.btn_build_graph,
            self.btn_cluster,
        ):
            b.setEnabled(checked)

    def render_current(self):
        self.pointcloud_view.render_pointcloud(
            show_full_view=self.checkbox_full_view.isChecked(),
            show_cluster_view=self.checkbox_cluster_view.isChecked(),
        )

    def _confirm_stage_rerun(self):
        """Re-running a pipeline stage invalidates saved annotations; ask only
        when an annotation file actually exists for the current file."""
        if not self.dest_folder_path or not self.pointcloud_files:
            return True
        filename = self.pointcloud_files[self.current_file_index]
        label_file_path = os.path.join(
            self.dest_folder_path, os.path.basename(filename) + ".json"
        )
        if not os.path.exists(label_file_path):
            return True
        reply = QMessageBox.question(
            self,
            "Confirmation",
            "Re-running this stage invalidates the saved annotation data for this file. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            os.remove(label_file_path)
            self.pointcloud_view.cluster_labels = {}
            self.pointcloud_view.label_map = {}
            return True
        return False

    def run_load_file(self):
        view = self.pointcloud_view
        if not view.current_file:
            return
        try:
            view.load_file(view.current_file)
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return
        self.populate_cluster_dims()
        self.render_current()

    def run_downsampling(self):
        if not self._confirm_stage_rerun():
            return
        self.pointcloud_view.apply_downsampling(
            self.downsampling_algorithm_selector.currentData(),
            self.target_point_count.value(),
        )
        self.render_current()

    def run_build_graph(self):
        if not self._confirm_stage_rerun():
            return
        self.pointcloud_view.build_graph(
            self.superpoint_graph_algorithm_selector.currentData(),
            self.k_value.value(),
        )
        self.render_current()

    def run_cluster(self):
        if not self._confirm_stage_rerun():
            return
        self.pointcloud_view.apply_clustering(
            self.community_detection_selector.currentData()
        )
        self.render_current()
        self.update_cluster_label()

    def _rerun_pipeline(self):
        """Re-run the pipeline on the current file with the current selector
        settings. No-op in manual mode (the stage buttons run the stages)."""
        if settings.current_settings["manual_pipeline"]:
            return
        if self.pointcloud_view.current_folder_path:
            self.show_loading_indicator()
            self.pointcloud_view.load_and_display_pointcloud(
                self.pointcloud_view.current_folder_path,
                downsampling_method=self.downsampling_algorithm_selector.currentData(),
                target_count=self.target_point_count.value(),
                show_full_view=self.checkbox_full_view.isChecked(),
                show_cluster_view=self.checkbox_cluster_view.isChecked(),
                superpoint_graph_method=self.superpoint_graph_algorithm_selector.currentData(),
                k_value=self.k_value.value(),
                clustering_algorithm=self.community_detection_selector.currentData(),
            )

    def on_cluster_dim_weight_changed(self, name, value):
        weight = max(0.0, float(value))  # clamp at 0: no negative weights
        weights = dict(settings.current_settings.get("cluster_dim_weights", {}))
        weights[name] = weight
        settings.current_settings["cluster_dim_weights"] = weights
        self.pointcloud_view.cluster_dim_weights = weights
        self._rerun_pipeline()

    def on_cluster_dims_changed(self, item):
        selected = self.selected_cluster_dims()
        if not selected:
            # At least one dimension must stay selected: revert this toggle
            self.cluster_dims_list.blockSignals(True)
            item.setCheckState(Qt.Checked)
            self.cluster_dims_list.blockSignals(False)
            self._sync_dim_row(item)
            return
        settings.current_settings["cluster_dims"] = selected
        self.pointcloud_view.cluster_dims = selected
        self._rerun_pipeline()

    def show_loading_indicator(self):
        self.file_name_display.setText("Loading...")

    def prev_file(self):
        if self.pointcloud_files:
            self.show_loading_indicator()
            QApplication.processEvents()  # Process any pending GUI events
            self.current_file_index = (self.current_file_index - 1) % len(
                self.pointcloud_files
            )

            filename = self.pointcloud_files[self.current_file_index]
            self.file_name_display.setText(filename)
            self.pointcloud_view.current_file = filename

            if self.dest_folder_path:
                # check if there is a destination file path for current file
                label_file_path = os.path.join(
                    self.dest_folder_path, os.path.basename(filename) + ".json"
                )
                print("Looking for save data at filepath: ", label_file_path + ".json")

                if not os.path.exists(label_file_path):
                    print("Did not find save data, creating new partition")
                    self.display_current_file()
                else:
                    print("Found save data, loading partition")
                self.pointcloud_view.set_destination_folder(self.dest_folder_path)
            else:
                self.display_current_file()
            self.pointcloud_view.render_pointcloud(
                show_full_view=self.checkbox_full_view.isChecked(),
                show_cluster_view=self.checkbox_cluster_view.isChecked(),
            )
            self.update_cluster_label()

    def next_file(self):
        if self.pointcloud_files:
            self.show_loading_indicator()
            QApplication.processEvents()  # Process any pending GUI events
            self.current_file_index = (self.current_file_index + 1) % len(
                self.pointcloud_files
            )

            filename = self.pointcloud_files[self.current_file_index]
            self.file_name_display.setText(filename)
            self.pointcloud_view.current_file = filename

            if self.dest_folder_path:
                # check if there is a destination file path for current file
                label_file_path = os.path.join(
                    self.dest_folder_path, os.path.basename(filename) + ".json"
                )
                print("Looking for save data at filepath: ", label_file_path + ".json")

                if not os.path.exists(label_file_path):
                    print("Did not find save data, creating new partition")
                    self.display_current_file()
                else:
                    print("Found save data, loading partition")

                self.pointcloud_view.set_destination_folder(self.dest_folder_path)
            else:
                self.display_current_file()
            self.pointcloud_view.render_pointcloud(
                show_full_view=self.checkbox_full_view.isChecked(),
                show_cluster_view=self.checkbox_cluster_view.isChecked(),
            )
            self.update_cluster_label()

    def change_file(self, event):
        # Logic to change file on click
        pass

    def toggle_night_mode(self):
        is_night_mode = self.night_mode_checkbox.isChecked()
        settings.current_settings["night_mode"] = is_night_mode
        self.pointcloud_view.set_night_mode(is_night_mode)

    def on_point_size_changed(self, value):
        settings.current_settings["point_size"] = value
        if (
            self.pointcloud_view.current_folder_path
        ):  # Check if a point cloud folder is loaded
            self.pointcloud_view.render_pointcloud(
                show_full_view=self.checkbox_full_view.isChecked(),
                show_cluster_view=self.checkbox_cluster_view.isChecked(),
            )

    def on_cluster_point_size_changed(self, value):
        settings.current_settings["cluster_point_size"] = value
        if (
            self.pointcloud_view.current_folder_path
        ):  # Check if a point cloud folder is loaded
            self.pointcloud_view.render_pointcloud(
                show_full_view=self.checkbox_full_view.isChecked(),
                show_cluster_view=self.checkbox_cluster_view.isChecked(),
            )

    def on_downsampling_changed(self):
        print("Running function: on_downsampling_changed")
        # Update the settings with the new selection
        current_selection = self.downsampling_algorithm_selector.currentData()
        settings.current_settings["Subsampling"] = current_selection
        current_subsample_size = int(self.target_point_count.value())
        settings.current_settings["Subsample_size"] = current_subsample_size
        if settings.current_settings["manual_pipeline"]:
            return  # manual mode: the Downsample button runs this stage
        self.show_loading_indicator()  # This may take a while depending on the size of the point cloud
        if self.confirm_deletion():
            if self.pointcloud_view.current_folder_path:
                downsampling_method = self.downsampling_algorithm_selector.currentData()
                target_count = self.target_point_count.value()
                self.pointcloud_view.load_and_display_pointcloud(
                    self.pointcloud_view.current_folder_path,
                    downsampling_method,
                    target_count,
                    show_full_view=self.checkbox_full_view.isChecked(),
                    show_cluster_view=self.checkbox_cluster_view.isChecked(),
                    superpoint_graph_method=self.superpoint_graph_algorithm_selector.currentData(),
                    k_value=self.k_value.value(),
                    clustering_algorithm=self.community_detection_selector.currentData(),
                )
        try:
            filename = self.pointcloud_files[self.current_file_index]
            self.file_name_display.setText(filename)
        except IndexError:
            print("No files to display changes for")

    def on_community_detection_changed(self):
        print("Running function: on_community_detection_changed")
        # Update the settings with the new selection
        current_selection = self.community_detection_selector.currentData()
        settings.current_settings["Community_detection"] = current_selection
        if settings.current_settings["manual_pipeline"]:
            return  # manual mode: the Cluster button runs this stage
        self.show_loading_indicator()  # This may take a while depending on the size of the point cloud
        if self.confirm_deletion():
            if self.pointcloud_view.current_folder_path:
                self.pointcloud_view.load_and_display_pointcloud(
                    self.pointcloud_view.current_folder_path,
                    downsampling_method=self.downsampling_algorithm_selector.currentData(),
                    target_count=self.target_point_count.value(),
                    show_full_view=self.checkbox_full_view.isChecked(),
                    show_cluster_view=self.checkbox_cluster_view.isChecked(),
                    superpoint_graph_method=self.superpoint_graph_algorithm_selector.currentData(),
                    k_value=self.k_value.value(),
                    clustering_algorithm=self.community_detection_selector.currentData(),
                )
        try:
            filename = self.pointcloud_files[self.current_file_index]
            self.file_name_display.setText(filename)
        except IndexError:
            print("No files to display changes for")

    def create_superpoint_graph(self):
        # Update the settings with the new selection
        current_selection = self.superpoint_graph_algorithm_selector.currentData()
        settings.current_settings["superpoint_graph"] = current_selection
        settings.current_settings["KNN_graph"] = self.k_value.value()

        index = self.superpoint_graph_algorithm_selector.findData(current_selection)
        print("index on spg: ", index)
        print("is k_value enabled: ", self.k_value.isEnabled())
        if index != 0:
            self.k_value.setEnabled(False)
        if index == 0 and self.k_value.isEnabled() == False:
            self.k_value.setEnabled(True)
            try:
                filename = self.pointcloud_files[self.current_file_index]
                self.file_name_display.setText(filename)
            except IndexError:
                print("No files to display changes for")
            return  # allow user to select k value

        if settings.current_settings["manual_pipeline"]:
            return  # manual mode: the Build Graph button runs this stage
        print("Running function: create_superpoint_graph")
        self.show_loading_indicator()  # This may take a while depending on the size of the point cloud
        if self.confirm_deletion():
            if self.pointcloud_view.current_folder_path:
                self.pointcloud_view.load_and_display_pointcloud(
                    self.pointcloud_view.current_folder_path,
                    downsampling_method=self.downsampling_algorithm_selector.currentData(),
                    target_count=self.target_point_count.value(),
                    show_full_view=self.checkbox_full_view.isChecked(),
                    show_cluster_view=self.checkbox_cluster_view.isChecked(),
                    superpoint_graph_method=self.superpoint_graph_algorithm_selector.currentData(),
                    k_value=self.k_value.value(),
                    clustering_algorithm=self.community_detection_selector.currentData(),
                )

        try:
            filename = self.pointcloud_files[self.current_file_index]
            self.file_name_display.setText(filename)
        except IndexError:
            print("No files to display changes for")

    def toggle_full_view(self, checked):
        settings.current_settings["show_full_view"] = checked
        # refresh point cloud view
        if not checked and self.pointcloud_view.fig_full is not None:
            self.pointcloud_view.close_figure("fig_full")
        self.pointcloud_view.render_pointcloud(
            show_full_view=self.checkbox_full_view.isChecked(),
            show_cluster_view=self.checkbox_cluster_view.isChecked(),
        )

    def toggle_cluster_view(self, checked):
        settings.current_settings["show_cluster_view"] = checked
        # refresh point cloud view
        if not checked and self.pointcloud_view.fig_zoom is not None:
            self.pointcloud_view.close_figure("fig_zoom")
        self.pointcloud_view.render_pointcloud(
            show_full_view=self.checkbox_full_view.isChecked(),
            show_cluster_view=self.checkbox_cluster_view.isChecked(),
        )

    def toggle_eco_mode(self):
        is_eco_mode = self.eco_mode_checkbox.isChecked()
        settings.current_settings["eco_mode"] = is_eco_mode
        self.point_size_selector.setEnabled(not is_eco_mode)
        self.cluster_point_size_selector.setEnabled(not is_eco_mode)
        if self.pointcloud_view.current_folder_path:
            # Refresh the point cloud view with the new eco mode setting
            self.pointcloud_view.render_pointcloud(
                show_full_view=self.checkbox_full_view.isChecked(),
                show_cluster_view=self.checkbox_cluster_view.isChecked(),
            )

    def prev_cluster(self):
        if self.pointcloud_view.clusters:
            self.pointcloud_view.current_cluster_index = (
                self.pointcloud_view.current_cluster_index - 1
            ) % len(self.pointcloud_view.clusters)
            self.pointcloud_view.render_pointcloud(
                show_full_view=self.checkbox_full_view.isChecked(),
                show_cluster_view=self.checkbox_cluster_view.isChecked(),
            )
            self.update_cluster_label()

    def next_cluster(self):
        if self.pointcloud_view.clusters:
            self.pointcloud_view.current_cluster_index = (
                self.pointcloud_view.current_cluster_index + 1
            ) % len(self.pointcloud_view.clusters)
            self.pointcloud_view.render_pointcloud(
                show_full_view=self.checkbox_full_view.isChecked(),
                show_cluster_view=self.checkbox_cluster_view.isChecked(),
            )
            self.update_cluster_label()

    def update_cluster_label(self):
        if self.pointcloud_view.clusters:
            total_clusters = len(self.pointcloud_view.clusters)
            current_index = (
                self.pointcloud_view.current_cluster_index + 1
            )  # +1 for human-readable indexing
            self.cluster_label.setText(f"Cluster {current_index} of {total_clusters}")
            # Show the current cluster's saved label in the entry box (for
            # easy reference when revisiting a coded cluster); clear it when
            # the cluster is unlabeled.
            label = self.pointcloud_view.cluster_labels.get(
                self.pointcloud_view.current_cluster_index, -1
            )
            self.label_entry_textbox.setText(
                "" if label is None or str(label) == "-1" else str(label)
            )
        else:
            self.cluster_label.setText("No clusters")
            self.label_entry_textbox.clear()

    def confirm_deletion(self):
        # Check if there is an active file, if not return False
        # check if pointcloud view is an empty list
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setIcon(QMessageBox.Warning)
        if not self.pointcloud_files:
            msg.setText("Please select a PointCloud Folder and try again")
            msg.exec_()
            print("No active file to delete save data for")
            return False
        filename = self.pointcloud_files[self.current_file_index]

        label_file_path = None
        if self.dest_folder_path:
            # check if there is a destination file path for current file
            label_file_path = os.path.join(
                self.dest_folder_path, os.path.basename(filename) + ".json"
            )
        else:
            msg.setText("Please select a Destination Folder and try again")
            msg.exec_()
            print("No destination folder set")
            return False
        # print("Looking for save data at filepath: ", label_file_path + ".json")

        # If there is no destination folder, or if there is no save data, return False
        # to indicate that deletion is not possible
        if not os.path.exists(label_file_path):
            print("No save data found")
            return False

        reply = QMessageBox.question(
            self,
            "Confirmation",
            "Proceeding will delete the current annotation save data. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.delete_save_and_reload()
            return True
        return False

    def delete_save_and_reload(self):
        filename = self.pointcloud_files[self.current_file_index]

        # check if there is a destination file path for current file
        label_file_path = os.path.join(
            self.dest_folder_path, os.path.basename(filename) + ".json"
        )
        print("Looking for save data at filepath: ", label_file_path + ".json")
        # delete the save data
        os.remove(label_file_path)
        self.display_current_file()
        if self.dest_folder_path:
            self.dest_folder_path_display.setText(self.dest_folder_path)
            self.pointcloud_view.set_destination_folder(self.dest_folder_path)
