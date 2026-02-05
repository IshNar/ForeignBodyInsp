import cv2
import numpy as np
import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QCheckBox, QFileDialog,
                             QListWidget, QGroupBox, QSplitter, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, QSize, QEvent, QPointF
from PyQt6.QtGui import QImage, QPixmap

from src.core.detection import ForeignBodyDetector
from src.core.classification import ParticleClassifier
from src.hardware.basler_camera import BaslerCamera
from src.hardware.file_camera import FileCamera

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vial Foreign Body Inspection")
        self.resize(1200, 800)

        self.camera = None
        self.detector = ForeignBodyDetector()
        self.classifier = ParticleClassifier()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_inspecting = False
        self.current_contours = []
        self.current_frame_bgr = None           # Original frame for defect ROI extraction
        self.original_frame_full = None         # Full resolution original (no processing)
        self.current_display_frame = None       # full-res with overlays
        self.display_base_frame = None          # base frame with all contours (no selection highlight)
        self.selected_contour_idx = -1          # Currently selected contour index
        self.zoom_factor = 1.0
        self.zoom_min = 0.2
        self.zoom_max = 6.0
        self.view_cx = None
        self.view_cy = None
        self.is_panning = False
        self.pan_last_pos = QPointF(0, 0)
        self.pan_start_pos = QPointF(0, 0)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.central_widget = central_widget
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Side: Image Display
        self.image_label = QLabel("Camera Feed / Image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        self.image_label.setScaledContents(False)  # keep pixmap size; use scroll for large
        self.image_label.setMinimumSize(400, 300)

        # Scroll area to prevent window resizing with large images
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Capture wheel events for zoom control
        self.image_scroll.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)
        # Accept drag/drop on main and child widgets
        self.setAcceptDrops(True)
        central_widget.setAcceptDrops(True)
        self.image_scroll.setAcceptDrops(True)
        self.image_scroll.viewport().setAcceptDrops(True)
        self.image_label.setAcceptDrops(True)
        # Also listen for drag events on these widgets
        central_widget.installEventFilter(self)
        self.image_scroll.installEventFilter(self)

        # Right Side: Controls
        control_panel = QWidget()
        self.control_panel = control_panel
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)
        control_panel.setAcceptDrops(True)
        control_panel.installEventFilter(self)

        # Source Selection
        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout()
        self.btn_connect_cam = QPushButton("Connect Basler Camera")
        self.btn_connect_cam.clicked.connect(self.connect_camera)
        self.btn_load_file = QPushButton("Load Image File")
        self.btn_load_file.clicked.connect(self.load_file)
        source_layout.addWidget(self.btn_connect_cam)
        source_layout.addWidget(self.btn_load_file)
        source_group.setLayout(source_layout)
        control_layout.addWidget(source_group)

        # Inspection Controls
        insp_group = QGroupBox("Inspection")
        insp_layout = QVBoxLayout()
        self.btn_start = QPushButton("Start Inspection")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self.toggle_inspection)
        insp_layout.addWidget(self.btn_start)
        insp_group.setLayout(insp_layout)
        control_layout.addWidget(insp_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        # Threshold with value label
        threshold_label_layout = QHBoxLayout()
        threshold_label_layout.addWidget(QLabel("Threshold:"))
        self.lbl_thresh_value = QLabel("100")
        self.lbl_thresh_value.setStyleSheet("font-weight: bold; color: blue;")
        threshold_label_layout.addStretch()
        threshold_label_layout.addWidget(self.lbl_thresh_value)
        settings_layout.addLayout(threshold_label_layout)
        
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(100)
        self.slider_thresh.valueChanged.connect(lambda v: self.lbl_thresh_value.setText(str(v)))
        settings_layout.addWidget(self.slider_thresh)

        # Min Area with value label
        area_label_layout = QHBoxLayout()
        area_label_layout.addWidget(QLabel("Min Area:"))
        self.lbl_area_value = QLabel("10")
        self.lbl_area_value.setStyleSheet("font-weight: bold; color: blue;")
        area_label_layout.addStretch()
        area_label_layout.addWidget(self.lbl_area_value)
        settings_layout.addLayout(area_label_layout)
        
        self.slider_area = QSlider(Qt.Orientation.Horizontal)
        self.slider_area.setRange(0, 500)
        self.slider_area.setValue(10)
        self.slider_area.valueChanged.connect(lambda v: self.lbl_area_value.setText(str(v)))
        settings_layout.addWidget(self.slider_area)

        self.chk_adaptive = QCheckBox("Adaptive Threshold")
        self.chk_adaptive.setChecked(True)
        settings_layout.addWidget(self.chk_adaptive)

        settings_group.setLayout(settings_layout)
        control_layout.addWidget(settings_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.lbl_status = QLabel("Status: WAIT")
        self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: gray;")
        results_layout.addWidget(self.lbl_status)

        self.list_results = QListWidget()
        self.list_results.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_results.setStyleSheet(
            "QListWidget::item:selected { background-color: #0078d7; color: white; }"
        )
        self.list_results.itemSelectionChanged.connect(self.on_selection_changed)
        results_layout.addWidget(self.list_results)
        results_group.setLayout(results_layout)
        control_layout.addWidget(results_group)

        # Defect View
        defect_group = QGroupBox("Defect View")
        defect_layout = QVBoxLayout()
        self.lbl_defect_view = QLabel("Select a defect")
        self.lbl_defect_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_defect_view.setStyleSheet("background-color: black; color: white;")
        self.lbl_defect_view.setFixedSize(260, 260)
        defect_layout.addWidget(self.lbl_defect_view)
        defect_group.setLayout(defect_layout)
        control_layout.addWidget(defect_group)

        control_layout.addStretch()

        # Splitter to allow resizing
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter
        splitter.addWidget(self.image_scroll)
        splitter.addWidget(control_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setAcceptDrops(True)
        splitter.installEventFilter(self)

        main_layout.addWidget(splitter)

    def connect_camera(self):
        if self.camera:
            self.camera.close()

        # Reset view state for new camera
        self.view_cx = None
        self.view_cy = None
        self.zoom_factor = 1.0

        self.camera = BaslerCamera()
        if self.camera.open():
            self.lbl_status.setText("Camera Connected")
            self.timer.start(33) # ~30 FPS
        else:
            self.lbl_status.setText("Camera Error")
            self.camera = None

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            self.load_image_path(fname)

    def load_image_path(self, path: str):
        if not os.path.isfile(path):
            self.lbl_status.setText("File Error")
            return

        if self.camera:
            self.camera.close()
            self.timer.stop()

        # Reset view state for new image
        self.view_cx = None
        self.view_cy = None
        self.zoom_factor = 1.0

        self.camera = FileCamera(path)
        if self.camera.open():
            self.lbl_status.setText("File Loaded")
            # Trigger one update to show the image
            self.update_frame()
            self.timer.start(100) # Slower update for static file
        else:
            self.lbl_status.setText("File Error")

    def toggle_inspection(self):
        if self.btn_start.isChecked():
            self.is_inspecting = True
            self.btn_start.setText("Stop Inspection")
        else:
            self.is_inspecting = False
            self.btn_start.setText("Start Inspection")
            self.lbl_status.setText("Stopped")

    def update_frame(self):
        # TODO: Move image acquisition and processing to a separate QThread
        # to prevent UI freezing with high-resolution (12MP) images.
        if self.camera:
            frame = self.camera.grab_frame()
            if frame is not None:
                # Store original full-res frame for defect ROI extraction
                if len(frame.shape) == 2:
                    self.original_frame_full = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    frame_bgr = self.original_frame_full
                else:
                    self.original_frame_full = frame.copy()
                    frame_bgr = self.original_frame_full

                # Process if inspection is on
                if self.is_inspecting:
                    contours, processed_img = self.detector.detect_static(
                        frame,
                        threshold=self.slider_thresh.value(),
                        min_area=self.slider_area.value(),
                        use_adaptive=self.chk_adaptive.isChecked()
                    )

                    # Draw results
                    display_frame = frame_bgr.copy()

                    cv2.drawContours(display_frame, contours, -1, (0, 0, 255), 2)

                    # Remember current selection
                    prev_selected = self.list_results.currentRow()
                    
                    self.current_contours = contours
                    # Use original full-res frame for defect view (not display_frame)
                    self.current_frame_bgr = self.original_frame_full.copy()

                    if contours:
                        self.lbl_status.setText("NG: Foreign Body")
                        self.lbl_status.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")

                        # Block signals to prevent selection changes during update
                        self.list_results.blockSignals(True)
                        
                        # Update list items (add/remove as needed)
                        current_count = self.list_results.count()
                        for i, cnt in enumerate(contours):
                            result = self.classifier.classify(cnt)
                            item_text = f"#{i+1}: {result['label']} (Area: {result['area']:.0f})"
                            
                            if i < current_count:
                                # Update existing item
                                self.list_results.item(i).setText(item_text)
                            else:
                                # Add new item
                                self.list_results.addItem(item_text)

                            # Draw label on image
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.putText(display_frame, f"{result['label']}", (x, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Remove extra items if list is now shorter
                        while self.list_results.count() > len(contours):
                            self.list_results.takeItem(self.list_results.count() - 1)
                        
                        self.list_results.blockSignals(False)
                        
                        # Save base frame (all contours in red, no selection highlight)
                        self.display_base_frame = display_frame.copy()
                        
                        # Restore previous selection if valid, otherwise select first item
                        if 0 <= prev_selected < len(contours):
                            self.selected_contour_idx = prev_selected
                            self.list_results.setCurrentRow(prev_selected)
                        else:
                            self.selected_contour_idx = 0
                            self.list_results.setCurrentRow(0)
                        
                        # Draw selection highlight
                        self.redraw_with_highlight()
                    else:
                        self.lbl_status.setText("OK")
                        self.lbl_status.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
                        self.current_contours = []
                        self.current_frame_bgr = None
                        self.list_results.clearSelection()
                        self.clear_defect_view()
                else:
                    display_frame = frame_bgr
                    self.current_contours = []
                    self.current_frame_bgr = None
                    self.list_results.clearSelection()
                    self.clear_defect_view()
                    # Keep full-res display frame for crisp zoom/crop rendering
                    self.current_display_frame = display_frame.copy()

                # Initialize view center only if not set yet
                if self.view_cx is None or self.view_cy is None:
                    if self.current_display_frame is not None:
                        h0, w0 = self.current_display_frame.shape[:2]
                        self.view_cx = w0 // 2
                        self.view_cy = h0 // 2
                self.render_main_view()

    def closeEvent(self, event):
        if self.camera:
            self.camera.close()
        event.accept()

    def on_selection_changed(self):
        """Handle Result list selection change: update highlight and defect view"""
        self.selected_contour_idx = self.list_results.currentRow()
        
        # Move view to center on selected defect
        if 0 <= self.selected_contour_idx < len(self.current_contours):
            cnt = self.current_contours[self.selected_contour_idx]
            # Get bounding rect center
            x, y, w, h = cv2.boundingRect(cnt)
            defect_cx = x + w // 2
            defect_cy = y + h // 2
            # Update view center to defect center
            self.view_cx = float(defect_cx)
            self.view_cy = float(defect_cy)
        
        self.redraw_with_highlight()
        self.update_defect_view()

    def redraw_with_highlight(self):
        """Redraw display frame with selected contour highlighted in yellow"""
        if self.display_base_frame is None or not self.current_contours:
            return
        
        # Start from base frame (all contours in red)
        display_frame = self.display_base_frame.copy()
        
        # Draw selected contour in yellow
        if 0 <= self.selected_contour_idx < len(self.current_contours):
            selected_cnt = self.current_contours[self.selected_contour_idx]
            cv2.drawContours(display_frame, [selected_cnt], -1, (0, 255, 255), 3)  # Yellow, thicker
        
        # Update display frame and render
        self.current_display_frame = display_frame
        self.render_main_view()

    def handle_image_click(self, view_pos):
        """Handle click on image view to select contour"""
        if not self.current_contours or self.current_display_frame is None:
            return
        
        # Convert view position to original image coordinates
        img_pos = self.view_to_image_coords(view_pos)
        if img_pos is None:
            return
        
        # Find which contour contains this point
        for i, cnt in enumerate(self.current_contours):
            dist = cv2.pointPolygonTest(cnt, img_pos, False)
            if dist >= 0:  # Point is inside or on the contour
                # Select this contour
                self.list_results.blockSignals(True)
                self.list_results.setCurrentRow(i)
                self.list_results.blockSignals(False)
                self.selected_contour_idx = i
                self.redraw_with_highlight()
                self.update_defect_view()
                break

    def view_to_image_coords(self, view_pos):
        """Convert viewport position to original image coordinates"""
        if self.current_display_frame is None:
            return None
        
        fh, fw, _ = self.current_display_frame.shape
        
        # Get current ROI being displayed
        zoom = max(self.zoom_factor, 1e-6)
        roi_w = min(fw, int(fw / zoom)) if zoom >= 1.0 else fw
        roi_h = min(fh, int(fh / zoom)) if zoom >= 1.0 else fh
        roi_w = max(1, roi_w)
        roi_h = max(1, roi_h)
        
        # Center
        if self.view_cx is None or self.view_cy is None:
            cx, cy = fw // 2, fh // 2
        else:
            cx = int(self.view_cx)
            cy = int(self.view_cy)
        
        half_w = roi_w // 2
        half_h = roi_h // 2
        cx = min(max(cx, half_w), max(fw - half_w, half_w))
        cy = min(max(cy, half_h), max(fh - half_h, half_h))
        
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        
        # Get viewport size
        target_size = self.image_scroll.viewport().size()
        target_w = max(1, target_size.width())
        target_h = max(1, target_size.height())
        
        # Convert view position to ROI coordinates
        scale_x = roi_w / target_w
        scale_y = roi_h / target_h
        
        img_x = x1 + view_pos.x() * scale_x
        img_y = y1 + view_pos.y() * scale_y
        
        return (float(img_x), float(img_y))

    def clear_defect_view(self):
        self.lbl_defect_view.setText("No selection")
        self.lbl_defect_view.setPixmap(QPixmap())

    def update_defect_view(self):
        if (
            not self.list_results.selectedIndexes()
            or not self.current_contours
            or self.current_frame_bgr is None
        ):
            self.clear_defect_view()
            return

        index = self.list_results.currentRow()
        if index < 0 or index >= len(self.current_contours):
            self.clear_defect_view()
            return

        cnt = self.current_contours[index]
        x, y, w, h = cv2.boundingRect(cnt)

        # Defect center and ROI size = defect size * 2 (with minimum 150x150)
        defect_cx = x + w // 2
        defect_cy = y + h // 2
        roi_w = max(w * 2, 150)  # Minimum 150 pixels
        roi_h = max(h * 2, 150)

        img_h, img_w, _ = self.current_frame_bgr.shape
        
        # Calculate ROI bounds centered on defect
        half_w = roi_w // 2
        half_h = roi_h // 2
        x1 = max(0, defect_cx - half_w)
        y1 = max(0, defect_cy - half_h)
        x2 = min(img_w, defect_cx + half_w)
        y2 = min(img_h, defect_cy + half_h)

        # Ensure valid ROI
        if x2 <= x1 or y2 <= y1:
            self.clear_defect_view()
            return

        roi = self.current_frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            self.clear_defect_view()
            return

        # Ensure contiguous array for QImage
        roi = np.ascontiguousarray(roi)

        # Resize ROI to fill DefectView label
        target_size = self.lbl_defect_view.size()
        target_w = max(1, target_size.width())
        target_h = max(1, target_size.height())
        
        roi_h_actual, roi_w_actual = roi.shape[:2]
        if roi_w_actual <= 0 or roi_h_actual <= 0:
            self.clear_defect_view()
            return
        
        # Always resize to DefectView size (upscale or downscale)
        interp = cv2.INTER_CUBIC if roi_w_actual < target_w or roi_h_actual < target_h else cv2.INTER_AREA
        resized = cv2.resize(roi, (target_w, target_h), interpolation=interp)

        # Ensure contiguous array for QImage
        resized = np.ascontiguousarray(resized)
        
        h_r, w_r, ch = resized.shape
        bytes_per_line = ch * w_r
        qt_img = QImage(resized.data, w_r, h_r, bytes_per_line, QImage.Format.Format_BGR888)
        self.lbl_defect_view.setPixmap(QPixmap.fromImage(qt_img))
        self.lbl_defect_view.setText("")

    def render_main_view(self):
        """Render ROI from full-res display frame using zoom crop to viewport size."""
        if self.current_display_frame is None:
            return

        frame = self.current_display_frame
        fh, fw, _ = frame.shape

        # Viewport target size (use scroll viewport for display size)
        target_size = self.image_scroll.viewport().size()
        target_w = max(1, target_size.width())
        target_h = max(1, target_size.height())

        # Compute ROI size based on zoom
        zoom = max(self.zoom_factor, 1e-6)
        roi_w = min(fw, int(fw / zoom)) if zoom >= 1.0 else fw
        roi_h = min(fh, int(fh / zoom)) if zoom >= 1.0 else fh
        roi_w = max(1, roi_w)
        roi_h = max(1, roi_h)

        # Center (with clamping)
        if self.view_cx is None or self.view_cy is None:
            cx, cy = fw // 2, fh // 2
        else:
            cx = int(self.view_cx)
            cy = int(self.view_cy)
        half_w = roi_w // 2
        half_h = roi_h // 2
        cx = min(max(cx, half_w), max(fw - half_w, half_w))
        cy = min(max(cy, half_h), max(fh - half_h, half_h))
        self.view_cx, self.view_cy = cx, cy

        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(fw, x1 + roi_w)
        y2 = min(fh, y1 + roi_h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        # Resize ROI to viewport; choose interpolation by scale direction
        interp = cv2.INTER_AREA if roi.shape[1] > target_w or roi.shape[0] > target_h else cv2.INTER_CUBIC
        resized = cv2.resize(roi, (target_w, target_h), interpolation=interp)

        # Ensure contiguous array for QImage
        resized = np.ascontiguousarray(resized)

        h, w, ch = resized.shape
        bytes_per_line = ch * w
        qt_img = QImage(resized.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(qt_img)
        if not pix.isNull():
            self.image_label.setPixmap(pix)
            self.image_label.resize(pix.size())


    # ---- Drag & Drop ----
    def _handle_drag_event(self, event, is_drop=False):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    ext = os.path.splitext(path)[1].lower()
                    if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                        event.setDropAction(Qt.DropAction.CopyAction)
                        event.acceptProposedAction()
                        if is_drop:
                            self.load_image_path(path)
                        return True
        return False

    def dragEnterEvent(self, event):
        if not self._handle_drag_event(event, is_drop=False):
            event.ignore()

    def dragMoveEvent(self, event):
        if not self._handle_drag_event(event, is_drop=False):
            event.ignore()

    def dropEvent(self, event):
        if not self._handle_drag_event(event, is_drop=True):
            event.ignore()

    # ---- Zoom with Ctrl + Wheel ----
    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor = min(self.zoom_factor * 1.1, self.zoom_max)
            else:
                self.zoom_factor = max(self.zoom_factor / 1.1, self.zoom_min)
            self.render_main_view()
            event.accept()
        else:
            super().wheelEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            if obj in (self.image_scroll.viewport(), self.image_label):
                delta = event.angleDelta().y()
                if delta > 0:
                    self.zoom_factor = min(self.zoom_factor * 1.1, self.zoom_max)
                else:
                    self.zoom_factor = max(self.zoom_factor / 1.1, self.zoom_min)
                self.render_main_view()
                return True
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if obj in (self.image_scroll.viewport(), self.image_label):
                self.is_panning = True
                self.pan_start_pos = event.position()
                self.pan_last_pos = event.position()
                event.accept()
                return True
        if event.type() == QEvent.Type.MouseMove and self.is_panning:
            if obj in (self.image_scroll.viewport(), self.image_label):
                pos = event.position()
                delta = pos - self.pan_last_pos
                self.pan_last_pos = pos

                if self.current_display_frame is not None:
                    fh, fw, _ = self.current_display_frame.shape
                    zoom = max(self.zoom_factor, 1e-6)
                    roi_w = min(fw, int(fw / zoom)) if zoom >= 1.0 else fw
                    roi_h = min(fh, int(fh / zoom)) if zoom >= 1.0 else fh
                    roi_w = max(1, roi_w)
                    roi_h = max(1, roi_h)

                    target_size = self.image_scroll.viewport().size()
                    target_w = max(1, target_size.width())
                    target_h = max(1, target_size.height())
                    scale_x = roi_w / target_w
                    scale_y = roi_h / target_h

                    self.view_cx -= delta.x() * scale_x
                    self.view_cy -= delta.y() * scale_y

                    half_w = roi_w // 2
                    half_h = roi_h // 2
                    self.view_cx = min(max(self.view_cx, half_w), max(fw - half_w, half_w))
                    self.view_cy = min(max(self.view_cy, half_h), max(fh - half_h, half_h))

                    self.render_main_view()
                event.accept()
                return True
        if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self.is_panning:
                self.is_panning = False
                # Check if this was a click (small movement) vs drag
                if obj in (self.image_scroll.viewport(), self.image_label):
                    release_pos = event.position()
                    drag_dist = (release_pos - self.pan_start_pos).manhattanLength()
                    if drag_dist < 5:  # Click threshold: 5 pixels
                        self.handle_image_click(release_pos)
                event.accept()
                return True
        if event.type() in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            if obj in (
                self.image_scroll.viewport(),
                self.image_label,
                self.image_scroll,
                self.central_widget,
                self.control_panel,
                self.splitter,
                self,
            ):
                handled = self._handle_drag_event(event, is_drop=event.type() == QEvent.Type.Drop)
                if handled:
                    return True
        return super().eventFilter(obj, event)
