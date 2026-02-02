import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QCheckBox, QFileDialog,
                             QListWidget, QGroupBox, QSplitter)
from PyQt6.QtCore import Qt, QTimer
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

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Side: Image Display
        self.image_label = QLabel("Camera Feed / Image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        self.image_label.setScaledContents(True) # Fit image to label

        # Right Side: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)

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

        settings_layout.addWidget(QLabel("Threshold:"))
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(100)
        settings_layout.addWidget(self.slider_thresh)

        settings_layout.addWidget(QLabel("Min Area:"))
        self.slider_area = QSlider(Qt.Orientation.Horizontal)
        self.slider_area.setRange(0, 500)
        self.slider_area.setValue(10)
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
        results_layout.addWidget(self.list_results)
        results_group.setLayout(results_layout)
        control_layout.addWidget(results_group)

        control_layout.addStretch()

        # Splitter to allow resizing
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.image_label)
        splitter.addWidget(control_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def connect_camera(self):
        if self.camera:
            self.camera.close()

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
            if self.camera:
                self.camera.close()
                self.timer.stop()

            self.camera = FileCamera(fname)
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
                # Process if inspection is on
                if self.is_inspecting:
                    contours, processed_img = self.detector.detect_static(
                        frame,
                        threshold=self.slider_thresh.value(),
                        min_area=self.slider_area.value(),
                        use_adaptive=self.chk_adaptive.isChecked()
                    )

                    # Draw results
                    display_frame = frame.copy()
                    if len(frame.shape) == 2:
                        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

                    cv2.drawContours(display_frame, contours, -1, (0, 0, 255), 2)

                    self.list_results.clear()
                    if contours:
                        self.lbl_status.setText("NG: Foreign Body")
                        self.lbl_status.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")

                        for i, cnt in enumerate(contours):
                            result = self.classifier.classify(cnt)
                            self.list_results.addItem(f"#{i+1}: {result['label']} (Area: {result['area']:.0f})")

                            # Draw label on image
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.putText(display_frame, f"{result['label']}", (x, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        self.lbl_status.setText("OK")
                        self.lbl_status.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
                else:
                    display_frame = frame
                    if len(display_frame.shape) == 2:
                        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

                # Convert to Qt Image
                h, w, ch = display_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        if self.camera:
            self.camera.close()
        event.accept()
