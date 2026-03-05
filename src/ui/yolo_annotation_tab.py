"""YOLO 어노테이션 + 학습 탭.

기능:
  - 원본 이미지 위에 bbox 드래그로 그리기 (수동 어노테이션)
  - 기존 threshold+RuleBase 자동 라벨링
  - bbox 수정/삭제
  - YOLO 데이터셋 내보내기
  - YOLO 모델 학습 (백그라운드)
"""

import os
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QListWidget, QGroupBox, QComboBox, QLineEdit, QProgressBar, QScrollArea,
    QSplitter, QMessageBox, QSpinBox, QListWidgetItem, QCheckBox,
    QSlider, QDoubleSpinBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QPointF, QThread, pyqtSignal, QTimer, QRectF
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont,
    QShortcut, QKeySequence,
)

from src.core.yolo_dataset import (
    YOLODatasetManager, bbox_xyxy_to_yolo, yolo_to_bbox_xyxy, DEFAULT_CLASS_NAMES,
)
from src.core.yolo_detector import YOLODetector
from src.core.detection import ForeignBodyDetector, BubbleDetectorParams
from src.core.classification import RuleBasedClassifier


# ──────────────────────────────────────────────────────────────
#  학습 Worker
# ──────────────────────────────────────────────────────────────

class YOLOTrainWorker(QThread):
    """백그라운드에서 YOLO 모델 학습."""
    progress = pyqtSignal(str)          # 로그 메시지
    finished_signal = pyqtSignal(bool, str)  # success, best_model_path or error

    def __init__(self, data_yaml, epochs, imgsz, batch, model_size,
                 project, name, parent=None):
        super().__init__(parent)
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.model_size = model_size
        self.project = project
        self.name = name

    def run(self):
        try:
            detector = YOLODetector()
            ok, result = detector.train(
                data_yaml=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                model_size=self.model_size,
                project=self.project,
                name=self.name,
            )
            if ok:
                self.finished_signal.emit(True, result)
            else:
                self.finished_signal.emit(False, result)
        except Exception as e:
            self.finished_signal.emit(False, str(e))


# ──────────────────────────────────────────────────────────────
#  Bbox 어노테이션 캔버스
# ──────────────────────────────────────────────────────────────

_LABEL_COLORS = {
    "Noise_Dust": QColor(128, 128, 128, 160),
    "Bubble":     QColor(0, 180, 255, 160),
    "Fiber":      QColor(255, 165, 0, 160),
    "Particle":   QColor(255, 0, 0, 160),
    "Unknown":    QColor(200, 200, 200, 160),
}


def _color_for_label(label: str) -> QColor:
    return _LABEL_COLORS.get(label, QColor(255, 255, 0, 160))


class BboxCanvas(QLabel):
    """이미지 위에 bbox를 그리는 캔버스."""

    bbox_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #222;")
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)

        self._image_bgr = None      # 원본 BGR 이미지
        self._bboxes: list[dict] = []  # [{label, x1, y1, x2, y2}, ...]
        self._selected_idx = -1
        self._current_label = "Particle"

        self._drawing = False
        self._draw_start = None      # QPointF (widget coords)
        self._draw_end = None

        self._resizing = False
        self._resize_handle = None   # "tl", "tr", "bl", "br"
        self._resize_idx = -1

        self._moving = False
        self._move_offset = None

        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

    def set_image(self, img_bgr: np.ndarray, keep_bboxes: bool = False):
        if not keep_bboxes:
            self._bboxes = []
            self._selected_idx = -1
        self._image_bgr = img_bgr.copy() if img_bgr is not None else None
        self._update_display()

    def set_bboxes(self, bboxes: list[dict]):
        self._bboxes = list(bboxes)
        self._selected_idx = min(self._selected_idx, len(self._bboxes) - 1)
        self._update_display()

    def get_bboxes(self) -> list[dict]:
        return list(self._bboxes)

    def set_current_label(self, label: str):
        self._current_label = label

    def add_bbox(self, x1, y1, x2, y2, label: str):
        self._bboxes.append({"label": label, "x1": int(x1), "y1": int(y1),
                              "x2": int(x2), "y2": int(y2)})
        self._selected_idx = len(self._bboxes) - 1
        self.bbox_changed.emit()
        self._update_display()

    def remove_selected(self):
        if 0 <= self._selected_idx < len(self._bboxes):
            del self._bboxes[self._selected_idx]
            self._selected_idx = min(self._selected_idx, len(self._bboxes) - 1)
            self.bbox_changed.emit()
            self._update_display()

    def clear_bboxes(self):
        self._bboxes = []
        self._selected_idx = -1
        self.bbox_changed.emit()
        self._update_display()

    def _widget_to_image(self, pos: QPointF):
        """위젯 좌표 → 이미지 좌표."""
        if self._image_bgr is None or self._scale == 0:
            return None
        ix = (pos.x() - self._offset_x) / self._scale
        iy = (pos.y() - self._offset_y) / self._scale
        return (ix, iy)

    def _image_to_widget(self, ix, iy):
        """이미지 좌표 → 위젯 좌표."""
        wx = ix * self._scale + self._offset_x
        wy = iy * self._scale + self._offset_y
        return (wx, wy)

    def _update_display(self):
        if self._image_bgr is None:
            self.setPixmap(QPixmap())
            self.setText("이미지 없음")
            return

        h, w = self._image_bgr.shape[:2]
        lw, lh = self.width(), self.height()
        if lw <= 0 or lh <= 0:
            return

        scale_x = lw / w
        scale_y = lh / h
        self._scale = min(scale_x, scale_y)
        disp_w = int(w * self._scale)
        disp_h = int(h * self._scale)
        self._offset_x = (lw - disp_w) / 2
        self._offset_y = (lh - disp_h) / 2

        rgb = cv2.cvtColor(self._image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        resized = np.ascontiguousarray(resized)
        qimg = QImage(resized.data, disp_w, disp_h, disp_w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for i, bb in enumerate(self._bboxes):
            color = _color_for_label(bb["label"])
            wx1, wy1 = self._image_to_widget(bb["x1"], bb["y1"])
            wx2, wy2 = self._image_to_widget(bb["x2"], bb["y2"])
            wx1 -= self._offset_x
            wy1 -= self._offset_y
            wx2 -= self._offset_x
            wy2 -= self._offset_y

            pen = QPen(color, 2 if i != self._selected_idx else 3)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
            painter.drawRect(QRectF(wx1, wy1, wx2 - wx1, wy2 - wy1))

            font = QFont("Arial", 9, QFont.Weight.Bold)
            painter.setFont(font)
            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.drawText(int(wx1), max(int(wy1) - 4, 12), bb["label"])

        if self._drawing and self._draw_start and self._draw_end:
            pen = QPen(QColor(0, 255, 0, 200), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 255, 0, 30)))
            sx, sy = self._draw_start
            ex, ey = self._draw_end
            painter.drawRect(QRectF(sx - self._offset_x, sy - self._offset_y,
                                    ex - sx, ey - sy))

        painter.end()

        self.setPixmap(pix)
        self.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._image_bgr is None:
            return
        pos = event.position()
        img_pos = self._widget_to_image(pos)
        if img_pos is None:
            return

        clicked_idx = self._find_bbox_at(img_pos)
        if clicked_idx >= 0:
            self._selected_idx = clicked_idx
            self._moving = True
            bb = self._bboxes[clicked_idx]
            self._move_offset = (img_pos[0] - bb["x1"], img_pos[1] - bb["y1"])
            self.bbox_changed.emit()
            self._update_display()
            return

        self._drawing = True
        self._draw_start = (pos.x(), pos.y())
        self._draw_end = (pos.x(), pos.y())

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._drawing:
            self._draw_end = (pos.x(), pos.y())
            self._update_display()
        elif self._moving and 0 <= self._selected_idx < len(self._bboxes):
            img_pos = self._widget_to_image(pos)
            if img_pos:
                bb = self._bboxes[self._selected_idx]
                w = bb["x2"] - bb["x1"]
                h = bb["y2"] - bb["y1"]
                bb["x1"] = int(img_pos[0] - self._move_offset[0])
                bb["y1"] = int(img_pos[1] - self._move_offset[1])
                bb["x2"] = bb["x1"] + w
                bb["y2"] = bb["y1"] + h
                self._update_display()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._moving:
            self._moving = False
            self._move_offset = None
            self.bbox_changed.emit()
            self._update_display()
            return
        if self._drawing and self._draw_start and self._draw_end:
            self._drawing = False
            img_start = self._widget_to_image(QPointF(*self._draw_start))
            img_end = self._widget_to_image(QPointF(*self._draw_end))
            if img_start and img_end:
                x1 = int(min(img_start[0], img_end[0]))
                y1 = int(min(img_start[1], img_end[1]))
                x2 = int(max(img_start[0], img_end[0]))
                y2 = int(max(img_start[1], img_end[1]))
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.add_bbox(x1, y1, x2, y2, self._current_label)
            self._draw_start = None
            self._draw_end = None
            self._update_display()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_selected()

    def _find_bbox_at(self, img_pos) -> int:
        """이미지 좌표가 어떤 bbox 안에 있는지 찾기. -1이면 없음."""
        ix, iy = img_pos
        for i in range(len(self._bboxes) - 1, -1, -1):
            bb = self._bboxes[i]
            if bb["x1"] <= ix <= bb["x2"] and bb["y1"] <= iy <= bb["y2"]:
                return i
        return -1


# ──────────────────────────────────────────────────────────────
#  YOLO 탭 메인 위젯
# ──────────────────────────────────────────────────────────────

class YOLOAnnotationTab(QWidget):
    """YOLO 어노테이션 + 학습 통합 탭."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset_mgr: YOLODatasetManager | None = None
        self._image_list: list[str] = []
        self._current_idx = -1
        self._current_split = "train"
        self._train_worker: YOLOTrainWorker | None = None
        self.init_ui()

    def _app_root(self):
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ══════ 왼쪽 패널: 데이터셋 + 자동라벨링 파라미터 + 학습 ══════
        left = QWidget()
        left.setFixedWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)
        splitter.addWidget(left)

        # --- 데이터셋 ---
        ds_group = QGroupBox("YOLO 데이터셋")
        ds_layout = QVBoxLayout()
        ds_layout.setSpacing(4)

        row_ds = QHBoxLayout()
        self.txt_dataset_dir = QLineEdit()
        self.txt_dataset_dir.setPlaceholderText("데이터셋 폴더 경로...")
        row_ds.addWidget(self.txt_dataset_dir)
        btn_browse_ds = QPushButton("...")
        btn_browse_ds.setFixedWidth(30)
        btn_browse_ds.clicked.connect(self._browse_dataset)
        row_ds.addWidget(btn_browse_ds)
        ds_layout.addLayout(row_ds)

        btn_create_ds = QPushButton("데이터셋 생성/초기화")
        btn_create_ds.setToolTip("선택한 폴더에 YOLO 데이터셋 구조와 data.yaml을 생성합니다.")
        btn_create_ds.clicked.connect(self._create_dataset)
        ds_layout.addWidget(btn_create_ds)

        self.lbl_ds_stats = QLabel("데이터셋 미설정")
        self.lbl_ds_stats.setWordWrap(True)
        self.lbl_ds_stats.setStyleSheet("font-size: 11px; color: #aaa;")
        ds_layout.addWidget(self.lbl_ds_stats)

        ds_group.setLayout(ds_layout)
        left_layout.addWidget(ds_group)

        # --- 자동 라벨링 파라미터 (스크롤) ---
        auto_scroll = QScrollArea()
        auto_scroll.setWidgetResizable(True)
        auto_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        auto_inner = QWidget()
        auto_inner_layout = QVBoxLayout(auto_inner)
        auto_inner_layout.setContentsMargins(0, 0, 0, 0)
        auto_inner_layout.setSpacing(4)

        # ── 일반 검출 (Particle/Fiber/Noise) ──
        gen_group = QGroupBox("일반 검출 (Particle/Fiber/Noise)")
        gen_group.setCheckable(True)
        gen_group.setChecked(True)
        self.grp_general = gen_group
        gen_layout = QVBoxLayout()
        gen_layout.setSpacing(2)

        def _hrow(label_text, widget):
            r = QHBoxLayout()
            r.addWidget(QLabel(label_text))
            r.addWidget(widget)
            return r

        self.spin_auto_thresh = QSpinBox()
        self.spin_auto_thresh.setRange(0, 255)
        self.spin_auto_thresh.setValue(100)
        gen_layout.addLayout(_hrow("Threshold:", self.spin_auto_thresh))

        self.spin_auto_area = QSpinBox()
        self.spin_auto_area.setRange(0, 5000)
        self.spin_auto_area.setValue(10)
        gen_layout.addLayout(_hrow("Min Area:", self.spin_auto_area))

        self.chk_auto_adaptive = QCheckBox("Adaptive Threshold")
        self.chk_auto_adaptive.setChecked(True)
        gen_layout.addWidget(self.chk_auto_adaptive)

        self.spin_morph_open = QSpinBox()
        self.spin_morph_open.setRange(1, 15)
        self.spin_morph_open.setValue(2)
        self.spin_morph_open.setToolTip("Opening 커널 크기 (작은 노이즈 제거)")
        gen_layout.addLayout(_hrow("Open 커널:", self.spin_morph_open))

        self.spin_morph_close = QSpinBox()
        self.spin_morph_close.setRange(1, 15)
        self.spin_morph_close.setValue(3)
        self.spin_morph_close.setToolTip("Closing 커널 크기 (끊어진 영역 연결)")
        gen_layout.addLayout(_hrow("Close 커널:", self.spin_morph_close))

        gen_group.setLayout(gen_layout)
        auto_inner_layout.addWidget(gen_group)

        # ── Bubble 검출 ──
        bub_group = QGroupBox("Bubble 검출")
        bub_group.setCheckable(True)
        bub_group.setChecked(True)
        self.grp_bubble = bub_group
        bub_layout = QVBoxLayout()
        bub_layout.setSpacing(2)

        # 배경 평탄화
        bub_layout.addWidget(QLabel("── 배경 평탄화 ──"))
        self.spin_bub_bg_ksize = QSpinBox()
        self.spin_bub_bg_ksize.setRange(3, 201)
        self.spin_bub_bg_ksize.setSingleStep(2)
        self.spin_bub_bg_ksize.setValue(61)
        bub_layout.addLayout(_hrow("커널 크기:", self.spin_bub_bg_ksize))

        self.spin_bub_bg_smooth = QDoubleSpinBox()
        self.spin_bub_bg_smooth.setRange(0.0, 100.0)
        self.spin_bub_bg_smooth.setDecimals(1)
        self.spin_bub_bg_smooth.setValue(15.0)
        self.spin_bub_bg_smooth.setToolTip("0=자동(커널/4)")
        bub_layout.addLayout(_hrow("스무딩 σ:", self.spin_bub_bg_smooth))

        self.chk_bub_use_clahe = QCheckBox("CLAHE 사용")
        self.chk_bub_use_clahe.setChecked(False)
        bub_layout.addWidget(self.chk_bub_use_clahe)

        # DoG 밴드패스
        bub_layout.addWidget(QLabel("── DoG 밴드패스 ──"))
        self.spin_bub_sigma_small = QDoubleSpinBox()
        self.spin_bub_sigma_small.setRange(0.1, 20.0)
        self.spin_bub_sigma_small.setDecimals(1)
        self.spin_bub_sigma_small.setValue(1.2)
        bub_layout.addLayout(_hrow("σ small:", self.spin_bub_sigma_small))

        self.spin_bub_sigma_large = QDoubleSpinBox()
        self.spin_bub_sigma_large.setRange(0.5, 50.0)
        self.spin_bub_sigma_large.setDecimals(1)
        self.spin_bub_sigma_large.setValue(6.0)
        bub_layout.addLayout(_hrow("σ large:", self.spin_bub_sigma_large))

        # MAD 임계
        bub_layout.addWidget(QLabel("── MAD 임계 ──"))
        self.spin_bub_thr_k = QDoubleSpinBox()
        self.spin_bub_thr_k.setRange(0.5, 20.0)
        self.spin_bub_thr_k.setDecimals(1)
        self.spin_bub_thr_k.setValue(4.0)
        bub_layout.addLayout(_hrow("k (민감도):", self.spin_bub_thr_k))

        # 형태학
        bub_layout.addWidget(QLabel("── 형태학 ──"))
        self.spin_bub_morph_close = QSpinBox()
        self.spin_bub_morph_close.setRange(1, 31)
        self.spin_bub_morph_close.setSingleStep(2)
        self.spin_bub_morph_close.setValue(5)
        bub_layout.addLayout(_hrow("Close 커널:", self.spin_bub_morph_close))

        self.spin_bub_morph_open = QSpinBox()
        self.spin_bub_morph_open.setRange(1, 31)
        self.spin_bub_morph_open.setSingleStep(2)
        self.spin_bub_morph_open.setValue(3)
        bub_layout.addLayout(_hrow("Open 커널:", self.spin_bub_morph_open))

        # 형상 필터
        bub_layout.addWidget(QLabel("── 형상 필터 ──"))
        self.spin_bub_min_diam = QSpinBox()
        self.spin_bub_min_diam.setRange(1, 500)
        self.spin_bub_min_diam.setValue(8)
        bub_layout.addLayout(_hrow("최소 직경:", self.spin_bub_min_diam))

        self.spin_bub_max_diam = QSpinBox()
        self.spin_bub_max_diam.setRange(1, 1000)
        self.spin_bub_max_diam.setValue(100)
        bub_layout.addLayout(_hrow("최대 직경:", self.spin_bub_max_diam))

        self.spin_bub_circularity = QDoubleSpinBox()
        self.spin_bub_circularity.setRange(0.0, 1.0)
        self.spin_bub_circularity.setDecimals(2)
        self.spin_bub_circularity.setSingleStep(0.05)
        self.spin_bub_circularity.setValue(0.35)
        bub_layout.addLayout(_hrow("최소 원형도:", self.spin_bub_circularity))

        self.spin_bub_solidity = QDoubleSpinBox()
        self.spin_bub_solidity.setRange(0.0, 1.0)
        self.spin_bub_solidity.setDecimals(2)
        self.spin_bub_solidity.setSingleStep(0.05)
        self.spin_bub_solidity.setValue(0.70)
        bub_layout.addLayout(_hrow("최소 볼록도:", self.spin_bub_solidity))

        self.spin_bub_max_aspect = QDoubleSpinBox()
        self.spin_bub_max_aspect.setRange(1.0, 10.0)
        self.spin_bub_max_aspect.setDecimals(1)
        self.spin_bub_max_aspect.setSingleStep(0.1)
        self.spin_bub_max_aspect.setValue(2.2)
        bub_layout.addLayout(_hrow("최대 종횡비:", self.spin_bub_max_aspect))

        bub_group.setLayout(bub_layout)
        auto_inner_layout.addWidget(bub_group)

        auto_scroll.setWidget(auto_inner)
        left_layout.addWidget(auto_scroll, 1)  # stretch=1 so scroll takes remaining space

        # --- 학습 ---
        train_group = QGroupBox("YOLO 학습")
        train_layout = QVBoxLayout()
        train_layout.setSpacing(3)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("모델:"))
        self.combo_model_size = QComboBox()
        self.combo_model_size.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self.combo_model_size.setCurrentText("yolov8s")
        row_model.addWidget(self.combo_model_size)
        train_layout.addLayout(row_model)

        row_epochs = QHBoxLayout()
        row_epochs.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        row_epochs.addWidget(self.spin_epochs)
        train_layout.addLayout(row_epochs)

        row_imgsz = QHBoxLayout()
        row_imgsz.addWidget(QLabel("ImgSize:"))
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(128, 1280)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        row_imgsz.addWidget(self.spin_imgsz)
        train_layout.addLayout(row_imgsz)

        row_batch = QHBoxLayout()
        row_batch.addWidget(QLabel("Batch:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 128)
        self.spin_batch.setValue(16)
        row_batch.addWidget(self.spin_batch)
        train_layout.addLayout(row_batch)

        btn_split = QPushButton("Train/Val 분할 (20%)")
        btn_split.setToolTip("Train 이미지의 20%를 Val로 자동 이동합니다.")
        btn_split.clicked.connect(self._split_train_val)
        train_layout.addWidget(btn_split)

        self.btn_start_train = QPushButton("학습 시작")
        self.btn_start_train.clicked.connect(self._start_training)
        train_layout.addWidget(self.btn_start_train)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        train_layout.addWidget(self.progress_bar)

        self.lbl_train_status = QLabel("")
        self.lbl_train_status.setWordWrap(True)
        self.lbl_train_status.setStyleSheet("font-size: 11px;")
        train_layout.addWidget(self.lbl_train_status)

        train_group.setLayout(train_layout)
        left_layout.addWidget(train_group)

        left_layout.addStretch()

        # ══════ 중앙: 캔버스 ══════
        self.canvas = BboxCanvas()
        splitter.addWidget(self.canvas)
        self.canvas.bbox_changed.connect(self._on_bbox_changed)

        # ══════ 오른쪽 패널: 이미지 탐색 + 어노테이션 ══════
        right = QWidget()
        right.setFixedWidth(280)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)
        splitter.addWidget(right)

        # --- 이미지 탐색 ---
        img_group = QGroupBox("이미지")
        img_layout = QVBoxLayout()
        img_layout.setSpacing(3)

        row_load = QHBoxLayout()
        btn_add_images = QPushButton("이미지 추가")
        btn_add_images.setToolTip("이미지를 데이터셋에 복사하여 추가합니다.")
        btn_add_images.clicked.connect(self._add_images)
        row_load.addWidget(btn_add_images)
        btn_add_folder = QPushButton("폴더 자동라벨링")
        btn_add_folder.setToolTip("폴더의 이미지를 threshold+RuleBase로 자동 라벨링하여 추가합니다.")
        btn_add_folder.clicked.connect(self._auto_label_folder)
        row_load.addWidget(btn_add_folder)
        img_layout.addLayout(row_load)

        self.list_images = QListWidget()
        self.list_images.currentRowChanged.connect(self._on_image_selected)
        img_layout.addWidget(self.list_images)

        nav_row = QHBoxLayout()
        btn_prev = QPushButton("< 이전")
        btn_prev.clicked.connect(self._prev_image)
        nav_row.addWidget(btn_prev)
        btn_next = QPushButton("다음 >")
        btn_next.clicked.connect(self._next_image)
        nav_row.addWidget(btn_next)
        img_layout.addLayout(nav_row)

        img_group.setLayout(img_layout)
        right_layout.addWidget(img_group)

        # --- 어노테이션 ---
        ann_group = QGroupBox("어노테이션")
        ann_layout = QVBoxLayout()
        ann_layout.setSpacing(3)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("라벨:"))
        self.combo_label = QComboBox()
        self.combo_label.addItems(DEFAULT_CLASS_NAMES)
        self.combo_label.setCurrentText("Particle")
        self.combo_label.currentTextChanged.connect(self.canvas.set_current_label)
        label_row.addWidget(self.combo_label)
        ann_layout.addLayout(label_row)

        btn_row = QHBoxLayout()
        btn_del = QPushButton("선택 삭제")
        btn_del.clicked.connect(self.canvas.remove_selected)
        btn_row.addWidget(btn_del)
        btn_clear = QPushButton("전체 삭제")
        btn_clear.clicked.connect(self.canvas.clear_bboxes)
        btn_row.addWidget(btn_clear)
        ann_layout.addLayout(btn_row)

        btn_auto = QPushButton("현재 이미지 자동 라벨링")
        btn_auto.setToolTip("현재 이미지에 threshold+RuleBase를 적용하여 bbox를 자동 생성합니다.")
        btn_auto.clicked.connect(self._auto_label_current)
        ann_layout.addWidget(btn_auto)

        btn_save_ann = QPushButton("어노테이션 저장")
        btn_save_ann.clicked.connect(self._save_current_annotations)
        ann_layout.addWidget(btn_save_ann)

        self.list_bboxes = QListWidget()
        self.list_bboxes.setMaximumHeight(150)
        ann_layout.addWidget(self.list_bboxes)

        ann_group.setLayout(ann_layout)
        right_layout.addWidget(ann_group)

        right_layout.addStretch()

        splitter.setStretchFactor(0, 0)  # left panel fixed
        splitter.setStretchFactor(1, 3)  # canvas stretches
        splitter.setStretchFactor(2, 0)  # right panel fixed

    # ── 데이터셋 관리 ──

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "YOLO 데이터셋 폴더", self._app_root())
        if path:
            self.txt_dataset_dir.setText(path)
            self._load_dataset(path)

    def _create_dataset(self):
        path = self.txt_dataset_dir.text().strip()
        if not path:
            QMessageBox.warning(self, "오류", "데이터셋 폴더 경로를 지정하세요.")
            return
        self._dataset_mgr = YOLODatasetManager(path, list(DEFAULT_CLASS_NAMES))
        self._dataset_mgr.create_structure()
        self._dataset_mgr.write_data_yaml()
        self._refresh_image_list()
        self._refresh_stats()
        QMessageBox.information(self, "완료", f"데이터셋이 초기화되었습니다.\n{path}")

    def _load_dataset(self, path: str):
        self._dataset_mgr = YOLODatasetManager(path, list(DEFAULT_CLASS_NAMES))
        if not os.path.isfile(self._dataset_mgr.data_yaml_path):
            self._dataset_mgr.create_structure()
            self._dataset_mgr.write_data_yaml()
        self._refresh_image_list()
        self._refresh_stats()

    def _refresh_image_list(self):
        self.list_images.clear()
        self._image_list = []
        if self._dataset_mgr is None:
            return
        img_dir = (self._dataset_mgr.images_train_dir
                   if self._current_split == "train"
                   else self._dataset_mgr.images_val_dir)
        if not os.path.isdir(img_dir):
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))
        self._image_list = files
        for f in files:
            self.list_images.addItem(f)
        if files:
            self.list_images.setCurrentRow(0)

    def _refresh_stats(self):
        if self._dataset_mgr is None:
            self.lbl_ds_stats.setText("데이터셋 미설정")
            return
        stats = self._dataset_mgr.get_stats()
        lines = [
            f"Train: {stats['train_images']}장 / Val: {stats['val_images']}장",
            "클래스별:"
        ]
        for name, cnt in stats["class_counts"].items():
            lines.append(f"  {name}: {cnt}")
        self.lbl_ds_stats.setText("\n".join(lines))

    # ── 이미지 탐색 ──

    def _on_image_selected(self, row):
        if row < 0 or row >= len(self._image_list):
            self.canvas.set_image(None)
            self._current_idx = -1
            return
        self._current_idx = row
        fname = self._image_list[row]
        img_dir = (self._dataset_mgr.images_train_dir
                   if self._current_split == "train"
                   else self._dataset_mgr.images_val_dir)
        img_path = os.path.join(img_dir, fname)
        try:
            buf = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is not None:
            self.canvas.set_image(img)
            anns = self._dataset_mgr.load_image_annotations(fname, self._current_split)
            bboxes = [{"label": a["label"], "x1": a["bbox_xyxy"][0], "y1": a["bbox_xyxy"][1],
                        "x2": a["bbox_xyxy"][2], "y2": a["bbox_xyxy"][3]} for a in anns]
            self.canvas.set_bboxes(bboxes)
            self._refresh_bbox_list()

    def _prev_image(self):
        if self._current_idx > 0:
            self.list_images.setCurrentRow(self._current_idx - 1)

    def _next_image(self):
        if self._current_idx < len(self._image_list) - 1:
            self.list_images.setCurrentRow(self._current_idx + 1)

    # ── 어노테이션 ──

    def _on_bbox_changed(self):
        self._refresh_bbox_list()

    def _refresh_bbox_list(self):
        self.list_bboxes.clear()
        for i, bb in enumerate(self.canvas.get_bboxes()):
            w = bb["x2"] - bb["x1"]
            h = bb["y2"] - bb["y1"]
            self.list_bboxes.addItem(f"#{i+1} {bb['label']} ({w}x{h})")

    def _save_current_annotations(self):
        if self._dataset_mgr is None or self._current_idx < 0:
            return
        fname = self._image_list[self._current_idx]
        bboxes = self.canvas.get_bboxes()
        img = self.canvas._image_bgr
        if img is None:
            return
        img_h, img_w = img.shape[:2]
        anns = [{"label": bb["label"], "bbox_xyxy": (bb["x1"], bb["y1"], bb["x2"], bb["y2"])}
                for bb in bboxes]
        self._dataset_mgr.save_image_annotations(fname, anns, img_w, img_h, self._current_split)
        self._refresh_stats()

    def _build_bubble_params(self) -> BubbleDetectorParams:
        """UI 위젯 값을 BubbleDetectorParams 객체로 변환."""
        bp = BubbleDetectorParams()
        bp.enabled = self.grp_bubble.isChecked()
        bp.bg_open_ksize = self.spin_bub_bg_ksize.value()
        bp.bg_smooth_sigma = self.spin_bub_bg_smooth.value()
        bp.use_clahe = self.chk_bub_use_clahe.isChecked()
        bp.sigma_small = self.spin_bub_sigma_small.value()
        bp.sigma_large = self.spin_bub_sigma_large.value()
        bp.thr_k = self.spin_bub_thr_k.value()
        bp.morph_close_size = self.spin_bub_morph_close.value()
        bp.morph_open_size = self.spin_bub_morph_open.value()
        bp.min_diameter = self.spin_bub_min_diam.value()
        bp.max_diameter = self.spin_bub_max_diam.value()
        bp.circularity_min = self.spin_bub_circularity.value()
        bp.solidity_min = self.spin_bub_solidity.value()
        bp.max_aspect_ratio = self.spin_bub_max_aspect.value()
        return bp

    def _auto_label_current(self):
        """현재 캔버스 이미지에 일반검출 + Bubble검출 자동 라벨링."""
        img = self.canvas._image_bgr
        if img is None:
            QMessageBox.information(self, "알림", "이미지를 먼저 선택하세요.")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        all_contours: list = []
        all_labels: list[str] = []

        # ── 1) 일반 검출 (Particle / Fiber / Noise_Dust) ──
        if self.grp_general.isChecked():
            detector_gen = ForeignBodyDetector()
            contours_gen, _ = detector_gen.detect_static(
                gray,
                threshold=self.spin_auto_thresh.value(),
                min_area=self.spin_auto_area.value(),
                use_adaptive=self.chk_auto_adaptive.isChecked(),
                detect_bubbles=False,
                open_kernel=self.spin_morph_open.value(),
                close_kernel=self.spin_morph_close.value(),
            )
            classifier = RuleBasedClassifier()
            for cnt in contours_gen:
                all_contours.append(cnt)
                all_labels.append(classifier.classify(cnt)["label"])

        # ── 2) Bubble 검출 ──
        if self.grp_bubble.isChecked():
            detector_bub = ForeignBodyDetector()
            detector_bub.bubble_params = self._build_bubble_params()
            bub_contours, _ = detector_bub.detect_bubbles(gray)
            for cnt in bub_contours:
                all_contours.append(cnt)
                all_labels.append("Bubble")

        if not all_contours:
            QMessageBox.information(self, "결과", "검출된 객체가 없습니다.")
            return

        new_bboxes = list(self.canvas.get_bboxes())
        added = 0
        for cnt, label in zip(all_contours, all_labels):
            x, y, w, h = cv2.boundingRect(cnt)
            new_bboxes.append({
                "label": label,
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
            })
            added += 1
        self.canvas.set_bboxes(new_bboxes)
        self._refresh_bbox_list()
        QMessageBox.information(self, "완료", f"{added}개 bbox가 추가되었습니다.\n"
                                f"(일반: {len(all_contours) - sum(1 for l in all_labels if l == 'Bubble')}, "
                                f"Bubble: {sum(1 for l in all_labels if l == 'Bubble')})")

    # ── 이미지 추가 ──

    def _add_images(self):
        """이미지 파일을 선택하여 데이터셋에 복사."""
        if self._dataset_mgr is None:
            QMessageBox.warning(self, "오류", "먼저 데이터셋을 설정/생성하세요.")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "이미지 추가", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All (*.*)",
        )
        if not files:
            return
        self._dataset_mgr.create_structure()
        img_dir = self._dataset_mgr.images_train_dir
        lbl_dir = self._dataset_mgr.labels_train_dir
        added = 0
        for f in files:
            basename = os.path.basename(f)
            dst = os.path.join(img_dir, basename)
            counter = 1
            while os.path.exists(dst):
                name, ext = os.path.splitext(basename)
                dst = os.path.join(img_dir, f"{name}_{counter}{ext}")
                counter += 1
            try:
                import shutil
                shutil.copy2(f, dst)
                lbl_name = os.path.splitext(os.path.basename(dst))[0] + ".txt"
                lbl_path = os.path.join(lbl_dir, lbl_name)
                if not os.path.exists(lbl_path):
                    with open(lbl_path, "w") as lf:
                        pass
                added += 1
            except Exception as e:
                print(f"이미지 추가 실패: {e}")
        self._refresh_image_list()
        self._refresh_stats()
        QMessageBox.information(self, "완료", f"{added}장이 데이터셋에 추가되었습니다.")

    def _auto_label_folder(self):
        """폴더의 모든 이미지를 자동 라벨링하여 데이터셋에 추가."""
        if self._dataset_mgr is None:
            QMessageBox.warning(self, "오류", "먼저 데이터셋을 설정/생성하세요.")
            return
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택", "")
        if not folder:
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if f.lower().endswith(exts)]
        if not files:
            QMessageBox.information(self, "알림", "이미지 파일이 없습니다.")
            return

        do_general = self.grp_general.isChecked()
        do_bubble = self.grp_bubble.isChecked()
        bubble_params = self._build_bubble_params() if do_bubble else None

        added = 0
        for f in files:
            result = self._dataset_mgr.auto_label_image(
                f,
                threshold=self.spin_auto_thresh.value(),
                min_area=self.spin_auto_area.value(),
                use_adaptive=self.chk_auto_adaptive.isChecked(),
                detect_general=do_general,
                detect_bubbles=do_bubble,
                bubble_params=bubble_params,
                open_kernel=self.spin_morph_open.value() if do_general else 2,
                close_kernel=self.spin_morph_close.value() if do_general else 3,
                split="train",
            )
            if result:
                added += 1

        self._refresh_image_list()
        self._refresh_stats()
        QMessageBox.information(self, "완료",
                                f"{len(files)}장 중 {added}장이 라벨링되어 추가되었습니다.")

    # ── Train/Val 분할 ──

    def _split_train_val(self):
        if self._dataset_mgr is None:
            QMessageBox.warning(self, "오류", "먼저 데이터셋을 설정/생성하세요.")
            return
        self._dataset_mgr.split_train_val(val_ratio=0.2)
        self._refresh_image_list()
        self._refresh_stats()
        QMessageBox.information(self, "완료", "Train 이미지의 20%가 Val로 이동되었습니다.")

    # ── 학습 ──

    def _start_training(self):
        if self._dataset_mgr is None:
            QMessageBox.warning(self, "오류", "먼저 데이터셋을 설정/생성하세요.")
            return

        yaml_path = self._dataset_mgr.data_yaml_path
        if not os.path.isfile(yaml_path):
            self._dataset_mgr.write_data_yaml()

        stats = self._dataset_mgr.get_stats()
        if stats["train_images"] == 0:
            QMessageBox.warning(self, "오류", "Train 이미지가 없습니다. 이미지를 먼저 추가하세요.")
            return

        project_dir = os.path.join(self._dataset_mgr.dataset_dir, "runs")
        self.btn_start_train.setEnabled(False)
        self.progress_bar.show()
        self.lbl_train_status.setText("학습 시작 중...")

        self._train_worker = YOLOTrainWorker(
            data_yaml=yaml_path,
            epochs=self.spin_epochs.value(),
            imgsz=self.spin_imgsz.value(),
            batch=self.spin_batch.value(),
            model_size=self.combo_model_size.currentText(),
            project=project_dir,
            name="train",
            parent=self,
        )
        self._train_worker.finished_signal.connect(self._on_train_finished)
        self._train_worker.start()

    def _on_train_finished(self, success, result):
        self.btn_start_train.setEnabled(True)
        self.progress_bar.hide()
        if success:
            self.lbl_train_status.setText(f"학습 완료!\n모델: {result}")
            QMessageBox.information(self, "학습 완료",
                                    f"YOLO 모델 학습이 완료되었습니다.\n\n"
                                    f"Best 모델: {result}\n\n"
                                    f"Main 탭에서 YOLO 모드로 전환 후 이 모델을 로드하세요.")
        else:
            self.lbl_train_status.setText(f"학습 실패: {result}")
            QMessageBox.warning(self, "학습 실패", f"학습 중 오류가 발생했습니다.\n\n{result}")
        self._train_worker = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self._prev_image()
        elif event.key() == Qt.Key.Key_Right:
            self._next_image()
        else:
            super().keyPressEvent(event)
