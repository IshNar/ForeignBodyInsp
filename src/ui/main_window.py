import cv2
import hashlib
import json
import numpy as np
import os
import sys
import concurrent.futures
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QCheckBox, QFileDialog,
                             QListWidget, QGroupBox, QSplitter, QScrollArea, QGridLayout,
                             QMessageBox, QTabWidget, QMenuBar, QMenu, QApplication,
                             QDialog, QDialogButtonBox, QComboBox, QStyle, QSizePolicy,
                             QDoubleSpinBox, QRadioButton, QInputDialog, QStatusBar,
                             QLineEdit
                             )
from PyQt6.QtCore import Qt, QTimer, QSize, QEvent, QPointF, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence, QAction, QIcon, QFont, QShowEvent

from src.core.detection import ForeignBodyDetector
from src.core.classification import ParticleClassifier, DefectImageSaver, CLASSIFICATION_INPUT_SIZE
from src.core.yolo_detector import YOLODetector
from src.hardware.basler_camera import BaslerCamera
from src.hardware.file_camera import FileCamera
from src.ui.classification_tab import ClassificationTab
from src.ui.yolo_annotation_tab import YOLOAnnotationTab
from src.ui.basler_settings_dialog import BaslerSettingsDialog
from src.ui.rule_params_dialog import RuleParamsDialog


class DetectionWorker(QThread):
    """Runs detection and optional MainView drawing in a background thread."""
    result_ready = pyqtSignal(list, object, object, object, object, object, dict)  # contours, display_frame, display_base_frame, last_debug, raw_frame, full_frame, tact_times

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_bgr = None
        self._full_frame_bgr = None
        self._threshold = 100
        self._min_area = 10
        self._use_adaptive = True
        self._draw_on_mainview = True
        self._debug_dir = None
        self._classification_enabled = True
        self._general_enabled = True
        self._bubble_show_text = True

    def run_detection(self, frame_bgr, full_frame_bgr, threshold, min_area, use_adaptive, draw_on_mainview, debug_dir, classifier=None, yolo_detector=None, use_yolo=False, yolo_conf=0.15, bubble_params=None, open_kernel=2, close_kernel=3, classification_enabled=True, general_enabled=True, bubble_show_text=True):
        self._frame_bgr = frame_bgr.copy() if frame_bgr is not None else None
        self._full_frame_bgr = full_frame_bgr.copy() if full_frame_bgr is not None else None
        self._threshold = threshold
        self._min_area = min_area
        self._use_adaptive = use_adaptive
        self._draw_on_mainview = draw_on_mainview
        self._debug_dir = debug_dir
        self._classifier = classifier
        self._yolo_detector = yolo_detector
        self._use_yolo = use_yolo
        self._yolo_conf = float(yolo_conf)
        self._bubble_params = bubble_params
        self._open_kernel = open_kernel
        self._close_kernel = close_kernel
        self._classification_enabled = classification_enabled
        self._general_enabled = general_enabled
        self._bubble_show_text = bubble_show_text
        self.start()

    def run(self):
        try:
            if self._frame_bgr is None:
                self.result_ready.emit([], None, None, None, None, self._full_frame_bgr, {})
                return
            frame = self._frame_bgr

            if getattr(self, "_use_yolo", False) and getattr(self, "_yolo_detector", None) is not None and self._yolo_detector.is_loaded():
                self._run_yolo(frame)
            else:
                self._run_threshold(frame)
        except Exception as e:
            print(f"DetectionWorker error: {e}")
            import traceback
            traceback.print_exc()
            self.result_ready.emit([], None, None, None, None, self._full_frame_bgr, {})
        finally:
            self._frame_bgr = None
            self._full_frame_bgr = None

    def _run_threshold(self, frame):
        """기존 Threshold+분류 파이프라인 (최적화 + 프로파일링)."""
        import time
        t_start = time.perf_counter()

        # Detector 재사용 (매 프레임마다 새로 생성하지 않음)
        if not hasattr(self, '_cached_detector'):
            self._cached_detector = ForeignBodyDetector()
        detector = self._cached_detector
        if getattr(self, "_bubble_params", None):
            detector.bubble_params = self._bubble_params
            
        classifier = getattr(self, "_classifier", None) or ParticleClassifier()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        do_bubble = False
        if getattr(self, "_bubble_params", None) and getattr(self._bubble_params, "enabled", False):
            do_bubble = True

        t_prep = time.perf_counter()

        # 1·2. 일반 검출과 Bubble 검출을 병렬 실행 (고정 비용 절반 수준으로 단축)
        import concurrent.futures
        valid_contours = []
        bubble_contours = []
        bubble_debug = None

        def run_general():
            if not getattr(self, "_general_enabled", True):
                return [], None
            c, closed = detector.detect_static(
                gray,
                threshold=self._threshold,
                min_area=self._min_area,
                use_adaptive=self._use_adaptive,
                debug_dir=None,
                detect_bubbles=False,
                open_kernel=getattr(self, "_open_kernel", 2),
                close_kernel=getattr(self, "_close_kernel", 3),
            )
            return c, closed

        def run_bubble_only():
            if not do_bubble:
                return [], None
            return detector.detect_bubbles(gray)

        if not hasattr(self, '_detect_pool'):
            self._detect_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        fut_gen = self._detect_pool.submit(run_general)
        fut_bub = self._detect_pool.submit(run_bubble_only)
        valid_contours, closed = fut_gen.result()
        bubble_contours, bubble_debug = fut_bub.result()

        if not getattr(self, "_general_enabled", True):
            detector.last_debug = {"gray": gray}
        if bubble_debug and getattr(detector, "last_debug", None) is not None:
            detector.last_debug["bubble_debug"] = bubble_debug

        t_detect = time.perf_counter()
        t_bubble = t_detect
                
        # 3. 우선순위 병합 (Bubble 우선)
        if bubble_contours and valid_contours:
            merged = detector._merge_contours(bubble_contours, valid_contours)
            non_overlap_valid = merged[len(bubble_contours):]
        else:
            non_overlap_valid = valid_contours

        # 모든 contour 합치기: bubble + 일반 (중복 제거 후)
        final_contours = list(bubble_contours) + list(non_overlap_valid)

        t_merge = time.perf_counter()

        # 4. 분류
        use_dl = (classifier and hasattr(classifier, 'use_deep_learning')
                  and classifier.use_deep_learning
                  and hasattr(classifier, 'dl_classifier')
                  and classifier.dl_classifier.is_loaded())

        n_bubble = len(bubble_contours)
        n_gen = len(non_overlap_valid)

        if use_dl and final_contours:
            # DL 모드: 전체 contour (bubble+일반) 배치 추론
            final_results = classifier.classify_batch(final_contours, frame)
            from collections import Counter
            label_counts = Counter(r.get("label", "?") for r in final_results)
            print(f"[DL 분류] 전체 {len(final_contours)}개 (Bub:{n_bubble}+Gen:{n_gen}) → {dict(label_counts)}")
        else:
            # RuleBase 모드: bubble은 직접 라벨, 일반은 RuleBase 분류
            final_results = []
            for cnt in bubble_contours:
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                circ = 4.0 * np.pi * area / (peri * peri) if peri > 0 else 0
                final_results.append({
                    "label": "Bubble", "confidence": 1.0,
                    "area": float(area), "circularity": circ, "aspect_ratio": 1.0,
                })
            if non_overlap_valid and classifier:
                gen_results = classifier.classify_batch(non_overlap_valid, frame)
                final_results.extend(gen_results)

        self._rule_detections = final_results

        t_classify = time.perf_counter()

        last_debug = getattr(detector, "last_debug", None)
        draw_on_main = self._draw_on_mainview and final_contours
        if draw_on_main:
            display_frame = frame.copy()
            # 라벨별 contour 그룹화 → 한 번에 그리기
            COLOR_MAP = {
                "Bubble": (0, 255, 0),
                "Particle": (0, 0, 255),
                "Noise_Dust": (255, 0, 0),
            }
            DEFAULT_COLOR = (0, 255, 0)
            groups = {}
            for i, cnt in enumerate(final_contours):
                label = final_results[i]["label"]
                groups.setdefault(label, []).append(cnt)
            for label, cnts in groups.items():
                color = COLOR_MAP.get(label, DEFAULT_COLOR)
                cv2.drawContours(display_frame, cnts, -1, color, 1)
            display_base_frame = display_frame.copy()
        else:
            display_frame = frame.copy()
            display_base_frame = display_frame
        raw_frame = frame

        t_draw = time.perf_counter()
        total = t_draw - t_start

        tact_times = {
            "prep": t_prep - t_start,
            "detect": t_detect - t_prep,
            "bubble": t_bubble - t_detect,
            "merge": t_merge - t_bubble,
            "classify": t_classify - t_merge,
            "draw": t_draw - t_classify,
            "total": total,
            "n_contours": len(final_contours),
            "n_bubble": n_bubble,
            "n_gen_classify": n_gen,
        }

        self.result_ready.emit(final_contours, display_frame, display_base_frame, last_debug, raw_frame, self._full_frame_bgr, tact_times)

    def _run_yolo(self, frame):
        """YOLO 통합 검출+분류 파이프라인."""
        conf = getattr(self, "_yolo_conf", 0.15)
        detections = self._yolo_detector.detect(frame, conf_threshold=conf)
        self._yolo_detections = list(detections)

        contours = [d["contour"] for d in detections]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        last_debug = {"gray": gray.copy()}

        draw_on_main = self._draw_on_mainview and detections
        if draw_on_main:
            display_frame = frame.copy()
            display_base_frame = frame.copy()
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                color = (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
                cv2.rectangle(display_base_frame, (x1, y1), (x2, y2), color, 1)
                label_text = f"{d['label']} {d['confidence']:.2f}"
                cv2.putText(display_frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display_base_frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            display_frame = frame.copy()
            display_base_frame = frame.copy()

        raw_frame = frame.copy()
        self.result_ready.emit(contours, display_frame, display_base_frame, last_debug, raw_frame, self._full_frame_bgr, {"total": 0, "yolo": 0})


class MainWindow(QMainWindow):
    # Maker 모드 비밀번호 해시 (SHA-256)
    _MAKER_HASH = "338b7af5ca7ccae82dfac6f2f764de033dc521aa4cbf7c981539502eaef25c79"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vial Foreign Body Inspection [User]")
        self.resize(1200, 800)
        self._set_window_icon()
        self._maker_mode = False  # User/Maker 모드 플래그
        self._viewer_mode = False  # Viewer 모드: 좌측 Debug/Defect 패널 숨김, Main View 확대
        self._layout_warmup_done = False  # 표시 후 1회 레이아웃 warm-up 실행 여부

        self.camera = None
        self.detector = ForeignBodyDetector()
        self.classifier = ParticleClassifier()
        self.yolo_detector = YOLODetector()
        self.use_yolo_mode = False  # True=YOLO 통합 검출, False=Threshold+분류
        self.open_kernel = 2
        self.close_kernel = 3
        self.classification_enabled = True
        self.general_enabled = True
        self.bubble_show_text = True
        self._defect_saver = None          # cached DefectImageSaver
        self._save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_inspecting = False
        self._worker_busy = False
        self._detection_worker = DetectionWorker(self)
        self._detection_worker.result_ready.connect(self._on_detection_result)
        self._detection_worker.finished.connect(self._on_worker_finished)
        self._display_frame_clean = None
        self.current_contours = []
        self.current_frame_bgr = None           # Original frame for defect ROI extraction
        self.original_frame_full = None         # Full resolution original (no processing)
        self.current_display_frame = None       # full-res with overlays
        self.display_base_frame = None          # base frame with all contours (no selection highlight)
        self.selected_contour_idx = -1          # Currently selected contour index
        self.zoom_factor = 1.0
        self.zoom_min = 0.2
        self.zoom_max = 1000.0  # 1:1 픽셀 확대까지 허용 (실제 상한은 render 시 뷰포트 기준 계산)
        self._zoom_before_1to1 = None  # 휠 더블클릭으로 1:1 전 전 배율 복원용
        self.view_cx = None
        self.view_cy = None
        self.is_panning = False
        self.pan_last_pos = QPointF(0, 0)
        self.pan_start_pos = QPointF(0, 0)
        self.current_gray = None  # Grayscale image for GrayValue readout
        self.inspection_roi = None  # (x, y, w, h) in image coords or None
        self.is_roi_drawing = False
        self.roi_drag_start = None  # (x, y) image coords
        self.roi_drag_end = None
        self.is_annotation_roi_drawing = False
        self.annotation_roi_start = None
        self.annotation_roi_end = None
        self._last_roi_offset = None  # (rx, ry) when last detection was on cropped ROI
        self._timer_was_paused = False  # 탭 전환으로 타이머 정지 여부
        self._timer_interval = 33      # 타이머 간격 백업
        self._user_paused = False      # 사용자 Pause 여부 (Resume/Stop 시 False)
        self._last_annotation_label = None  # 어노테이션 ROI 다이얼로그에서 마지막 선택한 라벨

        self.init_ui()
        self._load_settings()

    DEFECT_SAVE_DIR = r"D:\1. Projects\3. 2025\8. 검토\3. 바이오시료 이물검사\DefectImages"

    def init_ui(self):
        # === 메뉴 ===
        menubar = self.menuBar()
        menu_settings = menubar.addMenu("설정")
        act_rule_params = QAction("검사 파라미터 설정", self)
        act_rule_params.triggered.connect(self._open_rule_params_dialog)
        menu_settings.addAction(act_rule_params)
        menu_help = menubar.addMenu("도움말")
        act_restart = QAction("재시작 (업데이트 적용)", self)
        act_restart.setShortcut(QKeySequence("Ctrl+Shift+R"))
        act_restart.triggered.connect(self._restart_application)
        menu_help.addAction(act_restart)

        # === 탭 위젯 ===
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # 설정/도움말(메뉴 바)과 같은 줄에 로고 + NOVAnix-FVW101 오버레이
        self._tab_branding = QWidget(self)
        self._tab_branding.setStyleSheet("background: transparent;")
        self._tab_branding.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        branding_layout = QHBoxLayout(self._tab_branding)
        branding_layout.setContentsMargins(0, 0, 0, 0)
        branding_layout.setSpacing(10)
        self.header_logo_label = QLabel()
        self.header_logo_label.setScaledContents(False)
        self.header_logo_label.setStyleSheet("background: transparent;")
        logo_path = self._logo_path()
        if logo_path and os.path.isfile(logo_path):
            pix = QPixmap(logo_path)
            if not pix.isNull():
                self._logo_pixmap_orig = pix
                self.header_logo_label.setPixmap(pix.scaledToHeight(56, Qt.TransformationMode.SmoothTransformation))
        branding_layout.addWidget(self.header_logo_label)
        self.header_title_label = QLabel("NOVAnix-DVI")
        self.header_title_label.setStyleSheet("color: white; font-size: 45px; font-weight: normal; background: transparent;")
        branding_layout.addWidget(self.header_title_label)
        self._tab_branding.adjustSize()
        self._tab_branding.show()
        self.installEventFilter(self)
        self.tab_widget.tabBar().installEventFilter(self)
        QTimer.singleShot(50, self._update_tab_branding_pos)

        # --- Main Tab ---
        main_tab = QWidget()
        self.main_tab = main_tab
        main_layout = QHBoxLayout(main_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout = main_layout
        self.tab_widget.addTab(main_tab, "Main (검사)")

        # --- Classification Tab ---
        self.classification_tab = ClassificationTab()
        # 학습 완료 시 메인 윈도우의 모델 로드 함수 자동 호출
        self.classification_tab.model_trained_signal.connect(self._load_dl_model)
        self.tab_widget.addTab(self.classification_tab, "Classification (학습)")

        # --- YOLO Tab --- (Maker 모드에서만 표시)
        self.yolo_tab = YOLOAnnotationTab()
        self._yolo_tab_title = "YOLO (어노테이션/학습)"
        self._yolo_tab_index = self.tab_widget.addTab(self.yolo_tab, self._yolo_tab_title)

        # Tab 키로 탭 전환
        sc_tab = QShortcut(QKeySequence(Qt.Key.Key_Tab), self)
        sc_tab.activated.connect(self._toggle_tab)
        # Main 탭에서만: R = 어노테이션 ROI, Ctrl+R = 검사 ROI
        sc_annot_roi = QShortcut(QKeySequence(Qt.Key.Key_R), self)
        sc_annot_roi.activated.connect(self._shortcut_annotation_roi)
        sc_inspect_roi = QShortcut(QKeySequence("Ctrl+R"), self)
        sc_inspect_roi.activated.connect(self._shortcut_inspection_roi)
        sc_space_pause = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        sc_space_pause.activated.connect(self._shortcut_space_pause)
        # 탭 전환 시 포커스 전달 + 영상 타이머 관리
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Left Side: Image Display
        self.image_label = QLabel("Camera Feed / Image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        self.image_label.setScaledContents(False)  # keep pixmap size; use scroll for large
        # 로딩된 이미지/동영상 크기로 레이아웃 최소 높이가 커지지 않도록 축소 허용
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        # Scroll area to prevent window resizing with large images
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_label)
        self.image_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll.setMinimumHeight(0)
        # Capture wheel events for zoom control
        self.image_scroll.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)
        # Require mouse tracking so MouseMove is received without button press
        self.image_scroll.viewport().setMouseTracking(True)
        self.image_label.setMouseTracking(True)

        # Debug panel (left): Static Box 크기는 260 유지, 이미지만 230/238로 표시 + 이미지 간 여백
        self.debug_labels = {}
        debug_panel = QWidget()
        self._debug_panel = debug_panel
        debug_panel_layout = QVBoxLayout(debug_panel)
        debug_panel_layout.setContentsMargins(4, 4, 4, 4)
        debug_panel_layout.setSpacing(12)

        _cell_sz = 260
        debug_group = QGroupBox("Debug Image")
        debug_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        self.debug_tabs = QTabWidget()
        debug_group_layout = QVBoxLayout(debug_group)
        debug_group_layout.setContentsMargins(4, 12, 4, 4)
        debug_group_layout.addWidget(self.debug_tabs)
        
        # 1. 일반 검출 탭
        normal_tab = QWidget()
        normal_layout = QGridLayout(normal_tab)
        normal_layout.setContentsMargins(12, 12, 12, 12)
        normal_layout.setSpacing(12)
        for key, title, row, col in [("gray", "gray", 0, 0), ("blurred", "blurred", 0, 1), 
                                     ("threshold", "threshold", 1, 0), ("closed", "Morphology", 1, 1)]:
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: #222; color: #888; font-size: 10px;")
            lbl.setFixedSize(_cell_sz, _cell_sz)
            lbl.setMinimumSize(_cell_sz, _cell_sz)
            self.debug_labels[key] = lbl
            normal_layout.addWidget(lbl, row, col)
        
        self.debug_tabs.addTab(normal_tab, "일반 검출")
        
        # 2. 버블 검출 탭
        bubble_tab = QWidget()
        bubble_layout = QGridLayout(bubble_tab)
        bubble_layout.setContentsMargins(12, 12, 12, 12)
        bubble_layout.setSpacing(12)
        for key, title, row, col in [("clahe", "CLAHE / Flat", 0, 0), ("diff_map", "DoG Diff", 0, 1), 
                                     ("binary", "MAD Binary", 1, 0), ("bubble_candidates", "Bubble Result", 1, 1)]:
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: #222; color: #888; font-size: 10px;")
            lbl.setFixedSize(_cell_sz, _cell_sz)
            lbl.setMinimumSize(_cell_sz, _cell_sz)
            self.debug_labels[key] = lbl
            bubble_layout.addWidget(lbl, row, col)
            
        self.debug_tabs.addTab(bubble_tab, "버블 검출")
        debug_panel_layout.addWidget(debug_group)

        defect_group = QGroupBox("Defect View")
        defect_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        defect_group.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        defect_group.setFixedSize(286, 296)
        defect_layout = QVBoxLayout(defect_group)
        defect_layout.setContentsMargins(12, 12, 14, 14)
        self.lbl_defect_view = QLabel("Select a defect")
        self.lbl_defect_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_defect_view.setStyleSheet("background-color: black; color: white;")
        self.lbl_defect_view.setFixedSize(_cell_sz, _cell_sz)
        self.lbl_defect_view.setMinimumSize(_cell_sz, _cell_sz)
        self.lbl_defect_view.setScaledContents(False)
        defect_layout.addWidget(self.lbl_defect_view, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        debug_panel_layout.addWidget(defect_group)

        debug_panel.setFixedWidth(556)  # 2*260 + spacing 12 + margins 12*2
        debug_panel.setMinimumHeight(0)
        debug_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Ignored)

        # Mouse position / GrayValue display (original scale) and Defect Counts
        info_layout = QHBoxLayout()
        self.lbl_mouse_info = QLabel("Position: —  GrayValue: —")
        self.lbl_mouse_info.setStyleSheet("padding: 4px; font-family: monospace;")
        self.lbl_defect_counts = QLabel("")
        self.lbl_defect_counts.setStyleSheet("padding: 4px; font-family: monospace; color: #ff9900;")
        self.lbl_defect_counts.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.lbl_tact_info = QLabel("Tact: -")
        self.lbl_tact_info.setStyleSheet("padding: 4px; font-family: monospace; color: #00ff00; margin-left: 10px;")
        self.lbl_tact_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        info_layout.addWidget(self.lbl_mouse_info)
        info_layout.addStretch()
        info_layout.addWidget(self.lbl_defect_counts)
        info_layout.addWidget(self.lbl_tact_info)
        self._info_layout = info_layout
        self._info_row_widget = QWidget()
        self._info_row_widget.setLayout(info_layout)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.image_scroll)
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(1)
        self.video_slider.setValue(0)
        self.video_slider.setToolTip("동영상 재생 위치 (드래그하여 이동)")
        self.video_slider.hide()
        self._video_slider_block = False
        self._video_slider_just_sought = False  # 사용자가 방금 시크했으면 한 프레임 동안 슬라이더 동기화 스킵
        self.video_slider.sliderReleased.connect(self._on_video_slider_released)
        left_layout.addWidget(self.video_slider)
        # 미리보기 하단: 파일 정보 (학습창과 유사하게 추가)
        self.lbl_file_info = QLabel("")
        self.lbl_file_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 배율과 폰트 크기는 메인 윈도우 크기에 맞춰 조정 (27px은 너무 클 수 있으므로 22px 권장)
        self.lbl_file_info.setStyleSheet("color: #888; font-size: 22px; padding: 4px;")
        self.lbl_file_info.setFixedHeight(45)
        left_layout.addWidget(self.lbl_file_info)
        left_layout.addWidget(self._info_row_widget)
        # Accept drag/drop on main and child widgets
        self.setAcceptDrops(True)
        main_tab.setAcceptDrops(True)
        self.image_scroll.setAcceptDrops(True)
        self.image_scroll.viewport().setAcceptDrops(True)
        self.image_label.setAcceptDrops(True)
        # Also listen for drag events on these widgets
        main_tab.installEventFilter(self)
        self.image_scroll.installEventFilter(self)

        # Right Side: Controls
        control_panel = QWidget()
        self.control_panel = control_panel
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)
        control_panel.setAcceptDrops(True)
        control_panel.installEventFilter(self)

        # Source Selection (한 줄 줄여서 높이 축소)
        source_group = QGroupBox("Basler Cam Control")
        source_layout = QVBoxLayout()
        source_layout.setSpacing(4)
        source_layout.setContentsMargins(8, 8, 8, 8)
        connect_row = QHBoxLayout()
        self.btn_connect_cam = QPushButton("Connect")
        self.btn_connect_cam.clicked.connect(self.connect_camera)
        connect_row.addWidget(self.btn_connect_cam)
        self.chk_realtime_view = QCheckBox("RealTime View")
        self.chk_realtime_view.setChecked(True)
        self.chk_realtime_view.setToolTip("체크: 카메라에서 연속 수신. 해제: Start Inspection 시 1회만 Grab 후 검사.")
        self.chk_realtime_view.toggled.connect(self._on_realtime_view_toggled)
        connect_row.addWidget(self.chk_realtime_view)
        source_layout.addLayout(connect_row)
        self.btn_load_file = QPushButton("Load Image / Video")
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_pause_resume = QPushButton()
        self.btn_pause_resume.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.btn_pause_resume.setToolTip("영상 일시 정지 / 재개")
        self.btn_pause_resume.clicked.connect(self._on_pause_resume)
        self.btn_pause_resume.setEnabled(False)
        self.btn_pause_resume.setFixedSize(32, 32)
        self.btn_stop_play = QPushButton()
        self.btn_stop_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.btn_stop_play.setToolTip("영상 정지 후 처음부터 다시 재생")
        self.btn_stop_play.clicked.connect(self._on_stop_play)
        self.btn_stop_play.setEnabled(False)
        self.btn_stop_play.setFixedSize(32, 32)

        basler_opt_row = QHBoxLayout()
        self.btn_basler_settings = QPushButton("Cam 설정")
        self.btn_basler_settings.clicked.connect(self._open_basler_settings)
        self.btn_basler_settings.setEnabled(False)
        self.btn_basler_settings.setFixedWidth(80)

        self.btn_grab_image = QPushButton("Grab&Save")
        self.btn_grab_image.clicked.connect(self._on_grab_image)
        self.btn_grab_image.setEnabled(False)
        self.btn_grab_image.setFixedWidth(80)

        self.chk_save_video = QCheckBox("영상 저장")
        self.chk_save_video.setEnabled(False)
        self.chk_save_video.setToolTip("검사 시작 시 영상 녹화, 중단 시 파일 저장")

        basler_opt_row.addWidget(self.btn_basler_settings)
        basler_opt_row.addWidget(self.btn_grab_image)
        basler_opt_row.addWidget(self.chk_save_video)
        source_layout.addLayout(basler_opt_row)
        
        load_row = QHBoxLayout()
        load_row.addWidget(self.btn_load_file)
        load_row.addWidget(self.btn_pause_resume)
        load_row.addWidget(self.btn_stop_play)
        source_layout.addLayout(load_row)
        source_group.setLayout(source_layout)
        control_layout.addWidget(source_group)
        self._viewer_basler_group = source_group

        # Inspection Controls
        insp_group = QGroupBox("Inspection")
        insp_layout = QVBoxLayout()
        self.btn_start = QPushButton("Start Inspection")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self.toggle_inspection)
        insp_layout.addWidget(self.btn_start)
        insp_group.setLayout(insp_layout)
        control_layout.addWidget(insp_group)
        self._viewer_inspection_group = insp_group

        # Viewer 모드용 Position/GrayValue/Tact 복제 라벨 (레이아웃 이동 없이 show/hide만)
        self._viewer_info_widget = QWidget()
        _vi_layout = QVBoxLayout(self._viewer_info_widget)
        _vi_layout.setContentsMargins(4, 4, 4, 4)
        _vi_layout.setSpacing(2)
        self._viewer_lbl_mouse = QLabel("Position: —  GrayValue: —")
        self._viewer_lbl_mouse.setStyleSheet("padding: 2px; font-family: monospace; font-size: 15px;")
        self._viewer_lbl_mouse.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._viewer_lbl_tact = QLabel("Tact: -")
        self._viewer_lbl_tact.setStyleSheet("padding: 2px; font-family: monospace; color: #00ff00; font-size: 15px;")
        self._viewer_lbl_tact.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _vi_layout.addWidget(self._viewer_lbl_mouse)
        _vi_layout.addWidget(self._viewer_lbl_tact)
        control_layout.addWidget(self._viewer_info_widget)
        self._viewer_info_widget.setVisible(False)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        # Threshold with value label
        threshold_label_layout = QHBoxLayout()
        threshold_label_layout.addWidget(QLabel("Threshold:"))
        self.lbl_thresh_value = QLabel("100")
        self.lbl_thresh_value.setStyleSheet("font-weight: bold; color: white;")
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
        self.lbl_area_value.setStyleSheet("font-weight: bold; color: white;")
        area_label_layout.addStretch()
        area_label_layout.addWidget(self.lbl_area_value)
        settings_layout.addLayout(area_label_layout)
        
        self.slider_area = QSlider(Qt.Orientation.Horizontal)
        self.slider_area.setRange(0, 500)
        self.slider_area.setValue(10)
        self.slider_area.valueChanged.connect(lambda v: self.lbl_area_value.setText(str(v)))
        self.slider_area.valueChanged.connect(self._save_settings)
        self.slider_thresh.valueChanged.connect(self._save_settings)
        settings_layout.addWidget(self.slider_area)

        chk_row = QHBoxLayout()
        self.chk_adaptive = QCheckBox("Adaptive Threshold")
        self.chk_adaptive.setChecked(True)
        self.chk_adaptive.toggled.connect(self._save_settings)
        chk_row.addWidget(self.chk_adaptive)
        self.chk_draw_on_mainview = QCheckBox("MainView에 표시")
        self.chk_draw_on_mainview.setChecked(True)
        self.chk_draw_on_mainview.setToolTip("체크 시 MainView에 Defect 사각형·명칭을 그립니다.")
        self.chk_draw_on_mainview.toggled.connect(self._on_draw_on_mainview_toggled)
        chk_row.addWidget(self.chk_draw_on_mainview)
        chk_row.addStretch()
        settings_layout.addLayout(chk_row)

        self.btn_roi = QPushButton("검사 ROI 설정")
        self.btn_roi.clicked.connect(self._on_roi_button_clicked)
        settings_layout.addWidget(self.btn_roi)
        self.btn_annotation_roi = QPushButton("어노테이션 ROI 설정")
        self.btn_annotation_roi.setToolTip("드래그하여 박스를 그리면, 중앙을 기준으로 128x128 크롭 이미지가 학습 데이터로 저장됩니다.")
        self.btn_annotation_roi.clicked.connect(self._on_annotation_roi_button_clicked)
        settings_layout.addWidget(self.btn_annotation_roi)

        self.chk_show_roi = QCheckBox("검사 ROI표시")
        self.chk_show_roi.setChecked(True)
        self.chk_show_roi.setToolTip("체크 시 MainView에 설정된 검사 ROI를 초록 사각형으로 표시합니다.")
        self.chk_show_roi.toggled.connect(lambda: self.render_main_view())

        settings_layout.addWidget(self.chk_show_roi)

        # 검사 모드 선택: Threshold+분류 vs YOLO  (Maker 전용)
        self._yolo_widgets = []  # Maker 모드에서만 보이는 위젯 목록

        self._mode_container = QWidget()
        _mode_lay = QHBoxLayout(self._mode_container)
        _mode_lay.setContentsMargins(0, 0, 0, 0)
        _mode_lay.addWidget(QLabel("검사 모드:"))
        self.combo_inspect_mode = QComboBox()
        self.combo_inspect_mode.addItems(["Threshold + 분류", "YOLO 통합 검출"])
        self.combo_inspect_mode.setCurrentIndex(0)
        self.combo_inspect_mode.currentIndexChanged.connect(self._on_inspect_mode_changed)
        _mode_lay.addWidget(self.combo_inspect_mode)
        settings_layout.addWidget(self._mode_container)
        self._yolo_widgets.append(self._mode_container)

        # DeepLearning 분류 옵션 (Threshold 모드에서만 의미 있음)
        dl_row = QHBoxLayout()
        self.chk_use_dl = QCheckBox("Use Classification")
        self.chk_use_dl.setChecked(False)
        self.chk_use_dl.setToolTip("학습된 딥러닝 모델로 분류합니다. (Threshold 모드 전용)")
        self.chk_use_dl.toggled.connect(self._on_use_dl_toggled)
        dl_row.addWidget(self.chk_use_dl)
        self.btn_load_model = QPushButton("모델 로드")
        self.btn_load_model.setFixedWidth(80)
        self.btn_load_model.clicked.connect(self._on_load_model)
        dl_row.addWidget(self.btn_load_model)
        self.lbl_dl_device = QLabel("—")
        self.lbl_dl_device.setToolTip("Classification 추론 디바이스 (모델 로드 후 표시)")
        self.lbl_dl_device.setStyleSheet("color: #666; font-size: 11px;")
        dl_row.addWidget(self.lbl_dl_device)
        dl_row.addStretch()
        settings_layout.addLayout(dl_row)

        # YOLO 모델 로드 (Maker 전용)
        self._yolo_load_container = QWidget()
        _yolo_lay = QHBoxLayout(self._yolo_load_container)
        _yolo_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_load_yolo = QPushButton("YOLO 모델 로드")
        self.btn_load_yolo.setToolTip("학습된 YOLO 모델(.pt)을 로드합니다.")
        self.btn_load_yolo.clicked.connect(self._on_load_yolo_model)
        _yolo_lay.addWidget(self.btn_load_yolo)
        self.lbl_yolo_status = QLabel("미로드")
        self.lbl_yolo_status.setStyleSheet("color: #888; font-size: 11px;")
        _yolo_lay.addWidget(self.lbl_yolo_status)
        _yolo_lay.addStretch()
        settings_layout.addWidget(self._yolo_load_container)
        self._yolo_widgets.append(self._yolo_load_container)

        # YOLO confidence (Maker 전용)
        self._yolo_conf_container = QWidget()
        _yconf_lay = QHBoxLayout(self._yolo_conf_container)
        _yconf_lay.setContentsMargins(0, 0, 0, 0)
        _yconf_lay.addWidget(QLabel("YOLO Confidence:"))
        self.spin_yolo_conf = QDoubleSpinBox()
        self.spin_yolo_conf.setRange(0.01, 0.99)
        self.spin_yolo_conf.setDecimals(2)
        self.spin_yolo_conf.setSingleStep(0.05)
        self.spin_yolo_conf.setValue(0.15)
        self.spin_yolo_conf.setToolTip("이 값 이상인 검출만 표시. 검출이 안 보이면 0.1 이하로 낮춰 보세요.")
        self.spin_yolo_conf.setEnabled(False)  # YOLO 모드일 때만 활성화
        _yconf_lay.addWidget(self.spin_yolo_conf)
        _yconf_lay.addStretch()
        settings_layout.addWidget(self._yolo_conf_container)
        self._yolo_widgets.append(self._yolo_conf_container)

        # Defect 이미지 저장 옵션
        self.chk_save_defects = QCheckBox("Defect 이미지 저장")
        self.chk_save_defects.setChecked(False)
        self.chk_save_defects.setToolTip("검출된 Defect을 라벨별 폴더에 BMP로 저장합니다.")
        settings_layout.addWidget(self.chk_save_defects)

        settings_group.setLayout(settings_layout)
        control_layout.addWidget(settings_group)
        self._viewer_settings_group = settings_group

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.lbl_status = QLabel("Status: WAIT")
        self.lbl_status.setStyleSheet("font-size: 16px; font-weight: bold; color: gray;")
        
        self.chk_show_text = QCheckBox("글자 보기")
        self.chk_show_text.setChecked(True)
        self.chk_show_text.setToolTip("체크 시 검출된 이물 주변에 라벨 글자를 표시합니다.")
        self.chk_show_text.toggled.connect(self._on_draw_on_mainview_toggled)
        
        status_row = QHBoxLayout()
        status_row.addWidget(self.lbl_status)
        status_row.addStretch()
        status_row.addWidget(self.chk_show_text)
        results_layout.addLayout(status_row)
        # 라디오 버튼 필터: 지정된 라벨만 표시
        result_filter_row = QHBoxLayout()
        result_filter_row.addWidget(QLabel("표시:"))
        
        self.radio_result_all = QRadioButton("ALL")
        self.radio_result_all.setChecked(True)
        result_filter_row.addWidget(self.radio_result_all)
        
        self.radio_result_particle = QRadioButton("Particle")
        result_filter_row.addWidget(self.radio_result_particle)
        
        self.radio_result_noise = QRadioButton("Noise")
        result_filter_row.addWidget(self.radio_result_noise)
        
        self.radio_result_bubble = QRadioButton("Bubble")
        result_filter_row.addWidget(self.radio_result_bubble)
        
        result_filter_row.addStretch()
        results_layout.addLayout(result_filter_row)
        
        for radio in (self.radio_result_all, self.radio_result_particle, self.radio_result_noise, self.radio_result_bubble):
            radio.toggled.connect(self._on_result_filter_toggled)
        self._result_row_to_contour_index = []  # list row → original contour index (필터 시)
        self.current_classify_results = []  # 최근 검사 분류 결과 (label 등)

        self.list_results = QListWidget()
        self.list_results.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_results.setStyleSheet(
            "QListWidget::item:selected { background-color: #0078d7; color: white; }"
        )
        self.list_results.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_results.itemDoubleClicked.connect(self.on_item_double_clicked)
        results_layout.addWidget(self.list_results)
        results_group.setLayout(results_layout)
        control_layout.addWidget(results_group, 1)
        self._viewer_results_group = results_group

        # Splitter: debug panel | main view | control panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter
        splitter.addWidget(debug_panel)
        splitter.addWidget(left_widget)
        splitter.addWidget(control_panel)
        splitter.setStretchFactor(0, 0)   # debug panel fixed
        splitter.setStretchFactor(1, 4)   # main view
        splitter.setStretchFactor(2, 1)   # control panel
        # 세로 최소 높이 전파 차단 (Viewer 전환 시 status bar 밀림 방지)
        left_widget.setMinimumHeight(0)
        left_widget.setSizePolicy(left_widget.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        control_panel.setMinimumHeight(0)
        control_panel.setSizePolicy(control_panel.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        # 탭 내부 스플리터 초기 사이즈 명시 (debug_panel 폭 보장)
        splitter.setSizes([556, 600, 300])
        splitter.setCollapsible(0, True)  # Viewer 모드 시 폭 0으로 접기 허용
        splitter.setAcceptDrops(True)
        splitter.installEventFilter(self)

        main_layout.addWidget(splitter)
        self._main_splitter = splitter

        # exe/프로젝트 루트에 rule_params.json 있으면 RuleBase + Bubble 파라미터 자동 로드
        try:
            _rule_params_path = os.path.join(self._app_root(), "rule_params.json")
        except Exception:
            _rule_params_path = None
        if _rule_params_path and os.path.isfile(_rule_params_path):
            try:
                with open(_rule_params_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cls_data = data.get("classification", data)
                self.classifier.rule_based.set_params(cls_data)
                self.slider_thresh.blockSignals(True)
                self.slider_area.blockSignals(True)
                self.chk_adaptive.blockSignals(True)
                try:
                    if "threshold" in cls_data:
                        self.slider_thresh.setValue(int(cls_data["threshold"]))
                    if "min_area" in cls_data:
                        self.slider_area.setValue(int(cls_data["min_area"]))
                    if "use_adaptive" in cls_data:
                        self.chk_adaptive.setChecked(bool(cls_data["use_adaptive"]))
                    if "open_kernel" in cls_data:
                        self.open_kernel = int(cls_data["open_kernel"])
                    if "close_kernel" in cls_data:
                        self.close_kernel = int(cls_data["close_kernel"])
                    if "classification_enabled" in cls_data:
                        self.classification_enabled = bool(cls_data["classification_enabled"])
                    if "general_enabled" in cls_data:
                        self.general_enabled = bool(cls_data["general_enabled"])
                    if "bubble_show_text" in cls_data:
                        self.bubble_show_text = bool(cls_data["bubble_show_text"])
                    if "bubble_detection" in data:
                        self.detector.bubble_params.set_params(data["bubble_detection"])
                        if "bubble_show_text" in data["bubble_detection"]:
                            self.bubble_show_text = bool(data["bubble_detection"]["bubble_show_text"])
                finally:
                    self.slider_thresh.blockSignals(False)
                    self.slider_area.blockSignals(False)
                    self.chk_adaptive.blockSignals(False)
                self.lbl_thresh_value.setText(str(self.slider_thresh.value()))
                self.lbl_area_value.setText(str(self.slider_area.value()))
                dl_data = data.get("deep_learning", {})
                self.classifier.dl_classifier.set_optimization_level(
                    dl_data.get("optimization_level", 0))
                self.classifier.dl_classifier.set_openvino_device(
                    dl_data.get("openvino_device", "AUTO"))
            except Exception as e:
                print(f"[MainWindow] init_ui 설정 로드 실패: {_rule_params_path} — {e}")
        # exe 실행 시 exe 폴더에 classification_model.onnx 있으면 자동 로드
        if hasattr(sys, '_MEIPASS'):
            dl_model_loaded = False
            default_onnx = os.path.join(self._app_root(), "classification_model.onnx")
            if os.path.isfile(default_onnx):
                if self._load_dl_model(default_onnx):
                    dl_model_loaded = True
            
            if not dl_model_loaded:
                default_pth = os.path.join(self._app_root(), "classification_model.pth")
                if os.path.isfile(default_pth):
                    if self._load_dl_model(default_pth):
                        dl_model_loaded = True
            
            if dl_model_loaded:
                self.chk_use_dl.setChecked(True)
        # exe/프로젝트 루트에 best.pt 있으면 YOLO 모델 자동 로드
        for yolo_name in ("best.pt", "yolo_model.pt"):
            _yolo_path = os.path.join(self._app_root(), yolo_name)
            if os.path.isfile(_yolo_path):
                ok, _ = self.yolo_detector.load_model(_yolo_path)
                if ok:
                    self.lbl_yolo_status.setText(f"로드됨 ({len(self.yolo_detector.class_names)}클래스)")
                    self.lbl_yolo_status.setStyleSheet("color: #0f0; font-size: 11px;")
                    break

        # --- Status Bar: 모드 표시, Viewer 토글, Maker 로그인 ---
        status_bar = self.statusBar()
        self._lbl_mode = QLabel("  🔒 User 모드  ")
        self._lbl_mode.setStyleSheet("font-weight: bold; color: #aaa;")
        status_bar.addPermanentWidget(self._lbl_mode)
        self._btn_viewer = QPushButton("👁 Viewer")
        self._btn_viewer.setFixedHeight(22)
        self._btn_viewer.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        self._btn_viewer.setCheckable(True)
        self._btn_viewer.clicked.connect(self._toggle_viewer_mode)
        status_bar.addPermanentWidget(self._btn_viewer)
        self._btn_login = QPushButton("🔐 Maker 로그인")
        self._btn_login.setFixedHeight(22)
        self._btn_login.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        self._btn_login.clicked.connect(self._toggle_maker_mode)
        status_bar.addPermanentWidget(self._btn_login)

        # 초기 모드 적용 (User 모드: YOLO 숨김)
        self._apply_mode()

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        if not self._layout_warmup_done:
            QTimer.singleShot(0, self._post_show_layout_warmup)

    def _post_show_layout_warmup(self):
        """창 표시 후 1회만 실행: Viewer 토글 대상 위젯의 sizeHint를 실제 표시 상태에서 안정화."""
        if self._layout_warmup_done:
            return
        self._layout_warmup_done = True
        _lp = self._main_splitter.widget(0)
        _lp.hide()
        _lp.show()
        self._viewer_settings_group.hide()
        self._viewer_settings_group.show()
        self._viewer_results_group.hide()
        self._viewer_results_group.show()
        self._viewer_info_widget.show()
        self._viewer_info_widget.hide()

    def _on_tab_changed(self):
        """탭 전환 시: 포커스 전달 + Main 탭 벗어나면 영상 타이머 일시 정지."""
        current = self.tab_widget.currentWidget()
        if current is not None:
            current.setFocus()

        if current is self.main_tab:
            # Main 탭 복귀: 타이머가 중지됐었으면 재개
            if self.camera is not None and not self.timer.isActive():
                if hasattr(self, '_timer_was_paused') and self._timer_was_paused:
                    self.timer.start(self._timer_interval)
                    self._timer_was_paused = False
        else:
            # Main 탭 이탈: 영상 타이머 일시 정지
            if self.timer.isActive():
                self._timer_interval = self.timer.interval()
                self._timer_was_paused = True
                self.timer.stop()

    # ═══════════════════════ User / Maker / Viewer 모드 ═══════════════════════

    def _toggle_viewer_mode(self):
        """Viewer 모드 토글: 좌측 Debug Image / Defect View 숨김, Main View 확대."""
        self._viewer_mode = self._btn_viewer.isChecked()
        self._apply_mode()

    def _toggle_maker_mode(self):
        """Maker 모드 로그인/로그아웃 토글."""
        if self._maker_mode:
            # Maker → User 전환 (즉시)
            self._maker_mode = False
            self._apply_mode()
            return

        # User → Maker 전환 (비밀번호 확인)
        pw, ok = QInputDialog.getText(
            self, "Maker 로그인",
            "비밀번호를 입력하세요:",
            QLineEdit.EchoMode.Password  # 입력 마스킹 (●●●●)
        )
        if not ok or not pw:
            return

        pw_hash = hashlib.sha256(pw.encode()).hexdigest()
        if pw_hash == self._MAKER_HASH:
            self._maker_mode = True
            self._apply_mode()
        else:
            QMessageBox.warning(self, "인증 실패", "비밀번호가 올바르지 않습니다.")

    def _apply_mode(self):
        """현재 _maker_mode, _viewer_mode에 따라 YOLO·좌측 패널·상태바 갱신."""
        maker = self._maker_mode
        viewer = self._viewer_mode

        # 1) YOLO 관련 위젯 표시/숨김
        for w in self._yolo_widgets:
            w.setVisible(maker)

        # 2) YOLO 탭 관리
        tab_idx = self.tab_widget.indexOf(self.yolo_tab)
        if maker and tab_idx == -1:
            self.tab_widget.addTab(self.yolo_tab, self._yolo_tab_title)
        elif not maker and tab_idx != -1:
            self.tab_widget.removeTab(tab_idx)
            if self.use_yolo_mode:
                self.combo_inspect_mode.setCurrentIndex(0)

        # 3) Viewer 모드 (좌측 패널은 setVisible 대신 스플리터 폭 0으로만 접기)
        if viewer:
            self._saved_splitter_sizes = self._main_splitter.sizes()
            total = sum(self._saved_splitter_sizes) if self._saved_splitter_sizes else 1456
            right_w = self._saved_splitter_sizes[2] if len(self._saved_splitter_sizes) > 2 else 300
            self._main_splitter.setSizes([0, max(400, total - right_w), right_w])
            if hasattr(self, "_debug_panel"):
                self._debug_panel.setMaximumHeight(0)
            self._viewer_settings_group.setVisible(False)
            self._viewer_results_group.setVisible(True)
            self._viewer_info_widget.setVisible(True)
            # 중앙 하단은 hide/show 대신 높이만 0으로 접어 세로 레이아웃 재계산 부작용 방지
            self._info_row_widget.setMinimumHeight(0)
            self._info_row_widget.setMaximumHeight(0)
            self.lbl_file_info.setMinimumHeight(0)
            self.lbl_file_info.setMaximumHeight(0)
        else:
            saved = getattr(self, "_saved_splitter_sizes", [556, 600, 300])
            self._main_splitter.setSizes(saved)
            if hasattr(self, "_debug_panel"):
                self._debug_panel.setMaximumHeight(16777215)
            self._viewer_settings_group.setVisible(True)
            self._viewer_results_group.setVisible(True)
            self._viewer_info_widget.setVisible(False)
            self._info_row_widget.setMinimumHeight(0)
            self._info_row_widget.setMaximumHeight(16777215)
            self.lbl_file_info.setMinimumHeight(45)
            self.lbl_file_info.setMaximumHeight(45)

        # 4) 타이틀바 갱신
        if viewer:
            mode_str = "Viewer"
        else:
            mode_str = "Maker" if maker else "User"
        self.setWindowTitle(f"Vial Foreign Body Inspection [{mode_str}]")

        # 5) 상태바 갱신
        self._btn_viewer.setChecked(viewer)
        if viewer:
            self._lbl_mode.setText("  👁 Viewer 모드  ")
            self._lbl_mode.setStyleSheet("font-weight: bold; color: #81c784;")
            self._btn_viewer.setText("👁 Viewer 끄기")
        else:
            self._btn_viewer.setText("👁 Viewer")
            if maker:
                self._lbl_mode.setText("  🔓 Maker 모드  ")
                self._lbl_mode.setStyleSheet("font-weight: bold; color: #4fc3f7;")
                self._btn_login.setText("🔒 User로 전환")
            else:
                self._lbl_mode.setText("  🔒 User 모드  ")
                self._lbl_mode.setStyleSheet("font-weight: bold; color: #aaa;")
                self._btn_login.setText("🔐 Maker 로그인")

    # ═══════════════════════════════════════════════════════════════

    def _toggle_tab(self):
        """Tab 키로 Main ↔ Classification 탭 전환."""
        idx = self.tab_widget.currentIndex()
        new_idx = (idx + 1) % self.tab_widget.count()
        self.tab_widget.setCurrentIndex(new_idx)

    def _shortcut_annotation_roi(self):
        """R: Main 탭 = 어노테이션 ROI 토글, Classification 탭 = Brush 초기화."""
        current = self.tab_widget.currentWidget()
        if current is self.main_tab:
            self._on_annotation_roi_button_clicked()
        elif current is self.classification_tab:
            # Classification 탭에서는 어노테이션 ROI 토글 시 기존 브러시 초기화
            self.classification_tab._reset_brush()

    def _shortcut_inspection_roi(self):
        """Ctrl+R: Main 탭에서만 검사 ROI 설정 토글."""
        if self.tab_widget.currentWidget() is not self.main_tab:
            return
        self._on_roi_button_clicked()

    def _shortcut_space_pause(self):
        """Space: Main 탭에서만 Pause/재생 토글."""
        if self.tab_widget.currentWidget() is not self.main_tab:
            return
        if not self._has_playable_source():
            return
        self._on_pause_resume()

    def _is_basler_connected(self):
        return isinstance(self.camera, BaslerCamera) and self.camera.is_connected()

    def _update_basler_ui_state(self):
        """Basler 연결 상태에 따라 버튼 텍스트·설정 버튼 활성화 갱신."""
        is_conn = self._is_basler_connected()
        if is_conn:
            self.btn_connect_cam.setText("Disconnect Basler Camera")
        else:
            self.btn_connect_cam.setText("Connect Basler Camera")
        self.btn_basler_settings.setEnabled(is_conn)
        self.btn_grab_image.setEnabled(is_conn)
        self.chk_save_video.setEnabled(is_conn)

    def _on_grab_image(self):
        """Basler 카메라에서 프레임 1장 Grab 하여 저장"""
        if not self._is_basler_connected() or self.camera is None:
            return
        frame = self.camera.grab_frame()
        if frame is None:
            QMessageBox.warning(self, "Grab 실패", "프레임을 가져올 수 없습니다.")
            return

        import datetime
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        grab_dir = os.path.join(self.DEFAULT_IMAGE_FOLDER, "Basler Grab Image")
        os.makedirs(grab_dir, exist_ok=True)
        fname = os.path.join(grab_dir, f"{now_str}.bmp")
        try:
            # OpenCV의 영어 이외의 경로 저장 에러 방지를 위해 imencode 사용
            ok, buf = cv2.imencode(".bmp", frame)
            if ok:
                buf.tofile(fname)
                print(f"이미지 Grab 완료: {fname}")
            else:
                QMessageBox.warning(self, "저장 오류", "이미지 인코딩에 실패했습니다.")
        except Exception as e:
            QMessageBox.warning(self, "저장 오류", f"Grab 이미지 저장 중 오류 발생:\n{e}")

    def connect_camera(self):
        # 이미 Basler가 연결된 상태면 해제
        if self._is_basler_connected():
            self.timer.stop()
            self.camera.close()
            self.camera = None
            self._user_paused = False
            self.video_slider.hide()
            self.lbl_status.setText("Stopped")
            self._update_basler_ui_state()
            self._update_play_buttons()
            return

        if self.camera:
            self.camera.close()
            self.camera = None
        self.timer.stop()

        self.view_cx = None
        self.view_cy = None
        self.zoom_factor = 1.0

        self.camera = BaslerCamera()
        if self.camera.open():
            self.video_slider.hide()
            self.lbl_status.setText("Camera Connected")
            self._update_basler_ui_state()
            self._user_paused = False
            if self.chk_realtime_view.isChecked():
                self._timer_interval = 33
                self.timer.start(33)
            else:
                self.timer.stop()
            self._update_play_buttons()
        else:
            err = self.camera.get_last_error()
            self.camera = None
            self._update_basler_ui_state()
            self.lbl_status.setText("Camera Error")
            QMessageBox.warning(
                self,
                "Basler 카메라 연결 실패",
                "카메라에 연결할 수 없습니다.\n\n" + err + "\n\n"
                "필요 시:\n"
                "• pip install pypylon\n"
                "• Basler Pylon SDK 설치 (드라이버 포함)\n"
                "• GigE 카메라: 네트워크 IP/서브넷 설정 확인",
            )

    DEFAULT_IMAGE_FOLDER = r"D:\1. Projects\3. 2025\8. 검토\3. 바이오시료 이물검사\영상"
    DEBUG_IMAGE_FOLDER = r"D:\1. Projects\3. 2025\8. 검토\3. 바이오시료 이물검사\영상\DeBugImage"

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Image / Video",
            self.DEFAULT_IMAGE_FOLDER,
            "Images (*.png *.jpg *.jpeg *.bmp);;Videos (*.avi *.mp4 *.mov *.mkv *.wmv);;All (*.*)"
        )
        if fname:
            self.load_image_path(fname)

    def load_image_path(self, path: str):
        if not os.path.isfile(path):
            self.lbl_status.setText("File Error")
            return

        if self.camera:
            self.camera.close()
            self.timer.stop()
        self.video_slider.hide()

        # Reset view state for new image
        self.view_cx = None
        self.view_cy = None
        self.zoom_factor = 1.0
        self.inspection_roi = None
        self.is_roi_drawing = False
        self.image_label.unsetCursor()
        self.roi_drag_start = None
        self.roi_drag_end = None
        self.is_annotation_roi_drawing = False
        self.annotation_roi_start = None
        self.annotation_roi_end = None
        if hasattr(self, "btn_roi"):
            self.btn_roi.setText("검사 ROI 설정")
        if hasattr(self, "btn_annotation_roi"):
            self.btn_annotation_roi.setText("어노테이션 ROI 설정")

        self.camera = FileCamera(path)
        self._update_basler_ui_state()
        if self.camera.open():
            self.lbl_status.setText("Video Loaded" if self.camera.is_video() else "File Loaded")
            if self.camera.is_video():
                n = self.camera.get_frame_count()
                self.video_slider.setMaximum(max(0, n - 1))
                self.video_slider.setValue(0)
                self.video_slider.show()
            else:
                self.video_slider.hide()
            self.update_frame()
            if self.camera.is_video():
                interval = 33
            else:
                interval = 100
            self._timer_interval = interval
            self._user_paused = False
            self.timer.start(interval)
            self._update_play_buttons()
        else:
            self.video_slider.hide()
            self.lbl_status.setText("File Error")
            self._update_play_buttons()

    def _on_realtime_view_toggled(self):
        """RealTime View 체크 시 타이머 가동, 해제 시 타이머 정지 (Basler만)."""
        if not self._is_basler_connected():
            return
        if self.chk_realtime_view.isChecked():
            if not self.timer.isActive():
                self._timer_interval = 33
                self.timer.start(33)
        else:
            self.timer.stop()
        self._update_play_buttons()

    def _has_playable_source(self):
        """영상/실시간 소스가 있고 재생 가능한지 (Pause/Stop/Play 버튼 활성화 조건)."""
        if self.camera is None:
            return False
        if isinstance(self.camera, FileCamera):
            return self.camera.is_video()
        if self._is_basler_connected():
            return self.chk_realtime_view.isChecked()
        return False

    def _update_play_buttons(self):
        """Pause/Resume, Stop/Play 버튼 아이콘·활성화 상태 갱신."""
        style = self.style()
        icon_pause = style.standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        icon_play = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        icon_stop = style.standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        playable = self._has_playable_source()
        running = self.timer.isActive()
        if not playable:
            self.btn_pause_resume.setEnabled(False)
            self.btn_pause_resume.setIcon(icon_pause)
            self.btn_stop_play.setEnabled(False)
            self.btn_stop_play.setIcon(icon_stop)
            return
        self.btn_stop_play.setEnabled(True)
        if running:
            self.btn_pause_resume.setEnabled(True)
            self.btn_pause_resume.setIcon(icon_pause)
            self.btn_stop_play.setIcon(icon_stop)
        elif self._user_paused:
            self.btn_pause_resume.setEnabled(True)
            self.btn_pause_resume.setIcon(icon_play)
            self.btn_stop_play.setIcon(icon_stop)
        else:
            self.btn_pause_resume.setEnabled(False)
            self.btn_pause_resume.setIcon(icon_pause)
            self.btn_stop_play.setIcon(icon_play)

    def _on_pause_resume(self):
        """Pause: 타이머 정지. Resume: 타이머 재시작."""
        if not self._has_playable_source():
            return
        if self.timer.isActive():
            self._timer_interval = self.timer.interval()
            self.timer.stop()
            self._user_paused = True
        else:
            self.timer.start(self._timer_interval)
            self._user_paused = False
        self._update_play_buttons()

    def _on_stop_play(self):
        """Stop: 타이머 정지 + 동영상이면 처음으로. Play: 타이머 재시작."""
        if not self._has_playable_source():
            return
        if self.timer.isActive() or self._user_paused:
            self.timer.stop()
            self._user_paused = False
            if isinstance(self.camera, FileCamera) and self.camera.is_video() and getattr(self.camera, "cap", None) is not None:
                self.camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._update_play_buttons()
        else:
            self.timer.start(self._timer_interval)
            self._update_play_buttons()

    def _open_basler_settings(self):
        if not self._is_basler_connected():
            QMessageBox.warning(self, "오류", "Basler 카메라를 먼저 연결하세요.")
            return
        dlg = BaslerSettingsDialog(self.camera, self)
        dlg.exec()

    def toggle_inspection(self):
        if self.btn_start.isChecked():
            self.is_inspecting = True
            self.btn_start.setText("Stop Inspection")
            
            # --- 영상 저장 셋업 ---
            if getattr(self, "chk_save_video", None) and self.chk_save_video.isChecked() and self._is_basler_connected():
                import datetime
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_dir = os.path.join(self.DEFAULT_IMAGE_FOLDER, "Basler Video File")
                os.makedirs(video_dir, exist_ok=True)
                self._current_video_path = os.path.join(video_dir, f"{now_str}.avi")
                self.video_writer = None  # 첫 프레임 올 때 초기화
            else:
                self._current_video_path = None
                self.video_writer = None

            # 단일 이미지: 1회만 update_frame
            if isinstance(self.camera, FileCamera) and not self.camera.is_video():
                QTimer.singleShot(0, self.update_frame)
            # Basler 단일 Grab: 타이머 정지 후 1회 update_frame
            elif self._is_basler_connected() and not self.chk_realtime_view.isChecked():
                self.timer.stop()
                QTimer.singleShot(0, self.update_frame)
            # 동영상: 재생 중이면 타이머가 update_frame 호출. 멈춤/일시정지 상태면 현재 프레임에서 1회 검사
            elif isinstance(self.camera, FileCamera) and self.camera.is_video() and not self.timer.isActive():
                QTimer.singleShot(0, self.update_frame)
        else:
            self.is_inspecting = False
            self.btn_start.setText("Start Inspection")
            self.lbl_status.setText("Stopped")
            
            # --- 영상 저장 종료 ---
            if getattr(self, "video_writer", None) is not None:
                self.video_writer.release()
                self.video_writer = None
                if getattr(self, "_current_video_path", None):
                    import shutil
                    temp_path = os.path.join(self._app_root(), "temp_video.avi")
                    if os.path.exists(temp_path):
                        try:
                            shutil.move(temp_path, self._current_video_path)
                            print(f"영상 저장 완료: {self._current_video_path}")
                        except Exception as e:
                            print(f"영상 파일 이동 실패: {e}")

    def _on_worker_finished(self):
        """QThread가 완전히 종료된 후 호출. _worker_busy 해제."""
        self._worker_busy = False

    def _on_detection_result(self, contours, display_frame, display_base_frame, last_debug, raw_frame, full_frame, tact_times):
        """Called in main thread when DetectionWorker finishes."""
        try:
            # _worker_busy는 finished signal에서 해제 (QThread 종료 보장)
            self.detector.last_debug = last_debug
            self.update_debug_panel()
            contours = contours if contours else []

            # --- 프레임 및 contour 준비 ---
            if self._last_roi_offset is not None and full_frame is not None:
                rx, ry = self._last_roi_offset
                if contours:
                    contours = [(cnt + np.array([[[rx, ry]]])).astype(np.int32) for cnt in contours]
                self.original_frame_full = full_frame.copy()
                self._display_frame_clean = self.original_frame_full
                display_frame = full_frame.copy()
                display_base_frame = full_frame.copy()
                self.current_gray = cv2.cvtColor(self.original_frame_full, cv2.COLOR_BGR2GRAY) if len(full_frame.shape) == 3 else full_frame
            elif raw_frame is not None:
                self.original_frame_full = raw_frame.copy()
                self._display_frame_clean = self.original_frame_full
                display_frame = display_frame if display_frame is not None else raw_frame.copy()
                display_base_frame = display_base_frame if display_base_frame is not None else raw_frame.copy()
                self.current_gray = cv2.cvtColor(self.original_frame_full, cv2.COLOR_BGR2GRAY) if len(raw_frame.shape) == 3 else raw_frame
            else:
                # 프레임 없음 — 기존 프레임 유지
                if (self.view_cx is None or self.view_cy is None) and self.current_display_frame is not None:
                    h0, w0 = self.current_display_frame.shape[:2]
                    self.view_cx = w0 // 2
                    self.view_cy = h0 // 2
                self.render_main_view()
                return

            # --- 검출+분류 ---
            classify_frame = self.original_frame_full
            # YOLO 모드: DetectionWorker에서 이미 라벨이 포함된 contour를 반환.
            #   _detection_worker._run_yolo()에서 detections을 생성하므로 여기서는 _yolo_detections 참조.
            # Threshold 모드: 기존 classify_batch 호출.
            if self.use_yolo_mode and self.yolo_detector.is_loaded():
                yolo_dets = getattr(self._detection_worker, "_yolo_detections", None)
                if yolo_dets and len(yolo_dets) == len(contours):
                    classify_results = yolo_dets
                else:
                    classify_results = self.yolo_detector.detect(self.original_frame_full) if contours else []
                    if len(classify_results) != len(contours):
                        classify_results = [{"label": "Unknown", "confidence": 0.0,
                                             "area": cv2.contourArea(c), "circularity": 0.0,
                                             "aspect_ratio": 0.0} for c in contours]
            else:
                rule_dets = getattr(self._detection_worker, "_rule_detections", None)
                if rule_dets and len(rule_dets) == len(contours):
                    classify_results = rule_dets
                else:
                    classify_frame = self.original_frame_full
                    classify_results = self.classifier.classify_batch(contours, classify_frame) if contours else []

            # --- MainView에 contour/label 그리기 ---
            if contours and self.chk_draw_on_mainview.isChecked():
                allowed = self._get_result_filter_labels()
                show_text = getattr(self, "chk_show_text", None)
                show_text_val = show_text.isChecked() if show_text else True
                
                for i, cnt in enumerate(contours):
                    r = classify_results[i] if i < len(classify_results) else {"label": "Unknown"}
                    label = r.get("label", "Unknown")
                    
                    if allowed and label not in allowed:
                        continue
                        
                    x, y, w, h = cv2.boundingRect(cnt)
                    if label == "Particle":
                        color = (0, 0, 255)  # Red
                    elif label == "Noise_Dust":
                        color = (255, 0, 0)  # Blue
                    elif label == "Bubble":
                        color = (0, 255, 0)  # Green
                    else:
                        color = (0, 255, 0)  # Green fallback
                    text_color = color
                    
                    cv2.drawContours(display_frame, [cnt], -1, color, 1)
                    cv2.drawContours(display_base_frame, [cnt], -1, color, 1)
                    
                    # 라벨 표시 여부 결정: "글자 보기" 체크박스가 마스터 컨트롤
                    actual_show = show_text_val
                        
                    if actual_show:
                        cv2.putText(display_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                        cv2.putText(display_base_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # --- Defect 이미지 저장 (백그라운드) ---
            if contours and self.chk_save_defects.isChecked() and classify_frame is not None:
                try:
                    source_name = None
                    if isinstance(self.camera, FileCamera) and hasattr(self.camera, 'source_path'):
                        source_path = self.camera.source_path
                        if source_path:
                            source_name = os.path.splitext(os.path.basename(source_path))[0]
                    # Saver 캐싱 (매 프레임마다 _initialize_counters 재스캔 방지)
                    if self._defect_saver is None:
                        self._defect_saver = DefectImageSaver(self.DEFECT_SAVE_DIR)
                    saver = self._defect_saver
                    # 저장할 데이터를 복사하여 백그라운드 스레드에 전달
                    save_frame = classify_frame.copy()
                    save_contours = [cnt.copy() for cnt in contours]
                    save_results = [dict(r) for r in (classify_results if classify_results else [])]
                    def _bg_save(saver, frame, cnts, results, src_name):
                        try:
                            if src_name:
                                saver.save_original(frame, source_name=src_name)
                            for i, cnt in enumerate(cnts):
                                r = results[i] if i < len(results) else {"label": "Unknown"}
                                saver.save(cnt, frame, r["label"], source_name=src_name)
                        except Exception as e:
                            print(f"Defect 이미지 저장 오류 (BG): {e}")
                    self._save_executor.submit(_bg_save, saver, save_frame, save_contours, save_results, source_name)
                except Exception as save_err:
                    print(f"Defect 이미지 저장 준비 오류: {save_err}")

            # --- 파일 정보 레이블 업데이트 (학습창 스타일) ---
            if raw_frame is not None or full_frame is not None:
                # 영상 크기는 항상 원본(전체 프레임) 기준
                ref = full_frame if full_frame is not None else raw_frame
                fh, fw = ref.shape[:2]
                source_name = "Camera (Live)"
                size_str = "-"
                
                if isinstance(self.camera, FileCamera) and hasattr(self.camera, 'source_path'):
                    path = self.camera.source_path
                    if path and os.path.isfile(path):
                        source_name = os.path.basename(path)
                        fsize = os.path.getsize(path)
                        if fsize < 1024 * 1024:
                            size_str = f"{fsize / 1024:.1f} KB"
                        else:
                            size_str = f"{fsize / 1024 / 1024:.1f} MB"
                elif self.camera and self._is_basler_connected():
                    source_name = "Basler Camera"
                
                info_parts = [source_name, f"영상 크기: {fw}×{fh}"]
                if self.inspection_roi is not None:
                    rx, ry, rw, rh = self.inspection_roi
                    info_parts.append(f"검사 ROI: {int(rw)}×{int(rh)}")
                info_parts.append(size_str)
                self.lbl_file_info.setText("  |  ".join(info_parts))

            # --- 상태 업데이트 ---
            self.current_contours = contours
            self.current_frame_bgr = self.original_frame_full
            self.current_display_frame = display_frame
            self.display_base_frame = display_base_frame

            self.current_classify_results = list(classify_results) if classify_results else []
            if contours:
                self.lbl_status.setText("NG: Foreign Body")
                self.lbl_status.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
                self._refresh_results_list()
                # 선택 유지: 이전에 선택된 contour index가 필터 후 리스트에 있으면 해당 행 선택
                try:
                    row = self._result_row_to_contour_index.index(self.selected_contour_idx)
                    self.list_results.setCurrentRow(row)
                except (ValueError, AttributeError):
                    if self._result_row_to_contour_index:
                        self.selected_contour_idx = self._result_row_to_contour_index[0]
                        self.list_results.setCurrentRow(0)
                    else:
                        self.selected_contour_idx = 0
                self.redraw_with_highlight()
            else:
                self.lbl_status.setText("OK")
                self.lbl_status.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
                self.current_classify_results = []
                self._result_row_to_contour_index = []
                self.list_results.clear()
                self.list_results.clearSelection()
                self.clear_defect_view()
            self.update_defect_view()
            
            # --- Tact Time 업데이트 ---
            if tact_times:
                if self.use_yolo_mode:
                     tact_str = f"Tact: {tact_times.get('total', 0)*1000:.0f}ms (YOLO)"
                elif self._maker_mode:
                    p_t = tact_times.get('prep', 0) * 1000
                    d_t = tact_times.get('detect', 0) * 1000
                    b_t = tact_times.get('bubble', 0) * 1000
                    m_t = tact_times.get('merge', 0) * 1000
                    c_t = tact_times.get('classify', 0) * 1000
                    dr_t = tact_times.get('draw', 0) * 1000
                    total_t = tact_times.get('total', 0) * 1000
                    nc = tact_times.get('n_contours', 0)
                    nb = tact_times.get('n_bubble', 0)
                    ng = tact_times.get('n_gen_classify', 0)
                    tact_str = f"Tact: {total_t:.0f}ms (Rule:{d_t:.0f}, Bub:{b_t:.0f}, Class:{c_t:.0f}[{ng}개]) | 검출:{nc} (Bub:{nb}, Gen:{ng})"
                else:
                    total_t = tact_times.get('total', 0) * 1000
                    tact_str = f"Tact: {total_t:.0f}ms"
                self.lbl_tact_info.setText(tact_str)
                self._viewer_lbl_tact.setText(tact_str)
            else:
                self.lbl_tact_info.setText("Tact: -")
                self._viewer_lbl_tact.setText("Tact: -")

            # 이미지 / Basler 단일 Grab / 동영상 멈춤 상태 1회 검사: 완료 후 버튼을 Start로 복귀
            single_shot = False
            if isinstance(self.camera, FileCamera) and not self.camera.is_video():
                single_shot = True
            if self._is_basler_connected() and not self.chk_realtime_view.isChecked():
                single_shot = True
            if isinstance(self.camera, FileCamera) and self.camera.is_video() and not self.timer.isActive():
                single_shot = True
            if single_shot:
                self.timer.stop()
                self.is_inspecting = False
                self.btn_start.setChecked(False)
                self.btn_start.setText("Start Inspection")
            if (self.view_cx is None or self.view_cy is None) and self.current_display_frame is not None:
                h0, w0 = self.current_display_frame.shape[:2]
                self.view_cx = w0 // 2
                self.view_cy = h0 // 2
            self.render_main_view()
        except Exception as e:
            print(f"검사 결과 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            # 에러 발생해도 현재 프레임을 표시 (MainView 멈춤 방지)
            if full_frame is not None:
                self.current_display_frame = full_frame.copy()
                self.display_base_frame = full_frame.copy()
                self.original_frame_full = full_frame.copy()
            elif raw_frame is not None:
                self.current_display_frame = raw_frame.copy()
                self.display_base_frame = raw_frame.copy()
                self.original_frame_full = raw_frame.copy()
            if self.current_display_frame is not None:
                if self.view_cx is None or self.view_cy is None:
                    h0, w0 = self.current_display_frame.shape[:2]
                    self.view_cx = w0 // 2
                    self.view_cy = h0 // 2
                self.render_main_view()

    def _on_inspect_mode_changed(self, index):
        """검사 모드 전환: 0=Threshold+분류, 1=YOLO."""
        self.use_yolo_mode = (index == 1)
        is_threshold = not self.use_yolo_mode
        self.chk_use_dl.setEnabled(is_threshold)
        self.btn_load_model.setEnabled(is_threshold)
        self.slider_thresh.setEnabled(is_threshold)
        self.slider_area.setEnabled(is_threshold)
        self.chk_adaptive.setEnabled(is_threshold)
        self.spin_yolo_conf.setEnabled(self.use_yolo_mode)
        if self.use_yolo_mode and not self.yolo_detector.is_loaded():
            QMessageBox.information(self, "알림",
                "YOLO 모델이 로드되지 않았습니다.\n"
                "'YOLO 모델 로드' 버튼으로 학습된 모델(.pt)을 로드하세요.")

    def _on_load_yolo_model(self):
        """YOLO 모델 파일 로드."""
        start_dir = self._app_root()
        fname, _ = QFileDialog.getOpenFileName(
            self, "YOLO 모델 로드", start_dir,
            "YOLO Model (*.pt);;All (*.*)",
        )
        if fname:
            ok, err = self.yolo_detector.load_model(fname)
            if ok:
                names = self.yolo_detector.class_names
                self.lbl_yolo_status.setText(f"로드됨 ({len(names)}클래스)")
                self.lbl_yolo_status.setStyleSheet("color: #0f0; font-size: 11px;")
                QMessageBox.information(self, "YOLO 모델 로드 완료",
                    f"모델이 로드되었습니다.\n클래스: {', '.join(names)}")
            else:
                self.lbl_yolo_status.setText("로드 실패")
                self.lbl_yolo_status.setStyleSheet("color: red; font-size: 11px;")
                QMessageBox.warning(self, "오류",
                    "YOLO 모델 로드에 실패했습니다.\n\n" + (err or "알 수 없는 오류"))

    def _on_use_dl_toggled(self, checked):
        """Use Classification 체크박스 토글."""
        self.classifier.set_use_deep_learning(checked)
        self._update_dl_device_label()
        if checked and not self.classifier.dl_classifier.is_loaded():
            QMessageBox.information(self, "알림",
                "딥러닝 모델이 로드되지 않았습니다.\n"
                "'모델 로드' 버튼으로 학습된 모델(.onnx 또는 .pth)을 로드하세요.\n"
                "모델이 로드될 때까지 RuleBased 분류를 사용합니다.")

    def _update_dl_device_label(self):
        """Deep Learning 장치 상태를 메인 창에 표시."""
        if not hasattr(self, "lbl_dl_device"):
            return
        cls = self.classifier.dl_classifier
        short = cls.get_device_simple()
        if short == "—":
            self.lbl_dl_device.setText("—")
            self.lbl_dl_device.setStyleSheet("color: #666; font-size: 11px;")
            return

        if getattr(cls, "_ort_session_openvino", None) is not None:
            self.lbl_dl_device.setText(short)
        elif cls.ort_session is not None:
            self.lbl_dl_device.setText(f"{short} (ONNX)")
        elif cls.model is not None:
            self.lbl_dl_device.setText(f"{short} (PyTorch)")
        else:
            self.lbl_dl_device.setText(short)

        if short in ("GPU", "NPU"):
            self.lbl_dl_device.setStyleSheet("color: #0a0; font-size: 11px;")
        else:
            self.lbl_dl_device.setStyleSheet("color: #666; font-size: 11px;")

    def _app_root(self):
        """exe일 때 exe 폴더, 아니면 프로젝트 루트 (모델/설정 파일 기준 경로)."""
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _logo_path(self):
        """상단 헤더 로고 이미지 경로 (logo_alpha.png)."""  
        try:
            root = self._app_root()
            for rel in ("resource", "logo_alpha.png"), ("강로 UI 카피 해야됌", "resource", "logo_alpha.png"):
                path = os.path.join(root, *rel)
                if os.path.isfile(path):
                    return path
        except Exception:
            pass
        return None

    def _update_tab_branding_pos(self):
        """설정/도움말 + Main/Classification 탭 두 줄 전체 높이를 써서 로고·글자 표시 (잘리지 않음)."""
        if not hasattr(self, "_tab_branding"):
            return
        mb = self.menuBar()
        bar = self.tab_widget.tabBar()
        row1_h = mb.height() if mb else 28
        row2_h = bar.height()
        total_h = row1_h + row2_h
        # 두 줄 높이 안에 맞춤: 로고 3배·글자 2배 느낌 유지
        base_logo_h = max((total_h - 16) // 2, 24)
        logo_h = min(base_logo_h * 3, total_h - 12)
        base_font_px = max(12, (total_h - 16) // 2)
        font_px = min(base_font_px * 2, total_h - 8)
        self.header_title_label.setStyleSheet(
            "color: white; font-weight: normal; background: transparent;"
        )
        font = self.header_title_label.font()
        font.setPixelSize(font_px)
        font.setStretch(200)
        self.header_title_label.setFont(font)
        orig = getattr(self, "_logo_pixmap_orig", None)
        if orig and not orig.isNull():
            # 좌우 4배(400%), 위아래 1.3배
            logo_w = int(orig.width() * (logo_h / orig.height()) * 3)
            logo_h_scaled = int(logo_h * 1.3)
            self.header_logo_label.setPixmap(orig.scaled(
                logo_w, logo_h_scaled,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        self._tab_branding.setMinimumHeight(total_h)
        self._tab_branding.adjustSize()
        bw = self._tab_branding.sizeHint().width()
        bh = max(self._tab_branding.sizeHint().height(), total_h)
        w = self.width()
        x = max(0, (w - bw) // 2) + 120
        y = 0
        self._tab_branding.setGeometry(x, y, bw, total_h)
        self._tab_branding.raise_()
        self._tab_branding.show()

    def _set_window_icon(self):
        """좌상단 창 아이콘 설정. VialVolume2.ico 또는 icon.png 우선 사용."""
        try:
            root = self._app_root()
            candidates = [
                os.path.join(root, "강로 UI 카피 해야됌", "resource", "VialVolume2.ico"),
                os.path.join(root, "assets", "VialVolume2.ico"),
                os.path.join(root, "VialVolume2.ico"),
                os.path.join(root, "assets", "icon.png"),
                os.path.join(root, "icon.png"),
            ]
            for path in candidates:
                if os.path.isfile(path):
                    self.setWindowIcon(QIcon(path))
                    return
        except Exception:
            pass

    def _load_settings(self):
        """rule_params.json에서 설정을 로드하여 UI 및 객체에 반영."""
        try:
            path = os.path.join(self._app_root(), "rule_params.json")
        except Exception as e:
            print(f"[MainWindow] 설정 경로 확인 실패: {e}")
            return
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            gen = data.get("classification", {})
            bub = data.get("bubble_detection", {})
            # 로드 중 valueChanged/toggled가 _save_settings를 호출해 JSON을 덮어쓰는 것 방지
            self.slider_thresh.blockSignals(True)
            self.slider_area.blockSignals(True)
            self.chk_adaptive.blockSignals(True)
            try:
                self.slider_thresh.setValue(int(gen.get("threshold", 100)))
                self.slider_area.setValue(int(gen.get("min_area", 10)))
                self.chk_adaptive.setChecked(gen.get("use_adaptive", True))
                self.open_kernel = int(gen.get("open_kernel", 2))
                self.close_kernel = int(gen.get("close_kernel", 3))
                self.classification_enabled = gen.get("classification_enabled", True)
                self.general_enabled = gen.get("general_enabled", True)
                self.bubble_show_text = gen.get("bubble_show_text", True)
                self.classifier.rule_based.set_params(gen)
                self.detector.bubble_params.set_params(bub)
            finally:
                self.slider_thresh.blockSignals(False)
                self.slider_area.blockSignals(False)
                self.chk_adaptive.blockSignals(False)
            # 라벨은 시그널 차단으로 갱신 안 됐을 수 있음
            self.lbl_thresh_value.setText(str(self.slider_thresh.value()))
            self.lbl_area_value.setText(str(self.slider_area.value()))
            dl_data = data.get("deep_learning", {})
            self.classifier.dl_classifier.set_optimization_level(
                dl_data.get("optimization_level", 0))
            self.classifier.dl_classifier.set_openvino_device(
                dl_data.get("openvino_device", "AUTO"))
            self._update_dl_device_label()
            print(f"[MainWindow] 설정 로드 완료: {path}")
        except Exception as e:
            print(f"[MainWindow] 설정 로드 실패: {path} — {e}")
            import traceback
            traceback.print_exc()

    def _save_settings(self):
        """현재 UI 상태를 rule_params.json에 자동 저장."""
        data = {
            "classification": {
                "noise_contrast_threshold": self.classifier.rule_based.noise_contrast_threshold,
                "threshold": self.slider_thresh.value(),
                "min_area": self.slider_area.value(),
                "use_adaptive": self.chk_adaptive.isChecked(),
                "open_kernel": getattr(self, "open_kernel", 2),
                "close_kernel": getattr(self, "close_kernel", 3),
                "general_enabled": getattr(self, "general_enabled", True),
                "classification_enabled": getattr(self, "classification_enabled", True),
                "bubble_show_text": getattr(self, "bubble_show_text", True),
            },
            "bubble_detection": self.detector.bubble_params.get_params(),
            "deep_learning": {
                "optimization_level": self.classifier.dl_classifier.get_optimization_level(),
                "openvino_device": self.classifier.dl_classifier.get_openvino_device(),
            },
        }
        try:
            path = os.path.join(self._app_root(), "rule_params.json")
            root = os.path.dirname(path)
            if root:
                os.makedirs(root, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[MainWindow] 설정 저장 실패: {e}")
    def _open_rule_params_dialog(self):
        """RuleBase 파라미터 설정 다이얼로그 열기. ROI 설정 시 crop, 아니면 전체 프레임 전달."""
        current_frame = getattr(self, "original_frame_full", None)
        if current_frame is not None and self.inspection_roi is not None:
            fh, fw = current_frame.shape[:2]
            rx, ry, rw, rh = self.inspection_roi
            rx = max(0, min(int(rx), fw - 1))
            ry = max(0, min(int(ry), fh - 1))
            rw = max(1, min(int(rw), fw - rx))
            rh = max(1, min(int(rh), fh - ry))
            crop = current_frame[ry:ry + rh, rx:rx + rw]
            if crop.size > 0 and len(crop.shape) >= 2:
                current_frame = crop.copy()
        dlg = RuleParamsDialog(
            self, self.classifier.rule_based, self._app_root(),
            bubble_params=self.detector.bubble_params,
            source_frame=current_frame,
            dl_classifier=self.classifier.dl_classifier,
        )
        
        # UI 동기화
        dlg.spin_threshold.setValue(self.slider_thresh.value())
        dlg.spin_min_area.setValue(self.slider_area.value())
        dlg.chk_adaptive.setChecked(self.chk_adaptive.isChecked())
        dlg.spin_open_kernel.setValue(getattr(self, "open_kernel", 2))
        dlg.spin_close_kernel.setValue(getattr(self, "close_kernel", 3))
        dlg.group_classification.setChecked(getattr(self, "classification_enabled", True))
        dlg.chk_general_enabled.setChecked(getattr(self, "general_enabled", True))
        dlg.chk_bubble_show_text.setChecked(getattr(self, "bubble_show_text", True))

        if dlg.exec():
            # 저장된 값 다시 Main UI로 역반영
            gen = dlg._collect_params()
            self.slider_thresh.setValue(gen["threshold"])
            self.slider_area.setValue(gen["min_area"])
            self.chk_adaptive.setChecked(gen["use_adaptive"])
            self.open_kernel = gen["open_kernel"]
            self.close_kernel = gen["close_kernel"]
            self.classification_enabled = gen["classification_enabled"]
            self.general_enabled = gen["general_enabled"]
            self.bubble_show_text = gen["bubble_show_text"]
            # 확인 시 rule_params.json에 저장 (재실행 시 복원)
            self._save_settings()
            self._update_dl_device_label()
            # (선택) 설정 변경 즉시 재검사 하려면 주석 해제 (단, 정지 상태에서만)
            if not self.timer.isActive() and current_frame is not None:
                self.update_frame()

    def _load_dl_model(self, fname):
        """딥러닝 모델 파일 로드 (내부 헬퍼)."""
        ok, err = self.classifier.load_dl_model(fname)
        if ok:
            labels = self.classifier.dl_classifier.labels
            self._update_dl_device_label()
            disp = self.classifier.dl_classifier.get_device_display()
            QMessageBox.information(self, "모델 로드 완료",
                f"모델이 로드되었습니다.\n클래스: {', '.join(labels)}\n추론: {disp}")
        else:
            QMessageBox.warning(self, "오류",
                "모델 로드에 실패했습니다.\n\n" + (err or "알 수 없는 오류"))
            if hasattr(self, "lbl_dl_device"):
                self.lbl_dl_device.setText("—")
        return ok

    def _on_load_model(self):
        """딥러닝 모델 파일 로드."""
        start_dir = self._app_root()
        fname, _ = QFileDialog.getOpenFileName(
            self, "딥러닝 모델 로드", start_dir,
            "ONNX Model (*.onnx);;PyTorch Model (*.pth);;All (*.*)",
        )
        if fname:
            self._load_dl_model(fname)

    def _on_draw_on_mainview_toggled(self):
        """체크박스 토글 시 MainView 표시 갱신 (Results/Defect View는 그대로)."""
        if not self.current_contours:
            self.redraw_with_highlight()
            return
            
        allowed = self._get_result_filter_labels()
        show_text = getattr(self, "chk_show_text", None)
        show_text_val = show_text.isChecked() if show_text else True
        
        if self.chk_draw_on_mainview.isChecked():
            if getattr(self, "original_frame_full", None) is not None:
                base = self.original_frame_full.copy()
                
                for i, cnt in enumerate(self.current_contours):
                    r = self.current_classify_results[i] if self.current_classify_results and i < len(self.current_classify_results) else {"label": "Unknown"}
                    label = r.get("label", "Unknown")
                    
                    if allowed and label not in allowed:
                        continue
                        
                    x, y, w, h = cv2.boundingRect(cnt)
                    if label == "Particle":
                        color = (0, 0, 255)  # Red
                    elif label == "Noise_Dust":
                        color = (255, 0, 0)  # Blue
                    elif label == "Bubble":
                        color = (0, 255, 0)  # Green
                    else:
                        color = (0, 255, 0)  # Green fallback
                    text_color = color
                    
                    cv2.drawContours(base, [cnt], -1, color, 1)
                    
                    # 라벨 표시 여부 결정: "글자 보기" 체크박스가 마스터 컨트롤
                    actual_show = show_text_val
                        
                    if actual_show:
                        cv2.putText(base, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                        
                self.display_base_frame = base
        else:
            if getattr(self, "original_frame_full", None) is not None:
                self.display_base_frame = self.original_frame_full.copy()
            elif self._display_frame_clean is not None:
                self.display_base_frame = self._display_frame_clean.copy()
                
        self.redraw_with_highlight()

    def _on_video_slider_released(self):
        """동영상 슬라이더 놓으면 해당 프레임으로 이동."""
        if self.camera is None or not isinstance(self.camera, FileCamera) or not self.camera.is_video():
            return
        pos = self.video_slider.value()
        if self.camera.set_frame_position(pos):
            self._video_slider_just_sought = True  # 다음 update_frame에서 슬라이더를 카메라 위치로 덮어쓰지 않음
            self.update_frame()

    def update_frame(self):
        if self.camera is None:
            return
        frame = self.camera.grab_frame()
        if frame is None:
            return
            
        # --- 실시간 영상 녹화 (is_inspecting 상태일 때) ---
        if getattr(self, "is_inspecting", False) and getattr(self, "_current_video_path", None):
            if getattr(self, "video_writer", None) is None:
                fps = 30.0 # 기본 30fps
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                is_color = len(frame.shape) == 3
                temp_path = os.path.join(self._app_root(), "temp_video.avi")
                self.video_writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h), isColor=is_color)
            if self.video_writer.isOpened():
                self.video_writer.write(frame)

        if isinstance(self.camera, FileCamera) and self.camera.is_video() and self.video_slider.isVisible() and not self.video_slider.isSliderDown():
            if self._video_slider_just_sought:
                self._video_slider_just_sought = False
            else:
                self._video_slider_block = True
                try:
                    p = self.camera.get_frame_position()
                    idx = max(0, p - 1)
                    self.video_slider.setValue(min(idx, self.video_slider.maximum()))
                finally:
                    self._video_slider_block = False
        if len(frame.shape) == 2:
            self.original_frame_full = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame_bgr = self.original_frame_full
            self.current_gray = frame
        else:
            self.original_frame_full = frame.copy()
            frame_bgr = self.original_frame_full
            self.current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.is_inspecting:
            self.current_contours = []
            self.current_frame_bgr = None
            self.list_results.clearSelection()
            self.clear_defect_view()
            self.current_display_frame = frame_bgr.copy()
            self.display_base_frame = None
            # 비검사 상태에서도 디버그 패널에 기본 이미지 표시
            self.detector.last_debug = {"gray": self.current_gray}
            self.update_debug_panel()
        else:
            # 검사 중: 검사된 프레임만 표시(영상 늦춤). ROI가 있으면 해당 영역만 잘라서 검사
            if self._worker_busy:
                # 워커가 바쁠 때는 이전 프레임을 계속 표시 (검사 결과가 나올 때까지)
                if (self.view_cx is None or self.view_cy is None) and self.current_display_frame is not None:
                    h0, w0 = self.current_display_frame.shape[:2]
                    self.view_cx = w0 // 2
                    self.view_cy = h0 // 2
                self.render_main_view()
                return
            self._worker_busy = True
            fh, fw = frame_bgr.shape[:2]
            if self.inspection_roi is not None:
                try:
                    rx, ry, rw, rh = self.inspection_roi
                    rx = max(0, min(int(rx), fw - 1))
                    ry = max(0, min(int(ry), fh - 1))
                    rw = max(1, min(int(rw), fw - rx))
                    rh = max(1, min(int(rh), fh - ry))
                    if rx + rw > fw or ry + rh > fh:
                        self._worker_busy = False
                        if (self.view_cx is None or self.view_cy is None) and self.current_display_frame is not None:
                            h0, w0 = self.current_display_frame.shape[:2]
                            self.view_cx = w0 // 2
                            self.view_cy = h0 // 2
                        self.render_main_view()
                        return
                    crop = frame_bgr[ry : ry + rh, rx : rx + rw]
                    if crop.size == 0 or len(crop.shape) < 2:
                        self._worker_busy = False
                        if (self.view_cx is None or self.view_cy is None) and self.current_display_frame is not None:
                            h0, w0 = self.current_display_frame.shape[:2]
                            self.view_cx = w0 // 2
                            self.view_cy = h0 // 2
                        self.render_main_view()
                        return
                    crop = crop.copy()
                    self._last_roi_offset = (rx, ry)
                    self._detection_worker.run_detection(
                        crop,
                        frame_bgr,
                        self.slider_thresh.value(),
                        self.slider_area.value(),
                        self.chk_adaptive.isChecked(),
                        self.chk_draw_on_mainview.isChecked(),
                        self.DEBUG_IMAGE_FOLDER,
                        self.classifier,
                        yolo_detector=self.yolo_detector,
                        use_yolo=self.use_yolo_mode,
                        yolo_conf=self.spin_yolo_conf.value(),
                        bubble_params=self.detector.bubble_params,
                        open_kernel=getattr(self, "open_kernel", 2),
                        close_kernel=getattr(self, "close_kernel", 3),
                        classification_enabled=getattr(self, "classification_enabled", True),
                        general_enabled=getattr(self, "general_enabled", True),
                        bubble_show_text=getattr(self, "bubble_show_text", True),
                    )
                except Exception as e:
                    print(f"ROI 검사 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    self._worker_busy = False
                    if (self.view_cx is None or self.view_cy is None) and self.current_display_frame is not None:
                        h0, w0 = self.current_display_frame.shape[:2]
                        self.view_cx = w0 // 2
                        self.view_cy = h0 // 2
                    self.render_main_view()
                    return
            else:
                self._last_roi_offset = None
                self._detection_worker.run_detection(
                    frame_bgr,
                    frame_bgr,
                    self.slider_thresh.value(),
                    self.slider_area.value(),
                    self.chk_adaptive.isChecked(),
                    self.chk_draw_on_mainview.isChecked(),
                    self.DEBUG_IMAGE_FOLDER,
                    self.classifier,
                    yolo_detector=self.yolo_detector,
                    use_yolo=self.use_yolo_mode,
                    yolo_conf=self.spin_yolo_conf.value(),
                    bubble_params=self.detector.bubble_params,
                    open_kernel=getattr(self, "open_kernel", 2),
                    close_kernel=getattr(self, "close_kernel", 3),
                    classification_enabled=getattr(self, "classification_enabled", True),
                    general_enabled=getattr(self, "general_enabled", True),
                    bubble_show_text=getattr(self, "bubble_show_text", True),
                )
            # 검사 요청 후에도 이전 프레임을 표시 (결과가 나올 때까지)
            if (self.view_cx is None or self.view_cy is None) and self.current_display_frame is not None:
                h0, w0 = self.current_display_frame.shape[:2]
                self.view_cx = w0 // 2
                self.view_cy = h0 // 2
            self.render_main_view()
            return

        if (self.view_cx is None or self.view_cy is None) and self.current_display_frame is not None:
            h0, w0 = self.current_display_frame.shape[:2]
            self.view_cx = w0 // 2
            self.view_cy = h0 // 2
        self.render_main_view()

    def closeEvent(self, event):
        if self.camera:
            self.camera.close()
        if hasattr(self, '_save_executor') and self._save_executor:
            self._save_executor.shutdown(wait=False)
        event.accept()

    def _restart_application(self):
        """코드 변경 후 재시작하여 업데이트 적용. F5(디버거)로 실행해도 새 창이 뜨도록 subprocess로 새 프로세스 띄운 뒤 종료."""
        if self.camera:
            self.camera.close()
        import sys
        import subprocess
        # execv는 디버거(F5)로 실행 시 현재 프로세스만 바꿔치기해서 창이 꺼진 것처럼 보임. 대신 새 프로세스를 띄우고 우리는 종료.
        try:
            kwargs = {"cwd": os.getcwd()}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen([sys.executable] + sys.argv, **kwargs)
        except Exception as e:
            print(f"재시작 실패: {e}")
        QApplication.instance().quit()

    def _get_result_filter_labels(self):
        """선택된 라디오 버튼에 따른 필터 집합 반환. 비어 있으면 전부 표시(ALL)."""
        allowed = set()
        if self.radio_result_particle.isChecked():
            allowed.add("Particle")
        elif self.radio_result_noise.isChecked():
            allowed.add("Noise_Dust")
        elif self.radio_result_bubble.isChecked():
            allowed.add("Bubble")
        # ALL 일 경우는 셋이 비어있음
        return allowed

    def _refresh_results_list(self):
        """current_contours + current_classify_results 기준으로 리스트 갱신 (필터 적용)."""
        self.list_results.blockSignals(True)
        self.list_results.clear()
        self._result_row_to_contour_index = []
        if not self.current_contours or not self.current_classify_results:
            self.lbl_defect_counts.setText("")
            self.list_results.blockSignals(False)
            return
        allowed = self._get_result_filter_labels()
        
        counts = {}
        processed_counts = {"Particle": 0, "Noise_Dust": 0, "Bubble": 0}
        
        for i, cnt in enumerate(self.current_contours):
            result = self.current_classify_results[i] if i < len(self.current_classify_results) else {"label": "Unknown", "area": 0}
            label = result.get("label", "Unknown")
            counts[label] = counts.get(label, 0) + 1
            if label in processed_counts:
                processed_counts[label] += 1
            
            if allowed and label not in allowed:
                continue
            rect = cv2.minAreaRect(cnt)
            (_, (rw, rh), _) = rect
            major_axis = max(rw, rh)
            minor_axis = min(rw, rh)
            item_text = f"#{i+1}: {label} (Area: {result.get('area', 0):.0f}, 장축: {major_axis:.0f}, 단축: {minor_axis:.0f})"
            self.list_results.addItem(item_text)
            self._result_row_to_contour_index.append(i)
            
        count_text = f"Noise: {processed_counts['Noise_Dust']}, Particle: {processed_counts['Particle']}, Bubble: {processed_counts['Bubble']}"
        self.lbl_defect_counts.setText(count_text)
        
        self.list_results.blockSignals(False)

    def _on_result_filter_toggled(self):
        """Results 라디오 버튼 토글 시 리스트만 갱신. 전부 해제면 ALL(전부 표시)."""
        if not self.current_contours or not self.current_classify_results:
            return
        prev_contour_idx = self.selected_contour_idx
        self._refresh_results_list()
        try:
            row = self._result_row_to_contour_index.index(prev_contour_idx)
            self.list_results.blockSignals(True)
            self.list_results.setCurrentRow(row)
            self.list_results.blockSignals(False)
        except (ValueError, AttributeError):
            if self._result_row_to_contour_index:
                self.selected_contour_idx = self._result_row_to_contour_index[0]
                self.list_results.setCurrentRow(0)
            else:
                self.selected_contour_idx = 0
        self.update_defect_view()
        self._on_draw_on_mainview_toggled()

    def on_selection_changed(self):
        """Handle Result list selection change: update highlight and defect view"""
        row = self.list_results.currentRow()
        if self._result_row_to_contour_index and 0 <= row < len(self._result_row_to_contour_index):
            self.selected_contour_idx = self._result_row_to_contour_index[row]
        else:
            self.selected_contour_idx = row if 0 <= row < len(self.current_contours or []) else -1
        
        self.redraw_with_highlight()
        self.update_defect_view()

    def on_item_double_clicked(self, item):
        """Handle Result list double click: move view to center on selected defect"""
        row = self.list_results.row(item)
        if self._result_row_to_contour_index and 0 <= row < len(self._result_row_to_contour_index):
            idx = self._result_row_to_contour_index[row]
        else:
            idx = row
            
        if 0 <= idx < len(self.current_contours or []):
            cnt = self.current_contours[idx]
            # Get bounding rect center
            x, y, w, h = cv2.boundingRect(cnt)
            defect_cx = x + w // 2
            defect_cy = y + h // 2
            # Update view center to defect center
            self.view_cx = float(defect_cx)
            self.view_cy = float(defect_cy)
            
            # Force UI update
            self.redraw_with_highlight()
            self.update_defect_view()

    def redraw_with_highlight(self):
        """Redraw display frame with selected contour highlighted in yellow (only when MainView 표시 체크)."""
        if self.display_base_frame is None or not self.current_contours:
            return
        if not self.chk_draw_on_mainview.isChecked():
            self.current_display_frame = self.display_base_frame
            self.render_main_view()
            return
        display_frame = self.display_base_frame.copy()
        if 0 <= self.selected_contour_idx < len(self.current_contours):
            selected_cnt = self.current_contours[self.selected_contour_idx]
            cv2.drawContours(display_frame, [selected_cnt], -1, (0, 255, 255), 2)
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
                self.selected_contour_idx = i
                # 리스트에서 이 contour에 해당하는 행 선택 (필터 적용 시 행 번호가 다름)
                if self._result_row_to_contour_index:
                    try:
                        row = self._result_row_to_contour_index.index(i)
                        self.list_results.blockSignals(True)
                        self.list_results.setCurrentRow(row)
                        self.list_results.blockSignals(False)
                    except ValueError:
                        self.list_results.clearSelection()
                else:
                    self.list_results.blockSignals(True)
                    self.list_results.setCurrentRow(i)
                    self.list_results.blockSignals(False)
                self.redraw_with_highlight()
                self.update_defect_view()
                break

    def update_mouse_info(self, view_pos):
        """Update Position(x,y) and GrayValue display from mouse position (original scale)."""
        if self.current_gray is None:
            self.lbl_mouse_info.setText("Position: —  GrayValue: —")
            return
        # Only show when cursor is inside the displayed image (label) for correct mapping
        lw = self.image_label.width()
        lh = self.image_label.height()
        if lw <= 0 or lh <= 0:
            self.lbl_mouse_info.setText("Position: —  GrayValue: —")
            return
        if view_pos.x() < 0 or view_pos.y() < 0 or view_pos.x() >= lw or view_pos.y() >= lh:
            self.lbl_mouse_info.setText("Position: —  GrayValue: —")
            return
        img_pos = self.view_to_image_coords(view_pos)
        if img_pos is None:
            self.lbl_mouse_info.setText("Position: —  GrayValue: —")
            return
        x, y = img_pos
        h, w = self.current_gray.shape[:2]
        px = int(round(x))
        py = int(round(y))
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        gray_val = int(self.current_gray[py, px])
        txt = f"Position: ({px}, {py})  GrayValue: {gray_val}"
        self.lbl_mouse_info.setText(txt)
        self._viewer_lbl_mouse.setText(txt)

    def view_to_image_coords(self, view_pos):
        """Convert viewport position to original image coordinates"""
        if self.current_display_frame is None:
            return None
        
        fh, fw, _ = self.current_display_frame.shape
        
        zoom = max(self.zoom_factor, 1e-6)
        roi_w = min(fw, int(fw / zoom)) if zoom >= 1.0 else fw
        roi_h = min(fh, int(fh / zoom)) if zoom >= 1.0 else fh
        roi_w = max(1, roi_w)
        roi_h = max(1, roi_h)
        
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
        
        target_size = self.image_label.size()
        target_w = max(1, target_size.width())
        target_h = max(1, target_size.height())
        vx = max(0.0, min(view_pos.x(), target_w - 1e-6))
        vy = max(0.0, min(view_pos.y(), target_h - 1e-6))
        scale_x = roi_w / target_w
        scale_y = roi_h / target_h
        img_x = x1 + vx * scale_x
        img_y = y1 + vy * scale_y
        return (float(img_x), float(img_y))

    def update_debug_panel(self):
        """Update debug images. Static Box 260 유지, 이미지만 230으로 표시."""
        d = getattr(self.detector, "last_debug", None)
        bubble_d = d.get("bubble_debug", {}) if d else {}
        
        inner_sz = 230
        
        map_keys = [
            ("gray", d), ("blurred", d), ("threshold", d), ("closed", d),
            ("clahe", bubble_d), ("diff_map", bubble_d), ("binary", bubble_d), ("bubble_candidates", bubble_d)
        ]
        
        titles = {
            "gray": "Gray", "blurred": "Blurred", "threshold": "Threshold", "closed": "Morphology",
            "clahe": "CLAHE / Flat", "diff_map": "DoG Diff", "binary": "MAD Binary", "bubble_candidates": "Bubble Result"
        }
        
        for key, source_dict in map_keys:
            if key not in self.debug_labels:
                continue
            lbl = self.debug_labels[key]
            title = titles.get(key, key)
            
            if source_dict is None or key not in source_dict:
                lbl.setPixmap(QPixmap())
                lbl.setText(title)
                continue
                
            img = source_dict[key]
            if img is None or img.size == 0:
                lbl.setPixmap(QPixmap())
                lbl.setText(title)
                continue
                
            try:
                resized = cv2.resize(img, (inner_sz, inner_sz), interpolation=cv2.INTER_AREA)
                
                # --- 디버그 이미지 좌측 상단에 이름 표기 (동적 배경 박스) ---
                is_gray = (len(resized.shape) == 2)
                color_bg = 0 if is_gray else (0, 0, 0)
                color_fg = 255 if is_gray else (255, 255, 255)
                
                # 글자 길이에 맞춰 배경 박스 너비 조절
                (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(resized, (0, 0), (tw + 10, 24), color_bg, -1)
                cv2.putText(resized, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_fg, 1)
                
                if is_gray:
                    resized = np.ascontiguousarray(resized)
                    h, w = resized.shape
                    bytes_per_line = w
                    qt_img = QImage(resized.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
                else:
                    resized = np.ascontiguousarray(resized)
                    h, w, ch = resized.shape
                    bytes_per_line = ch * w
                    qt_img = QImage(resized.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                # .copy()로 QImage 데이터를 안전하게 복사 (numpy 참조 해제 대비)
                pix = QPixmap.fromImage(qt_img.copy())
                lbl.setPixmap(pix)
                lbl.setText("")
            except Exception:
                lbl.setPixmap(QPixmap())
                lbl.setText(key)

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

        row = self.list_results.currentRow()
        if row < 0:
            self.clear_defect_view()
            return
        if self._result_row_to_contour_index and row < len(self._result_row_to_contour_index):
            index = self._result_row_to_contour_index[row]
        else:
            index = row
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

        # Defect View 라벨 260 유지, 이미지만 238로 표시
        target_w = target_h = 238
        
        roi_h_actual, roi_w_actual = roi.shape[:2]
        if roi_w_actual <= 0 or roi_h_actual <= 0:
            self.clear_defect_view()
            return
        
        interp = cv2.INTER_CUBIC if roi_w_actual < target_w or roi_h_actual < target_h else cv2.INTER_AREA
        resized = cv2.resize(roi, (target_w, target_h), interpolation=interp)

        # Ensure contiguous array for QImage
        resized = np.ascontiguousarray(resized)
        
        h_r, w_r, ch = resized.shape
        bytes_per_line = ch * w_r
        qt_img = QImage(resized.data, w_r, h_r, bytes_per_line, QImage.Format.Format_BGR888)
        self.lbl_defect_view.setPixmap(QPixmap.fromImage(qt_img))
        self.lbl_defect_view.setText("")

    def _get_zoom_1to1(self):
        """1:1 픽셀 확대일 때의 zoom_factor (이미지 1픽셀 = 화면 1픽셀)."""
        if self.current_display_frame is None:
            return self.zoom_factor
        fh, fw, _ = self.current_display_frame.shape
        target_size = self.image_scroll.viewport().size()
        target_w = max(1, target_size.width())
        target_h = max(1, target_size.height())
        return max(fw / target_w, fh / target_h)

    def _on_roi_button_clicked(self):
        if self.is_roi_drawing:
            self.is_roi_drawing = False
            self.image_label.unsetCursor()
            self.roi_drag_start = None
            self.roi_drag_end = None
            self.btn_roi.setText("검사 ROI 설정")
        else:
            self.is_annotation_roi_drawing = False
            self.annotation_roi_start = None
            self.annotation_roi_end = None
            if hasattr(self, "btn_annotation_roi"):
                self.btn_annotation_roi.setText("어노테이션 ROI 설정")
            self.is_roi_drawing = True
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.roi_drag_start = None
            self.roi_drag_end = None
            self.btn_roi.setText("ROI 지정중..")
        self.render_main_view()

    def _on_annotation_roi_button_clicked(self):
        if self.current_display_frame is None:
            QMessageBox.information(self, "알림", "먼저 이미지 또는 영상을 로드하세요.")
            return
        if self.is_annotation_roi_drawing:
            self.is_annotation_roi_drawing = False
            self.image_label.unsetCursor()
            self.annotation_roi_start = None
            self.annotation_roi_end = None
            self.btn_annotation_roi.setText("어노테이션 ROI 설정")
        else:
            self.is_roi_drawing = False
            self.roi_drag_start = None
            self.roi_drag_end = None
            if hasattr(self, "btn_roi"):
                self.btn_roi.setText("검사 ROI 설정")
            self.is_annotation_roi_drawing = True
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.annotation_roi_start = None
            self.annotation_roi_end = None
            self.btn_annotation_roi.setText("드래그하여 지정중..")
        self.render_main_view()

    def _finish_annotation_rect(self, cx, cy):
        """그려진 박스의 중앙점 중심으로 128×128 크롭 후 지정한 라벨 폴더에 저장."""
        labels = []
        if hasattr(self, "classification_tab") and hasattr(self.classification_tab, "combo_label"):
            for i in range(self.classification_tab.combo_label.count()):
                labels.append(self.classification_tab.combo_label.itemText(i))
        if not labels:
            from src.core.classification import RuleBasedClassifier
            labels = list(RuleBasedClassifier.LABELS)

        # 단축키 지원을 위한 커스텀 다이얼로그 클래스
        class AnnotationLabelDialog(QDialog):
            def __init__(self, parent, labels, last_label):
                super().__init__(parent)
                self.setWindowTitle("어노테이션 추가")
                self.labels = labels
                layout = QVBoxLayout(self)
                
                msg = "선택한 영역의 중심을 기준으로 학습 데이터가 추가됩니다.\n어떤 분류로 저장하겠습니까?"
                msg += "\n(팁: 숫자키 1, 2, 3, 4로 즉시 선택 가능)"
                layout.addWidget(QLabel(msg))
                
                self.combo = QComboBox()
                self.combo.addItems(labels)
                if last_label in labels:
                    self.combo.setCurrentText(last_label)
                layout.addWidget(self.combo)
                
                bbox = QDialogButtonBox(
                    QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                )
                bbox.button(QDialogButtonBox.StandardButton.Ok).setText("추가")
                bbox.accepted.connect(self.accept)
                bbox.rejected.connect(self.reject)
                layout.addWidget(bbox)

                # 숫자 단축키 1 ~ 9 등록 (더 확실한 반응)
                for i in range(min(9, len(labels))):
                    key_seq = QKeySequence(str(i + 1))
                    shortcut = QShortcut(key_seq, self)
                    # lambda capture: i를 현재 값으로 고정 (i=i)
                    shortcut.activated.connect(lambda idx=i: self._on_shortcut_activated(idx))

            def _on_shortcut_activated(self, index):
                if 0 <= index < self.combo.count():
                    self.combo.setCurrentIndex(index)
                    self.accept()

        dlg = AnnotationLabelDialog(self, labels, getattr(self, "_last_annotation_label", None))

        def _cleanup_and_render():
            self.annotation_roi_start = None
            self.annotation_roi_end = None
            self.render_main_view()

        if dlg.exec() != QDialog.DialogCode.Accepted:
            _cleanup_and_render()
            return

        label = dlg.combo.currentText().strip()
        if not label:
            QMessageBox.warning(self, "오류", "분류 라벨을 선택하세요.")
            _cleanup_and_render()
            return
        self._last_annotation_label = label

        data_dir = None
        if hasattr(self, "classification_tab") and hasattr(self.classification_tab, "get_resolved_data_dir"):
            data_dir = self.classification_tab.get_resolved_data_dir()
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(
                self, "오류",
                "학습 데이터 폴더가 지정되지 않았거나 없습니다.\n"
                "Classification(학습) 탭에서 '데이터 폴더'를 먼저 지정하세요.",
            )
            _cleanup_and_render()
            return

        frame = self.current_display_frame.copy()
        saver = DefectImageSaver(data_dir)
        source_name = "MainAnnot"
        
        center_contour = np.array([[[cx, cy]]], dtype=np.int32)
        path = saver.save(
            center_contour,
            frame,
            label,
            source_name=source_name,
            roi_size=CLASSIFICATION_INPUT_SIZE,
        )

        self.is_annotation_roi_drawing = False
        self.btn_annotation_roi.setText("어노테이션 ROI 설정")
        _cleanup_and_render()

        if path:
            QMessageBox.information(
                self, "저장 완료",
                "어노테이션 1건이 저장되었습니다.\n(드래그한 영역의 중심 128×128)",
            )
        else:
            QMessageBox.warning(self, "오류", "저장에 실패했습니다.")

    def render_main_view(self):
        """Render ROI from full-res display frame using zoom crop to viewport size."""
        if self.current_display_frame is None:
            return

        frame = self.current_display_frame.copy()
        fh, fw, _ = frame.shape
        if self.chk_show_roi.isChecked() and self.inspection_roi is not None:
            x, y, w, h = self.inspection_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if self.is_roi_drawing and self.roi_drag_start is not None and self.roi_drag_end is not None:
            x1, y1 = int(self.roi_drag_start[0]), int(self.roi_drag_start[1])
            x2, y2 = int(self.roi_drag_end[0]), int(self.roi_drag_end[1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if self.is_annotation_roi_drawing and self.annotation_roi_start is not None and self.annotation_roi_end is not None:
            x1, y1 = self.annotation_roi_start
            x2, y2 = self.annotation_roi_end
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        target_size = self.image_scroll.viewport().size()
        target_w = max(1, target_size.width())
        target_h = max(1, target_size.height())

        zoom = max(self.zoom_factor, 1e-6)
        roi_w = min(fw, int(fw / zoom)) if zoom >= 1.0 else fw
        roi_h = min(fh, int(fh / zoom)) if zoom >= 1.0 else fh
        roi_w = max(1, roi_w)
        roi_h = max(1, roi_h)

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

        # 확대 시(업스케일): INTER_NEAREST로 픽셀을 정사각형 블록처럼 표시(ImageJ/Fiji 스타일). 축소 시: INTER_AREA
        upscaling = roi.shape[1] < target_w or roi.shape[0] < target_h
        interp = cv2.INTER_NEAREST if upscaling else cv2.INTER_AREA
        resized = cv2.resize(roi, (target_w, target_h), interpolation=interp)

        # Ensure contiguous array for QImage
        resized = np.ascontiguousarray(resized)

        h, w, ch = resized.shape
        bytes_per_line = ch * w
        qt_img = QImage(resized.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(qt_img)
        if not pix.isNull():
            self.image_label.setPixmap(pix)


    # ---- Drag & Drop ----
    def _handle_drag_event(self, event, is_drop=False):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            # .onnx / .pth 단일 파일 드롭 시 모델 로드 (탭 무관)
            if len(urls) == 1 and urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if os.path.isfile(path) and ext in (".onnx", ".pth"):
                    event.setDropAction(Qt.DropAction.CopyAction)
                    event.acceptProposedAction()
                    if is_drop:
                        ok, err = self.classifier.load_dl_model(path)
                        if ok:
                            labels = self.classifier.dl_classifier.labels
                            QMessageBox.information(self, "모델 로드 완료",
                                f"Classification 모델이 로드되었습니다.\n클래스: {', '.join(labels)}")
                        else:
                            QMessageBox.warning(self, "오류",
                                "모델 로드에 실패했습니다.\n\n" + (err or "알 수 없는 오류"))
                    return True
                if os.path.isfile(path) and ext == ".pt":
                    event.setDropAction(Qt.DropAction.CopyAction)
                    event.acceptProposedAction()
                    if is_drop:
                        ok, err = self.yolo_detector.load_model(path)
                        if ok:
                            names = self.yolo_detector.class_names
                            self.lbl_yolo_status.setText(f"로드됨 ({len(names)}클래스)")
                            self.lbl_yolo_status.setStyleSheet("color: #0f0; font-size: 11px;")
                            QMessageBox.information(self, "YOLO 모델 로드 완료",
                                f"YOLO 모델이 로드되었습니다.\n클래스: {', '.join(names)}")
                        else:
                            self.lbl_yolo_status.setText("로드 실패")
                            self.lbl_yolo_status.setStyleSheet("color: red; font-size: 11px;")
                            QMessageBox.warning(self, "오류",
                                "YOLO 모델 로드에 실패했습니다.\n\n" + (err or "알 수 없는 오류"))
                    return True
            for url in urls:
                if url.isLocalFile():
                    path = url.toLocalFile()
                    ext = os.path.splitext(path)[1].lower()
                    # Classification 탭이 활성화되어 있으면 해당 탭으로 전달
                    if self.tab_widget.currentWidget() is self.classification_tab:
                        # 이미지 파일 또는 폴더 허용
                        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"] or os.path.isdir(path):
                            event.setDropAction(Qt.DropAction.CopyAction)
                            event.acceptProposedAction()
                            if is_drop:
                                if os.path.isdir(path):
                                    self.classification_tab._load_folder(path)
                                else:
                                    self.classification_tab._load_image_path(path)
                            return True
                    else:
                        # Main 탭: 기존 동작 (이미지/영상 로드)
                        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".avi", ".mp4", ".mov", ".mkv", ".wmv"]:
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
            zoom_1to1 = self._get_zoom_1to1()
            effective_max = min(self.zoom_max, zoom_1to1 * 32)
            if delta > 0:
                self.zoom_factor = min(self.zoom_factor * 1.1, effective_max)
            else:
                self.zoom_factor = max(self.zoom_factor / 1.1, self.zoom_min)
            self.render_main_view()
            event.accept()
        else:
            super().wheelEvent(event)

    def eventFilter(self, obj, event):
        if obj == self and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._update_tab_branding_pos)
            return False
        if obj == self.tab_widget.tabBar() and event.type() in (QEvent.Type.Resize, QEvent.Type.Move):
            QTimer.singleShot(0, self._update_tab_branding_pos)
            return False
        if event.type() == QEvent.Type.Wheel and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            if obj in (self.image_scroll.viewport(), self.image_label):
                delta = event.angleDelta().y()
                zoom_1to1 = self._get_zoom_1to1()
                effective_max = min(self.zoom_max, zoom_1to1 * 32)
                if delta > 0:
                    self.zoom_factor = min(self.zoom_factor * 1.1, effective_max)
                else:
                    self.zoom_factor = max(self.zoom_factor / 1.1, self.zoom_min)
                self.render_main_view()
                return True
        if event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.MiddleButton:
            if obj in (self.image_scroll.viewport(), self.image_label) and self.current_display_frame is not None:
                zoom_1to1 = self._get_zoom_1to1()
                if self._zoom_before_1to1 is not None:
                    self.zoom_factor = self._zoom_before_1to1
                    self._zoom_before_1to1 = None
                else:
                    self._zoom_before_1to1 = self.zoom_factor
                    self.zoom_factor = zoom_1to1
                self.render_main_view()
                return True
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if obj in (self.image_scroll.viewport(), self.image_label):
                if self.is_annotation_roi_drawing and self.current_display_frame is not None:
                    p = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
                    img_pos = self.view_to_image_coords(QPointF(p))
                    if img_pos is not None:
                        self.annotation_roi_start = (float(img_pos[0]), float(img_pos[1]))
                        self.annotation_roi_end = self.annotation_roi_start
                    event.accept()
                    return True
                if self.is_roi_drawing and self.current_display_frame is not None:
                    p = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
                    img_pos = self.view_to_image_coords(QPointF(p))
                    if img_pos is not None:
                        self.roi_drag_start = (float(img_pos[0]), float(img_pos[1]))
                        self.roi_drag_end = self.roi_drag_start
                    event.accept()
                    return True
                self.is_panning = True
                self.pan_start_pos = event.position()
                self.pan_last_pos = event.position()
                event.accept()
                return True
        if event.type() == QEvent.Type.MouseMove and self.is_annotation_roi_drawing and self.annotation_roi_start is not None:
            if obj in (self.image_scroll.viewport(), self.image_label):
                p = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
                img_pos = self.view_to_image_coords(QPointF(p))
                if img_pos is not None:
                    self.annotation_roi_end = (float(img_pos[0]), float(img_pos[1]))
                    self.render_main_view()
                event.accept()
                return True
        if event.type() == QEvent.Type.MouseMove and self.is_roi_drawing and self.roi_drag_start is not None:
            if obj in (self.image_scroll.viewport(), self.image_label):
                p = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
                img_pos = self.view_to_image_coords(QPointF(p))
                if img_pos is not None:
                    self.roi_drag_end = (float(img_pos[0]), float(img_pos[1]))
                    self.render_main_view()
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
        if event.type() == QEvent.Type.MouseMove and not self.is_panning:
            if obj in (self.image_scroll.viewport(), self.image_label):
                # 항상 글로벌 좌표 → image_label 좌표로 변환 (최대화/복원 상태 무관하게 동작)
                p = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
                pos_in_label = QPointF(p)
                self.update_mouse_info(pos_in_label)
        if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self.is_annotation_roi_drawing and self.annotation_roi_start is not None and self.annotation_roi_end is not None:
                if obj in (self.image_scroll.viewport(), self.image_label):
                    x1, y1 = self.annotation_roi_start
                    x2, y2 = self.annotation_roi_end
                    rx, ry = min(x1, x2), min(y1, y2)
                    rw, rh = max(x1, x2) - rx, max(y1, y2) - ry
                    
                    if rw >= 2 and rh >= 2:
                        cx = int(rx + rw / 2)
                        cy = int(ry + rh / 2)
                        self._finish_annotation_rect(cx, cy)
                    else:
                        self.annotation_roi_start = None
                        self.annotation_roi_end = None
                        self.render_main_view()
                event.accept()
                return True
            if self.is_roi_drawing and self.roi_drag_start is not None and self.roi_drag_end is not None:
                if obj in (self.image_scroll.viewport(), self.image_label):
                    x1, y1 = self.roi_drag_start[0], self.roi_drag_start[1]
                    x2, y2 = self.roi_drag_end[0], self.roi_drag_end[1]
                    rx = min(x1, x2)
                    ry = min(y1, y2)
                    rw = max(x1, x2) - rx
                    rh = max(y1, y2) - ry
                    if rw < 2 or rh < 2:
                        self.roi_drag_start = None
                        self.roi_drag_end = None
                        self.render_main_view()
                        event.accept()
                        return True
                    msg = f"(설정된 ROI 좌표 x={int(rx)}, y={int(ry)}, width={int(rw)}, height={int(rh)})\n검사 ROI 설정하시겠습니까?"
                    reply = QMessageBox.question(
                        self, "검사 ROI 설정", msg,
                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Ok,
                    )
                    if reply == QMessageBox.StandardButton.Ok:
                        self.inspection_roi = (int(rx), int(ry), int(rw), int(rh))
                    self.is_roi_drawing = False
                    self.image_label.unsetCursor()
                    self.roi_drag_start = None
                    self.roi_drag_end = None
                    self.btn_roi.setText("검사 ROI 설정")
                    self.render_main_view()
                event.accept()
                return True
            if self.is_panning:
                self.is_panning = False
                # Check if this was a click (small movement) vs drag
                if obj in (self.image_scroll.viewport(), self.image_label):
                    release_pos = event.position()
                    drag_dist = (release_pos - self.pan_start_pos).manhattanLength()
                    if drag_dist < 5:  # Click threshold: 5 pixels
                        p = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
                        self.handle_image_click(QPointF(p))
                event.accept()
                return True
        if event.type() in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            if obj in (
                self.image_scroll.viewport(),
                self.image_label,
                self.image_scroll,
                self.main_tab,
                self.control_panel,
                self.splitter,
                self,
            ):
                handled = self._handle_drag_event(event, is_drop=event.type() == QEvent.Type.Drop)
                if handled:
                    return True
        return super().eventFilter(obj, event)
