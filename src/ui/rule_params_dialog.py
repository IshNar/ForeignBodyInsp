"""RuleBase 검사 조건 파라미터 + Bubble 검출 파라미터 설정 다이얼로그.

왼쪽: 실시간 디버깅 View (확대/축소 가능)
오른쪽: 파라미터 패널 (분류 조건 / Bubble 검출 탭)
"""
import os
import json
import time
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QGroupBox, QCheckBox, QRadioButton, QTabWidget, QWidget, QScrollArea,
    QSplitter, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QComboBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent, QPainter

from src.core.classification import RuleBasedClassifier
from src.core.detection import ForeignBodyDetector, BubbleDetectorParams

RULE_PARAMS_FILENAME = "rule_params.json"


def _default_rule_params_path(app_root: str) -> str:
    return os.path.join(app_root, RULE_PARAMS_FILENAME)


# ═══════════════════════════════════════════════════════════════
#  ZoomableImageView – QGraphicsView 기반 확대/축소/팬 뷰어
# ═══════════════════════════════════════════════════════════════

class ZoomableImageView(QGraphicsView):
    """마우스 휠 확대/축소 + 드래그 팬을 지원하는 이미지 뷰어."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._zoom = 1.0
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, img_bgr_or_gray: np.ndarray | None):
        """OpenCV 이미지를 표시. None이면 비움."""
        self._scene.clear()
        self._pixmap_item = None
        if img_bgr_or_gray is None:
            return
        img = img_bgr_or_gray
        if len(img.shape) == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, ch = img.shape
            if ch == 4:
                qimg = QImage(img.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        self._pixmap_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))

    def fit_in_view(self):
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
            self._zoom *= factor
        else:
            self.scale(1 / factor, 1 / factor)
            self._zoom /= factor

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item is not None and self._zoom == 1.0:
            self.fit_in_view()


# ═══════════════════════════════════════════════════════════════
#  DebugDetectionWorker – 백그라운드 검사 스레드
# ═══════════════════════════════════════════════════════════════

class DebugDetectionWorker(QThread):
    """파라미터를 받아 detect_static + detect_bubbles를 백그라운드에서 실행.

    target_view가 지정되면 해당 뷰에 필요한 단계까지만 실행하여 속도를 높임.
    """
    finished = pyqtSignal(dict)

    _VIEW_SCOPE = {
        0: "all",              # 최종 결과
        1: "none",             # 원본 Gray
        2: "general",          # Threshold
        3: "general",          # Opening
        4: "general",          # Closing
        5: "bubble_clahe",     # CLAHE
        6: "bubble_diff",      # 차이 맵
        7: "bubble_binary",    # 이진화
        8: "bubble_full",      # Bubble 최종
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gray: np.ndarray | None = None
        self._frame_bgr: np.ndarray | None = None
        self._params: dict = {}
        self._cancelled = False
        self._target_view: int | None = None

    def setup(self, frame_bgr: np.ndarray, gray: np.ndarray, params: dict,
              target_view: int | None = None):
        self._frame_bgr = frame_bgr
        self._gray = gray
        self._params = dict(params)
        self._cancelled = False
        self._target_view = target_view

    def cancel(self):
        self._cancelled = True

    def run(self):
        t0 = time.perf_counter()
        p = self._params
        gray = self._gray
        frame = self._frame_bgr
        if gray is None or frame is None:
            self.finished.emit({})
            return

        scope = self._VIEW_SCOPE.get(self._target_view, "all")

        debug_images: dict[str, np.ndarray] = {}
        debug_images["gray"] = gray

        detector = ForeignBodyDetector()
        bp = BubbleDetectorParams()
        bp.set_params(p.get("bubble", {}))
        detector.bubble_params = bp

        contours_gen = []
        bubble_contours = []

        if scope == "none":
            pass

        # ── 일반 검출: "general" 또는 "all" 일 때만 실행 ──
        elif scope in ("general", "all"):
            if p.get("general_enabled", True):
                contours_gen, _ = detector.detect_static(
                    gray,
                    threshold=p.get("threshold", 100),
                    min_area=p.get("min_area", 10),
                    use_adaptive=p.get("use_adaptive", True),
                    detect_bubbles=False,
                    open_kernel=p.get("open_kernel", 2),
                    close_kernel=p.get("close_kernel", 3),
                )
            else:
                contours_gen = []
                detector.last_debug = {"gray": gray}

            if self._cancelled:
                self.finished.emit({})
                return

            dbg = detector.last_debug or {}
            for dk in ("threshold", "opened", "closed"):
                if dk in dbg:
                    debug_images[dk] = dbg[dk]

            # "all" 이면 이어서 Bubble 검출도 실행
            if scope == "all" and p.get("bubble_enabled", True):
                bubble_contours, bubble_dbg = detector.detect_bubbles(gray)
                if self._cancelled:
                    self.finished.emit({})
                    return
                if bubble_dbg:
                    for bk in ("clahe", "diff_map", "binary",
                                "bubble_candidates"):
                        if bk in bubble_dbg:
                            debug_images[bk] = bubble_dbg[bk]

        # ── Bubble 부분만 실행 (stop_after로 조기 종료) ──
        elif scope.startswith("bubble_"):
            stop_map = {
                "bubble_clahe": "clahe",
                "bubble_diff": "diff_map",
                "bubble_binary": "binary",
                "bubble_full": None,
            }
            stop_after = stop_map.get(scope)

            if p.get("bubble_enabled", True):
                bubble_contours, bubble_dbg = detector.detect_bubbles(
                    gray, stop_after=stop_after)
                if self._cancelled:
                    self.finished.emit({})
                    return
                if bubble_dbg:
                    for bk in ("clahe", "diff_map", "binary",
                                "bubble_candidates"):
                        if bk in bubble_dbg:
                            debug_images[bk] = bubble_dbg[bk]

        # ── 최종 결과 시각화 (scope="all" 일 때만) ──
        n_noise = 0
        n_particle = 0
        
        view_filter = p.get("view_filter", "ALL")
            
        if scope == "all":
            result_vis = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cls_params = p.get("classification", {})
            classifier = RuleBasedClassifier()
            classifier.set_params(cls_params)

            for cnt in contours_gen:
                info = classifier.classify(cnt, image=gray)
                label = info["label"]
                if label == "Noise_Dust":
                    n_noise += 1
                elif label == "Particle":
                    n_particle += 1
                
                # Check filter to decide if we should draw this contour
                should_draw = False
                if view_filter == "ALL":
                    should_draw = True
                elif view_filter == "NOISE" and label == "Noise_Dust":
                    should_draw = True
                elif view_filter == "PARTICLE" and label == "Particle":
                    should_draw = True
                    
                if should_draw:
                    color = {"Noise_Dust": (180, 180, 180), "Bubble": (0, 255, 0),
                             "Fiber": (255, 100, 0), "Particle": (0, 0, 255)}.get(label, (0, 255, 0))
                    cv2.drawContours(result_vis, [cnt], -1, color, 1)
                    if p.get("show_text", True):
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.putText(result_vis, label, (x, max(y - 4, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Draw Bubbles only if we are viewing ALL or there isn't a strict noise/particle filter that hides them
            if view_filter == "ALL":
                for cnt in bubble_contours:
                    cv2.drawContours(result_vis, [cnt], -1, (0, 255, 0), 1)
                    if p.get("bubble_show_text", True):
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.putText(result_vis, "Bubble", (x, max(y - 4, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            debug_images["result"] = result_vis

        elapsed = time.perf_counter() - t0
        self.finished.emit({
            "debug_images": debug_images,
            "n_general": len(contours_gen),
            "n_noise": n_noise,
            "n_particle": n_particle,
            "n_bubble": len(bubble_contours),
            "elapsed": elapsed,
        })


# ═══════════════════════════════════════════════════════════════
#  RuleParamsDialog
# ═══════════════════════════════════════════════════════════════

class RuleParamsDialog(QDialog):
    """RuleBase 분류 + Bubble 검출 파라미터 편집 + 실시간 디버깅 View.

    source_frame: MainWindow에서 전달하는 현재 프레임 (BGR). None이면 디버깅 비활성.
    """

    def __init__(self, parent=None, rule_based=None, app_root="",
                 bubble_params=None, source_frame=None):
        super().__init__(parent)
        self.rule_based = rule_based or RuleBasedClassifier()
        self.bubble_params = bubble_params or BubbleDetectorParams()
        self.app_root = app_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._source_frame: np.ndarray | None = source_frame
        self._debug_images: dict[str, np.ndarray] = {}

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._run_debug_detection)

        self._worker: DebugDetectionWorker | None = None
        self._worker_pending = False

        self.setWindowTitle("RuleBase / Bubble 검출 파라미터")
        self.resize(1200, 720)
        self._build_ui()
        self._form_from_params(self.rule_based.get_params())
        self._bubble_from_params(self.bubble_params.get_params())
        self._connect_realtime_signals()

        if self._source_frame is not None:
            self.chk_realtime.setEnabled(True)
        else:
            self.chk_realtime.setEnabled(False)
            self.chk_realtime.setToolTip("MainView에 로드된 영상이 없어 비활성화됨")

    # ══════════════════════════════════════════════════════════
    #  UI 빌드
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        # ── 상단: 실시간 디버깅 체크박스 + 뷰 선택 콤보 ──
        top_row = QHBoxLayout()
        self.chk_realtime = QCheckBox("실시간 디버깅")
        self.chk_realtime.setChecked(False)
        self.chk_realtime.toggled.connect(self._on_realtime_toggled)
        top_row.addWidget(self.chk_realtime)

        self.chk_auto_view = QCheckBox("파라미터별 View")
        self.chk_auto_view.setChecked(True)
        self.chk_auto_view.setToolTip("파라미터 조정 시 해당 단계의 디버깅 영상으로 자동 전환")
        top_row.addWidget(self.chk_auto_view)

        top_row.addWidget(QLabel("표시:"))
        self.combo_debug_view = QComboBox()
        self.combo_debug_view.addItems([
            "최종 결과 (Result)",            # 0
            "원본 (Gray)",                   # 1
            "Threshold",                     # 2
            "Opening",                       # 3
            "Closing",                       # 4
            "평탄화+CLAHE (Flat)",           # 5
            "DoG 밴드패스 (Diff)",            # 6
            "MAD 이진화 (Binary)",            # 7
            "Bubble 최종",                   # 8
        ])
        self.combo_debug_view.currentIndexChanged.connect(self._on_view_combo_changed)
        top_row.addWidget(self.combo_debug_view)

        btn_fit = QPushButton("맞춤")
        btn_fit.setFixedWidth(50)
        btn_fit.setToolTip("이미지를 View에 맞춤")
        btn_fit.clicked.connect(lambda: self.debug_view.fit_in_view())
        top_row.addWidget(btn_fit)

        self.lbl_debug_info = QLabel("")
        self.lbl_debug_info.setStyleSheet("color: #888; font-size: 11px;")
        top_row.addWidget(self.lbl_debug_info)
        top_row.addStretch()
        root_layout.addLayout(top_row)

        # ── 메인 Splitter: 왼쪽(View) | 오른쪽(파라미터) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        # 왼쪽: 디버그 View
        self.debug_view = ZoomableImageView()
        splitter.addWidget(self.debug_view)

        # 오른쪽: 파라미터 패널
        right_panel = QWidget()
        right_panel.setFixedWidth(420)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 4, 0)

        tabs = QTabWidget()
        right_layout.addWidget(tabs)

        cls_tab = QWidget()
        cls_layout = QVBoxLayout(cls_tab)

        self.chk_general_enabled = QCheckBox("일반 검출 활성화")
        self.chk_general_enabled.setChecked(True)
        self.chk_general_enabled.setToolTip(
            "일반 이물 검출(Threshold 기반) 전체를 켜거나 끕니다.\n"
            "체크 해제 시: 일반 이물 검출(임계값 이진화 → 컨투어 탐색 → 분류)을\n"
            "완전히 건너뛰어, 버블 검출만 독립적으로 테스트할 수 있습니다.\n"
            "체크 시: 일반 이물 검출이 정상적으로 수행됩니다."
        )
        cls_layout.addWidget(self.chk_general_enabled)
        
        # 필터 라디오 버튼
        filter_group = QGroupBox("디버그 뷰 필터")
        filter_layout = QHBoxLayout()
        self.radio_view_all = QRadioButton("VIEW ALL")
        self.radio_view_noise = QRadioButton("NOISE")
        self.radio_view_particle = QRadioButton("PARTICLE")
        self.radio_view_all.setChecked(True)
        
        filter_layout.addWidget(self.radio_view_all)
        filter_layout.addWidget(self.radio_view_noise)
        filter_layout.addWidget(self.radio_view_particle)
        
        self.chk_show_text = QCheckBox("글자 보기")
        self.chk_show_text.setChecked(True)
        self.chk_show_text.setToolTip(
            "체크 시: 검출된 이물 옆에 분류 결과(Particle, Noise_Dust 등)를 텍스트로 표시합니다.\n"
            "체크 해제 시: 이물의 윤곽선만 표시하고 라벨 텍스트는 숨깁니다.\n"
            "이물이 많을 때 화면이 복잡하면 해제하면 깔끔합니다."
        )
        filter_layout.addWidget(self.chk_show_text)
        
        filter_group.setLayout(filter_layout)
        cls_layout.addWidget(filter_group)
        
        # 분류 조건
        group = QGroupBox("분류 조건")
        group.setCheckable(True)
        group.setChecked(True)
        self.group_classification = group
        form = QFormLayout()

        self.spin_noise_contrast_threshold = QSpinBox()
        self.spin_noise_contrast_threshold.setRange(1, 255)
        self.spin_noise_contrast_threshold.setToolTip(
            "검출된 이물이 '진짜 이물(Particle)'인지 '노이즈/먼지(Noise_Dust)'인지를\n"
            "구분하는 기준값입니다.\n\n"
            "• 이물과 주변 배경의 밝기 차이(Contrast)를 계산합니다.\n"
            "• 차이가 이 값 미만이면 → Noise_Dust (미세한 잡음)\n"
            "• 차이가 이 값 이상이면 → Particle (실제 이물)\n\n"
            "값을 높이면: Particle 판정이 까다로워져 노이즈로 분류되는 것이 많아집니다.\n"
            "값을 낮추면: 약한 대비의 이물도 Particle로 인식합니다."
        )
        form.addRow("Noise/Particle 대비 임계값:", self.spin_noise_contrast_threshold)

        group.setLayout(form)
        cls_layout.addWidget(group)

        # 일반 검출 파라미터
        detect_group = QGroupBox("일반 검출 (Threshold)")
        detect_form = QFormLayout()

        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_threshold.setValue(100)
        self.spin_threshold.setToolTip(
            "이미지를 흑/백 이진화할 때 사용하는 밝기 기준값 (0~255)입니다.\n\n"
            "• 그레이스케일 이미지에서 이 값보다 어두운 픽셀이 '이물 후보'가 됩니다.\n"
            "• 값을 높이면: 좀 더 어두운 이물만 검출 (엄격, 검출 수 감소)\n"
            "• 값을 낮추면: 밝은 이물까지 검출 (민감, 검출 수 증가, 오검출 가능)\n\n"
            "Adaptive Threshold를 켜면 이 값은 보조 역할을 합니다."
        )
        detect_form.addRow("Threshold:", self.spin_threshold)

        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(0, 5000)
        self.spin_min_area.setValue(10)
        self.spin_min_area.setToolTip(
            "검출된 이물 윤곽선의 최소 면적(Pixel²) 필터입니다.\n\n"
            "• 이 면적보다 작은 윤곽선은 노이즈로 간주하여 무시합니다.\n"
            "• 값을 높이면: 작은 점/노이즈가 걸러져 오검출이 줄어듭니다.\n"
            "• 값을 낮추면: 아주 작은 이물까지 검출하지만 노이즈도 늘어납니다.\n\n"
            "일반적으로 10~50 정도가 적당합니다."
        )
        detect_form.addRow("Min Area:", self.spin_min_area)

        self.chk_adaptive = QCheckBox("Adaptive Threshold")
        self.chk_adaptive.setChecked(True)
        self.chk_adaptive.setToolTip(
            "조명이 균일하지 않을 때 유용한 적응형 이진화 방식입니다.\n\n"
            "• 체크 시: 이미지의 각 영역마다 주변 밝기를 참고하여 임계값을 자동 조절합니다.\n"
            "  바이알 곡면처럼 한쪽이 밝고 한쪽이 어두운 경우에 효과적입니다.\n"
            "• 체크 해제 시: 위의 Threshold 값 하나로 전체 이미지를 일괄 이진화합니다.\n"
            "  조명이 매우 균일한 환경에서 적합합니다."
        )
        detect_form.addRow(self.chk_adaptive)

        self.spin_open_kernel = QSpinBox()
        self.spin_open_kernel.setRange(1, 15)
        self.spin_open_kernel.setValue(2)
        self.spin_open_kernel.setToolTip(
            "모폴로지 Opening 연산의 커널(구조체) 크기입니다.\n\n"
            "• Opening = 침식(Erosion) → 팽창(Dilation) 순서로 수행합니다.\n"
            "• 이진화 결과에서 작은 점 형태의 잡음을 제거하는 역할입니다.\n"
            "• 값을 키우면: 더 큰 잡음까지 제거하지만, 작은 실제 이물도 사라질 수 있습니다.\n"
            "• 값을 줄이면: 잡음 제거 효과가 약해지지만 작은 이물을 보존합니다.\n\n"
            "보통 2~3 정도로 설정합니다."
        )
        detect_form.addRow("Open 커널:", self.spin_open_kernel)

        self.spin_close_kernel = QSpinBox()
        self.spin_close_kernel.setRange(1, 15)
        self.spin_close_kernel.setValue(3)
        self.spin_close_kernel.setToolTip(
            "모폴로지 Closing 연산의 커널(구조체) 크기입니다.\n\n"
            "• Closing = 팽창(Dilation) → 침식(Erosion) 순서로 수행합니다.\n"
            "• 이물 내부의 작은 구멍이나 끊어진 부분을 메우는 역할입니다.\n"
            "• 값을 키우면: 끊어진 이물을 하나로 연결하지만, 가까운 이물끼리 합쳐질 수 있습니다.\n"
            "• 값을 줄이면: 이물 형태를 세밀하게 유지하지만 끊어짐이 남을 수 있습니다.\n\n"
            "보통 3~5 정도로 설정합니다."
        )
        detect_form.addRow("Close 커널:", self.spin_close_kernel)

        detect_group.setLayout(detect_form)
        cls_layout.addWidget(detect_group)

        # QFormLayout의 label에도 같은 툴팁 적용
        for widget in (self.spin_threshold, self.spin_min_area, self.spin_open_kernel, self.spin_close_kernel):
            lbl = detect_form.labelForField(widget)
            if lbl:
                lbl.setToolTip(widget.toolTip())
        lbl_nct = form.labelForField(self.spin_noise_contrast_threshold)
        if lbl_nct:
            lbl_nct.setToolTip(self.spin_noise_contrast_threshold.toolTip())

        cls_layout.addStretch()
        tabs.addTab(cls_tab, "분류/검출 조건")

        # ── 탭 2: Bubble 검출 (DoG 블롭) ──
        bub_tab = QWidget()
        bub_scroll = QScrollArea()
        bub_scroll.setWidgetResizable(True)
        bub_inner = QWidget()
        bub_layout = QVBoxLayout(bub_inner)
        bub_layout.setSpacing(6)

        # 상단 활성화 및 글자 보기 박스
        bub_top_layout = QHBoxLayout()
        self.chk_bubble_enabled = QCheckBox("Bubble 검출 활성화")
        self.chk_bubble_enabled.setChecked(True)
        self.chk_bubble_enabled.setToolTip(
            "버블 검출 기능 전체를 켜거나 끕니다.\n\n"
            "• 체크 시: DoG(Difference of Gaussians) 기반 버블 검출 파이프라인이 실행됩니다.\n"
            "  배경 평탄화 → 노이즈 제거 → 밴드패스 필터 → 이진화 → 형상 필터 순서로 처리됩니다.\n"
            "• 체크 해제 시: 버블 검출을 건너뛰어 일반 이물 검출만 수행합니다."
        )
        bub_top_layout.addWidget(self.chk_bubble_enabled)
        
        self.chk_bubble_show_text = QCheckBox("글자 보기")
        self.chk_bubble_show_text.setChecked(True)
        self.chk_bubble_show_text.setToolTip(
            "체크 시: 검출된 버블 옆에 'Bubble' 라벨을 텍스트로 표시합니다.\n"
            "체크 해제 시: 버블의 윤경선만 표시하고 텍스트는 숨깁니다.\n"
            "버블이 많을 때 글자가 겹쳐 보이면 해제하면 깔끔합니다."
        )
        bub_top_layout.addWidget(self.chk_bubble_show_text)
        bub_top_layout.addStretch()
        
        bub_layout.addLayout(bub_top_layout)

        # 배경 평탄화
        bg_group = QGroupBox("배경 평탄화 (Morph Opening)")
        bg_form = QFormLayout()
        self.spin_bg_open_ksize = QSpinBox()
        self.spin_bg_open_ksize.setRange(3, 201)
        self.spin_bg_open_ksize.setSingleStep(2)
        self.spin_bg_open_ksize.setToolTip(
            "배경 추정을 위한 Morphological Opening 커널 크기입니다 (홀수만 사용).\n\n"
            "• 바이알 이미지에서 매우 큰 커널로 Opening을 수행하면,\n"
            "  버블 같은 작은 구조가 제거되고 배경만 남습니다.\n"
            "• 이 배경을 원본에서 빼면 버블만 남는 평탄화된 이미지가 됩니다.\n"
            "• 값이 너무 작으면: 배경에 버블이 포함되어 검출 실패\n"
            "• 값이 너무 크면: 처리 시간이 증가하고 배경 추정이 야해짐\n\n"
            "바이알 곡면 스케일(수십~수백 px)보다 충분히 커야 합니다.\n"
            "예: 61, 101 등"
        )
        bg_form.addRow("커널 크기:", self.spin_bg_open_ksize)
        self.spin_bg_smooth_sigma = QDoubleSpinBox()
        self.spin_bg_smooth_sigma.setRange(0.0, 100.0)
        self.spin_bg_smooth_sigma.setDecimals(1)
        self.spin_bg_smooth_sigma.setSingleStep(1.0)
        self.spin_bg_smooth_sigma.setToolTip(
            "배경 이미지에 가우시안 블러를 적용하는 표준편차(σ)입니다.\n\n"
            "• Opening으로 추정한 배경에 남아있는 원형 아티팩트를 부드럽게 제거합니다.\n"
            "• 0 입력 시: 자동으로 커널/4 값을 사용합니다.\n"
            "• 값을 높이면: 배경이 더 부드럽지만 미세 변화가 속쓸 수 있음\n"
            "• 값을 낮추면: 배경 말람이 덤 되지만 국소적 변화는 보존"
        )
        bg_form.addRow("스무딩 σ:", self.spin_bg_smooth_sigma)
        bg_group.setLayout(bg_form)
        bub_layout.addWidget(bg_group)

        # CLAHE (선택)
        self.chk_use_clahe = QCheckBox("CLAHE 사용 (평탄화 후 대비 보정)")
        self.chk_use_clahe.setChecked(False)
        self.chk_use_clahe.setToolTip(
            "CLAHE(Contrast Limited Adaptive Histogram Equalization)를 사용할지 여부입니다.\n\n"
            "• 체크 시: 배경 평탄화 후 버블의 대비를 국소적으로 보정하여\n"
            "  어두운 영역에서도 버블이 더 잘 보이게 됩니다.\n"
            "• 체크 해제 시: CLAHE를 건너뛰어 평탄화된 연상 그대로 사용.\n\n"
            "대부분의 경우 해제 상태로 충분합니다.\n"
            "바이알 내부가 매우 어두운 경우에만 활성화해 보세요."
        )
        bub_layout.addWidget(self.chk_use_clahe)
        clahe_group = QGroupBox("CLAHE")
        clahe_form = QFormLayout()
        self.spin_clahe_clip = QDoubleSpinBox()
        self.spin_clahe_clip.setRange(0.1, 40.0)
        self.spin_clahe_clip.setDecimals(1)
        self.spin_clahe_clip.setToolTip(
            "CLAHE의 대비 제한 값(clipLimit)입니다.\n\n"
            "• 높을수록 대비 보정이 강해져 버블이 더 뚜렷해지지만,\n"
            "  너무 높으면 노이즈까지 강조될 수 있습니다.\n"
            "• 낮을수록 보정 효과가 약해짐\n\n"
            "기본값 2.0 정도가 적당합니다."
        )
        clahe_form.addRow("clipLimit:", self.spin_clahe_clip)
        self.spin_clahe_grid = QSpinBox()
        self.spin_clahe_grid.setRange(2, 64)
        self.spin_clahe_grid.setToolTip(
            "CLAHE를 적용할 타일(tile) 크기입니다.\n\n"
            "• 이미지를 NxN 블록으로 나누어 각 블록마다 히스토그램 균등화를 수행합니다.\n"
            "• 값이 작으면: 더 국소적으로 보정 (미세한 변화에 민감)\n"
            "• 값이 크면: 광범위하게 보정 (전체적인 변화에 반응)\n\n"
            "기본값 8 정도가 적당합니다."
        )
        clahe_form.addRow("tileGridSize:", self.spin_clahe_grid)
        clahe_group.setLayout(clahe_form)
        bub_layout.addWidget(clahe_group)

        # 노이즈 제거
        denoise_group = QGroupBox("노이즈 제거")
        denoise_form = QFormLayout()
        self.combo_denoise = QComboBox()
        self.combo_denoise.addItems(["median", "bilateral", "none"])
        self.combo_denoise.setToolTip(
            "DoG 필터 전에 적용할 노이즈 제거 방식입니다.\n\n"
            "• median: 미디언 필터로 소금 노이즈(salt & pepper) 제거에 효과적\n"
            "• bilateral: 엣지를 보존하면서 부드럽게 제거 (처리 느림)\n"
            "• none: 노이즈 제거 없이 원본 사용\n\n"
            "보통 median이 적합합니다."
        )
        denoise_form.addRow("방식:", self.combo_denoise)
        self.spin_median_k = QSpinBox()
        self.spin_median_k.setRange(3, 31)
        self.spin_median_k.setSingleStep(2)
        self.spin_median_k.setToolTip(
            "Median 필터의 커널 크기입니다 (홀수만 사용).\n\n"
            "• 각 픽셀을 주변 NxN 영역의 중앙값(median)으로 대체합니다.\n"
            "• 값이 크면: 강력한 노이즈 제거 효과이지만 이미지가 흐려질 수 있음\n"
            "• 값이 작으면: 원본에 가까운 결과이지만 노이즈가 남을 수 있음\n\n"
            "기본값 5 정도가 적당합니다."
        )
        denoise_form.addRow("Median 크기:", self.spin_median_k)
        denoise_group.setLayout(denoise_form)
        bub_layout.addWidget(denoise_group)

        # DoG 밴드패스
        dog_group = QGroupBox("DoG 밴드패스")
        dog_form = QFormLayout()
        self.spin_sigma_small = QDoubleSpinBox()
        self.spin_sigma_small.setRange(0.1, 20.0)
        self.spin_sigma_small.setDecimals(1)
        self.spin_sigma_small.setSingleStep(0.1)
        self.spin_sigma_small.setToolTip(
            "DoG(Difference of Gaussians) 밴드패스 필터의 작은 σ 값입니다.\n\n"
            "• 가우시안 블러를 작은 σ와 큰 σ 두 번 적용한 후 차이를 구합니다.\n"
            "• 작은 σ = 검출할 버블의 '최소 두께'에 대응\n"
            "• 값을 높이면: 미세한 버블을 무시하고 큰 버블 위주로 검출\n"
            "• 값을 낮추면: 아주 작은 버블까지 검출하지만 노이즈도 늘어날 수 있음\n\n"
            "기본값 1.2 정도가 적당합니다."
        )
        dog_form.addRow("σ small:", self.spin_sigma_small)
        self.spin_sigma_large = QDoubleSpinBox()
        self.spin_sigma_large.setRange(0.5, 50.0)
        self.spin_sigma_large.setDecimals(1)
        self.spin_sigma_large.setSingleStep(0.5)
        self.spin_sigma_large.setToolTip(
            "DoG(Difference of Gaussians) 밴드패스 필터의 큰 σ 값입니다.\n\n"
            "• 큰 σ = 검출할 버블의 '최대 두께'에 대응\n"
            "• 값을 높이면: 더 큰 버블까지 검출 범위에 포함\n"
            "• 값을 낮추면: 큰 버블을 무시하고 작은 버블만 검출\n\n"
            "σ small보다 충분히 커야 합니다 (3배 이상 권장).\n"
            "기본값 6.0 정도가 적당합니다."
        )
        dog_form.addRow("σ large:", self.spin_sigma_large)
        dog_group.setLayout(dog_form)
        bub_layout.addWidget(dog_group)

        # MAD 적응 임계
        thr_group = QGroupBox("MAD 적응 임계")
        thr_form = QFormLayout()
        self.spin_thr_k = QDoubleSpinBox()
        self.spin_thr_k.setRange(0.5, 20.0)
        self.spin_thr_k.setDecimals(1)
        self.spin_thr_k.setSingleStep(0.5)
        self.spin_thr_k.setToolTip(
            "MAD(Median Absolute Deviation) 적응형 임계값의 민감도 계수입니다.\n\n"
            "• DoG 필터 결과에서 '일반적인 변동(MAD)'을 기준으로,\n"
            "  이 k배 이상 돌출하는 픽셀을 버블 후보로 판단합니다.\n"
            "• 값을 낮추면 (2~3): 약한 버블까지 검출 (민감, 오검출 가능)\n"
            "• 값을 높이면 (5~8): 뚜렷한 버블만 검출 (엄격, 미검출 가능)\n\n"
            "기본값 4.0 정도가 적당합니다."
        )
        thr_form.addRow("k (민감도):", self.spin_thr_k)
        thr_group.setLayout(thr_form)
        bub_layout.addWidget(thr_group)

        # 형태학
        morph_group = QGroupBox("형태학 정리")
        morph_form = QFormLayout()
        self.spin_morph_close = QSpinBox()
        self.spin_morph_close.setRange(1, 31)
        self.spin_morph_close.setSingleStep(2)
        self.spin_morph_close.setToolTip(
            "버블 이진화 결과에 적용하는 Closing 커널 크기입니다.\n\n"
            "• Closing = 팽창 → 침식 순서로, 버블 내부의 구멍을 메우고\n"
            "  버블 윤경선 끊어진 부분을 연결합니다.\n"
            "• 값을 키우면: 버블 형태가 매끄럽지만 가까운 버블이 합쳐질 수 있음\n"
            "• 값을 줄이면: 버블 형태 유지지만 끊어진 버블이 남을 수 있음\n\n"
            "기본값 5 정도가 적당합니다."
        )
        morph_form.addRow("Close 커널:", self.spin_morph_close)
        self.spin_morph_open = QSpinBox()
        self.spin_morph_open.setRange(1, 31)
        self.spin_morph_open.setSingleStep(2)
        self.spin_morph_open.setToolTip(
            "버블 이진화 결과에 적용하는 Opening 커널 크기입니다.\n\n"
            "• Opening = 침식 → 팽창 순서로, 이진화의 작은 노이즈 점을 제거합니다.\n"
            "• 값을 키우면: 더 큰 노이즈까지 제거하지만 작은 버블도 사라질 수 있음\n"
            "• 값을 줄이면: 작은 버블을 보존하지만 노이즈가 남을 수 있음\n\n"
            "기본값 3 정도가 적당합니다."
        )
        morph_form.addRow("Open 커널:", self.spin_morph_open)
        morph_group.setLayout(morph_form)
        bub_layout.addWidget(morph_group)

        # 형상 필터
        filter_group = QGroupBox("형상 필터")
        filter_form = QFormLayout()
        self.spin_min_diam = QSpinBox()
        self.spin_min_diam.setRange(1, 500)
        self.spin_min_diam.setToolTip(
            "버블로 인정할 최소 직경(px)입니다.\n\n"
            "• 검출된 후보의 등가원 직경을 계산하여,\n"
            "  이 값보다 작은 후보는 버블로 인정하지 않고 버립니다.\n"
            "• 값을 높이면: 아주 작은 버블을 무시 (노이즈 감소)\n"
            "• 값을 낮추면: 작은 버블까지 검출\n\n"
            "기본값 8px 정도가 적당합니다."
        )
        filter_form.addRow("최소 직경:", self.spin_min_diam)
        self.spin_max_diam = QSpinBox()
        self.spin_max_diam.setRange(1, 1000)
        self.spin_max_diam.setToolTip(
            "버블로 인정할 최대 직경(px)입니다.\n\n"
            "• 이 값보다 큰 후보는 버블이 아닌 다른 구조물로 판단하여 버립니다.\n"
            "• 값을 높이면: 큰 버블까지 허용\n"
            "• 값을 낮추면: 작은 버블만 검출\n\n"
            "기본값 100px 정도가 적당합니다."
        )
        filter_form.addRow("최대 직경:", self.spin_max_diam)
        self.spin_circularity = QDoubleSpinBox()
        self.spin_circularity.setRange(0.0, 1.0)
        self.spin_circularity.setDecimals(2)
        self.spin_circularity.setSingleStep(0.05)
        self.spin_circularity.setToolTip(
            "버블로 인정할 최소 원형도(Circularity)입니다.\n\n"
            "• 원형도 = 4π×면적 / 둘레²  (완벽한 원 = 1.0)\n"
            "• 값이 높을수록 동그란 형태만 버블로 인정\n"
            "• 값이 낮으면 다소 찌그러진 형태도 버블로 허용\n\n"
            "기본값 0.35 정도가 적당합니다."
        )
        filter_form.addRow("최소 원형도:", self.spin_circularity)
        self.spin_solidity = QDoubleSpinBox()
        self.spin_solidity.setRange(0.0, 1.0)
        self.spin_solidity.setDecimals(2)
        self.spin_solidity.setSingleStep(0.05)
        self.spin_solidity.setToolTip(
            "버블로 인정할 최소 볼록도(Solidity)입니다.\n\n"
            "• 볼록도 = 면적 / 볼록껍질(Convex Hull) 면적  (꽉 찬 = 1.0)\n"
            "• 값이 높을수록 꽉 찬 형태만 버블로 인정 (움푸악 만은 형태 제외)\n"
            "• 값이 낮으면 불규칙한 형태도 버블로 허용\n\n"
            "기본값 0.70 정도가 적당합니다."
        )
        filter_form.addRow("최소 볼록도:", self.spin_solidity)
        self.spin_max_aspect = QDoubleSpinBox()
        self.spin_max_aspect.setRange(1.0, 10.0)
        self.spin_max_aspect.setDecimals(1)
        self.spin_max_aspect.setSingleStep(0.1)
        self.spin_max_aspect.setToolTip(
            "버블로 인정할 최대 종횡비(Aspect Ratio)입니다.\n\n"
            "• 종횡비 = 더 긴 쪽 / 더 짧은 쪽  (완벽한 원 = 1.0)\n"
            "• 값이 작으면: 동그란 형태만 버블로 인정 (엄격)\n"
            "• 값이 크면: 타원형이나 길쫑한 형태도 버블로 허용\n\n"
            "기본값 2.2 정도가 적당합니다."
        )
        filter_form.addRow("최대 종횡비:", self.spin_max_aspect)
        filter_group.setLayout(filter_form)
        bub_layout.addWidget(filter_group)

        # Bubble Tab: 모든 QFormLayout label에도 툴팁 적용
        for frm, widgets in [
            (bg_form, [self.spin_bg_open_ksize, self.spin_bg_smooth_sigma]),
            (clahe_form, [self.spin_clahe_clip, self.spin_clahe_grid]),
            (denoise_form, [self.combo_denoise, self.spin_median_k]),
            (dog_form, [self.spin_sigma_small, self.spin_sigma_large]),
            (thr_form, [self.spin_thr_k]),
            (morph_form, [self.spin_morph_close, self.spin_morph_open]),
            (filter_form, [self.spin_min_diam, self.spin_max_diam, self.spin_circularity,
                           self.spin_solidity, self.spin_max_aspect]),
        ]:
            for w in widgets:
                lbl = frm.labelForField(w)
                if lbl:
                    lbl.setToolTip(w.toolTip())

        bub_layout.addStretch()
        bub_scroll.setWidget(bub_inner)
        bub_tab_layout = QVBoxLayout(bub_tab)
        bub_tab_layout.setContentsMargins(0, 0, 0, 0)
        bub_tab_layout.addWidget(bub_scroll)
        tabs.addTab(bub_tab, "Bubble 검출")

        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)

        # ── 하단 버튼 ──
        # (저장/불러오기 버튼 제거: 확인 버튼에서 자동 저장됨)

        ok_cancel = QHBoxLayout()
        ok_cancel.addStretch()
        self.btn_ok = QPushButton("확인")
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.clicked.connect(self.reject)
        ok_cancel.addWidget(self.btn_ok)
        ok_cancel.addWidget(self.btn_cancel)
        root_layout.addLayout(ok_cancel)

    # ══════════════════════════════════════════════════════════
    #  실시간 디버깅 시그널 연결
    # ══════════════════════════════════════════════════════════

    def _connect_realtime_signals(self):
        """모든 파라미터 위젯의 값 변경 시그널을 디바운스된 재검사에 연결.
        파라미터별 View 자동전환을 위해 각 위젯→콤보인덱스 매핑도 구축."""

        # 콤보 인덱스:
        # 0=Result, 1=Gray, 2=Threshold, 3=Opening, 4=Closing,
        # 5=CLAHE, 6=DoG응답, 7=DoG후보, 8=Bubble최종

        VIEW_RESULT = 0
        VIEW_THRESHOLD = 2
        VIEW_OPENING = 3
        VIEW_CLOSING = 4
        VIEW_CLAHE = 5
        VIEW_DIFF = 6
        VIEW_BINARY = 7
        VIEW_BUBBLE = 8

        self._widget_view_map: dict[int, int] = {}

        widget_view_pairs = [
            # 일반 검출
            (self.spin_threshold, VIEW_THRESHOLD),
            (self.spin_min_area, VIEW_CLOSING),
            (self.chk_adaptive, VIEW_THRESHOLD),
            (self.spin_open_kernel, VIEW_OPENING),
            (self.spin_close_kernel, VIEW_CLOSING),
            # 일반 활성화
            (self.chk_general_enabled, VIEW_RESULT),
            # 분류 조건 → 최종 결과
            (self.group_classification, VIEW_RESULT),
            (self.spin_noise_contrast_threshold, VIEW_RESULT),
            # Bubble 활성화 / 보기
            (self.chk_bubble_enabled, VIEW_RESULT),
            (self.chk_bubble_show_text, VIEW_RESULT),
            # Bubble: 배경 평탄화 / CLAHE
            (self.spin_bg_open_ksize, VIEW_CLAHE),
            (self.spin_bg_smooth_sigma, VIEW_CLAHE),
            (self.chk_use_clahe, VIEW_CLAHE),
            (self.spin_clahe_clip, VIEW_CLAHE),
            (self.spin_clahe_grid, VIEW_CLAHE),
            # Bubble: 노이즈 / DoG → diff_map
            (self.combo_denoise, VIEW_DIFF),
            (self.spin_median_k, VIEW_DIFF),
            (self.spin_sigma_small, VIEW_DIFF),
            (self.spin_sigma_large, VIEW_DIFF),
            # Bubble: MAD 임계 / 형태학 → binary
            (self.spin_thr_k, VIEW_BINARY),
            (self.spin_morph_close, VIEW_BINARY),
            (self.spin_morph_open, VIEW_BINARY),
            # Bubble: 형상 필터
            (self.spin_min_diam, VIEW_BUBBLE),
            (self.spin_max_diam, VIEW_BUBBLE),
            (self.spin_circularity, VIEW_BUBBLE),
            (self.spin_solidity, VIEW_BUBBLE),
            (self.spin_max_aspect, VIEW_BUBBLE),
            # Bubble: 활성화
            (self.chk_bubble_enabled, VIEW_BUBBLE),
        ]

        for widget, view_idx in widget_view_pairs:
            wid = id(widget)
            self._widget_view_map[wid] = view_idx
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda _, w=widget: self._schedule_debug_for(w))
            elif isinstance(widget, QCheckBox) or isinstance(widget, QGroupBox):
                widget.toggled.connect(lambda _, w=widget: self._schedule_debug_for(w))
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(lambda _, w=widget: self._schedule_debug_for(w))
                
        # Radio buttons and checkboxes are mapped to VIEW_RESULT
        for widget in (self.radio_view_all, self.radio_view_noise, self.radio_view_particle, self.chk_show_text):
            self._widget_view_map[id(widget)] = VIEW_RESULT
            if isinstance(widget, QRadioButton):
                widget.toggled.connect(lambda checked, w=widget: self._schedule_debug_for(w) if checked else None)
            else:
                widget.toggled.connect(lambda _, w=widget: self._schedule_debug_for(w))

    def _schedule_debug_for(self, widget):
        """파라미터 변경 시 디바운스 + 자동 뷰 전환."""
        if not self.chk_realtime.isChecked() or self._source_frame is None:
            return
        if self.chk_auto_view.isChecked():
            view_idx = self._widget_view_map.get(id(widget))
            if view_idx is not None:
                self.combo_debug_view.setCurrentIndex(view_idx)
        self._debounce_timer.start()

    def _on_realtime_toggled(self, checked: bool):
        if checked and self._source_frame is not None:
            self._run_debug_detection()

    def _on_view_combo_changed(self, _index: int):
        self._update_debug_view()

    # ══════════════════════════════════════════════════════════
    #  디버그 검사 실행 (백그라운드 스레드)
    # ══════════════════════════════════════════════════════════

    def _collect_params(self) -> dict:
        """UI에서 모든 파라미터를 dict로 수집."""
        
        view_filter = "ALL"
        if self.radio_view_noise.isChecked():
            view_filter = "NOISE"
        elif self.radio_view_particle.isChecked():
            view_filter = "PARTICLE"
            
        return {
            "threshold": self.spin_threshold.value(),
            "min_area": self.spin_min_area.value(),
            "use_adaptive": self.chk_adaptive.isChecked(),
            "open_kernel": self.spin_open_kernel.value(),
            "close_kernel": self.spin_close_kernel.value(),
            "general_enabled": self.chk_general_enabled.isChecked(),
            "classification_enabled": self.group_classification.isChecked(),
            "bubble_enabled": self.chk_bubble_enabled.isChecked(),
            "bubble_show_text": self.chk_bubble_show_text.isChecked(),
            "bubble": self._bubble_to_params(),
            "classification": self._form_to_params(),
            "show_text": self.chk_show_text.isChecked(),
            "view_filter": view_filter,
        }

    def _run_debug_detection(self):
        """현재 파라미터로 백그라운드 검사를 시작.

        "파라미터별 View" 활성화 시 현재 콤보 인덱스를 target_view로 전달하여
        해당 뷰에 필요한 단계까지만 실행 (불필요한 전체 검사 생략).
        """
        if self._source_frame is None:
            return

        if self._worker is not None and self._worker.isRunning():
            self._worker_pending = True
            self._worker.cancel()
            return

        frame = self._source_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

        target_view = None
        if self.chk_auto_view.isChecked():
            target_view = self.combo_debug_view.currentIndex()

        self.lbl_debug_info.setText("검사 중...")
        self.lbl_debug_info.setStyleSheet("color: #f90; font-size: 11px;")

        self._worker = DebugDetectionWorker(self)
        self._worker.setup(frame, gray, self._collect_params(),
                           target_view=target_view)
        self._worker.finished.connect(self._on_debug_finished)
        self._worker.start()

    def _on_debug_finished(self, result: dict):
        """백그라운드 검사 완료 콜백."""
        self._worker = None

        if not result:
            self.lbl_debug_info.setText("취소됨")
            self.lbl_debug_info.setStyleSheet("color: #888; font-size: 11px;")
            if self._worker_pending:
                self._worker_pending = False
                self._run_debug_detection()
            return

        self._debug_images = result.get("debug_images", {})
        n_gen = result.get("n_general", 0)
        n_noise = result.get("n_noise", 0)
        n_particle = result.get("n_particle", 0)
        n_bub = result.get("n_bubble", 0)
        elapsed = result.get("elapsed", 0)

        self.lbl_debug_info.setText(
            f"Noise: {n_noise}, Particle: {n_particle}, Bubble: {n_bub}  "
            f"({elapsed * 1000:.0f}ms)"
        )
        self.lbl_debug_info.setStyleSheet("color: #0c0; font-size: 11px;")

        self._update_debug_view()

        if self._worker_pending:
            self._worker_pending = False
            self._run_debug_detection()

    def _update_debug_view(self):
        """콤보박스 선택에 따라 디버그 이미지를 View에 표시."""
        key_map = {
            0: "result",
            1: "gray",
            2: "threshold",
            3: "opened",
            4: "closed",
            5: "clahe",
            6: "diff_map",
            7: "binary",
            8: "bubble_candidates",
        }
        idx = self.combo_debug_view.currentIndex()
        key = key_map.get(idx, "result")
        img = self._debug_images.get(key)
        if img is not None:
            self.debug_view.set_image(img)
        else:
            self.debug_view.set_image(None)

    # ══════════════════════════════════════════════════════════
    #  분류 파라미터 ↔ 폼
    # ══════════════════════════════════════════════════════════

    def _form_to_params(self):
        return {
            "noise_contrast_threshold": self.spin_noise_contrast_threshold.value(),
            "threshold": self.spin_threshold.value(),
            "min_area": self.spin_min_area.value(),
            "use_adaptive": self.chk_adaptive.isChecked(),
            "open_kernel": self.spin_open_kernel.value(),
            "close_kernel": self.spin_close_kernel.value(),
            "general_enabled": self.chk_general_enabled.isChecked(),
            "classification_enabled": self.group_classification.isChecked(),
        }

    def _form_from_params(self, params: dict):
        if not params:
            return
        self.spin_noise_contrast_threshold.setValue(
            params.get("noise_contrast_threshold", RuleBasedClassifier.DEFAULT_NOISE_CONTRAST_THRESHOLD))
            
        if "threshold" in params:
            self.spin_threshold.setValue(params["threshold"])
        if "min_area" in params:
            self.spin_min_area.setValue(params["min_area"])
        if "use_adaptive" in params:
            self.chk_adaptive.setChecked(params["use_adaptive"])
        if "open_kernel" in params:
            self.spin_open_kernel.setValue(params["open_kernel"])
        if "close_kernel" in params:
            self.spin_close_kernel.setValue(params["close_kernel"])
        if "general_enabled" in params:
            self.chk_general_enabled.setChecked(bool(params["general_enabled"]))
        if "classification_enabled" in params:
            self.group_classification.setChecked(bool(params["classification_enabled"]))

    # ══════════════════════════════════════════════════════════
    #  Bubble 파라미터 ↔ 폼
    # ══════════════════════════════════════════════════════════

    def _bubble_to_params(self) -> dict:
        return {
            "enabled": self.chk_bubble_enabled.isChecked(),
            "bg_open_ksize": self.spin_bg_open_ksize.value(),
            "bg_smooth_sigma": self.spin_bg_smooth_sigma.value(),
            "use_clahe": self.chk_use_clahe.isChecked(),
            "clahe_clip": self.spin_clahe_clip.value(),
            "clahe_grid": self.spin_clahe_grid.value(),
            "denoise_mode": self.combo_denoise.currentText(),
            "median_ksize": self.spin_median_k.value(),
            "sigma_small": self.spin_sigma_small.value(),
            "sigma_large": self.spin_sigma_large.value(),
            "thr_k": self.spin_thr_k.value(),
            "morph_close_size": self.spin_morph_close.value(),
            "morph_open_size": self.spin_morph_open.value(),
            "min_diameter": self.spin_min_diam.value(),
            "max_diameter": self.spin_max_diam.value(),
            "circularity_min": self.spin_circularity.value(),
            "solidity_min": self.spin_solidity.value(),
            "max_aspect_ratio": self.spin_max_aspect.value(),
            "bubble_show_text": self.chk_bubble_show_text.isChecked(),
        }

    def _bubble_from_params(self, params: dict):
        if not params:
            return
        self.chk_bubble_enabled.setChecked(params.get("enabled", True))
        self.spin_bg_open_ksize.setValue(params.get("bg_open_ksize", 61))
        self.spin_bg_smooth_sigma.setValue(params.get("bg_smooth_sigma", 15.0))
        self.chk_use_clahe.setChecked(params.get("use_clahe", False))
        self.spin_clahe_clip.setValue(params.get("clahe_clip", 2.0))
        self.spin_clahe_grid.setValue(params.get("clahe_grid", 8))
        dm = params.get("denoise_mode", "median")
        idx = self.combo_denoise.findText(dm)
        if idx >= 0:
            self.combo_denoise.setCurrentIndex(idx)
        self.spin_median_k.setValue(params.get("median_ksize", 5))
        self.spin_sigma_small.setValue(params.get("sigma_small", 1.2))
        self.spin_sigma_large.setValue(params.get("sigma_large", 6.0))
        self.spin_thr_k.setValue(params.get("thr_k", 4.0))
        self.spin_morph_close.setValue(params.get("morph_close_size", 5))
        self.spin_morph_open.setValue(params.get("morph_open_size", 3))
        self.spin_min_diam.setValue(params.get("min_diameter", 8))
        self.spin_max_diam.setValue(params.get("max_diameter", 100))
        self.spin_circularity.setValue(params.get("circularity_min", 0.35))
        self.spin_solidity.setValue(params.get("solidity_min", 0.70))
        self.spin_max_aspect.setValue(params.get("max_aspect_ratio", 2.2))
        self.chk_bubble_show_text.setChecked(params.get("bubble_show_text", True))

    # ══════════════════════════════════════════════════════════
    #  파일 저장/로드
    # ══════════════════════════════════════════════════════════

    def _save_to_file(self):
        default_path = _default_rule_params_path(self.app_root)
        path, _ = QFileDialog.getSaveFileName(
            self, "파라미터 저장", default_path,
            "JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        data = {
            "classification": self._form_to_params(),
            "bubble_detection": self._bubble_to_params(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "저장 완료", f"저장했습니다.\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "저장 실패", str(e))

    def _load_from_file(self):
        default_path = _default_rule_params_path(self.app_root)
        path, _ = QFileDialog.getOpenFileName(
            self, "파라미터 불러오기", default_path,
            "JSON (*.json);;All (*.*)",
        )
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "classification" in data:
                self._form_from_params(data["classification"])
            else:
                self._form_from_params(data)
            if "bubble_detection" in data:
                self._bubble_from_params(data["bubble_detection"])
            QMessageBox.information(self, "불러오기 완료",
                                    "파라미터를 적용했습니다. '확인'을 누르면 검사에 반영됩니다.")
        except Exception as e:
            QMessageBox.warning(self, "불러오기 실패", str(e))

    def _on_ok(self):
        self.rule_based.set_params(self._form_to_params())
        self.bubble_params.set_params(self._bubble_to_params())
        self._auto_save()
        self.accept()

    def _auto_save(self):
        """rule_params.json에 현재 파라미터를 자동 저장 (앱 재시작 시 복원)."""
        path = _default_rule_params_path(self.app_root)
        data = {
            "classification": self._form_to_params(),
            "bubble_detection": self._bubble_to_params(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
