"""Classification Tab: 이미지 로드 → 라벨 선택 → 분류 저장 → 학습 UI."""
import cv2
import numpy as np
import os
import shutil
import sys
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QListWidget, QGroupBox, QComboBox, QLineEdit, QProgressBar,
    QSplitter, QMessageBox, QSpinBox, QListWidgetItem,
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QSettings, QEvent
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence

from src.core.classification import (
    train_classifier, RuleBasedClassifier, DefectImageSaver,
    BUBBLE_LABEL, LEGACY_BUBBLE_LABELS, LEGACY_BUBBLE_FOLDERS,
)


class TrainWorker(QThread):
    """백그라운드에서 모델 학습 수행."""
    progress = pyqtSignal(int, int, float, float)  # epoch, total, loss, acc
    finished_signal = pyqtSignal(bool, str, str)  # success, error_message, model_path

    def __init__(self, data_dir, save_path, epochs, parent=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.save_path = save_path
        self.epochs = epochs

    def run(self):
        result = train_classifier(
            self.data_dir, self.save_path, self.epochs,
            progress_callback=lambda e, t, l, a: self.progress.emit(e, t, l, a),
        )
        success = result[0] if isinstance(result, tuple) else result
        error_msg = result[1] if isinstance(result, tuple) and len(result) > 1 else ""
        self.finished_signal.emit(success, error_msg or "", self.save_path)


# ═══════════════════════════════════════════════════════════════
# 경로 유틸리티 (기존 유지)
# ═══════════════════════════════════════════════════════════════

def _project_root():
    """실행 프로젝트 기준 루트 디렉터리 (절대 경로). exe로 실행 시 exe가 있는 폴더."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_path(path: str) -> str:
    """상대 경로면 exe/프로젝트 루트 기준 절대 경로로 변환."""
    if not path or not path.strip():
        return ""
    path = path.strip()
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_project_root(), path))


def _resolve_data_dir(path: str) -> str:
    """데이터 폴더 경로 해석 (상대 → exe/프로젝트 루트 기준)."""
    return _resolve_path(path)


# exe/프로젝트 루트에 두는 경로 설정 JSON
PATH_CONFIG_FILENAME = "path_config.json"
KEY_DATA_DIR = "ClassificationData"


def _path_config_path():
    return os.path.join(_project_root(), PATH_CONFIG_FILENAME)


def _load_path_config():
    """path_config.json에서 ClassificationData 경로 읽기. 없거나 키 없으면 ''."""
    path = _path_config_path()
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(KEY_DATA_DIR, "").strip()
    except Exception:
        return ""


def _save_path_config(data_dir: str):
    """path_config.json에 ClassificationData 경로 저장."""
    path = _path_config_path()
    try:
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        data[KEY_DATA_DIR] = data_dir
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"path_config 저장 실패: {e}")


# ═══════════════════════════════════════════════════════════════
# Classification Tab
# ═══════════════════════════════════════════════════════════════

class ClassificationTab(QWidget):
    """
    (V2.2 개편) Classification 이미지 분류 및 재학습 데이터 구축 전용 탭.
    드래그 앤 드롭 및 다중 선택, 라벨 일괄 변경 기능 포함.
    """
    
    # 모델 학습 완료 시 메인 윈도우에 알리기 위한 시그널 (새 모델 경로 전달)
    model_trained_signal = pyqtSignal(str)

    DEFAULT_DATA_DIR = "ClassificationData"
    DEFAULT_MODEL_PATH = "classification_model.onnx"

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._train_worker = None

        # 등록된 이미지 목록
        # 각 항목: {"path": str, "label": str, "image": np.ndarray | None}
        self._image_items = []

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # === 왼쪽: 이미지 미리보기 ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_preview = QLabel("이미지를 등록하세요\n\nDrag & Drop 또는 '이미지 로드' 버튼")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("background-color: #222; color: #888; font-size: 14px;")
        self.lbl_preview.setMinimumSize(400, 400)
        self.lbl_preview.setScaledContents(False)
        left_layout.addWidget(self.lbl_preview, 1)  # 이미지 영역 최대화

        # 미리보기 하단: 파일 정보 (컴팩트)
        self.lbl_file_info = QLabel("")
        self.lbl_file_info.setStyleSheet("color: gray; font-size: 27px; padding: 8px;")
        self.lbl_file_info.setFixedHeight(90)
        left_layout.addWidget(self.lbl_file_info, 0)  # 고정 크기

        # === 오른쪽: 이미지 목록 + 라벨 + 저장 + 학습 ===
        right_panel = QWidget()
        right_panel.setFixedWidth(360)
        right_layout = QVBoxLayout(right_panel)

        # ── 이미지 등록 ──
        load_group = QGroupBox("이미지 등록")
        load_layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        self.btn_load_images = QPushButton("이미지 로드 (L)")
        self.btn_load_images.setToolTip("여러 이미지를 선택하여 목록에 등록합니다.")
        self.btn_load_images.clicked.connect(self._load_images)
        btn_row.addWidget(self.btn_load_images)

        self.btn_load_folder = QPushButton("폴더 로드")
        self.btn_load_folder.setToolTip("폴더 내의 모든 이미지를 일괄 등록합니다.\n하위 라벨폴더 구조가 있으면 자동 인식합니다.")
        self.btn_load_folder.clicked.connect(self._load_folder)
        btn_row.addWidget(self.btn_load_folder)
        load_layout.addLayout(btn_row)

        # 이미지 목록
        self.list_images = QListWidget()
        self.list_images.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_images.currentRowChanged.connect(self._on_item_selected)
        self.list_images.setMinimumHeight(200)
        self.list_images.installEventFilter(self)
        load_layout.addWidget(self.list_images)

        # 라벨 변경 행
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("라벨:"))
        self.combo_label = QComboBox()
        self.combo_label.addItems(RuleBasedClassifier.LABELS)
        self.combo_label.setCurrentText("Particle")
        label_row.addWidget(self.combo_label)

        self.btn_apply_label = QPushButton("라벨 적용")
        self.btn_apply_label.setToolTip("선택된 항목들의 라벨을 변경합니다.")
        self.btn_apply_label.clicked.connect(self._apply_label)
        label_row.addWidget(self.btn_apply_label)
        load_layout.addLayout(label_row)

        # 삭제 행
        del_row = QHBoxLayout()
        self.btn_delete_selected = QPushButton("선택 삭제 (Del)")
        self.btn_delete_selected.clicked.connect(self._delete_selected)
        del_row.addWidget(self.btn_delete_selected)

        self.btn_clear_all = QPushButton("전체 삭제")
        self.btn_clear_all.clicked.connect(self._clear_all)
        del_row.addWidget(self.btn_clear_all)
        load_layout.addLayout(del_row)

        # 분류 저장 버튼
        self.btn_save_classification = QPushButton("▶ 분류 저장")
        self.btn_save_classification.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 6px; "
            "background-color: #2d5f2d; color: white;"
        )
        self.btn_save_classification.setToolTip(
            "등록된 이미지를 데이터 폴더의 라벨별 하위 폴더에 복사합니다.\n"
            "예: ClassificationData/Particle/, ClassificationData/Noise_Dust/"
        )
        self.btn_save_classification.clicked.connect(self._save_classification)
        load_layout.addWidget(self.btn_save_classification)

        self.lbl_save_status = QLabel("")
        self.lbl_save_status.setStyleSheet("color: gray; font-size: 11px;")
        load_layout.addWidget(self.lbl_save_status)

        load_group.setLayout(load_layout)
        right_layout.addWidget(load_group)

        # ── 학습 데이터 ──
        data_group = QGroupBox("학습 데이터")
        data_layout = QVBoxLayout()
        data_layout.addWidget(QLabel("데이터 폴더:"))
        dir_row = QHBoxLayout()
        _saved_data_dir = _load_path_config()
        if not _saved_data_dir:
            _saved_data_dir = QSettings().value("ClassificationTab/data_dir", "", type=str)
        self.txt_data_dir = QLineEdit((_saved_data_dir or "").strip() or self.DEFAULT_DATA_DIR)
        dir_row.addWidget(self.txt_data_dir)
        self.btn_browse_data = QPushButton("...")
        self.btn_browse_data.setFixedWidth(30)
        self.btn_browse_data.clicked.connect(self._browse_data_dir)
        dir_row.addWidget(self.btn_browse_data)
        data_layout.addLayout(dir_row)

        self.lbl_data_summary = QLabel("폴더를 선택하세요")
        self.lbl_data_summary.setWordWrap(True)
        self.lbl_data_summary.setStyleSheet("color: gray; font-size: 11px;")
        data_layout.addWidget(self.lbl_data_summary)

        self.btn_refresh_data = QPushButton("데이터 폴더 새로고침")
        self.btn_refresh_data.clicked.connect(self._refresh_data_summary)
        data_layout.addWidget(self.btn_refresh_data)

        self.get_resolved_data_dir = lambda: _resolve_data_dir(self.txt_data_dir.text().strip())

        data_group.setLayout(data_layout)
        right_layout.addWidget(data_group)

        # ── 학습 ──
        train_group = QGroupBox("학습")
        train_layout = QVBoxLayout()

        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(5, 500)
        self.spin_epochs.setValue(50)  # Fine-tuning을 위해 기본 50 epoch 권장
        epoch_row.addWidget(self.spin_epochs)
        train_layout.addLayout(epoch_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("모델 저장:"))
        self.txt_model_path = QLineEdit(self.DEFAULT_MODEL_PATH)
        model_row.addWidget(self.txt_model_path)
        self.btn_browse_model = QPushButton("...")
        self.btn_browse_model.setFixedWidth(30)
        self.btn_browse_model.clicked.connect(self._browse_model_path)
        model_row.addWidget(self.btn_browse_model)
        train_layout.addLayout(model_row)

        self.btn_train = QPushButton("학습 시작")
        self.btn_train.setStyleSheet("font-weight: bold; font-size: 14px; padding: 8px;")
        self.btn_train.clicked.connect(self._start_training)
        train_layout.addWidget(self.btn_train)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        train_layout.addWidget(self.progress_bar)

        self.lbl_train_status = QLabel("대기 중")
        self.lbl_train_status.setStyleSheet("font-size: 11px;")
        train_layout.addWidget(self.lbl_train_status)

        train_group.setLayout(train_layout)
        right_layout.addWidget(train_group)

        right_layout.addStretch()

        # 스플리터로 배치
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # 드래그앤드롭 지원
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ========== 단축키 ==========

    def eventFilter(self, source, event):
        if source == self.list_images and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5,
                       Qt.Key.Key_L, Qt.Key.Key_Delete):
                self.keyPressEvent(event)
                return True
        return super().eventFilter(source, event)

    def keyPressEvent(self, event):
        """Classification 탭 전용 단축키 처리."""
        key = event.key()
        if key == Qt.Key.Key_L:
            self._load_images()
        elif key == Qt.Key.Key_Delete:
            self._delete_selected()
        elif key == Qt.Key.Key_1:
            if self.combo_label.count() > 0:
                self.combo_label.setCurrentIndex(0)
                self._apply_label()
        elif key == Qt.Key.Key_2:
            if self.combo_label.count() > 1:
                self.combo_label.setCurrentIndex(1)
                self._apply_label()
        elif key == Qt.Key.Key_3:
            if self.combo_label.count() > 2:
                self.combo_label.setCurrentIndex(2)
                self._apply_label()
        elif key == Qt.Key.Key_4:
            if self.combo_label.count() > 3:
                self.combo_label.setCurrentIndex(3)
                self._apply_label()
        elif key == Qt.Key.Key_5:
            if self.combo_label.count() > 4:
                self.combo_label.setCurrentIndex(4)
                self._apply_label()
        elif key == Qt.Key.Key_Up:
            current = self.list_images.currentRow()
            if current > 0:
                self.list_images.setCurrentRow(current - 1)
        elif key == Qt.Key.Key_Down:
            current = self.list_images.currentRow()
            if current < self.list_images.count() - 1:
                self.list_images.setCurrentRow(current + 1)
        else:
            super().keyPressEvent(event)

    # ========== 드래그앤드롭 ==========

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    ext = os.path.splitext(path)[1].lower()
                    if ext in self.IMAGE_EXTENSIONS or os.path.isdir(path):
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = []
            folder_paths = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if os.path.isdir(path):
                        folder_paths.append(path)
                    else:
                        ext = os.path.splitext(path)[1].lower()
                        if ext in self.IMAGE_EXTENSIONS:
                            paths.append(path)
            # 폴더가 있으면 폴더 로드
            for fp in folder_paths:
                self._load_folder_path(fp)
            # 개별 이미지 파일 등록
            if paths:
                self._register_images(paths)
            event.acceptProposedAction()
            return
        event.ignore()

    # ========== 이미지 로드 ==========

    def _load_images(self):
        """다중 파일 선택 다이얼로그로 이미지 등록."""
        fnames, _ = QFileDialog.getOpenFileNames(
            self, "이미지 로드", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All (*.*)",
        )
        if fnames:
            self._register_images(fnames)

    def _load_folder(self):
        """폴더 선택 다이얼로그로 폴더 내 이미지 일괄 등록."""
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택", "")
        if folder:
            self._load_folder_path(folder)

    def _load_folder_path(self, folder_path: str):
        """폴더 내의 모든 이미지를 등록. 하위 라벨 폴더 구조 자동 인식."""
        if not os.path.isdir(folder_path):
            return

        existing_paths = {item["path"] for item in self._image_items}
        new_items = []
        exclude_dirs = {"_originals", "_annotations", "__pycache__"}

        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            # 현재 폴더명을 라벨 후보로 사용
            folder_name = os.path.basename(root)
            # 라벨 폴더인지 판단 (알려진 라벨명과 일치하면)
            known_labels = set(RuleBasedClassifier.LABELS)
            # Bubble 레거시 폴더명도 인식
            legacy_labels = set(LEGACY_BUBBLE_FOLDERS) if hasattr(DefectImageSaver, '__class__') else set()
            auto_label = None
            if folder_name in known_labels:
                auto_label = folder_name
            elif folder_name in LEGACY_BUBBLE_FOLDERS:
                auto_label = BUBBLE_LABEL

            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext not in self.IMAGE_EXTENSIONS:
                    continue
                full_path = os.path.normpath(os.path.join(root, file))
                if full_path in existing_paths:
                    continue
                new_items.append({
                    "path": full_path,
                    "label": auto_label or self.combo_label.currentText(),
                    "image": None,
                })

        if not new_items:
            QMessageBox.information(self, "알림", "폴더 내에 새로운 이미지가 없습니다.")
            return

        self._image_items.extend(new_items)
        self._refresh_list()
        self.lbl_save_status.setText(f"{len(new_items)}개 이미지 등록됨")

    def _register_images(self, paths: list):
        """경로 목록에서 이미지를 등록."""
        existing_paths = {item["path"] for item in self._image_items}
        added = 0
        default_label = self.combo_label.currentText()
        for p in paths:
            norm_path = os.path.normpath(p)
            if norm_path in existing_paths:
                continue
            self._image_items.append({
                "path": norm_path,
                "label": default_label,
                "image": None,
            })
            existing_paths.add(norm_path)
            added += 1

        if added > 0:
            self._refresh_list()
            # 첫 번째 새로 추가된 항목 선택
            self.list_images.setCurrentRow(len(self._image_items) - added)
            self.lbl_save_status.setText(f"{added}개 이미지 등록됨")
        else:
            self.lbl_save_status.setText("이미 등록된 이미지입니다.")

    # ========== 목록 관리 ==========

    def _refresh_list(self):
        """이미지 목록 UI 갱신."""
        self.list_images.blockSignals(True)
        current_row = self.list_images.currentRow()
        self.list_images.clear()
        for item in self._image_items:
            fname = os.path.basename(item["path"])
            label = item["label"]
            display_text = f"[{label}]  {fname}"
            list_item = QListWidgetItem(display_text)
            # 라벨별 색상 표시
            label_colors = {
                "Particle": "#ff4444",
                "Noise_Dust": "#8888ff",
                BUBBLE_LABEL: "#ffff44",
                "Fiber": "#ff8800",
            }
            color = label_colors.get(label, "#88ff88")
            list_item.setForeground(QLabel().palette().text().color())
            list_item.setToolTip(item["path"])
            self.list_images.addItem(list_item)

        # 선택 복원
        if current_row >= 0 and current_row < self.list_images.count():
            self.list_images.setCurrentRow(current_row)
        elif self.list_images.count() > 0:
            self.list_images.setCurrentRow(0)
        self.list_images.blockSignals(False)

        # 선택 상태 갱신
        if self.list_images.currentRow() >= 0:
            self._on_item_selected(self.list_images.currentRow())

    def _on_item_selected(self, row: int):
        """목록에서 항목 선택 시 미리보기 표시."""
        if row < 0 or row >= len(self._image_items):
            self.lbl_preview.setText("이미지를 등록하세요\n\nDrag & Drop 또는 '이미지 로드' 버튼")
            self.lbl_file_info.setText("")
            return

        item = self._image_items[row]

        # 이미지 로드 (캐시)
        if item["image"] is None:
            try:
                buf = np.fromfile(item["path"], dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                item["image"] = img
            except Exception:
                item["image"] = None

        if item["image"] is None:
            self.lbl_preview.setText(f"이미지를 열 수 없습니다\n{item['path']}")
            self.lbl_file_info.setText("")
            return

        # 이미지 표시
        img = item["image"]
        h, w = img.shape[:2]
        target_w = max(1, self.lbl_preview.width())
        target_h = max(1, self.lbl_preview.height())
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw < 1 or nh < 1:
            return

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        resized = np.ascontiguousarray(resized)
        qt_img = QImage(resized.data, nw, nh, 3 * nw, QImage.Format.Format_BGR888)
        self.lbl_preview.setPixmap(QPixmap.fromImage(qt_img.copy()))

        # 파일 정보 표시
        file_size = os.path.getsize(item["path"]) if os.path.isfile(item["path"]) else 0
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / 1024 / 1024:.1f} MB"
        self.lbl_file_info.setText(
            f"{os.path.basename(item['path'])}  |  {w}×{h}  |  {size_str}  |  라벨: {item['label']}"
        )

        # 콤보에 현재 라벨 반영
        self.combo_label.blockSignals(True)
        idx = self.combo_label.findText(item["label"])
        if idx >= 0:
            self.combo_label.setCurrentIndex(idx)
        self.combo_label.blockSignals(False)

    def _apply_label(self):
        """선택된 항목들의 라벨을 변경."""
        selected_rows = [idx.row() for idx in self.list_images.selectedIndexes()]
        if not selected_rows:
            # 현재 행이라도 적용
            current = self.list_images.currentRow()
            if current >= 0:
                selected_rows = [current]
            else:
                return

        new_label = self.combo_label.currentText()
        for row in selected_rows:
            if 0 <= row < len(self._image_items):
                self._image_items[row]["label"] = new_label

        self._refresh_list()
        self.lbl_save_status.setText(f"{len(selected_rows)}개 항목 라벨 → {new_label}")

    def _delete_selected(self):
        """선택된 항목들을 목록에서 삭제."""
        selected_rows = sorted([idx.row() for idx in self.list_images.selectedIndexes()], reverse=True)
        if not selected_rows:
            current = self.list_images.currentRow()
            if current >= 0:
                selected_rows = [current]
            else:
                return

        for row in selected_rows:
            if 0 <= row < len(self._image_items):
                del self._image_items[row]

        self._refresh_list()
        if not self._image_items:
            self.lbl_preview.setText("이미지를 등록하세요\n\nDrag & Drop 또는 '이미지 로드' 버튼")
            self.lbl_file_info.setText("")
        self.lbl_save_status.setText(f"{len(selected_rows)}개 항목 삭제됨")

    def _clear_all(self):
        """전체 목록 초기화."""
        if not self._image_items:
            return
        reply = QMessageBox.question(
            self, "확인", f"등록된 {len(self._image_items)}개 이미지를 모두 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._image_items.clear()
        self._refresh_list()
        self.lbl_preview.setText("이미지를 등록하세요\n\nDrag & Drop 또는 '이미지 로드' 버튼")
        self.lbl_file_info.setText("")
        self.lbl_save_status.setText("전체 삭제됨")

    # ========== 분류 저장 ==========

    def _save_classification(self):
        """등록된 이미지를 데이터 폴더의 라벨별 하위 폴더에 복사."""
        if not self._image_items:
            QMessageBox.information(self, "알림", "등록된 이미지가 없습니다.")
            return

        try:
            from PyQt6.QtWidgets import QApplication
            
            data_dir = _resolve_data_dir(self.txt_data_dir.text().strip())
            if not data_dir:
                QMessageBox.warning(self, "오류", "데이터 폴더를 설정하세요.")
                return

            saved = 0
            skipped = 0
            errors = 0
            total = len(self._image_items)

            self.btn_save_classification.setEnabled(False)

            for idx, item in enumerate(self._image_items):
                try:
                    label = item["label"]
                    # Bubble 레거시 라벨 정리
                    if label in LEGACY_BUBBLE_LABELS:
                        label = BUBBLE_LABEL
                    safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")

                    label_dir = os.path.join(data_dir, safe_label)
                    os.makedirs(label_dir, exist_ok=True)

                    src_path = item["path"]
                    dst_name = os.path.basename(src_path)
                    dst_path = os.path.join(label_dir, dst_name)

                    # 이미 같은 이름 파일이 있으면 넘버링
                    if os.path.exists(dst_path) and os.path.normpath(src_path) != os.path.normpath(dst_path):
                        base, ext = os.path.splitext(dst_name)
                        counter = 1
                        while os.path.exists(dst_path):
                            dst_path = os.path.join(label_dir, f"{base}_{counter}{ext}")
                            counter += 1

                    if os.path.normpath(src_path) == os.path.normpath(dst_path):
                        skipped += 1
                        continue
                    shutil.copy2(src_path, dst_path)
                    saved += 1
                except Exception as e:
                    print(f"분류 저장 오류: {e}")
                    errors += 1

                # UI 응답 없음(프리징) 방지를 위해 10개마다 이벤트 루프 처리
                if idx % 10 == 0:
                    self.lbl_save_status.setText(f"저장 중... ({idx+1}/{total})")
                    QApplication.processEvents()

            # 데이터 폴더 경로 저장
            _save_path_config(data_dir)

            msg = f"저장 완료: {saved}개 저장"
            if skipped:
                msg += f", {skipped}개 이미 존재"
            if errors:
                msg += f", {errors}개 오류"
            self.lbl_save_status.setText(msg)
            
            try:
                self._refresh_data_summary()
            except Exception as e:
                print(f"새로고침 중 오류: {e}")

            QMessageBox.information(self, "분류 저장 완료", msg)

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            QMessageBox.critical(self, "치명적 오류 방지", f"저장 중 예상치 못한 시스템 오류가 발생했습니다.\n(프로그램 종료 방지됨)\n\n오류 내용:\n{str(e)}")
            print(err_msg)
        finally:
            self.btn_save_classification.setEnabled(True)

    # ========== 학습 데이터/모델 ==========

    def _browse_data_dir(self):
        d = QFileDialog.getExistingDirectory(self, "데이터 폴더 선택", "")
        if d:
            self.txt_data_dir.setText(d)
            _save_path_config(d)
            self._refresh_data_summary()

    def _browse_model_path(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "모델 저장 경로",
            _resolve_path(self.txt_model_path.text()),
            "ONNX Model (*.onnx);;PyTorch Model (*.pth);;All (*.*)",
        )
        if fname:
            self.txt_model_path.setText(fname)

    def _refresh_data_summary(self):
        """학습에 사용되는 데이터만 표기 (_originals, _annotations 제외)."""
        data_dir = _resolve_data_dir(self.txt_data_dir.text().strip())
        if not data_dir or not os.path.isdir(data_dir):
            self.lbl_data_summary.setText("폴더가 존재하지 않습니다.")
            return

        exclude_dirs = ("_originals", "_annotations")
        summary_parts = []
        total = 0
        for item in sorted(os.listdir(data_dir)):
            if item.startswith(".") or item in exclude_dirs:
                continue
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                count = sum(1 for f in os.listdir(item_path)
                            if os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS)
                if count > 0:
                    summary_parts.append(f"  {item}: {count}장")
                    total += count

        if summary_parts:
            self.lbl_data_summary.setText(f"총 {total}장\n" + "\n".join(summary_parts))
        else:
            self.lbl_data_summary.setText("데이터가 없습니다.")

    # ========== 학습 ==========

    def _start_training(self):
        if self._train_worker and self._train_worker.isRunning():
            QMessageBox.warning(self, "알림", "이미 학습이 진행 중입니다.")
            return

        data_dir = _resolve_data_dir(self.txt_data_dir.text().strip())
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(self, "오류", f"데이터 폴더가 존재하지 않습니다.\n{data_dir}")
            return

        model_path = _resolve_path(self.txt_model_path.text().strip())
        if not model_path:
            QMessageBox.warning(self, "오류", "모델 저장 경로를 설정하세요.")
            return

        epochs = self.spin_epochs.value()
        self.btn_train.setEnabled(False)
        self.lbl_train_status.setText("학습 시작...")
        self.progress_bar.setValue(0)

        self._train_worker = TrainWorker(data_dir, model_path, epochs, self)
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished_signal.connect(self._on_train_finished)
        self._train_worker.start()

    def _on_train_progress(self, epoch, total, loss, acc):
        pct = int(epoch / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.lbl_train_status.setText(f"Epoch {epoch}/{total}  loss={loss:.4f}  acc={acc:.2f}%")

    def _on_train_finished(self, success, error_message="", model_path=""):
        self.btn_train.setEnabled(True)
        if success:
            self.progress_bar.setValue(100)
            self.lbl_train_status.setText("학습 완료!")
            QMessageBox.information(self, "학습 완료", "모델 학습이 완료되었습니다.")
            
            # 새로 학습된 ONNX 모델 경로로 메인 윈도우 갱신 시그널 발생 (확장자 변경)
            onnx_path = model_path.rsplit('.', 1)[0] + '.onnx'
            self.model_trained_signal.emit(onnx_path)
        else:
            self.lbl_train_status.setText(f"학습 실패: {error_message}")
            QMessageBox.warning(self, "학습 실패",
                f"학습 중 오류가 발생했습니다.\n\n{error_message}")
