"""Basler 카메라 설정 다이얼로그: 노출/게인 조절, 현재값 로드, 파일 저장/로드."""
import os
import sys
import json
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QComboBox, QGroupBox, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt


def _project_root():
    """exe로 실행 시 exe가 있는 폴더, 아니면 프로젝트 루트."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_SETTINGS_PATH = os.path.join(_project_root(), "basler_settings.json")


class BaslerSettingsDialog(QDialog):
    def __init__(self, basler_camera, parent=None):
        super().__init__(parent)
        self.basler = basler_camera
        self.setWindowTitle("Basler 카메라 설정")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        # Exposure
        exp_group = QGroupBox("노출 (Exposure)")
        exp_layout = QVBoxLayout()
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("ExposureAuto:"))
        self.combo_exposure_auto = QComboBox()
        self.combo_exposure_auto.addItems(["Off", "Once", "Continuous"])
        exp_row.addWidget(self.combo_exposure_auto)
        exp_layout.addLayout(exp_row)
        exp_row2 = QHBoxLayout()
        exp_row2.addWidget(QLabel("ExposureTime (μs):"))
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(1, 1000000)
        self.spin_exposure.setDecimals(0)
        self.spin_exposure.setSuffix(" μs")
        exp_row2.addWidget(self.spin_exposure)
        exp_layout.addLayout(exp_row2)
        exp_group.setLayout(exp_layout)
        layout.addWidget(exp_group)

        # Gain
        gain_group = QGroupBox("게인 (Gain)")
        gain_layout = QVBoxLayout()
        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("GainAuto:"))
        self.combo_gain_auto = QComboBox()
        self.combo_gain_auto.addItems(["Off", "Once", "Continuous"])
        gain_row.addWidget(self.combo_gain_auto)
        gain_layout.addLayout(gain_row)
        gain_row2 = QHBoxLayout()
        gain_row2.addWidget(QLabel("Gain:"))
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0, 24)
        self.spin_gain.setDecimals(2)
        gain_row2.addWidget(self.spin_gain)
        gain_layout.addLayout(gain_row2)
        gain_group.setLayout(gain_layout)
        layout.addWidget(gain_group)

        # Buttons: Load from camera, Apply to camera, Save to file, Load from file
        btn_layout = QHBoxLayout()
        self.btn_load_cam = QPushButton("현재 세팅 로드 (카메라→화면)")
        self.btn_load_cam.clicked.connect(self.load_from_camera)
        btn_layout.addWidget(self.btn_load_cam)
        self.btn_apply = QPushButton("적용 (화면→카메라)")
        self.btn_apply.clicked.connect(self.apply_to_camera)
        btn_layout.addWidget(self.btn_apply)
        layout.addLayout(btn_layout)

        file_layout = QHBoxLayout()
        self.btn_save_file = QPushButton("파일에 저장")
        self.btn_save_file.clicked.connect(self.save_to_file)
        file_layout.addWidget(self.btn_save_file)
        self.btn_load_file = QPushButton("파일에서 로드")
        self.btn_load_file.clicked.connect(self.load_from_file)
        file_layout.addWidget(self.btn_load_file)
        layout.addLayout(file_layout)

        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.load_from_camera()

    def _params_to_ui(self, params: dict):
        exp_auto = str(params.get("ExposureAuto", "Off"))
        idx = self.combo_exposure_auto.findText(exp_auto, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.combo_exposure_auto.setCurrentIndex(idx)
        self.spin_exposure.setValue(float(params.get("ExposureTime", 10000)))

        gain_auto = str(params.get("GainAuto", "Off"))
        idx = self.combo_gain_auto.findText(gain_auto, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.combo_gain_auto.setCurrentIndex(idx)
        self.spin_gain.setValue(float(params.get("Gain", 0)))

    def _ui_to_params(self) -> dict:
        return {
            "ExposureAuto": self.combo_exposure_auto.currentText(),
            "ExposureTime": self.spin_exposure.value(),
            "GainAuto": self.combo_gain_auto.currentText(),
            "Gain": self.spin_gain.value(),
        }

    def load_from_camera(self):
        try:
            if self.basler is None or not self.basler.is_connected():
                QMessageBox.warning(self, "오류", "Basler 카메라가 연결되어 있지 않습니다.")
                return
            params = self.basler.get_parameters_dict()
            if not params:
                QMessageBox.warning(self, "오류", "카메라에서 설정을 읽어오지 못했습니다.")
                return
            self._params_to_ui(params)
            QMessageBox.information(self, "로드 완료", "카메라에서 현재 설정을 읽었습니다.")
        except Exception as e:
            print(f"BaslerSettingsDialog.load_from_camera exception: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "오류", f"카메라 설정을 불러오는 동안 오류가 발생했습니다.\n{e}")

    def apply_to_camera(self):
        if self.basler is None or not self.basler.is_connected():
            QMessageBox.warning(self, "오류", "Basler 카메라가 연결되어 있지 않습니다.")
            return
        params = self._ui_to_params()
        self.basler.set_parameters_dict(params)

        # 적용 후 다시 읽어서 실제 반영 상황 확인
        try:
            applied = self.basler.get_parameters_dict()
            if applied:
                self._params_to_ui(applied)

            if "ExposureTime" in params and "ExposureTime" in applied:
                requested = float(params["ExposureTime"])
                actual = float(applied["ExposureTime"])
                if abs(actual - requested) > 1.0:
                    QMessageBox.warning(
                        self,
                        "적용 경고",
                        f"ExposureTime이 요청값 {requested}μs로 설정되지 않았습니다.\n현재 {actual}μs입니다."
                    )
                else:
                    QMessageBox.information(
                        self,
                        "적용 확인",
                        f"ExposureTime이 {actual}μs로 정상 적용되었습니다."
                    )
        except Exception as e:
            print(f"BaslerSettingsDialog.apply_to_camera exception: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "오류", f"적용 후 값 확인 중 오류가 발생했습니다.\n{e}")
            return

        QMessageBox.information(self, "적용 완료", "설정을 카메라에 적용했습니다.")

    def save_to_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "설정 저장", DEFAULT_SETTINGS_PATH,
            "JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        params = self._ui_to_params()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "저장 완료", f"설정을 저장했습니다.\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "오류", f"저장 실패: {e}")

    def load_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "설정 로드", _project_root(),
            "JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                params = json.load(f)
            self._params_to_ui(params)
            QMessageBox.information(self, "로드 완료", "파일에서 설정을 불러왔습니다. '적용'을 누르면 카메라에 반영됩니다.")
        except Exception as e:
            QMessageBox.warning(self, "오류", f"로드 실패: {e}")
