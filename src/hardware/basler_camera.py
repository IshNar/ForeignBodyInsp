from .camera_interface import CameraSource
import numpy as np
import os
try:
    from pypylon import pylon
except ImportError:
    pylon = None

PREFERRED_MODEL = "acA1920-25gm"


def _find_basler_device():
    """첫 번째 Basler 디바이스 반환. PREFERRED_MODEL이 있으면 해당 모델 우선."""
    if pylon is None:
        return None
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        return None
    for dev in devices:
        name = dev.GetModelName() or ""
        if PREFERRED_MODEL and PREFERRED_MODEL in name:
            return factory.CreateDevice(dev)
    return factory.CreateDevice(devices[0])


class BaslerCamera(CameraSource):
    def __init__(self):
        self.camera = None
        self.converter = None
        self._last_error = None  # 연결 실패 시 원인 메시지
        if pylon is not None:
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def is_connected(self):
        return self.camera is not None and self.camera.IsOpen()

    def get_last_error(self):
        """마지막 연결 실패 시 오류 메시지 (UI 표시용)."""
        return self._last_error or "Unknown error"

    def open(self):
        """연결 시도. 성공 시 True, 실패 시 False (원인은 get_last_error())."""
        self._last_error = None
        if pylon is None:
            self._last_error = (
                "pypylon이 설치되어 있지 않습니다.\n"
                "터미널에서: pip install pypylon\n"
                "그리고 Basler Pylon SDK(런타임)를 설치해야 합니다."
            )
            return False

        try:
            dev = _find_basler_device()
            if dev is None:
                self._last_error = (
                    "Basler 카메라를 찾을 수 없습니다.\n\n"
                    "• acA1920-25gm은 GigE 카메라입니다. 랜 케이블 연결 후:\n"
                    "  - PC 네트워크 어댑터 IP를 카메라와 같은 대역으로 설정 (예: 192.168.1.10)\n"
                    "  - 또는 Basler 'pylon IP Configurator'로 카메라 IP 확인/설정\n"
                    "• Basler Pylon SDK(드라이버 포함) 설치 여부 확인\n"
                    "  https://www.baslerweb.com/en/downloads/software-downloads/"
                )
                return False
            self.camera = pylon.InstantCamera(dev)
            self.camera.Open()
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            grabResult = self.camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
            if grabResult and grabResult.GrabSucceeded():
                grabResult.Release()
            return True
        except Exception as e:
            self._last_error = str(e)
            print(f"Error opening Basler camera: {e}")
            import traceback
            traceback.print_exc()
            if self.camera is not None:
                try:
                    if self.camera.IsGrabbing():
                        self.camera.StopGrabbing()
                    if self.camera.IsOpen():
                        self.camera.Close()
                except Exception:
                    pass
                self.camera = None
            return False

    def close(self):
        if self.camera is not None:
            try:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
            except Exception as e:
                print(f"Error closing Basler camera: {e}")
            self.camera = None  # 해제 후 재연결 시 새 인스턴스 사용

    def grab_frame(self) -> np.ndarray:
        if self.camera and self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                grabResult.Release()
                return img.copy()
            grabResult.Release()
        return None

    def set_exposure(self, value):
        if self.camera and self.camera.IsOpen():
            try:
                self._safe_set_property("ExposureAuto", "Off")
                self._safe_set_property("ExposureTime", float(value))
            except Exception as e:
                print(f"Error setting exposure: {e}")

    def get_exposure(self):
        if self.camera and self.camera.IsOpen():
            try:
                val = self._safe_get_property("ExposureTime")
                return float(val) if val is not None else 0
            except Exception:
                return 0
        return 0

    def _has_property(self, name: str) -> bool:
        if self.camera is None or not self.camera.IsOpen():
            return False
        try:
            return hasattr(self.camera, name)
        except Exception as e:
            print(f"BaslerCamera._has_property({name}) failed: {e}")
            return False

    def _resolve_property_name(self, name: str) -> str:
        if name == "ExposureTime":
            for opt in ["ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"]:
                if self._has_property(opt): return opt
        elif name == "Gain":
            for opt in ["Gain", "GainAbs", "GainRaw"]:
                if self._has_property(opt): return opt
        return name

    def _safe_get_property(self, name: str):
        real_name = self._resolve_property_name(name)
        if not self._has_property(real_name):
            return None
        try:
            prop = getattr(self.camera, real_name)
        except Exception as e:
            print(f"BaslerCamera._safe_get_property({name}): getattr failed: {e}")
            return None
        try:
            if hasattr(prop, "GetValue"):
                return prop.GetValue()
        except Exception as e:
            print(f"BaslerCamera._safe_get_property({name}) failed: {e}")
        return None

    def _safe_set_property(self, name: str, value) -> bool:
        real_name = self._resolve_property_name(name)
        if not self._has_property(real_name):
            print(f"BaslerCamera._safe_set_property({name}): property missing (resolved: {real_name})")
            return False
        try:
            prop = getattr(self.camera, real_name)
        except Exception as e:
            print(f"BaslerCamera._safe_set_property({name}): getattr failed: {e}")
            return False

        try:
            if hasattr(prop, "IsWritable") and not prop.IsWritable():
                print(f"BaslerCamera._safe_set_property({name}): property not writable")
                return False
        except Exception as e:
            print(f"BaslerCamera._safe_set_property({name}): IsWritable check failed: {e}")
            return False

        # ExposureTime/Gain 값이 제한 범위 안에 있도록 정밀하게 조정
        try:
            if name in ("ExposureTime", "Gain") and hasattr(prop, "GetMin") and hasattr(prop, "GetMax") and hasattr(prop, "GetInc"):
                min_v = prop.GetMin()
                max_v = prop.GetMax()
                inc = prop.GetInc()
                
                # Raw의 경우 정수형 처리, 아닐 경우 float
                is_raw = "Raw" in real_name
                if is_raw:
                    value = int(float(value))
                    min_v, max_v, inc = int(min_v), int(max_v), int(inc)
                else:
                    value = float(value)
                    
                if value < min_v:
                    value = min_v
                elif value > max_v:
                    value = max_v
                if inc > 0:
                    value = min_v + round((value - min_v) / inc) * inc
                    
                if is_raw:
                    value = int(value)
        except Exception as e:
            print(f"BaslerCamera._safe_set_property({name}) range adjust failed: {e}")

        try:
            if hasattr(prop, "SetValue"):
                print(f"BaslerCamera._safe_set_property: setting {real_name} to {value}")
                prop.SetValue(value)
                # 읽어서 값 확인
                actual = None
                try:
                    if hasattr(prop, "GetValue"):
                        actual = prop.GetValue()
                        print(f"BaslerCamera._safe_set_property: {real_name} actual after set = {actual}")
                except Exception as e2:
                    print(f"BaslerCamera._safe_set_property: {real_name} GetValue after set failed: {e2}")
                if actual is not None and float(actual) != float(value):
                    print(f"BaslerCamera._safe_set_property: {real_name} set mismatch: requested={value}, actual={actual}")
                return True
        except Exception as e:
            print(f"BaslerCamera._safe_set_property({name}) failed: {e}")
        return False

    def get_parameters_dict(self):
        out = {}
        if not self.camera or not self.camera.IsOpen():
            return out

        exposure = self._safe_get_property("ExposureTime")
        if exposure is not None:
            out["ExposureTime"] = exposure

        gain = self._safe_get_property("Gain")
        if gain is not None:
            out["Gain"] = gain

        exposure_auto = self._safe_get_property("ExposureAuto")
        if exposure_auto is not None:
            out["ExposureAuto"] = str(exposure_auto)

        gain_auto = self._safe_get_property("GainAuto")
        if gain_auto is not None:
            out["GainAuto"] = str(gain_auto)

        return out

    def set_parameters_dict(self, params: dict):
        if not self.camera or not self.camera.IsOpen():
            print("BaslerCamera.set_parameters_dict: camera not open")
            return

        # 변화를 적용하기 전에 프레임 스트리밍 일시 중지
        was_grabbing = False
        try:
            was_grabbing = self.camera.IsGrabbing()
            if was_grabbing:
                self.camera.StopGrabbing()
        except Exception as e:
            print(f"BaslerCamera.set_parameters_dict: stop grabbing failed: {e}")

        if "ExposureAuto" in params:
            self._safe_set_property("ExposureAuto", str(params["ExposureAuto"]))
        if "GainAuto" in params:
            self._safe_set_property("GainAuto", str(params["GainAuto"]))
        if "ExposureTime" in params:
            self._safe_set_property("ExposureAuto", "Off")
            self._safe_set_property("ExposureTime", float(params["ExposureTime"]))
        if "Gain" in params:
            self._safe_set_property("GainAuto", "Off")
            self._safe_set_property("Gain", float(params["Gain"]))

        # 변경 후에 스트리밍 다시 시작
        try:
            if was_grabbing and not self.camera.IsGrabbing():
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            print(f"BaslerCamera.set_parameters_dict: restart grabbing failed: {e}")
