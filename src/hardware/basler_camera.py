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
                self.camera.ExposureAuto.SetValue("Off")
                self.camera.ExposureTime.SetValue(float(value))
            except Exception as e:
                print(f"Error setting exposure: {e}")

    def get_exposure(self):
        if self.camera and self.camera.IsOpen():
            try:
                return self.camera.ExposureTime.GetValue()
            except Exception:
                return 0
        return 0

    def get_parameters_dict(self):
        out = {}
        if not self.camera or not self.camera.IsOpen():
            return out
        try:
            out["ExposureTime"] = self.camera.ExposureTime.GetValue()
        except Exception:
            pass
        try:
            out["Gain"] = self.camera.Gain.GetValue()
        except Exception:
            pass
        try:
            out["ExposureAuto"] = str(self.camera.ExposureAuto.GetValue()) if hasattr(self.camera.ExposureAuto, "GetValue") else "Off"
        except Exception:
            pass
        try:
            out["GainAuto"] = str(self.camera.GainAuto.GetValue()) if hasattr(self.camera.GainAuto, "GetValue") else "Off"
        except Exception:
            pass
        return out

    def set_parameters_dict(self, params: dict):
        if not self.camera or not self.camera.IsOpen():
            return
        try:
            if "ExposureAuto" in params and hasattr(self.camera, "ExposureAuto"):
                self.camera.ExposureAuto.SetValue(str(params["ExposureAuto"]))
        except Exception as e:
            print(f"Set ExposureAuto: {e}")
        try:
            if "GainAuto" in params and hasattr(self.camera, "GainAuto"):
                self.camera.GainAuto.SetValue(str(params["GainAuto"]))
        except Exception as e:
            print(f"Set GainAuto: {e}")
        try:
            if "ExposureTime" in params:
                self.camera.ExposureAuto.SetValue("Off")
                self.camera.ExposureTime.SetValue(float(params["ExposureTime"]))
        except Exception as e:
            print(f"Set ExposureTime: {e}")
        try:
            if "Gain" in params:
                self.camera.GainAuto.SetValue("Off")
                self.camera.Gain.SetValue(float(params["Gain"]))
        except Exception as e:
            print(f"Set Gain: {e}")
