from .camera_interface import CameraSource
import numpy as np
try:
    from pypylon import pylon
except ImportError:
    pylon = None

class BaslerCamera(CameraSource):
    def __init__(self):
        self.camera = None
        self.converter = None
        if pylon is not None:
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def open(self):
        if pylon is None:
            raise RuntimeError("pypylon not installed")

        try:
            # Create an instant camera object with the camera device found first.
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            # Start Grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            return True
        except Exception as e:
            print(f"Error opening camera: {e}")
            return False

    def close(self):
        if self.camera and self.camera.IsOpen():
            self.camera.StopGrabbing()
            self.camera.Close()

    def grab_frame(self) -> np.ndarray:
        if self.camera and self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                grabResult.Release()
                return img
            grabResult.Release()
        return None

    def set_exposure(self, value):
        if self.camera and self.camera.IsOpen():
            try:
                # Value usually in microseconds
                # Ensure AutoExposure is off
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
