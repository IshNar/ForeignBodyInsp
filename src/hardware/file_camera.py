from .camera_interface import CameraSource
import cv2
import numpy as np
import os

class FileCamera(CameraSource):
    def __init__(self, source_path):
        self.source_path = source_path
        self.image = None

    def open(self):
        # Basic existence check
        exists = os.path.exists(self.source_path)
        print(f"Opening image: {self.source_path} | exists={exists}")
        if not exists:
            print(f"File not found: {self.source_path}")
            return False

        try:
            size = os.path.getsize(self.source_path)
            print(f"File size: {size} bytes")
        except Exception as e:
            print(f"Could not get file size: {e}")

        # Try a Unicode-safe read: load bytes then decode with OpenCV
        try:
            with open(self.source_path, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Failed to read file bytes: {e}")
            return False

        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                print("cv2.imdecode returned None, trying cv2.imread fallback")
                img = cv2.imread(self.source_path)

            if img is None:
                print(f"Failed to load image from {self.source_path} (imdecode+imread)")
                return False

            self.image = img
            print(f"Loaded image shape: {self.image.shape}")
            return True
        except Exception as e:
            print(f"Exception while decoding image: {e}")
            return False

    def close(self):
        self.image = None

    def grab_frame(self) -> np.ndarray:
        # In a real loop, we might want to simulate frame rate or return sequence
        # For now, just return the loaded static image
        if self.image is not None:
            return self.image.copy()
        return None

    def set_exposure(self, value):
        # Not applicable for file source
        pass

    def get_exposure(self):
        return 0
