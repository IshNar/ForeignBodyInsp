from .camera_interface import CameraSource
import cv2
import numpy as np
import os
from src.core.utils import imread_safe

class FileCamera(CameraSource):
    def __init__(self, source_path):
        self.source_path = source_path
        self.image = None

    def open(self):
        if os.path.exists(self.source_path):
            self.image = imread_safe(self.source_path)
            if self.image is None:
                print(f"Failed to load image from {self.source_path}")
                return False
            return True
        else:
            print(f"File not found: {self.source_path}")
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
