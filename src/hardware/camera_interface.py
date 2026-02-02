from abc import ABC, abstractmethod
import numpy as np

class CameraSource(ABC):
    @abstractmethod
    def open(self):
        """Opens the camera connection."""
        pass

    @abstractmethod
    def close(self):
        """Closes the camera connection."""
        pass

    @abstractmethod
    def grab_frame(self) -> np.ndarray:
        """Captures and returns a single frame (numpy array)."""
        pass

    @abstractmethod
    def set_exposure(self, value):
        """Sets the exposure time."""
        pass

    @abstractmethod
    def get_exposure(self):
        """Gets the current exposure time."""
        pass
