import cv2
import numpy as np

class ForeignBodyDetector:
    def __init__(self):
        pass

    def detect_static(self, image: np.ndarray, threshold: int = 100, min_area: int = 10, use_adaptive: bool = False):
        """
        Detects foreign bodies in a static image.
        Returns a list of contours.
        """
        if image is None:
            return []

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding
        if use_adaptive:
            # Adaptive thresholding is often better for varying lighting conditions
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 3)
        else:
            # Standard binary threshold using the slider value
            # Assuming dark particles on light background or vice versa.
            # If particles are dark on light, we use THRESH_BINARY_INV.
            _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Opening (Erosion followed by Dilation) removes small noise
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        # Closing (Dilation followed by Erosion) closes small holes inside foreground objects
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        return valid_contours, closed

    def detect_motion(self, frames: list):
        """
        Placeholder for motion detection logic.
        Intended for 'Spin & Stop' sequences where particles move against a static background.
        """
        pass
