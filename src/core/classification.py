import cv2
import numpy as np

class ParticleClassifier:
    def classify(self, contour):
        """
        Classifies a contour based on features.
        Returns a dictionary with classification details.
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return {"label": "Unknown", "confidence": 0.0}

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # Bounding rect (Rotated to account for diagonal fibers)
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect

        # Ensure w is the longer side for aspect ratio calculation consistency
        if w < h:
            w, h = h, w

        aspect_ratio = float(w) / h if h != 0 else 0

        # Simple Logic based on shape descriptors
        # This can be expanded with more sophisticated logic or ML later.

        label = "Unknown"

        if area < 50:
            # Very small dots
            label = "Noise/Dust"
        elif circularity > 0.75:
            # Round objects
            label = "Bubble"
        elif aspect_ratio > 3.0:
            # Elongated objects (Rotated aspect ratio)
            label = "Fiber"
        else:
            # Irregular shapes
            label = "Particle"

        return {
            "label": label,
            "area": area,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio
        }
