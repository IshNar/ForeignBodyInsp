import cv2
import numpy as np
import os

def imread_safe(path: str) -> np.ndarray:
    """
    Reads an image from a file path, handling non-ASCII paths (e.g. Korean characters) correctly.

    Args:
        path (str): The file path to the image.

    Returns:
        np.ndarray: The loaded image, or None if loading failed.
    """
    if not os.path.exists(path):
        return None

    try:
        # np.fromfile handles Unicode paths on Windows correctly
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def imwrite_safe(path: str, img: np.ndarray) -> bool:
    """
    Writes an image to a file path, handling non-ASCII paths correctly.

    Args:
        path (str): The file path to save the image.
        img (np.ndarray): The image to save.

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        ext = os.path.splitext(path)[1]
        if not ext:
            pass

        is_success, buffer = cv2.imencode(ext, img)
        if is_success:
            buffer.tofile(path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error writing image {path}: {e}")
        return False
