from .camera_interface import CameraSource
import cv2
import numpy as np
import os

# 동영상으로 시도할 확장자 (OpenCV VideoCapture 지원)
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


def _is_video_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


class FileCamera(CameraSource):
    def __init__(self, source_path):
        self.source_path = source_path
        self.image = None
        self.cap = None  # cv2.VideoCapture when source is video

    def open(self):
        exists = os.path.exists(self.source_path)
        print(f"Opening: {self.source_path} | exists={exists}")
        if not exists:
            print(f"File not found: {self.source_path}")
            return False

        # 1) 동영상이면 VideoCapture로 열기
        if _is_video_path(self.source_path):
            try:
                # 한글 경로 대응: 짧은 경로나 8.3 형식 시도, 실패하면 원본 경로 사용
                self.cap = cv2.VideoCapture(self.source_path, cv2.CAP_ANY)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"Opened video, frame shape: {frame.shape}")
                        return True
                    self.cap.release()
                self.cap = None
            except Exception as e:
                print(f"Failed to open as video: {e}")
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
            print("Falling back to image open")
            # fall through to image load (e.g. some .avi might be misnamed image)

        # 2) 이미지로 열기
        try:
            with open(self.source_path, "rb") as f:
                data = f.read()
        except Exception as e:
            print(f"Failed to read file: {e}")
            return False

        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Failed to load image from {self.source_path}")
                return False
            self.image = img
            print(f"Loaded image shape: {self.image.shape}")
            return True
        except Exception as e:
            print(f"Exception while decoding image: {e}")
            return False

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.image = None

    def grab_frame(self) -> np.ndarray:
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # 끝까지 재생됐으면 처음부터 반복 (GIF처럼 무한 반복)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # 일부 코덱은 seek(0)이 안 되므로, 닫았다가 다시 열기
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source_path)
                    if self.cap.isOpened():
                        ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            return None
        if self.image is not None:
            return self.image.copy()
        return None

    def set_exposure(self, value):
        pass

    def get_exposure(self):
        return 0

    def is_video(self) -> bool:
        """True if source is video (AVI etc.), False if static image."""
        return self.cap is not None

    def get_frame_count(self) -> int:
        """동영상일 때 총 프레임 수. 이미지일 때는 1."""
        if self.cap is not None:
            n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return max(1, n)
        return 1

    def get_frame_position(self) -> int:
        """동영상일 때 현재 프레임 인덱스(0-based). 이미지일 때는 0."""
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return 0

    def set_frame_position(self, frame_index: int) -> bool:
        """동영상 재생 위치를 지정한 프레임으로 이동. 성공 여부 반환."""
        if self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        return True
