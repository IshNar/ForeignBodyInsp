"""YOLO 기반 통합 검출+분류 래퍼.

ultralytics YOLOv8/v11 모델을 사용하여 단일 추론으로
검출(Detection)과 분류(Classification)를 동시에 수행한다.
"""

import os
import cv2
import numpy as np

# 기본 클래스 라벨 (학습 시 data.yaml의 names와 동일해야 함)
DEFAULT_CLASS_NAMES = ["Noise_Dust", "Bubble", "Fiber", "Particle", "Unknown"]


class YOLODetector:
    """ultralytics YOLO 모델 래퍼: 로드 / 추론 / 결과 변환."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.class_names: list[str] = []
        self._device = "cpu"
        if model_path and os.path.isfile(model_path):
            self.load_model(model_path)

    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self, model_path: str):
        """학습된 YOLO .pt 모델 로드.

        Returns:
            (True, None) on success, (False, error_message) on failure.
        """
        try:
            from ultralytics import YOLO
            import torch

            path_abs = os.path.abspath(os.path.normpath(model_path))
            if not os.path.isfile(path_abs):
                return False, f"파일이 없습니다: {path_abs}"

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(path_abs)
            self.model.to(self._device)

            names = getattr(self.model, "names", None)
            if isinstance(names, dict):
                self.class_names = [names[k] for k in sorted(names.keys())]
            elif isinstance(names, (list, tuple)):
                self.class_names = list(names)
            else:
                self.class_names = list(DEFAULT_CLASS_NAMES)

            self.model_path = path_abs
            print(f"[YOLODetector] 모델 로드 완료: {path_abs}, "
                  f"클래스: {self.class_names}, device: {self._device}")
            return True, None
        except Exception as e:
            err = str(e)
            print(f"[YOLODetector] 모델 로드 실패: {err}")
            self.model = None
            return False, err

    def detect(self, frame_bgr: np.ndarray,
               conf_threshold: float = 0.15,
               iou_threshold: float = 0.45) -> list[dict]:
        """이미지에서 객체 검출+분류 수행.

        Args:
            frame_bgr: BGR 이미지 (numpy).
            conf_threshold: confidence 임계값.
            iou_threshold: NMS IoU 임계값.

        Returns:
            list of dict, 각 dict는:
                label (str), confidence (float),
                bbox (tuple[int]): (x1, y1, x2, y2),
                area (float), circularity (float), aspect_ratio (float),
                contour (np.ndarray): bbox를 contour로 변환한 값.
        """
        if not self.is_loaded() or frame_bgr is None:
            return []

        try:
            results = self.model.predict(
                source=frame_bgr,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as e:
            print(f"[YOLODetector] 추론 오류: {e}")
            return []

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else "Unknown"

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                area = float(w * h)
                aspect_ratio = float(max(w, h)) / float(min(w, h)) if min(w, h) > 0 else 0.0
                perimeter = 2.0 * (w + h)
                circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0.0

                contour = np.array([
                    [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
                ], dtype=np.int32)

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "area": area,
                    "circularity": circularity,
                    "aspect_ratio": aspect_ratio,
                    "contour": contour,
                })
        return detections

    def train(self, data_yaml: str, epochs: int = 100,
              imgsz: int = 640, batch: int = 16,
              model_size: str = "yolov8s",
              project: str = "runs/detect", name: str = "train",
              progress_callback=None):
        """YOLO 모델 학습.

        Args:
            data_yaml: data.yaml 경로.
            epochs: 학습 에폭 수.
            imgsz: 입력 이미지 크기.
            batch: 배치 크기.
            model_size: 사전학습 모델 (yolov8n, yolov8s, yolov8m 등).
            project: 결과 저장 프로젝트 디렉토리.
            name: 결과 저장 서브 디렉토리 이름.
            progress_callback: fn(epoch, total_epochs, metrics_dict) or None.

        Returns:
            (True, best_model_path) on success, (False, error_message) on failure.
        """
        try:
            from ultralytics import YOLO

            pretrained = f"{model_size}.pt"
            model = YOLO(pretrained)

            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project,
                name=name,
                exist_ok=True,
                verbose=True,
            )

            best_path = os.path.join(project, name, "weights", "best.pt")
            if os.path.isfile(best_path):
                self.load_model(best_path)
                return True, best_path

            last_path = os.path.join(project, name, "weights", "last.pt")
            if os.path.isfile(last_path):
                self.load_model(last_path)
                return True, last_path

            return False, "학습은 완료되었으나 모델 파일을 찾을 수 없습니다."
        except Exception as e:
            err = str(e)
            print(f"[YOLODetector] 학습 오류: {err}")
            import traceback
            traceback.print_exc()
            return False, err
