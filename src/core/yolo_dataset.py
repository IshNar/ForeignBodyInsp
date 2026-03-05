"""YOLO 데이터셋 생성/관리 유틸리티.

기능:
  - YOLO 학습용 폴더 구조 생성 (images/train, images/val, labels/train, labels/val)
  - 기존 threshold 검출 contour + RuleBase 분류 결과를 YOLO txt 포맷으로 자동 변환
  - data.yaml 자동 생성
  - train/val 분할
  - 개별 이미지+라벨 추가/삭제
"""

import os
import shutil
import random
import cv2
import numpy as np

from src.core.classification import RuleBasedClassifier
from src.core.detection import ForeignBodyDetector, BubbleDetectorParams
from src.core.yolo_detector import DEFAULT_CLASS_NAMES


def contour_to_yolo_bbox(contour, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """contour → YOLO normalized bbox (cx, cy, w, h)."""
    x, y, w, h = cv2.boundingRect(contour)
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return (cx, cy, nw, nh)


def bbox_xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int,
                       img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """(x1,y1,x2,y2) absolute → YOLO normalized (cx, cy, w, h)."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h
    return (cx, cy, w, h)


def yolo_to_bbox_xyxy(cx: float, cy: float, w: float, h: float,
                       img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """YOLO normalized → (x1, y1, x2, y2) absolute."""
    abs_cx = cx * img_w
    abs_cy = cy * img_h
    abs_w = w * img_w
    abs_h = h * img_h
    x1 = int(abs_cx - abs_w / 2)
    y1 = int(abs_cy - abs_h / 2)
    x2 = int(abs_cx + abs_w / 2)
    y2 = int(abs_cy + abs_h / 2)
    return (x1, y1, x2, y2)


class YOLODatasetManager:
    """YOLO 데이터셋 폴더 구조 생성 및 관리."""

    def __init__(self, dataset_dir: str, class_names: list[str] = None):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.class_names = list(class_names or DEFAULT_CLASS_NAMES)
        self._class_to_id = {name: i for i, name in enumerate(self.class_names)}

    @property
    def images_train_dir(self) -> str:
        return os.path.join(self.dataset_dir, "images", "train")

    @property
    def images_val_dir(self) -> str:
        return os.path.join(self.dataset_dir, "images", "val")

    @property
    def labels_train_dir(self) -> str:
        return os.path.join(self.dataset_dir, "labels", "train")

    @property
    def labels_val_dir(self) -> str:
        return os.path.join(self.dataset_dir, "labels", "val")

    @property
    def data_yaml_path(self) -> str:
        return os.path.join(self.dataset_dir, "data.yaml")

    def create_structure(self):
        """데이터셋 디렉토리 구조 생성."""
        for d in [self.images_train_dir, self.images_val_dir,
                  self.labels_train_dir, self.labels_val_dir]:
            os.makedirs(d, exist_ok=True)

    def write_data_yaml(self):
        """data.yaml 파일 생성 (외부 yaml 라이브러리 없이 직접 작성)."""
        self.create_structure()
        names_list = ", ".join(f'"{n}"' for n in self.class_names)
        content = (
            f'path: "{self.dataset_dir.replace(chr(92), "/")}"\n'
            f'train: images/train\n'
            f'val: images/val\n'
            f'nc: {len(self.class_names)}\n'
            f'names: [{names_list}]\n'
        )
        with open(self.data_yaml_path, "w", encoding="utf-8") as f:
            f.write(content)
        return self.data_yaml_path

    def class_id(self, label: str) -> int:
        """라벨 이름 → 클래스 ID. 없으면 -1."""
        return self._class_to_id.get(label, -1)

    def add_image_with_labels(self, image_path: str, annotations: list[dict],
                               split: str = "train"):
        """이미지와 어노테이션을 데이터셋에 추가.

        Args:
            image_path: 원본 이미지 경로.
            annotations: list of dict, 각 dict는:
                class_id (int) 또는 label (str),
                bbox_xyxy (tuple[int]) = (x1,y1,x2,y2) 또는
                bbox_yolo (tuple[float]) = (cx,cy,w,h) normalized.
            split: "train" 또는 "val".
        """
        self.create_structure()
        img = cv2.imread(image_path) if isinstance(image_path, str) else None
        if img is None:
            try:
                buf = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                return None
        if img is None:
            return None

        img_h, img_w = img.shape[:2]
        basename = os.path.splitext(os.path.basename(image_path))[0]

        img_dir = self.images_train_dir if split == "train" else self.images_val_dir
        lbl_dir = self.labels_train_dir if split == "train" else self.labels_val_dir

        dst_img = os.path.join(img_dir, basename + ".jpg")
        counter = 1
        while os.path.exists(dst_img):
            dst_img = os.path.join(img_dir, f"{basename}_{counter}.jpg")
            counter += 1
        actual_basename = os.path.splitext(os.path.basename(dst_img))[0]

        cv2.imwrite(dst_img, img)

        label_lines = []
        for ann in annotations:
            if "class_id" in ann:
                cid = ann["class_id"]
            elif "label" in ann:
                cid = self.class_id(ann["label"])
            else:
                continue
            if cid < 0:
                continue

            if "bbox_yolo" in ann:
                cx, cy, w, h = ann["bbox_yolo"]
            elif "bbox_xyxy" in ann:
                x1, y1, x2, y2 = ann["bbox_xyxy"]
                cx, cy, w, h = bbox_xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            else:
                continue
            label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        dst_lbl = os.path.join(lbl_dir, actual_basename + ".txt")
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        return dst_img

    def add_frame_with_contours(self, frame_bgr: np.ndarray,
                                 contours: list, labels: list[str],
                                 frame_name: str = None,
                                 split: str = "train"):
        """프레임 + contour/라벨 목록을 YOLO 데이터셋에 추가.

        Args:
            frame_bgr: BGR 이미지.
            contours: OpenCV contours 리스트.
            labels: 각 contour에 대응하는 라벨 리스트.
            frame_name: 저장 파일 이름 (확장자 제외). None이면 자동 생성.
            split: "train" 또는 "val".
        """
        self.create_structure()
        if frame_bgr is None or not contours:
            return None

        img_h, img_w = frame_bgr.shape[:2]
        if frame_name is None:
            import datetime
            frame_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        img_dir = self.images_train_dir if split == "train" else self.images_val_dir
        lbl_dir = self.labels_train_dir if split == "train" else self.labels_val_dir

        dst_img = os.path.join(img_dir, frame_name + ".jpg")
        counter = 1
        while os.path.exists(dst_img):
            dst_img = os.path.join(img_dir, f"{frame_name}_{counter}.jpg")
            counter += 1
        actual_name = os.path.splitext(os.path.basename(dst_img))[0]

        ok, buf = cv2.imencode(".jpg", frame_bgr)
        if ok:
            with open(dst_img, "wb") as f:
                f.write(buf.tobytes())

        label_lines = []
        for cnt, label in zip(contours, labels):
            cid = self.class_id(label)
            if cid < 0:
                continue
            cx, cy, w, h = contour_to_yolo_bbox(cnt, img_w, img_h)
            label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        dst_lbl = os.path.join(lbl_dir, actual_name + ".txt")
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        return dst_img

    def auto_label_image(self, image_path: str,
                          threshold: int = 100, min_area: int = 10,
                          use_adaptive: bool = True,
                          detect_general: bool = True,
                          detect_bubbles: bool = True,
                          bubble_params: "BubbleDetectorParams | None" = None,
                          open_kernel: int = 2, close_kernel: int = 3,
                          split: str = "train") -> str | None:
        """이미지를 일반검출 + Bubble검출로 자동 라벨링하여 데이터셋에 추가.

        Args:
            detect_general: 일반 검출(Particle/Fiber/Noise) 수행 여부.
            detect_bubbles: Bubble 검출 수행 여부.
            bubble_params: Bubble 검출 파라미터. None이면 기본값 사용.
            open_kernel: Morphology Opening 커널 크기.
            close_kernel: Morphology Closing 커널 크기.

        Returns:
            저장된 이미지 경로, 실패 시 None.
        """
        try:
            buf = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        all_contours: list = []
        all_labels: list[str] = []

        if detect_general:
            detector = ForeignBodyDetector()
            classifier = RuleBasedClassifier()
            contours_gen, _ = detector.detect_static(
                gray, threshold=threshold, min_area=min_area,
                use_adaptive=use_adaptive, detect_bubbles=False,
                open_kernel=open_kernel, close_kernel=close_kernel,
            )
            for cnt in contours_gen:
                all_contours.append(cnt)
                all_labels.append(classifier.classify(cnt)["label"])

        if detect_bubbles:
            detector_bub = ForeignBodyDetector()
            if bubble_params is not None:
                detector_bub.bubble_params = bubble_params
            bub_contours, _ = detector_bub.detect_bubbles(gray)
            for cnt in bub_contours:
                all_contours.append(cnt)
                all_labels.append("Bubble")

        if not all_contours:
            return None

        frame_name = os.path.splitext(os.path.basename(image_path))[0]
        return self.add_frame_with_contours(img, all_contours, all_labels,
                                             frame_name=frame_name, split=split)

    def split_train_val(self, val_ratio: float = 0.2):
        """train 이미지를 val_ratio 비율로 val로 이동 (랜덤 분할)."""
        train_imgs = [f for f in os.listdir(self.images_train_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if not train_imgs:
            return

        random.shuffle(train_imgs)
        n_val = max(1, int(len(train_imgs) * val_ratio))
        val_imgs = train_imgs[:n_val]

        for fname in val_imgs:
            basename = os.path.splitext(fname)[0]
            src_img = os.path.join(self.images_train_dir, fname)
            dst_img = os.path.join(self.images_val_dir, fname)
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
            src_lbl = os.path.join(self.labels_train_dir, basename + ".txt")
            dst_lbl = os.path.join(self.labels_val_dir, basename + ".txt")
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

    def get_stats(self) -> dict:
        """데이터셋 통계 반환."""
        stats = {"train_images": 0, "val_images": 0,
                 "train_labels": 0, "val_labels": 0,
                 "class_counts": {name: 0 for name in self.class_names}}

        for split, img_dir, lbl_dir in [
            ("train", self.images_train_dir, self.labels_train_dir),
            ("val", self.images_val_dir, self.labels_val_dir),
        ]:
            if os.path.isdir(img_dir):
                imgs = [f for f in os.listdir(img_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
                stats[f"{split}_images"] = len(imgs)
            if os.path.isdir(lbl_dir):
                lbls = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
                stats[f"{split}_labels"] = len(lbls)
                for lbl_file in lbls:
                    try:
                        with open(os.path.join(lbl_dir, lbl_file), "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    cid = int(parts[0])
                                    if 0 <= cid < len(self.class_names):
                                        stats["class_counts"][self.class_names[cid]] += 1
                    except Exception:
                        pass
        return stats

    def load_image_annotations(self, image_filename: str,
                                split: str = "train") -> list[dict]:
        """특정 이미지의 YOLO 어노테이션 로드.

        Returns:
            list of dict with keys: class_id, label, bbox_yolo (cx,cy,w,h),
            bbox_xyxy (x1,y1,x2,y2).
        """
        img_dir = self.images_train_dir if split == "train" else self.images_val_dir
        lbl_dir = self.labels_train_dir if split == "train" else self.labels_val_dir

        basename = os.path.splitext(image_filename)[0]
        lbl_path = os.path.join(lbl_dir, basename + ".txt")
        img_path = os.path.join(img_dir, image_filename)

        if not os.path.isfile(lbl_path):
            return []

        try:
            img = cv2.imread(img_path)
            if img is None:
                buf = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img_h, img_w = img.shape[:2] if img is not None else (1, 1)
        except Exception:
            img_h, img_w = 1, 1

        annotations = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                label = self.class_names[cid] if 0 <= cid < len(self.class_names) else "Unknown"
                x1, y1, x2, y2 = yolo_to_bbox_xyxy(cx, cy, w, h, img_w, img_h)
                annotations.append({
                    "class_id": cid,
                    "label": label,
                    "bbox_yolo": (cx, cy, w, h),
                    "bbox_xyxy": (x1, y1, x2, y2),
                })
        return annotations

    def save_image_annotations(self, image_filename: str,
                                annotations: list[dict],
                                img_w: int, img_h: int,
                                split: str = "train"):
        """특정 이미지의 어노테이션을 YOLO txt로 저장."""
        lbl_dir = self.labels_train_dir if split == "train" else self.labels_val_dir
        basename = os.path.splitext(image_filename)[0]
        lbl_path = os.path.join(lbl_dir, basename + ".txt")

        lines = []
        for ann in annotations:
            if "class_id" in ann:
                cid = ann["class_id"]
            elif "label" in ann:
                cid = self.class_id(ann["label"])
            else:
                continue
            if cid < 0:
                continue
            if "bbox_yolo" in ann:
                cx, cy, w, h = ann["bbox_yolo"]
            elif "bbox_xyxy" in ann:
                x1, y1, x2, y2 = ann["bbox_xyxy"]
                cx, cy, w, h = bbox_xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            else:
                continue
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        os.makedirs(lbl_dir, exist_ok=True)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
