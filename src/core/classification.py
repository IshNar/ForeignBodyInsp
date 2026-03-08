import cv2
import numpy as np
import os
import datetime
import torch
import torch.nn as nn
import copy
import shutil

# Bubble: 코드 전체에서 하나의 라벨만 사용. 하위 호환용 이전 이름은 여기만 정의.
BUBBLE_LABEL = "Bubble"
LEGACY_BUBBLE_LABELS = ("Small Bubble", "Big Bubble", "BackGround Bubble")  # 표시/저장 시 Bubble로 통일
LEGACY_BUBBLE_FOLDERS = ("Small_Bubble", "Big_Bubble", "BackGround_Bubble")  # 폴더명 → Bubble로 통일

# 분류 모델 입력 크기 (학습·추론 공통). 224=표준 고해상도 (ResNet50 기본 규격).
CLASSIFICATION_INPUT_SIZE = 224


class RuleBasedClassifier:
    """기존 규칙 기반 분류기 (형태 특성 기반). 파라미터는 인스턴스 속성으로 설정 가능."""

    # 지원하는 라벨 목록 (Bubble 단일)
    # 지원하는 라벨 목록 (Bubble 단일)
    LABELS = [
        "Noise_Dust",
        BUBBLE_LABEL,
        "Particle",
        "Unknown",
    ]

    # 기본값 (클래스 상수)
    DEFAULT_NOISE_CONTRAST_THRESHOLD = 30

    def __init__(self):
        self.noise_contrast_threshold = self.DEFAULT_NOISE_CONTRAST_THRESHOLD

    def get_params(self):
        """현재 파라미터를 dict로 반환 (JSON/UI 저장용)."""
        return {
            "noise_contrast_threshold": self.noise_contrast_threshold,
        }

    def set_params(self, params: dict):
        """dict에서 파라미터 적용. 없는 키는 무시."""
        if not params:
            return
        if "noise_contrast_threshold" in params:
            self.noise_contrast_threshold = int(params["noise_contrast_threshold"])

    def classify(self, contour, image=None):
        results = self.classify_batch([contour], image)
        return results[0] if results else {
            "label": "Unknown", "area": 0, "circularity": 0, "aspect_ratio": 0
        }

    def classify_batch(self, contours, image=None):
        """Optimized batch classification for rule-based approach."""
        if not contours:
            return []
            
        gray = None
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            img_h, img_w = gray.shape

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0

            rect = cv2.minAreaRect(contour)
            (x, y), (w, h), angle = rect
            if w < h:
                w, h = h, w
            aspect_ratio = float(w) / h if h != 0 else 0

            label = "Unknown"
            
            if gray is None:
                label = "Particle"
            else:
                x, y, w, h = cv2.boundingRect(contour)
                # Fast ROI without drawing masks for contrast calculation
                # Use a larger bounding box to sample background
                pad = 3
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
                
                # Core vs Periphery estimation (much faster than drawing mask for every particle)
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    # Simple heuristic: compare center vs border intensities
                    rh, rw = roi.shape
                    
                    if rh > pad*2 and rw > pad*2:
                        center_roi = roi[pad:rh-pad, pad:rw-pad]
                        fg_mean = np.mean(center_roi)
                        
                        # Full mean
                        full_mean = np.mean(roi)
                        
                        # Approximate background by taking difference
                        # (Sum_full - Sum_center) / (Area_full - Area_center)
                        sum_full = full_mean * roi.size
                        sum_center = fg_mean * center_roi.size
                        bg_pts = roi.size - center_roi.size
                        
                        bg_mean = (sum_full - sum_center) / bg_pts if bg_pts > 0 else fg_mean
                        
                        contrast = abs(fg_mean - bg_mean)
                        label = "Noise_Dust" if contrast < self.noise_contrast_threshold else "Particle"
                    else:
                        label = "Particle" # Too small to check contrast
                else:
                    label = "Particle"
                    
            results.append({
                "label": label,
                "area": area,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
            })
            
        return results


class DeepLearningClassifier:
    """딥러닝 기반 분류기 (학습된 모델 사용)."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.ort_session = None
        self.model_path = model_path
        self.labels = []  # 학습 시 사용된 라벨 목록
        self._device = "cpu"
        if model_path and os.path.isfile(model_path):
            self.load_model(model_path)

    def is_loaded(self) -> bool:
        return self.model is not None or self.ort_session is not None

    def get_device(self) -> str:
        """추론에 사용 중인 디바이스. 'cuda' 또는 'cpu'."""
        return getattr(self, "_device", "cpu")

    def get_device_display(self) -> str:
        """UI 표시용: GPU 사용 여부 및 백엔드."""
        if not self.is_loaded():
            return "—"
        dev = self.get_device()
        if self.ort_session is not None:
            return "GPU (ONNX)" if dev == "cuda" else "CPU (ONNX)"
        return "GPU (PyTorch)" if dev == "cuda" else "CPU (PyTorch)"

    def load_model(self, model_path: str):
        """저장된 모델 로드. 반환: (성공 여부, 실패 시 오류 메시지)."""
        try:
            path_abs = os.path.abspath(os.path.normpath(model_path))
            if not os.path.isfile(path_abs):
                return False, f"파일이 없습니다: {path_abs}"

            # 1. ONNX 모델 로드
            if path_abs.endswith('.onnx'):
                import onnxruntime as ort
                
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = (os.cpu_count() or 2) // 2
                
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                try:
                    self.ort_session = ort.InferenceSession(path_abs, sess_options=sess_options, providers=providers)
                except Exception as e:
                    print(f"[DL Classifier] CUDA 로드 실패, CPU로 폴백: {e}")
                    self.ort_session = ort.InferenceSession(path_abs, sess_options=sess_options, providers=['CPUExecutionProvider'])
                
                # 라벨 파일 로드 (*_labels.json)
                label_path = path_abs.replace('.onnx', '_labels.json')
                if os.path.exists(label_path):
                    import json
                    with open(label_path, 'r', encoding='utf-8') as f:
                        self.labels = json.load(f)
                else:
                    self.labels = ["Bubble", "Noise_Dust", "Particle", "Unknown"]

                self.model = None # PyTorch 모델은 사용 안 함
                active_providers = self.ort_session.get_providers()
                self._device = "cuda" if "CUDAExecutionProvider" in active_providers else "cpu"
                self._use_fp16 = False # ONNX 런타임 자체 최적화에 맡김
                self.model_path = path_abs
                print(f"[DL Classifier] ONNX 모델 로드 완료: {path_abs}, 클래스: {self.labels}, device: {active_providers[0]}")
                return True, None

            # 2. PyTorch 모델 로드 (하위 호환)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            with open(path_abs, "rb") as f:
                checkpoint = torch.load(f, map_location=self._device, weights_only=False)
            self.labels = checkpoint.get("labels", [])
            self.labels = [l.replace(" ", "_") for l in self.labels]
            num_classes = len(self.labels) if self.labels else 4
            self.model = _build_model(num_classes)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self._device)
            self.model.eval()
            self.ort_session = None # ORT 미사용
            
            # FP16 반정밀도 추론 (CUDA GPU에서 ~2배 속도 향상, 메모리 절반)
            self._use_fp16 = (self._device == "cuda")
            if self._use_fp16:
                self.model = self.model.half()
                print(f"[DL Classifier] FP16 반정밀도 모드 활성화 (CUDA)")
            
            self.model_path = path_abs
            print(f"[DL Classifier] PyTorch 모델 로드 완료: {path_abs}, 클래스: {self.labels}, device: {self._device}")
            return True, None
        except Exception as e:
            err = str(e)
            print(f"[DL Classifier] 모델 로드 실패: {err}")
            self.model = None
            self.ort_session = None
            return False, err

    def classify(self, contour, frame_bgr=None):
        """단일 contour 분류. 여러 개는 classify_batch 사용 권장."""
        results = self.classify_batch([contour], frame_bgr)
        return results[0] if results else {"label": "Unknown", "confidence": 0.0, "area": 0,
                "circularity": 0.0, "aspect_ratio": 0.0}

    # ImageNet normalization 상수 (numpy 기반 — torchvision 대비 빠름)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # CHW 형태로 미리 변환 (broadcast용)
    _MEAN_CHW = _MEAN.reshape(1, 3, 1, 1)
    _STD_CHW  = _STD.reshape(1, 3, 1, 1)

    def _extract_rois_chunk(self, bboxes, frame_bgr, start, end, img_w, img_h, sz):
        """CPU에서 ROI crop + resize + normalize를 수행하는 헬퍼 (파이프라인용)."""
        chunk_size = end - start
        # 결과 배열: (N, C, H, W) 형태로 직접 생성 (transpose 제거)
        roi_chw = np.empty((chunk_size, 3, sz, sz), dtype=np.float32)
        valid = np.zeros(chunk_size, dtype=bool)

        for j in range(chunk_size):
            i = start + j
            x, y, w, h = bboxes[i]
            side = max(w * 2, h * 2, sz)
            cx, cy = x + w // 2, y + h // 2
            x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
            x2, y2 = min(img_w, x1 + side), min(img_h, y1 + side)

            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # INTER_LINEAR: INTER_AREA보다 빠르고 차단율 좋음
            resized = cv2.resize(roi, (sz, sz), interpolation=cv2.INTER_LINEAR)
            # C레벨에서 uint8 -> float32 자동 형변환 및 대입 (루프 내부에서 numpy 배열 생성 비용 제거)
            roi_chw[j] = resized[:, :, ::-1].transpose(2, 0, 1)
            valid[j] = True

        # valid만 뽑아서 일괄 normalize (Python for루프 바깥에서 C단위 벡터 연산)
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            v = roi_chw[valid_indices]
            v /= 255.0
            v -= self._MEAN_CHW
            v /= self._STD_CHW
            roi_chw[valid_indices] = v

        return roi_chw, valid, valid_indices

    def classify_batch(self, contours, frame_bgr=None):
        """
        여러 contour를 한 번에 배치 추론 (10K+ contour 대응, CPU/GPU 파이프라인).
        frame_bgr: 원본 BGR 이미지.
        Returns: list of dict (label, confidence, area, ...)
        """
        import time
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.perf_counter()
        n = len(contours)
        if n == 0:
            return []

        # 공통 정보 계산 (area 포함 bbox 한 번에 계산)
        rects = []
        peris = []
        areas = np.zeros(n, dtype=np.float32)
        bboxes = np.zeros((n, 4), dtype=np.int32)
        
        for i, cnt in enumerate(contours):
            areas[i] = cv2.contourArea(cnt)
            peris.append(cv2.arcLength(cnt, True))
            rects.append(cv2.minAreaRect(cnt))
            bboxes[i] = cv2.boundingRect(cnt)

        if not self.is_loaded() or frame_bgr is None:
            infos = self._build_infos(areas, peris, rects)
            return [{"label": "Unknown", "confidence": 0.0, **infos[i]} for i in range(n)]

        try:
            img_h, img_w = frame_bgr.shape[:2]
            sz = CLASSIFICATION_INPUT_SIZE
            use_fp16 = getattr(self, '_use_fp16', False)

            CHUNK = 2048  # 더 큰 청크 → 오버헤드 감소
            MINI_BATCH = 128  # 속도를 위해 128 미니배치
            all_conf = np.zeros(n, dtype=np.float32)
            all_pred = np.zeros(n, dtype=np.int64)
            valid_mask = np.zeros(n, dtype=bool)

            # CPU/GPU 파이프라인 (작업 스레드를 2개 이상 주어 GPU 병목 완전 제거)
            chunk_ranges = [(s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]

            with torch.inference_mode(), ThreadPoolExecutor(max_workers=2) as executor:
                # 첫 번째 청크를 미리 준비
                future = executor.submit(
                    self._extract_rois_chunk, bboxes, frame_bgr,
                    chunk_ranges[0][0], chunk_ranges[0][1], img_w, img_h, sz
                )

                for ci, (chunk_start, chunk_end) in enumerate(chunk_ranges):
                    # 현재 청크의 CPU 작업 결과 받기
                    roi_chw, chunk_valid, valid_in_chunk = future.result()

                    # 다음 청크의 CPU 작업을 비동기로 시작 (GPU 추론과 병렬)
                    if ci + 1 < len(chunk_ranges):
                        next_s, next_e = chunk_ranges[ci + 1]
                        future = executor.submit(
                            self._extract_rois_chunk, bboxes, frame_bgr,
                            next_s, next_e, img_w, img_h, sz
                        )

                    if len(valid_in_chunk) == 0:
                        continue

                    # Tensor 변환 + GPU 전송
                    valid_rois = roi_chw[valid_in_chunk]

                    if getattr(self, 'ort_session', None) is not None:
                        # === ONNX Runtime 엔진 사용 ===
                        input_name = self.ort_session.get_inputs()[0].name
                        chunk_conf, chunk_idx = [], []
                        # ONNX도 일시적 VRAM 초과 방지를 위해 미니배치 단위로 추론
                        for mb_start in range(0, len(valid_rois), MINI_BATCH):
                            mb_rois = np.ascontiguousarray(valid_rois[mb_start:mb_start + MINI_BATCH])
                            ort_out = self.ort_session.run(None, {input_name: mb_rois})[0]
                            # NumPy로 Softmax 계산
                            max_out = np.max(ort_out, axis=1, keepdims=True)
                            exp_out = np.exp(ort_out - max_out)
                            probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
                            chunk_conf.append(np.max(probs, axis=1))
                            chunk_idx.append(np.argmax(probs, axis=1))
                        
                        conf_arr = np.concatenate(chunk_conf) if chunk_conf else np.array([])
                        idx_arr = np.concatenate(chunk_idx) if chunk_idx else np.array([])
                    else:
                        # === 기존 PyTorch 엔진 사용 ===
                        batch_tensor = torch.from_numpy(
                            np.ascontiguousarray(valid_rois)
                        ).to(self._device, non_blocking=True)

                        if use_fp16:
                            batch_tensor = batch_tensor.half()

                        # 미니배치 추론
                        chunk_conf, chunk_idx = [], []
                        for mb_start in range(0, len(batch_tensor), MINI_BATCH):
                            out = self.model(batch_tensor[mb_start:mb_start + MINI_BATCH])
                            probs = torch.softmax(out.float(), dim=1)
                            conf, pred_idx = probs.max(dim=1)
                            chunk_conf.append(conf.cpu().numpy())
                            chunk_idx.append(pred_idx.cpu().numpy())

                        conf_arr = np.concatenate(chunk_conf)
                        idx_arr = np.concatenate(chunk_idx)

                    for k, j in enumerate(valid_in_chunk):
                        gi = chunk_start + j
                        all_conf[gi] = conf_arr[k]
                        all_pred[gi] = idx_arr[k]
                        valid_mask[gi] = True

            # 결과 매핑
            infos = self._build_infos(areas, peris, rects)
            result_list = [{"label": "Unknown", "confidence": 0.0, **infos[i]} for i in range(n)]
            
            for i in range(n):
                if valid_mask[i]:
                    label = self.labels[all_pred[i]] if all_pred[i] < len(self.labels) else "Unknown"
                    result_list[i]["label"] = label
                    result_list[i]["confidence"] = float(all_conf[i])

            elapsed = time.perf_counter() - t0
            print(f"[DL Classifier] {n}개 contour 분류 완료 (FP16={use_fp16}, Pipeline): {elapsed:.3f}초")
            return result_list
        except Exception as e:
            print(f"[DL Classifier] 배치 추론 오류: {e}")
            import traceback; traceback.print_exc()
            infos = self._build_infos(areas, peris, rects)
            return [{"label": "Unknown", "confidence": 0.0, **infos[i]} for i in range(n)]

    @staticmethod
    def _build_infos(areas, peris, rects):
        infos = []
        for i in range(len(areas)):
            perimeter = peris[i]
            circularity = 4 * np.pi * (areas[i] / (perimeter * perimeter)) if perimeter > 0 else 0
            (_, (w, h), _) = rects[i]
            if w < h: w, h = h, w
            aspect_ratio = float(w) / h if h != 0 else 0
            infos.append({"area": float(areas[i]), "circularity": circularity, "aspect_ratio": aspect_ratio})
        return infos




class ParticleClassifier:
    """통합 분류기: RuleBased 또는 DeepLearning 선택 사용."""

    def __init__(self):
        self.rule_based = RuleBasedClassifier()
        self.dl_classifier = DeepLearningClassifier()
        self.use_deep_learning = False

    def classify(self, contour, frame_bgr=None):
        """현재 모드에 따라 분류 수행."""
        if self.use_deep_learning and self.dl_classifier.is_loaded():
            return self.dl_classifier.classify(contour, frame_bgr)
        return self.rule_based.classify(contour)

    def classify_batch(self, contours, frame_bgr=None):
        """여러 contour 한 번에 분류 (딥러닝 시 배치 추론으로 속도 개선)."""
        if not contours:
            return []
        if self.use_deep_learning and self.dl_classifier.is_loaded():
            return self.dl_classifier.classify_batch(contours, frame_bgr)
            
        return self.rule_based.classify_batch(contours, image=frame_bgr)

    def load_dl_model(self, model_path: str):
        """모델 로드. 반환: (성공 여부, 실패 시 오류 메시지)."""
        return self.dl_classifier.load_model(model_path)

    def set_use_deep_learning(self, enabled: bool):
        self.use_deep_learning = enabled


class DefectImageSaver:
    """검출된 Defect 이미지를 라벨별 폴더에 BMP로 저장.

    디렉토리 구조:
        base_dir/
            _annotations/        ← 어노테이션 데이터 JSON
            _originals/          ← 원본 프레임
            {label}/             ← Defect ROI 이미지 (Particle, Dust 등)
    """

    ORIGINALS_SUBDIR = "_originals"
    ANNOTATIONS_SUBDIR = "_annotations"

    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: 기본 저장 디렉토리 (예: ClassificationData)
        """
        self.base_dir = base_dir
        self.originals_dir = os.path.join(base_dir, self.ORIGINALS_SUBDIR)
        self.annotations_dir = os.path.join(base_dir, self.ANNOTATIONS_SUBDIR)
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(self.originals_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        self._defect_counter = {}  # 라벨별 카운터 {label: count}
        self._original_counter = {}  # source_name별 원본 카운터
        self._initialize_counters()  # 기존 파일 확인하여 카운터 초기화

    def _initialize_counters(self):
        """기존 파일들을 확인하여 카운터를 올바르게 초기화."""
        # 라벨별 Defect 파일 카운터 초기화
        if os.path.isdir(self.base_dir):
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path) and item not in (self.ORIGINALS_SUBDIR, self.ANNOTATIONS_SUBDIR):
                    # 라벨 폴더인 경우
                    label = item
                    max_idx = 0
                    for fname in os.listdir(item_path):
                        if fname.endswith('.bmp'):
                            # 파일명 형식: {label}_{idx}.bmp
                            try:
                                prefix = f"{label}_"
                                if fname.startswith(prefix):
                                    idx_str = fname[len(prefix):-4]  # .bmp 제거
                                    idx = int(idx_str)
                                    max_idx = max(max_idx, idx)
                            except (ValueError, IndexError):
                                pass
                    self._defect_counter[label] = max_idx
        
        # 원본 파일 카운터 초기화
        if os.path.isdir(self.originals_dir):
            source_name_counts = {}
            for fname in os.listdir(self.originals_dir):
                if fname.endswith('.bmp'):
                    # 파일명 형식: {source_name}_{idx}.bmp
                    try:
                        parts = fname[:-4].rsplit('_', 1)  # 마지막 _ 기준으로 분리
                        if len(parts) == 2:
                            source_name, idx_str = parts
                            idx = int(idx_str)
                            if source_name not in source_name_counts:
                                source_name_counts[source_name] = 0
                            source_name_counts[source_name] = max(source_name_counts[source_name], idx)
                    except (ValueError, IndexError):
                        pass
            self._original_counter = source_name_counts

    def save_original(self, frame_bgr: np.ndarray, source_name: str = None) -> str:
        """원본 프레임 저장. 간단한 파일명 사용."""
        if source_name is None:
            source_name = "unknown"
        safe_source_name = source_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        
        # 카운터 증가
        if safe_source_name not in self._original_counter:
            self._original_counter[safe_source_name] = 0
        self._original_counter[safe_source_name] += 1
        idx = self._original_counter[safe_source_name]
        
        filename = f"{safe_source_name}_{idx}.bmp"
        filepath = os.path.join(self.originals_dir, filename)
        
        try:
            if not os.path.exists(filepath):
                ok, buf = cv2.imencode(".bmp", frame_bgr)
                if ok:
                    with open(filepath, "wb") as f:
                        f.write(buf.tobytes())
            return filepath
        except Exception as e:
            print(f"[DefectImageSaver] 원본 저장 오류: {e}")
            return None

    def save(self, contour, frame_bgr: np.ndarray, label: str,
             defect_index: int = 0, source_name: str = None, roi_size: int = None):
        """Defect ROI를 라벨 폴더에 BMP로 저장. 간단한 파일명 사용."""
        try:
            if roi_size is None:
                roi_size = CLASSIFICATION_INPUT_SIZE
            if label in LEGACY_BUBBLE_LABELS:
                label = BUBBLE_LABEL
            safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")
            label_dir = os.path.join(self.base_dir, safe_label)
            os.makedirs(label_dir, exist_ok=True)

            # contour ROI 추출
            roi_img = _extract_contour_roi(contour, frame_bgr, size=roi_size)
            if roi_img is None:
                return None

            # 라벨별 카운터 증가
            if safe_label not in self._defect_counter:
                self._defect_counter[safe_label] = 0
            self._defect_counter[safe_label] += 1
            idx = self._defect_counter[safe_label]

            # 간단한 파일명: {label}_{index}.bmp
            filename = f"{safe_label}_{idx}.bmp"
            filepath = os.path.join(label_dir, filename)

            ok, buf = cv2.imencode(".bmp", roi_img)
            if ok:
                with open(filepath, "wb") as f:
                    f.write(buf.tobytes())
                return filepath
        except Exception as e:
            print(f"[DefectImageSaver] 저장 오류: {e}")
        return None

    @staticmethod
    def parse_defect_filename(filepath: str) -> dict:
        """DefectImage 파일명에서 메타데이터를 파싱.

        Returns:
            dict with keys: orig_id, x, y, w, h, label, idx
            or None if parsing fails.
        """
        try:
            basename = os.path.splitext(os.path.basename(filepath))[0]
            # 파일명: {orig_id}_{x}_{y}_{w}_{h}_{label}_{idx}
            # orig_id는 타임스탬프(YYYYMMDD_HHMMSS_ffffff)이므로 _로 분리 시 주의
            # 뒤에서부터 파싱: idx, label, h, w, y, x 순서로 추출
            parts = basename.split("_")
            if len(parts) < 7:
                return None
            idx = int(parts[-1])
            label = parts[-2]
            bh = int(parts[-3])
            bw = int(parts[-4])
            by = int(parts[-5])
            bx = int(parts[-6])
            orig_id = "_".join(parts[:-6])
            return {
                "orig_id": orig_id,
                "x": bx, "y": by, "w": bw, "h": bh,
                "label": label, "idx": idx,
            }
        except Exception:
            return None

    def get_original_path(self, orig_id: str) -> str:
        """orig_id에 해당하는 원본 프레임 경로 반환."""
        path = os.path.join(self.originals_dir, f"{orig_id}.bmp")
        return path if os.path.isfile(path) else None

    def load_original(self, orig_id: str) -> np.ndarray:
        """orig_id에 해당하는 원본 프레임 로드."""
        path = self.get_original_path(orig_id)
        if path is None:
            return None
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            return None


# ========== 유틸리티 함수 ==========

def _extract_contour_roi(contour, frame_bgr: np.ndarray, size: int = None) -> np.ndarray:
    """contour 주변 ROI를 정사각형으로 추출하고 size×size로 리사이즈."""
    if size is None:
        size = CLASSIFICATION_INPUT_SIZE
    if frame_bgr is None or contour is None:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    img_h, img_w = frame_bgr.shape[:2]

    # 정사각형 ROI (defect 크기의 2배, 최소 size)
    side = max(w * 2, h * 2, size)
    cx, cy = x + w // 2, y + h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(img_w, x1 + side)
    y2 = min(img_h, y1 + side)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    resized = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
    return resized


class FocalLoss(nn.Module):
    """
    Focal Loss: 불균형 데이터셋에 극도로 효과적인 손실 함수.
    잘 맞추고 있는 쉬운 샘플(Noise)에 대한 가중치를 낮추고,
    자꾸 틀리는 어려운 샘플(Bubble, Particle)에 매우 큰 패널티를 부여.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # 예측 확률
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def _build_model(num_classes: int):
    """사전 학습된 EfficientNet-B0(최고의 파라미터 대비 성능) 모델 생성."""
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    # 1. ImageNet으로 사전 학습된 초경량-고성능 모델(EfficientNet-B0) 불러오기
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 
    model = efficientnet_b0(weights=weights)
    
    # 2. 마지막 분류기(Classifier) 레이어를 우리 클래스 개수에 맞게 교체
    in_features = model.classifier[1].in_features
    
    # 단순 Linear 대신 특징을 모아주는 병목 Head 구축
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def train_classifier(data_dir: str, save_path: str, epochs: int = 50,
                     progress_callback=None):
    """
    data_dir 하위 폴더(라벨별)의 이미지로 분류 모델 학습.
    progress_callback(epoch, total_epochs, loss, accuracy): 진행 상황 콜백.
    """
    try:
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        import torchvision.transforms as T

        raw_folders = [d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))
                       and not d.startswith("_")]

        def _folder_to_label(folder_name):
            return BUBBLE_LABEL if folder_name in LEGACY_BUBBLE_FOLDERS else folder_name

        labels = sorted(set(_folder_to_label(d) for d in raw_folders))
        if len(labels) < 2:
            msg = f"최소 2개 클래스 필요. 현재: {labels}"
            print(f"[Train] {msg}")
            return False, msg

        images, targets, file_paths = [], [], []
        
        label_counts = {name: 0 for name in labels}
        for label_idx, label_name in enumerate(labels):
            folders_for_label = [f for f in raw_folders if _folder_to_label(f) == label_name]
            for folder_name in folders_for_label:
                label_dir = os.path.join(data_dir, folder_name)
                for fname in sorted(os.listdir(label_dir)):
                    fpath = os.path.join(label_dir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    try:
                        buf = np.fromfile(fpath, dtype=np.uint8)
                        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    except Exception:
                        img = None
                    if img is None:
                        continue
                    img = cv2.resize(img, (CLASSIFICATION_INPUT_SIZE, CLASSIFICATION_INPUT_SIZE), interpolation=cv2.INTER_AREA)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    targets.append(label_idx)
                    file_paths.append(fpath)
                    label_counts[label_name] += 1

        if len(images) < 10:
            msg = f"전체 이미지가 너무 적음: {len(images)}장 (최소 10장 필요)"
            print(f"[Train] {msg}")
            return False, msg
        
        min_per_class = 10
        for label_name, count in label_counts.items():
            if count < min_per_class:
                msg = f"클래스 '{label_name}' 이미지 부족: {count}장 (최소 {min_per_class}장 필요)"
                print(f"[Train] {msg}")
                return False, msg

        print(f"[Train] 클래스: {labels}, 총 이미지: {len(images)}장 (클래스별: {label_counts})")

        total_samples = len(images)
        class_weights = [total_samples / (len(labels) * max(1, label_counts[l])) for l in labels]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"[Train] Class weights: {class_weights}")

        class SimpleDataset(Dataset):
            def __init__(self, imgs, tgts):
                self.imgs = imgs
                self.tgts = tgts
                self.transform = T.Compose([
                    T.ToTensor(),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomRotation(degrees=180),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
            def __len__(self):
                return len(self.imgs)
                
            def __getitem__(self, idx):
                return self.transform(self.imgs[idx]), self.tgts[idx]

        dataset = SimpleDataset(images, targets)
        loader = DataLoader(dataset, batch_size=16, shuffle=True) # ResNet50은 무거우므로 배치 16

        model = _build_model(len(labels)).to(device)
        
        # CrossEntropyLoss 대신 극강의 불균형 데이터 처리용 FocalLoss 도입
        criterion = FocalLoss(weight=weights_tensor, gamma=2.0)
        
        # EfficientNet은 layer 설계가 다르므로 classifier 역할이 다름
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())

        model.train()
        for epoch in range(epochs):
            total_loss, correct, total = 0.0, 0, 0
            for batch_imgs, batch_tgts in loader:
                batch_imgs = batch_imgs.to(device)
                batch_tgts = batch_tgts.clone().detach().to(dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                out = model(batch_imgs)
                loss = criterion(out, batch_tgts)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_tgts)
                _, pred = out.max(1)
                correct += pred.eq(batch_tgts).sum().item()
                total += len(batch_tgts)

            # 에폭 끝날 때마다 학습률(LR) 감소
            scheduler.step()

            avg_loss = total_loss / max(total, 1)
            accuracy = correct / max(total, 1) * 100
            
            # Best Model Checkpointing: 로스가 최저일 때의 가중치 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"  --> Best model updated! (Loss: {best_loss:.4f})")
            
            if progress_callback:
                progress_callback(epoch + 1, epochs, avg_loss, accuracy)
            print(f"[Train] Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}  Acc: {accuracy:.1f}%")

        # 가장 학습이 잘 된(Best) 모델의 가중치로 복원
        print(f"[Train] 학습 완료. Best Loss: {best_loss:.4f}")
        model.load_state_dict(best_model_wts)

        # === 오답 분석 시스템 (Misclassification Analysis) ===
        try:
            mis_dir = os.path.join(data_dir, "_misclassified")
            if os.path.exists(mis_dir):
                shutil.rmtree(mis_dir)
            os.makedirs(mis_dir, exist_ok=True)
            
            model.eval()
            print(f"[Train] 오답 분석 중... (결과는 {mis_dir} 에서 확인 가능)")
            
            mis_count = 0
            with torch.no_grad():
                # 전체 데이터를 다시 하나씩 검증 (Augmentation 제외)
                eval_transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                for i in range(len(images)):
                    img_tensor = eval_transform(images[i]).unsqueeze(0).to(device)
                    out = model(img_tensor)
                    _, pred = out.max(1)
                    pred_idx = pred.item()
                    gt_idx = targets[i]
                    
                    if pred_idx != gt_idx:
                        mis_count += 1
                        gt_label = labels[gt_idx]
                        pred_label = labels[pred_idx]
                        orig_path = file_paths[i]
                        fname = os.path.basename(orig_path)
                        # 파일명 형식: [정답]_to_[오답]_원래파일명.bmp
                        new_name = f"[{gt_label}]_predicted_as_[{pred_label}]_{fname}"
                        shutil.copy2(orig_path, os.path.join(mis_dir, new_name))
            
            if mis_count > 0:
                print(f"[Train] 총 {mis_count}개의 오답 이미지 발견. '_misclassified' 폴더를 확인하세요.")
            else:
                print(f"[Train] 모든 이미지를 완벽하게 학습했습니다!")
        except Exception as e:
            print(f"[Train] 오답 분석 중 오류 발생: {e}")

        # PyTorch 모델 및 가중치 저장 (.pth)
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        base_path = save_path.rsplit('.', 1)[0]
        pth_path = base_path + '.pth'
        onnx_path = base_path + '.onnx'
        label_path = base_path + '_labels.json'
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "labels": labels,
        }, pth_path)
        print(f"[Train] PyTorch 모델 저장 완료: {pth_path}")

        # ONNX 포맷 형변환 및 저장 (.onnx)
        
        try:
            model.eval()
            dummy_input = torch.randn(1, 3, CLASSIFICATION_INPUT_SIZE, CLASSIFICATION_INPUT_SIZE, device=device)
            torch.onnx.export(
                model, 
                dummy_input, 
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            # 라벨 정보를 JSON으로 저장
            import json
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(labels, f)
            print(f"[Train] ONNX 가속 모델 저장 완료: {onnx_path}")
            
        except Exception as e:
            print(f"[Train] ONNX 모델 변환 실패: {e}\n.pth 파일은 정상 사용 가능합니다.")
        return True, None
    except Exception as e:
        err_msg = str(e)
        print(f"[Train] 학습 오류: {err_msg}")
        import traceback
        traceback.print_exc()
        return False, err_msg
