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
                pad = 3
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(img_w, x + w + pad), min(img_h, y + h + pad)

                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    rh, rw = roi.shape

                    if rh > pad*2 and rw > pad*2:
                        center_roi = roi[pad:rh-pad, pad:rw-pad]
                        fg_mean = np.mean(center_roi)

                        full_mean = np.mean(roi)

                        sum_full = full_mean * roi.size
                        sum_center = fg_mean * center_roi.size
                        bg_pts = roi.size - center_roi.size

                        bg_mean = (sum_full - sum_center) / bg_pts if bg_pts > 0 else fg_mean

                        contrast = abs(fg_mean - bg_mean)
                        label = "Noise_Dust" if contrast < self.noise_contrast_threshold else "Particle"
                    else:
                        label = "Particle"
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
        self.labels = []
        self._device = "cpu"
        self._optimization_level = 0
        self._compiled = False
        self._ort_session_openvino = None
        self._openvino_device = "AUTO"
        if model_path and os.path.isfile(model_path):
            self.load_model(model_path)

    def is_loaded(self) -> bool:
        return self.model is not None or self.ort_session is not None

    def get_device(self) -> str:
        """추론에 사용 중인 디바이스. 'cuda' 또는 'cpu'."""
        return getattr(self, "_device", "cpu")

    def get_device_display(self) -> str:
        """UI 표시용: GPU/NPU 사용 여부 및 백엔드."""
        if not self.is_loaded():
            return "—"
        if getattr(self, "_ort_session_openvino", None) is not None:
            dev = getattr(self, "_openvino_device", "AUTO")
            return f"OpenVINO ({dev})"
        dev = self.get_device()
        if self.ort_session is not None:
            return "GPU (ONNX)" if dev == "cuda" else "CPU (ONNX)"
        return "GPU (PyTorch)" if dev == "cuda" else "CPU (PyTorch)"

    def get_optimization_level(self) -> int:
        return self._optimization_level

    def set_optimization_level(self, level: int):
        """최적화 수준 설정 (0~5). Level 2+ torch.compile, Level 5 OpenVINO EP(CPU/GPU/NPU/AUTO)."""
        old = self._optimization_level
        self._optimization_level = min(max(0, int(level)), 5)

        if self._optimization_level == 5:
            self._ensure_openvino_session()
        else:
            self._clear_openvino_session()
            if self._optimization_level >= 2 and not self._compiled and self.model is not None and self._device == "cuda":
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self._compiled = True
                    print(f"[DL Classifier] torch.compile 적용 (level={self._optimization_level})")
                except Exception as e:
                    print(f"[DL Classifier] torch.compile 실패 (무시): {e}")

        if self._optimization_level < 2 and self._compiled:
            try:
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
                    self._compiled = False
                    print("[DL Classifier] torch.compile 해제")
            except Exception:
                pass

        if old != self._optimization_level:
            backend = "ONNX" if self.ort_session is not None else "PyTorch"
            if getattr(self, "_ort_session_openvino", None) is not None:
                dev = getattr(self, "_openvino_device", "AUTO")
                print(f"[DL Classifier] level: {old}→{self._optimization_level}, backend: OpenVINO ({dev})")
            else:
                cuda_str = "CUDA" if self._device == "cuda" else "CPU-only"
                print(f"[DL Classifier] level: {old}→{self._optimization_level}, backend: {backend}, {cuda_str}")

    def _clear_openvino_session(self):
        """OpenVINO EP 세션 해제."""
        self._ort_session_openvino = None

    def get_openvino_device(self) -> str:
        """Level 5일 때 OpenVINO 장치 (CPU / GPU / NPU / AUTO)."""
        return getattr(self, "_openvino_device", "AUTO")

    def set_openvino_device(self, device: str):
        """Level 5일 때 사용할 OpenVINO 장치. CPU / GPU / NPU / AUTO."""
        allowed = ("CPU", "GPU", "NPU", "AUTO")
        self._openvino_device = device if device in allowed else "AUTO"
        if self._optimization_level == 5:
            self._ensure_openvino_session()

    def _ensure_openvino_session(self):
        """Level 5 선택 시 ONNX Runtime + OpenVINO EP 세션 생성. 실패 시 Level 4로 폴백."""
        self._clear_openvino_session()
        if not getattr(self, "model_path", None) or not str(self.model_path).endswith(".onnx"):
            print("[DL Classifier] Level 5(OpenVINO)는 ONNX 모델에서만 사용 가능. 현재 PyTorch 모델 로드됨.")
            self._optimization_level = 4
            return
        if self.ort_session is None:
            return
        device = getattr(self, "_openvino_device", "AUTO")
        try:
            import onnxruntime as ort
            opts = {"device_type": device}
            try:
                self._ort_session_openvino = ort.InferenceSession(
                    self.model_path,
                    sess_options=ort.SessionOptions(),
                    providers=[("OpenVINOExecutionProvider", opts)],
                )
            except (ValueError, TypeError) as e:
                if "OpenVINOExecutionProvider" in str(e) or "Provider" in str(e):
                    print("[DL Classifier] OpenVINO EP를 사용하려면: pip install onnxruntime-openvino")
                    self._optimization_level = 4
                    return
                raise
            dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
            inp_name = self._ort_session_openvino.get_inputs()[0].name
            self._ort_session_openvino.run(None, {inp_name: dummy})
            print(f"[DL Classifier] OpenVINO EP 로드 완료 (device={device})")
        except Exception as e:
            print(f"[DL Classifier] OpenVINO EP 로드 실패 (Level 4로 폴백): {e}")
            self._optimization_level = 4
            self._clear_openvino_session()

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

                self.model = None
                self._clear_openvino_session()
                active_providers = self.ort_session.get_providers()
                self._device = "cuda" if "CUDAExecutionProvider" in active_providers else "cpu"
                self._use_fp16 = False
                self.model_path = path_abs
                if self._optimization_level == 5:
                    self._ensure_openvino_session()

                # Warmup: CUDA 메모리 사전 할당 (첫 추론 지연 제거)
                try:
                    input_name = self.ort_session.get_inputs()[0].name
                    input_shape = self.ort_session.get_inputs()[0].shape
                    dummy_h = input_shape[2] if isinstance(input_shape[2], int) else 224
                    dummy_w = input_shape[3] if isinstance(input_shape[3], int) else 224
                    dummy = np.random.randn(1, 3, dummy_h, dummy_w).astype(np.float32)
                    self.ort_session.run(None, {input_name: dummy})
                    print(f"[DL Classifier] ONNX warmup 완료 (CUDA workspace 사전할당)")
                except Exception:
                    pass

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
            
            self._compiled = False
            self.model_path = path_abs
            print(f"[DL Classifier] PyTorch 모델 로드 완료: {path_abs}, 클래스: {self.labels}, device: {self._device}")

            if self._optimization_level >= 2 and self._device == "cuda":
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    self._compiled = True
                    print("[DL Classifier] torch.compile 적용 완료")
                except Exception as e:
                    print(f"[DL Classifier] torch.compile 실패 (무시): {e}")

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

    # ROI 풀: 재사용 버퍼 2개 (스레드 2개와 번갈아 사용, 할당/GC 감소)
    _ROI_CHUNK_MAX = 2048

    def _get_roi_pool_buffers(self, chunk_size: int, sz: int):
        """캐시된 [buf0, buf1] 반환. 각 (CHUNK_MAX, 3, sz, sz)."""
        if not hasattr(self, "_roi_pool_buffers") or self._roi_pool_buffers is None:
            self._roi_pool_buffers = [
                np.empty((self._ROI_CHUNK_MAX, 3, sz, sz), dtype=np.float32),
                np.empty((self._ROI_CHUNK_MAX, 3, sz, sz), dtype=np.float32),
            ]
        cap = self._roi_pool_buffers[0].shape[0]
        if chunk_size > cap:
            self._roi_pool_buffers = [
                np.empty((max(chunk_size, self._ROI_CHUNK_MAX), 3, sz, sz), dtype=np.float32),
                np.empty((max(chunk_size, self._ROI_CHUNK_MAX), 3, sz, sz), dtype=np.float32),
            ]
        return self._roi_pool_buffers

    def _extract_rois_chunk(self, bboxes, frame_bgr, start, end, img_w, img_h, sz, out=None):
        """CPU에서 ROI crop + resize + normalize. out이 있으면 유효 ROI만 앞쪽에 채움."""
        chunk_size = end - start
        if out is not None and out.shape[0] >= chunk_size:
            roi_chw = out
        else:
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

            if roi.shape[0] == sz and roi.shape[1] == sz:
                patch = roi
            else:
                patch = cv2.resize(roi, (sz, sz), interpolation=cv2.INTER_LINEAR)
            roi_chw[j] = patch[:, :, ::-1].transpose(2, 0, 1)
            valid[j] = True

        valid_indices = np.where(valid)[0]
        n_valid = len(valid_indices)
        if n_valid == 0:
            return roi_chw, 0, valid_indices

        if out is roi_chw:
            # 유효 ROI만 앞쪽으로 밀어서 연속 구간으로 (풀 재사용 시)
            for k, idx in enumerate(valid_indices):
                if k != idx:
                    roi_chw[k] = roi_chw[idx]
            v = roi_chw[:n_valid]
        else:
            v = roi_chw[valid_indices]

        v /= 255.0
        v -= self._MEAN_CHW
        v /= self._STD_CHW
        if out is roi_chw:
            roi_chw[:n_valid] = v
        else:
            roi_chw[valid_indices] = v

        return roi_chw, n_valid, valid_indices

    def _classify_gpu_crop(self, bboxes, frame_bgr, img_w, img_h, sz, use_fp16, n):
        """PyTorch CUDA: 프레임 1장만 GPU 전송 → crop/resize/normalize GPU 처리.

        optimization_level별 전략:
          1-2: patches.append + torch.cat + softmax bypass
          3:   배치 버퍼 사전할당 (torch.cat 제거)
          4:   F.grid_sample (Python crop 루프 제거)
        """
        import torch.nn.functional as F
        opt = self._optimization_level
        dtype = torch.float16 if use_fp16 else torch.float32

        if not hasattr(self, '_norm_cache'):
            self._norm_cache = {}
        cache_key = (str(self._device), dtype)
        if cache_key not in self._norm_cache:
            m = torch.tensor([0.485, 0.456, 0.406], device=self._device, dtype=dtype).view(1, 3, 1, 1)
            s = torch.tensor([0.229, 0.224, 0.225], device=self._device, dtype=dtype).view(1, 3, 1, 1)
            self._norm_cache[cache_key] = (m, s)
        mean_gpu, std_gpu = self._norm_cache[cache_key]

        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        frame_gpu = torch.from_numpy(frame_rgb).to(self._device, non_blocking=True)
        frame_gpu = frame_gpu.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
        frame_gpu.div_(255.0).sub_(mean_gpu).div_(std_gpu)

        crop_coords = []
        valid_indices = []
        for i in range(n):
            x, y, w, h = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 2]), int(bboxes[i, 3])
            side = max(w * 2, h * 2, sz)
            cx, cy = x + w // 2, y + h // 2
            x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
            x2, y2 = min(img_w, x1 + side), min(img_h, y1 + side)
            if x2 > x1 and y2 > y1:
                crop_coords.append((y1, y2, x1, x2))
                valid_indices.append(i)

        all_conf = np.zeros(n, dtype=np.float32)
        all_pred = np.zeros(n, dtype=np.int64)
        valid_mask = np.zeros(n, dtype=bool)

        if not valid_indices:
            del frame_gpu
            return all_conf, all_pred, valid_mask

        MINI_BATCH = 128
        for mb_start in range(0, len(valid_indices), MINI_BATCH):
            mb_end = min(mb_start + MINI_BATCH, len(valid_indices))
            mb_size = mb_end - mb_start
            mb_crops = crop_coords[mb_start:mb_end]

            if opt >= 4:
                batch = self._grid_sample_crop(frame_gpu, mb_crops, img_h, img_w, sz, dtype)
            elif opt >= 3:
                batch = torch.empty(mb_size, 3, sz, sz, device=self._device, dtype=dtype)
                for k, (y1, y2, x1, x2) in enumerate(mb_crops):
                    patch = frame_gpu[:, :, y1:y2, x1:x2]
                    if patch.shape[2] != sz or patch.shape[3] != sz:
                        batch[k:k+1] = F.interpolate(patch, size=(sz, sz), mode='bilinear', align_corners=False)
                    else:
                        batch[k] = patch[0]
            else:
                patches = []
                for y1, y2, x1, x2 in mb_crops:
                    patch = frame_gpu[:, :, y1:y2, x1:x2]
                    if patch.shape[2] != sz or patch.shape[3] != sz:
                        patch = F.interpolate(patch, size=(sz, sz), mode='bilinear', align_corners=False)
                    patches.append(patch)
                batch = torch.cat(patches, dim=0)

            out = self.model(batch)
            pred_idx = out.argmax(dim=1)
            max_logits = out.gather(1, pred_idx.unsqueeze(1))
            exp_shifted = torch.exp(out.float() - max_logits.float())
            conf = 1.0 / exp_shifted.sum(dim=1)
            conf_np = conf.cpu().numpy()
            pred_np = pred_idx.cpu().numpy()

            for k in range(mb_size):
                gi = valid_indices[mb_start + k]
                all_conf[gi] = conf_np[k]
                all_pred[gi] = pred_np[k]
                valid_mask[gi] = True

        del frame_gpu
        return all_conf, all_pred, valid_mask

    def _grid_sample_crop(self, frame_gpu, crop_coords, img_h, img_w, sz, dtype):
        """Level 4: F.grid_sample로 모든 crop을 단일 GPU 연산으로 처리."""
        import torch.nn.functional as F
        n = len(crop_coords)
        coords_t = torch.tensor(crop_coords, device=self._device, dtype=torch.float32)
        y1s, y2s, x1s, x2s = coords_t[:, 0], coords_t[:, 1], coords_t[:, 2], coords_t[:, 3]

        t = torch.linspace(0, 1, sz, device=self._device, dtype=torch.float32)
        x_px = x1s.unsqueeze(1) + t.unsqueeze(0) * (x2s - x1s - 1).unsqueeze(1)
        y_px = y1s.unsqueeze(1) + t.unsqueeze(0) * (y2s - y1s - 1).unsqueeze(1)

        x_norm = 2.0 * x_px / max(img_w - 1, 1) - 1.0
        y_norm = 2.0 * y_px / max(img_h - 1, 1) - 1.0

        grid_x = x_norm.unsqueeze(1).expand(-1, sz, -1)
        grid_y = y_norm.unsqueeze(2).expand(-1, -1, sz)
        grid = torch.stack([grid_x, grid_y], dim=-1).to(dtype=dtype)

        frame_expanded = frame_gpu.expand(n, -1, -1, -1)
        return F.grid_sample(frame_expanded, grid, mode='bilinear',
                             align_corners=True, padding_mode='zeros')

    _gc_counter = 0
    _GC_INTERVAL = 50

    def classify_batch(self, contours, frame_bgr=None):
        """
        여러 contour를 한 번에 배치 추론 (10K+ contour 대응).
        PyTorch+CUDA → GPU-side crop, ONNX/CPU → 기존 CPU 파이프라인.
        frame_bgr: 원본 BGR 이미지.
        Returns: list of dict (label, confidence, area, ...)
        """
        import time
        t0 = time.perf_counter()
        n = len(contours)
        if n == 0:
            return []

        DeepLearningClassifier._gc_counter += 1
        if DeepLearningClassifier._gc_counter % self._GC_INTERVAL == 0:
            import gc
            gc.collect()
            if self._device == "cuda":
                torch.cuda.empty_cache()

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

            opt = self._optimization_level
            is_cuda = (self._device == "cuda")
            # GPU crop: Level 1+ AND PyTorch + CUDA. ONNX는 항상 CPU pipeline (GPU→CPU 왕복 없음).
            use_gpu_crop = (opt >= 1 and is_cuda and self.model is not None and self.ort_session is None)

            if use_gpu_crop:
                with torch.inference_mode():
                    all_conf, all_pred, valid_mask = self._classify_gpu_crop(
                        bboxes, frame_bgr, img_w, img_h, sz, use_fp16, n)
            else:
                import gc
                from concurrent.futures import ThreadPoolExecutor
                CHUNK = 2048
                MINI_BATCH = 128
                all_conf = np.zeros(n, dtype=np.float32)
                all_pred = np.zeros(n, dtype=np.int64)
                valid_mask = np.zeros(n, dtype=bool)

                chunk_ranges = [(s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]
                pool = self._get_roi_pool_buffers(CHUNK, sz)

                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        c0s, c0e = chunk_ranges[0][0], chunk_ranges[0][1]
                        future = executor.submit(
                            self._extract_rois_chunk, bboxes, frame_bgr,
                            c0s, c0e, img_w, img_h, sz, out=pool[0]
                        )

                        for ci, (chunk_start, chunk_end) in enumerate(chunk_ranges):
                            roi_chw, n_valid, valid_in_chunk = future.result()

                            if ci + 1 < len(chunk_ranges):
                                next_s, next_e = chunk_ranges[ci + 1]
                                future = executor.submit(
                                    self._extract_rois_chunk, bboxes, frame_bgr,
                                    next_s, next_e, img_w, img_h, sz,
                                    out=pool[(ci + 1) % 2]
                                )

                            if n_valid == 0:
                                continue

                            valid_rois = roi_chw[:n_valid]

                            ov_sess = getattr(self, "_ort_session_openvino", None)
                            if self.ort_session is not None or ov_sess is not None:
                                chunk_conf, chunk_idx = [], []
                                session = ov_sess if ov_sess is not None else self.ort_session
                                input_name = session.get_inputs()[0].name
                                for mb_start in range(0, len(valid_rois), MINI_BATCH):
                                    mb_rois = valid_rois[mb_start:mb_start + MINI_BATCH]
                                    ort_out = session.run(None, {input_name: mb_rois})[0]
                                    pred = np.argmax(ort_out, axis=1)
                                    max_vals = ort_out[np.arange(len(pred)), pred]
                                    exp_shifted = np.exp(ort_out - max_vals[:, None])
                                    chunk_conf.append((1.0 / exp_shifted.sum(axis=1)).astype(np.float32))
                                    chunk_idx.append(pred)

                                conf_arr = np.concatenate(chunk_conf) if chunk_conf else np.array([])
                                idx_arr = np.concatenate(chunk_idx) if chunk_idx else np.array([])
                            else:
                                batch_tensor = torch.from_numpy(valid_rois).to(
                                    self._device, non_blocking=True)

                                if use_fp16:
                                    batch_tensor = batch_tensor.half()

                                chunk_conf, chunk_idx = [], []
                                with torch.inference_mode():
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
                finally:
                    if gc_was_enabled:
                        gc.enable()

            infos = self._build_infos(areas, peris, rects)
            result_list = [{"label": "Unknown", "confidence": 0.0, **infos[i]} for i in range(n)]

            for i in range(n):
                if valid_mask[i]:
                    label = self.labels[all_pred[i]] if all_pred[i] < len(self.labels) else "Unknown"
                    result_list[i]["label"] = label
                    result_list[i]["confidence"] = float(all_conf[i])

            elapsed = time.perf_counter() - t0
            if getattr(self, "_ort_session_openvino", None) is not None:
                backend = f"OpenVINO({self._openvino_device})"
            else:
                backend = "ONNX" if self.ort_session is not None else "PyTorch"
            mode = f"GPU-Crop(L{opt},{backend})" if use_gpu_crop else f"CPU-Pipeline(L{opt},{backend})"
            print(f"[DL Classifier] {n}개 contour 분류 완료 (mode={mode}, FP16={use_fp16}): {elapsed:.3f}초")
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
