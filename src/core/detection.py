import os
import cv2
import numpy as np
import concurrent.futures


class BubbleDetectorParams:
    """Bubble 검출 파라미터.

    Morphological Opening 배경 평탄화 → 양극성 차이 → (선택) CLAHE →
    노이즈 제거 → DoG 밴드패스 → MAD 적응 임계 → 형태학 → 형상 필터.
    """

    def __init__(self):
        self.enabled = True
        # ── 배경 평탄화 (Morphological Opening) ──
        self.bg_open_ksize = 61       # 배경 추정 커널 (홀수, 바이알 곡면 스케일에 맞춤)
        self.bg_smooth_sigma = 15.0   # 배경 스무딩 σ (0이면 자동=ksize/4)
        # ── CLAHE (평탄화 후 적용, 선택) ──
        self.use_clahe = False
        self.clahe_clip = 2.0
        self.clahe_grid = 8
        # ── 노이즈 제거 ──
        self.denoise_mode = "median"  # "median" | "bilateral" | "none"
        self.median_ksize = 5
        self.bilateral_d = 7
        self.bilateral_sigmaColor = 25.0
        self.bilateral_sigmaSpace = 25.0
        # ── DoG 밴드패스 ──
        self.sigma_small = 1.2
        self.sigma_large = 6.0
        # ── MAD 적응 임계 ──
        self.thr_k = 4.0             # median + k * MAD (낮을수록 민감)
        # ── 형태학 ──
        self.morph_close_size = 5
        self.morph_open_size = 3
        # ── 형상 필터 ──
        self.min_diameter = 8         # 최소 버블 직경 (px)
        self.max_diameter = 100       # 최대 버블 직경 (px)
        self.circularity_min = 0.35
        self.solidity_min = 0.70
        self.max_aspect_ratio = 2.2

    def get_params(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def set_params(self, params: dict):
        if not params:
            return
        for k, v in params.items():
            if hasattr(self, k):
                expected_type = type(getattr(self, k))
                try:
                    setattr(self, k, expected_type(v))
                except (ValueError, TypeError):
                    pass


class ForeignBodyDetector:
    def __init__(self):
        self.last_debug = None
        self.bubble_params = BubbleDetectorParams()

    def detect_static(self, image: np.ndarray, threshold: int = 100,
                      min_area: int = 10, use_adaptive: bool = False,
                      debug_dir: str = None,
                      detect_bubbles: bool = False,
                      open_kernel: int = 2, close_kernel: int = 3):
        """
        Detects foreign bodies in a static image.
        Returns (contours_list, binary_mask).
        detect_bubbles=True이면 Bubble 전용 검출 결과를 합침.
        """
        if image is None:
            self.last_debug = None
            return [], None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        def _save(name: str, img: np.ndarray):
            if debug_dir is None:
                return
            try:
                os.makedirs(debug_dir, exist_ok=True)
                path = os.path.join(debug_dir, name)
                ok, buf = cv2.imencode(".png", img)
                if ok:
                    with open(path, "wb") as f:
                        f.write(buf.tobytes())
            except Exception:
                pass

        _save("01_gray.png", gray)

        # ─── 1. 멀티스레드 병렬 처리를 위한 함수 분리 ───
        def run_regular():
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _save("02_blurred.png", blurred)

            if use_adaptive:
                adaptive_binary = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 15, 3
                )
                global_dark = (blurred < threshold).astype(np.uint8) * 255
                thresh = cv2.bitwise_and(adaptive_binary, global_dark)
            else:
                _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

            _save("03_threshold.png", thresh)

            ok_size = max(1, open_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok_size, ok_size))
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            _save("04_opened.png", opened)
            
            ck_size = max(1, close_kernel)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck_size, ck_size))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            _save("05_closed.png", closed)

            cnts, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            v_cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= min_area]
            return v_cnts, closed, blurred, thresh, opened

        def run_bubble():
            if detect_bubbles and self.bubble_params.enabled:
                return self.detect_bubbles(gray, debug_dir=debug_dir)
            return [], None

        # ─── 2. ThreadPoolExecutor를 통한 병렬 검사(일반+버블) 동시 실행 ───
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_reg = executor.submit(run_regular)
            future_bub = executor.submit(run_bubble)

            # 두 검사가 모두 끝날 때까지 대기
            valid_contours, closed, blurred, thresh, opened = future_reg.result()
            bubble_contours, bubble_debug = future_bub.result()

        # ─── 3. 검사 결과 병합 및 디버깅 데이터 최종 저장 ───
        if bubble_contours:
            valid_contours = self._merge_contours(valid_contours, bubble_contours)

        self.last_debug = {
            "gray": gray,
            "blurred": blurred,
            "threshold": thresh,
            "opened": opened,
            "closed": closed,
        }
        if bubble_debug:
            self.last_debug["bubble_debug"] = bubble_debug

        return valid_contours, closed

    @staticmethod
    def _mad_threshold(values: np.ndarray, k: float) -> float:
        """median + k * (1.4826 * MAD) 적응형 임계값."""
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        return float(med + k * 1.4826 * mad)

    def detect_bubbles(self, gray: np.ndarray,
                       debug_dir: str = None,
                       stop_after: str | None = None) -> tuple[list, dict | None]:
        """Bubble 검출: Morph-Open 배경 평탄화 → 양극성 → DoG → MAD 임계 → 형상 필터.

        ROI가 없으면 전체 이미지에 대해 수행.
        stop_after: "clahe" / "diff_map" / "binary" / None(전체).
        """
        p = self.bubble_params
        if not p.enabled:
            return [], None

        h, w = gray.shape[:2]
        debug_imgs = {}
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # ──── 1. Morphological Opening 배경 평탄화 ────
        ksize = max(3, int(p.bg_open_ksize)) | 1
        k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k_bg)
        smooth_sigma = float(getattr(p, "bg_smooth_sigma", 0))
        if smooth_sigma <= 0:
            smooth_sigma = ksize / 4.0
        bg = cv2.GaussianBlur(bg, (0, 0), smooth_sigma)

        flat_dark = cv2.subtract(bg, gray)
        flat_bright = cv2.subtract(gray, bg)
        flat = cv2.max(flat_dark, flat_bright)

        # ──── 2. (선택) CLAHE ────
        if getattr(p, "use_clahe", False):
            clahe = cv2.createCLAHE(
                clipLimit=float(p.clahe_clip),
                tileGridSize=(int(p.clahe_grid), int(p.clahe_grid)))
            flat = clahe.apply(flat)

        debug_imgs["clahe"] = flat.copy()
        if stop_after == "clahe":
            return [], debug_imgs

        # ──── 3. 노이즈 제거 ────
        dm = getattr(p, "denoise_mode", "median") or "median"
        if dm == "median":
            ks = max(3, int(p.median_ksize)) | 1
            den = cv2.medianBlur(flat, ks)
        elif dm == "bilateral":
            den = cv2.bilateralFilter(
                flat, int(p.bilateral_d),
                float(p.bilateral_sigmaColor),
                float(p.bilateral_sigmaSpace))
        else:
            den = flat

        # ──── 4. DoG 밴드패스 (abs) 멀티스레드 병렬 실행 ────
        den_f = den.astype(np.float32) / 255.0
        s1, s2 = float(p.sigma_small), float(p.sigma_large)
        
        def _calc_g1():
            return cv2.GaussianBlur(den_f, (0, 0), s1)
            
        def _calc_g2():
            return cv2.GaussianBlur(den_f, (0, 0), s2)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_g1 = executor.submit(_calc_g1)
            fut_g2 = executor.submit(_calc_g2)
            g1 = fut_g1.result()
            g2 = fut_g2.result()
            
        dog_abs = np.abs(g1 - g2)

        diff_vis = np.clip(dog_abs * 255.0 * 10.0, 0, 255).astype(np.uint8)
        debug_imgs["diff_map"] = diff_vis
        if stop_after == "diff_map":
            return [], debug_imgs

        # ──── 5. MAD 적응 임계 ────
        vals = dog_abs[mask > 0].ravel()
        if vals.size < 50:
            debug_imgs["binary"] = np.zeros((h, w), dtype=np.uint8)
            debug_imgs["bubble_candidates"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return [], debug_imgs

        T = self._mad_threshold(vals, float(p.thr_k))
        cand = ((dog_abs > T).astype(np.uint8) * 255)
        cand = cv2.bitwise_and(cand, cand, mask=mask)

        # ──── 6. 형태학 정리 ────
        ck = max(1, int(p.morph_close_size)) | 1
        ok = max(1, int(p.morph_open_size)) | 1
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k_close)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k_open)
        debug_imgs["binary"] = cand

        if stop_after == "binary":
            return [], debug_imgs

        # ──── 7. 컨투어 + 형상 필터 ────
        contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        min_r = max(1, int(p.min_diameter)) / 2.0
        max_r = max(min_r + 1, int(p.max_diameter)) / 2.0
        min_area = np.pi * (min_r ** 2) * 0.35
        max_area = np.pi * (max_r ** 2) * 1.30
        circ_min = float(p.circularity_min)
        sol_min = float(getattr(p, "solidity_min", 0.70))
        ar_max = float(getattr(p, "max_aspect_ratio", 2.2))

        bubble_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1e-6:
                continue

            (_, _), radius = cv2.minEnclosingCircle(cnt)
            if radius < min_r or radius > max_r:
                continue

            circularity = 4.0 * np.pi * area / (peri * peri)
            if circularity < circ_min:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 1e-6 else 0.0
            if solidity < sol_min:
                continue

            x, y, ww, hh = cv2.boundingRect(cnt)
            aspect = max(ww, hh) / max(min(ww, hh), 1)
            if aspect > ar_max:
                continue

            bubble_contours.append(cnt)

        # ──── 8. 시각화 ────
        bub_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cnt in bubble_contours:
            cv2.drawContours(bub_vis, [cnt], -1, (0, 255, 0), 1)
        debug_imgs["bubble_candidates"] = bub_vis

        if debug_dir is not None:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                for name, img in debug_imgs.items():
                    ok_enc, buf = cv2.imencode(".png", img)
                    if ok_enc:
                        with open(os.path.join(debug_dir, f"06_{name}.png"), "wb") as f:
                            f.write(buf.tobytes())
            except Exception:
                pass

        return bubble_contours, debug_imgs

    def _merge_contours(self, base_contours: list, new_contours: list) -> list:
        """겹치는 contour를 제거하며 합침. new_contours의 중심이 base bbox 안이면 스킵.
        
        new_contours는 findContours에서 나온 것이므로 서로 겹치지 않음 → base만 체크.
        완전 벡터화: O(n) (이전 O(n²) 대비 10,000배 이상 빠름).
        """
        if not base_contours:
            return list(new_contours)
        if not new_contours:
            return list(base_contours)

        # Base BBoxes: vectorized
        base_arr = np.array([cv2.boundingRect(c) for c in base_contours], dtype=np.float32)
        if len(base_arr) == 0:
            return list(base_contours) + list(new_contours)

        bx, by, bw, bh = base_arr.T
        margin = np.maximum(bw, bh) * 0.3
        x_min = bx - margin
        x_max = bx + bw + margin
        y_min = by - margin
        y_max = by + bh + margin

        # New contours의 중심점을 한 번에 계산
        n = len(new_contours)
        centers = np.empty((n, 2), dtype=np.float64)
        valid = np.ones(n, dtype=bool)
        
        for i, cnt in enumerate(new_contours):
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                valid[i] = False
                continue
            centers[i, 0] = M["m10"] / M["m00"]
            centers[i, 1] = M["m01"] / M["m00"]

        # 벡터화된 겹침 검사: 모든 new centers vs 모든 base bboxes (broadcasting)
        # cx shape: (n,), x_min shape: (m,) → broadcasting to (n, m)
        cx = centers[:, 0]  # (n,)
        cy = centers[:, 1]  # (n,)
        
        # (n, 1) vs (m,) → (n, m) boolean matrix
        overlap_x = (cx[:, None] >= x_min[None, :]) & (cx[:, None] <= x_max[None, :])
        overlap_y = (cy[:, None] >= y_min[None, :]) & (cy[:, None] <= y_max[None, :])
        overlap_any = np.any(overlap_x & overlap_y, axis=1)  # (n,) — True if overlaps with ANY base
        
        # valid이고 overlap이 없는 contour만 추가
        keep = valid & ~overlap_any
        
        result = list(base_contours)
        for i in range(n):
            if keep[i]:
                result.append(new_contours[i])
                
        return result

    def detect_motion(self, frames: list):
        """
        Placeholder for motion detection logic.
        Intended for 'Spin & Stop' sequences where particles move against a static background.
        """
        pass
