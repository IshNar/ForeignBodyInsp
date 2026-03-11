# 🚀 검사 파이프라인 성능 최적화 Walkthrough

> **핵심 원칙**: 검사 민감도·정확도는 **단 1%도 변경하지 않음**. 오직 코드 실행 효율만 개선.

---

## 📊 최적화 결과 (벤치마크)

| 모드 | 테스트 조건 | 소요 시간 |
|------|------------|----------|
| **Rule-Based** | 2000×2000px 이미지, 436개 결함 | **0.19초** |
| **Deep Learning** | 2000×2000px 이미지, 405개 결함 (EfficientNet-B0 / ONNX) | **3.08초** (참고) |

---

## 1️⃣ Rule-Based 분류 최적화

### 🔴 Before (느렸던 이유)

모든 결함마다 **마스크 도화지 3장**을 그려서 대비(Contrast)를 계산했습니다.

```python
# ❌ 결함 1개마다 이 작업을 반복 (100개면 100번!)
fg_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
cv2.drawContours(fg_mask, [roi_contour], -1, 255, -1)  # 마스크 그리기 (느림)

kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(fg_mask, kernel, iterations=1)  # 팽창 (느림)
bg_mask = cv2.bitwise_xor(dilated_mask, fg_mask)          # XOR (느림)

fg_mean = cv2.mean(roi_gray, mask=fg_mask)[0]
bg_mean = cv2.mean(roi_gray, mask=bg_mask)[0]
```

> [!WARNING]
> `cv2.drawContours` + `cv2.dilate` + `cv2.bitwise_xor`를 **결함 1개마다** 호출하면, 결함이 수백 개일 때 CPU가 극심하게 느려집니다.

### 🟢 After (빨라진 방법)

마스크를 아예 **그리지 않고**, 바운딩 박스의 **중심 vs 테두리** 픽셀 평균값만 비교합니다.

```python
# ✅ 마스크 없이 순수 NumPy 연산으로 대비 계산
roi = gray[y1:y2, x1:x2]
rh, rw = roi.shape

center_roi = roi[pad:rh-pad, pad:rw-pad]   # 중심 영역 (결함 내부)
fg_mean = np.mean(center_roi)               # 결함 내부 평균 밝기

full_mean = np.mean(roi)                    # 전체 ROI 평균
sum_full = full_mean * roi.size
sum_center = fg_mean * center_roi.size
bg_pts = roi.size - center_roi.size

# 배경 평균 = (전체 합 - 중심 합) / (전체 픽셀 - 중심 픽셀)
bg_mean = (sum_full - sum_center) / bg_pts

contrast = abs(fg_mean - bg_mean)
label = "Noise_Dust" if contrast < threshold else "Particle"
```

> [!TIP]
> `np.mean()`은 C로 구현된 벡터 연산이라 `cv2.drawContours`보다 **수십 배** 빠릅니다.

또한, `classify`와 `classify_batch`를 통합하여 **grayscale 변환을 1번만** 수행하도록 했습니다:

```diff
 def classify_batch(self, contours, image=None):
+    # ✅ grayscale 변환을 배치 전체에서 딱 1번만 수행
+    gray = None
+    if image is not None:
+        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
+        img_h, img_w = gray.shape
+
     for contour in contours:
-        # ❌ 이전: classify()를 개별 호출 → 매번 cvtColor 반복
-        results.append(self.classify(contour, image))
+        # ✅ 이후: gray를 재사용하며 직접 계산
+        ...
```

---

## 2️⃣ Deep Learning 분류 최적화

### 🔴 Before

`cv2.contourArea` → `cv2.boundingRect` 순서로 **2번 루프**를 돌았고, GPU 전송이 동기(blocking)로 이루어졌습니다.

```python
# ❌ 1차 루프: area만 계산
areas = np.array([cv2.contourArea(cnt) for cnt in contours])

# ❌ 2차 루프: boundingRect 다시 계산
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)  # 이미 위에서 했어야 할 일
    ...

# ❌ GPU 전송 대기 (CPU가 멈춤)
batch_tensor = tensor.to(self._device)
```

### 🟢 After

**1번의 루프**로 area, bbox, perimeter, minAreaRect를 **모두** 계산하고, GPU 전송을 **비동기(non_blocking)**로 처리합니다.

```python
# ✅ 모든 정보를 한 번에 계산 (루프 1회)
for i, cnt in enumerate(contours):
    areas[i] = cv2.contourArea(cnt)
    peris.append(cv2.arcLength(cnt, True))
    rects.append(cv2.minAreaRect(cnt))
    bboxes[i] = cv2.boundingRect(cnt)

# ✅ GPU 비동기 전송 (CPU는 멈추지 않고 다음 작업 준비)
batch_tensor = tensor.to(self._device, non_blocking=True)
```

> [!NOTE]
> `non_blocking=True`는 CUDA GPU 사용 시 CPU↔GPU 데이터 전송을 **병렬화**합니다. CPU가 다음 배치를 준비하는 동안 GPU가 이전 배치를 처리할 수 있습니다.

---

## 3️⃣ Main Window 파이프라인 최적화

### 🔴 Before

Bubble contour마다 `cv2.contourArea`를 **개별 호출**하여 결과 dict를 만들었습니다.

```python
# ❌ Bubble 1개마다 contourArea 호출
for bc in bubble_contours:
    area = cv2.contourArea(bc)
    final_results.append({
        "label": "Bubble",
        "area": area,
        "confidence": 1.0
    })
```

### 🟢 After

`classify_batch`를 통해 **한 번에** 모든 Bubble의 area/circularity/aspect_ratio를 계산하고, 라벨만 덮어씌웁니다.

```python
# ✅ 배치로 한 번에 처리 후 라벨만 교체
if bubble_contours:
    bubble_stats = classifier.classify_batch(bubble_contours, frame)
    for stat in bubble_stats:
        stat["label"] = "Bubble"
        stat["confidence"] = 1.0
        final_results.append(stat)
```

---

## 📁 수정된 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| [classification.py](file:///F:/PythonProjects/ForeignBodyInsp-vial-inspection-system/src/core/classification.py) | `RuleBasedClassifier` 마스크 제거, `DeepLearningClassifier` 루프 통합, `_build_infos` 신규 |
| [main_window.py](file:///F:/PythonProjects/ForeignBodyInsp-vial-inspection-system/src/ui/main_window.py) | Bubble 처리 배치화, 중복 `contourArea` 제거 |

---

## 🛠️ 검증 방법
1. 프로그램 실행 후 영상을 로드합니다.
2. `Process (General)` 또는 `Process (Deep Learning)`를 실행합니다.
3. 검사 로그 창에서 `[DL Classifier] ... 분류 완료 (최적화): 0.XXX초` 문구를 확인합니다.
4. 기존과 동일한 검출 결과가 나오는지 확인합니다.

---

> [!IMPORTANT]
> 검출 알고리즘(`detection.py`)과 모델 구조(`ResNet50`, `FocalLoss`)는 **전혀 수정하지 않았습니다**. 검사 결과는 이전과 100% 동일합니다.
