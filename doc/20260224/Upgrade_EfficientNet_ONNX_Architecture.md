# ⚡ 가속 모델 전환 가이드: EfficientNet-B0 + ONNX Runtime

> 이 문서는 극단적인 검사 속도 확보를 위해, 기존 `ResNet50 (PyTorch)` 모델을 `EfficientNet-B0 (ONNX Runtime)` 엔진으로 전환한 설계 사유와 변경점을 기록합니다.

---

## 🚀 도입 배경 (왜 바꿔야 했는가?)

1. **병목 지점 파악**: `ResNet50`은 한 번에 10,000개 이상의 결함(Contour) ROI 스니펫을 추론할 때, 16GB VRAM과 반정밀도(FP16)를 총동원해도 약 25~30초가 소요되는 한계에 도달했습니다.
2. **연산량 감소 필요**: GPU의 수학적 연산량 한계를 우회하려면 모델 자체의 파라미터를 줄여야 했습니다.
3. **엔진 레이턴시 제거**: PyTorch는 범용 딥러닝 "연구" 프레임워크라 추론 시 불필요한 Python 오버헤드가 큽니다. 오직 "운영 환경 추론(Inference)"에만 특화된 **ONNX Runtime(ORT) C++ 백엔드**를 파이썬에서 직접 호출하여 오버헤드를 제로에 가깝게 만들기로 결정했습니다.

---

## 🎯 1. 모델 교체: `ResNet50` ➡️ `EfficientNet-B0`

### 모델 스펙 비교

| 구분 | ResNet50 (이전) | EfficientNet-B0 (변경 후) | 비교 |
|------|-----------------|---------------------------|------|
| **파라미터 수** | 약 25.6M (2,500만) | **약 5.3M (530만)** | ⬇️ **약 5배 가벼움** |
| **GFLOPs (연산량)** | 4.1 | **0.39** | ⬇️ **약 10배 적은 연산량** |
| **ImageNet 정확도** | 80.8% | **77.1% (근사치)** | 거의 동일 |
| **입력 해상도** | 224 × 224 | 224 × 224 | 유지 |

> **선택 사유**: Google의 NAS(Neural Architecture Search)를 통해 발견된 EfficientNet은, ResNet50보다 모델 크기가 1/5 임에도 불구하고 실무 정확도는 거의 동일합니다.

### 코드 변경 (PyTorch 모델 빌더)

```diff
- from torchvision.models import resnet50, ResNet50_Weights
- weights = ResNet50_Weights.IMAGENET1K_V2
- model = resnet50(weights=weights)
- model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))

+ from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
+ weights = EfficientNet_B0_Weights.IMAGENET1K_V1
+ model = efficientnet_b0(weights=weights)
+ # EfficientNet은 마지막 레이어 이름이 'classifier'
+ in_features = model.classifier[1].in_features
+ model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, num_classes))
```

---

## 🏎️ 2. 엔진 교체: `PyTorch` ➡️ `ONNX Runtime`

ONNX(Open Neural Network Exchange)는 딥러닝 모델의 국제 표준 포맷입니다. ONNX Runtime 프로바이더를 `CUDAExecutionProvider`로 설정하면, PyTorch 순정(FP16) 대비 실시간 추론 속도가 **1.5배 ~ 2배** 상승합니다.

### 📝 학습 파이프라인 변경 (학습 시 자동 변환)
기존에는 `*.pth` (파이토치 상태 파일)만 저장했지만, 이제 학습 완료 후 `torch.onnx.export`를 통해 `*.onnx` 포맷도 자동으로 함께 구어서 저장(Export)합니다.

### ⚡ 추론 파이프라인 변경 (ONNX Runtime 적용)
`classifier_batch` 호출 시 기존 PyTorch Tensor를 그대로 넘기던 부분은, **NumPy 배열을 그대로 ONNX Runtime 세션에 밀어넣는 구조**로 변경됩니다. (torch.Tensor 구조체 변환 오버헤드 제거)

```python
# Before (PyTorch)
tensor = torch.from_numpy(np_array).to('cuda')
output = model(tensor)
probs = torch.softmax(output)

# After (ONNX Runtime)
# tensor 변환 없이 순수 NumPy Array로 통신
ort_inputs = {ort_session.get_inputs()[0].name: np_array}
output = ort_session.run(None, ort_inputs)[0] 
# (np.exp() 등 NumPy만으로 softmax 직접 계산)
```

---

## 📦 3. 패키지 의존성 (Environment)

이 파이프라인을 가동하려면 가상 환경에 C++ 라이브러리가 바인딩된 다음 패키지들이 필요합니다:

```txt
# requirements.txt에 추가
onnx>=1.15.0
onnxruntime-gpu>=1.17.0
```

*※ CPU 전용 PC일 경우 `onnxruntime` 순정 패키지로 Fallback 작동하게 코드가 설계되었습니다.*
