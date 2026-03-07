# 🧠 컴퓨터 비전 & 딥러닝 완벽 마스터 가이드 (ForeignBodyInsp 프로젝트 기반)

이 문서는 본 **이물질 검사 시스템(ForeignBodyInsp) 프로젝트**에 적용된 모든 영상 처리(Classic Computer Vision) 및 딥러닝(Deep Learning) 기술을 **이론적 기초부터 프로젝트 실제 구현 코드까지 완벽하게 이해할 수 있도록** 작성된 학습 가이드입니다. 수학적 배경지식과 함께 읽으시면 더욱 깊이 있는 이해가 가능합니다.

---

## 📖 목차
1. **[Part 1: 기반 지식 - 고전적 영상 처리 (Image Processing & Classic CV)](#part-1-기반-지식---고전적-영상-처리-image-processing--classic-cv)**
2. **[Part 2: 딥러닝 기반 이미지 분류 (Deep Learning Classification)](#part-2-딥러닝-기반-이미지-분류-deep-learning-classification)**
3. **[Part 3: 영상 내 객체 탐지 (Object Detection & YOLO)](#part-3-영상-내-객체-탐지-object-detection--yolo)**
4. **[Part 4: 프로젝트 실제 파이프라인 (Project Implementation Roadmap)](#part-4-프로젝트-실제-파이프라인-project-implementation-roadmap)**

---

# 🔎 Part 1: 기반 지식 - 고전적 영상 처리 (Image Processing & Classic CV)

프로젝트의 `cv2`(OpenCV)를 활용한 기본적인 전처리 및 Rule-based 판단 로직의 기반이 되는 수학적/이론적 지식입니다.

### 1.1 그레이스케일(Grayscale) 변환과 이진화(Binarization)

이미지는 일반적으로 RED, GREEN, BLUE의 3채널(각각 0~255 값)로 이루어집니다. 이를 1채널 흑백(Grayscale)으로 변환하는 수학적 표준 식은 다음과 같습니다:
$$Y = 0.299R + 0.587G + 0.114B$$
사람의 눈이 녹색에 가장 민감하기 때문에 G의 가중치가 가장 높습니다.

**이진화(Thresholding)**: 그레이스케일 이미지를 계산하기 쉽게 0(검정)과 255(흰색) 두 가지 값으로만 나누는 기법입니다.
$$ dst(x,y) = \begin{cases} maxval & \text{if } src(x,y) > thresh \\ 0 & \text{otherwise} \end{cases} $$

- **전역 이진화(Global Thresholding)**: 이미지 전체에 동일한 임계값 $thresh$ 적용 (조명 변화에 취약)
- **적응형 이진화(Adaptive Thresholding)**: 이미지의 각 블록별로 따로 임계값을 계산 (그림자가 져도 객체 분리 가능)

---

### 1.2 적응형 임계값 알고리즘 (MAD - Median Absolute Deviation)
본 프로젝트의 기포(Bubble) 검출에서는 단순 평균/가우시안 적응형 이진화 대신 통계학의 **MAD(중앙값 절대 편차)** 를 활용한 강력한 동적 임계값을 직접 구현하여 사용합니다.

**MAD의 수학적 정의:**
$$ MAD = median(|X_i - median(X)|) $$
표준편차($\sigma$)와 비슷하게 데이터의 산포도를 나타내지만, 표준편차가 노이즈(극단적인 튀는 값)에 평균이 왜곡되는 반면 **MAD는 중앙값을 사용하므로 압도적으로 노이즈에 강건(Robust)합니다.**

**프로젝트 적용 공식:**
$$ Threshold(x, y) = median(window) + k \times MAD $$
이 식을 통해 국소 영역 내에서 배경 텍스처와 형태학적 노이즈를 완벽히 무시하고, 미세하게 튀는 결함만 추출해냅니다.

---

### 1.3 형태학적 변환 (Morphological Transformations)
바이너리 이미지 안의 모양이나 구조를 변경하는 수학적 기법입니다. (집합론에 기반) 커널(Kernel)이라는 구조 요소(Structuring Element)를 이미지 위로 슬라이딩시키며 계산합니다.

- **Dilation(팽창, $\oplus$)**: 객체 영역을 확장시킵니다. 내부 구멍을 메울 때 씁니다.
- **Erosion(침식, $\ominus$)**: 객체 영역을 축소시킵니다. 작은 잔노이즈를 지울 때 씁니다.

프로젝트에서는 이 두 가지를 조합한 **Opening(열기)** 와 **Closing(닫기)** 를 주로 사용합니다.
- **Opening** (침식 후 팽창, $A \circ B = (A \ominus B) \oplus B$): 가느다란 노이즈 제거, 객체와 객체 끊기 기능. **프로젝트에서 배경 조명을 평탄화할 때 거대한 `kernel=151`로 Opening을 수행**하여 배경만 얻어냅니다.
- **Closing** (팽창 후 침식, $A \bullet B = (A \oplus B) \ominus B$): 객체 안의 작은 구멍(Hole)을 메우거나 윤곽을 매끄럽게 합니다.

---

### 1.4 제한적 대비 평활화 (CLAHE)
Histogram Equalization(히스토그램 평활화)은 이미지 픽셀 값의 분포를 균일하게 펴서 대비(Contrast)를 높이는 기법입니다. 수식으로는 누적 적산 확률(CDF)을 선형으로 매핑합니다. 
하지만 전체를 대상으로 하면 밝은 곳은 너무 하얗게, 어두운 곳은 너무 까맣게 변하는 단점이 있습니다.

✨ **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
1. 이미지를 작은 그리드 블록(예: 8x8)으로 나눔 (Adaptive)
2. 각 블록에서 히스토그램 평활화 수행
3. 대비가 과하게 폭발적으로 증가하는 것을 막기 위해, 특정 높이(Clip Limit) 이상의 히스토그램을 잘라내어 다른 곳에 골고루 분배함 (Contrast Limited)

> **프로젝트 활용:** 유리 바이알의 곡면 반사율 차이에 구애받지 않고, 미세한 기포의 테두리 윤곽선만 날카롭게 살려내기 위해 사용됩니다.

---

### 1.5 주파수 필터링 (DoG - Difference of Gaussians)
영상에서 주파수(Frequency)란 픽셀 값의 변화 속도를 의미합니다.
- **저주파 지대**: 픽셀 밝기가 완만히 변하는 배경
- **고주파 지대**: 픽셀 밝기가 급변하는 객체의 테두리(Edge)나 날카로운 노이즈

**DoG의 원리:**
서로 다른 표준편차($\sigma_1$, $\sigma_2$)를 가진 두 개의 가우시안 블러를 구한 뒤 빼는(Difference) 연산입니다. 넓은 가우시안에서 좁은 가우시안을 빼면 특정 크기의 물체만 돋보이는 **대역 통과 필터(Band-pass Filter)** 역할을 하게 됩니다.

수학적 함수형:
$$ DoG(x,y) = \frac{1}{2\pi\sigma_1^2} e^{-\frac{x^2+y^2}{2\sigma_1^2}} - \frac{1}{2\pi\sigma_2^2} e^{-\frac{x^2+y^2}{2\sigma_2^2}} $$
> **프로젝트 활용:** 거대한 얼룩(버블 모양이 아님, 너무 낮은 주파수)이나 픽셀 1개짜리 화이트 노이즈(너무 높은 주파수)를 완벽히 억제하고 **목표 버블 크기만 극대화**하기 위해 사용합니다.

---

### 1.6 기하학적 특징 추출 (Contours Feature)
이진화된 결과물에서 외곽선(Contour)을 따낸 뒤, 각 객체들이 '무엇인지' 수학적으로 걸러내는 Rule-based 분류의 핵심입니다.

- **Area (면적)**: 픽셀 수. 너무 작으면 Noise로 판단.
- **Bounding Box (바운딩 박스)**: 객체를 감싸는 직사각형. Aspect Ratio (가로세로 비율, $w/h$) 계산에 사용 (Fiber나 긴 먼지를 걸러낼 때 유리).
- **Circularity (원형도)**: 형상이 얼마나 완벽한 원형에 가까운지 0.0 ~ 1.0으로 반환하는 지표입니다.
  $$ Circularity = \frac{4 \pi \times Area}{Perimeter^2} $$
  > **프로젝트 활용:** 기포(Bubble)는 표면장력으로 인해 거의 완벽한 원형($> 0.6$)을 띕니다. 이 원리를 통해 단순한 먼지와 기포를 구별($Circularity \ge 0.6$ 이면 Bubble)합니다.

<br>

---

# 🤖 Part 2: 딥러닝 기반 이미지 분류 (Deep Learning Classification)

현대 컴퓨터 비전에서는 수동 추출 특징(Hand-crafted Features: 원형도, 면적 등)의 한계를 극복하기 위해 **신경망이 데이터로부터 최적의 특징 공간(Feature space)을 스스로 학습하는 딥러닝**을 사용합니다. 프로젝트의 `classification.py` 중 DL 모듈이 여기에 해당합니다.

### 2.1 합성곱 신경망 (CNN - Convolutional Neural Networks)
이미지의 공간적 특성(픽셀 간의 연관성)을 유지하면서 특징을 뽑아내는 구조입니다.
1. **Convolution Layer (합성곱 층)**: 필터(가중치 행렬)를 이미지에 슬라이딩하며 곱하여 더합니다. 초기 층은 선/곡선을 찾고, 깊은 층은 복잡한 패턴(버블 표면 특성)을 찾습니다.
2. **Pooling Layer (풀링 층)**: 이미지 크기를 줄이면서(Max-pooling) 가장 뚜렷한 특징만 남겨 연산량을 줄입니다.
3. **Fully Connected Layer (완전 연결 층)**: 마지막 1D 벡터로 쫙 펴서 N개의 클래스 각각에 대한 확률(Softmax)을 계산합니다.

### 2.2 EfficientNet-B0 구조 및 특징
본 프로젝트는 **EfficientNet-B0** 전이 학습(Transfer Learning)을 사용합니다. 왜 수많은 모델 중 EfficientNet인가?

전통적인 CNN은 정확도를 높이기 위해 깊이(Depth - 층 수), 너비(Width - 채널 수), 해상도(Resolution) 중 하나만을 무식하게 키웠습니다.
EfficientNet은 **Compound Scaling Method**를 제안하여, 깊이/너비/해상도의 스케일업 비율 구조를 수학적 최적화 공식을 통해 찾아냈습니다.
$$ \text{Depth: } \alpha^\phi, \quad \text{Width: } \beta^\phi, \quad \text{Resolution: } \gamma^\phi $$
(단, $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$)

**결과적으로 파라미터(용량) 크기는 VGG16 같은 옛날 모델의 1/15 수준이면서, 분류 성능과 속도는 훨씬 빠릅니다.** 1초에 수십 장의 결함을 분류해야 하는 본 프로젝트의 공장 실시간 검사에 가장 적합한 모델입니다.

### 2.3 데이터 불균형(Class Imbalance) 문제와 Focal Loss
제조업 AI에서 가장 악명 높은 문제입니다🔥. 정상 제품과 단순 먼지(Noise_Dust) 데이터는 99%인 반면, 중요한 Particle이나 Bubble은 1%밖에 나오지 않습니다. 이를 흔한 Cross-Entropy(CE) Loss로 학습시키면, 신경망이 "무조건 먼지라고 찍으면 정답이 99%네?" 하고 게으른 학습을 해버립니다.

이를 해결하기 위해 프로젝트의 학습 스크립트(`train_classifier`)는 일반 CE 대신 **Focal Loss**라는 특별한 오차 함수를 탑재했습니다.

**Focal Loss 공식:**
$$ FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) $$
- 일반 CE: $-\log(p_t)$
- 조절인자(Modulating Factor): $(1-p_t)^\gamma$
- 현상: 예측 확률($p_t$)이 0.9인 이미 잘 맞추는 쉬운 데이터(먼지)는 $\gamma=2.0$ 적용 시 오차값이 $\approx 0$으로 소멸됩니다. 하지만 예측 확률이 0.1인 어려운 데이터(희귀한 기포)의 오차값은 거의 그대로 유지됩니다.
- **결과: 신경망이 쉽고 많은 데이터 학습을 멈추고, 드물고 어려운 결함을 뚫어지게 분석하도록 강제합니다.**

<br>

---

# 🎯 Part 3: 영상 내 객체 탐지 (Object Detection & YOLO)

Classification(분류)이 이미 잘려나간 결함 조각 1장을 주고 "이게 무슨 결함이냐?" 묻는 것이라면, Object Detection(탐지)은 "전체 화면(2048x2048) 안에서 어디에, 어떤 결함 박스들이 쳐져 있냐?"를 동시에 푸는 문제입니다. 프로젝트의 `yolo_detector.py` 모듈입니다.

### 3.1 1-Stage vs 2-Stage Detector
- **2-Stage (예: Faster R-CNN)**: 
  1. 객체가 파편으로 존재할 만한 수천 개의 후보 박스를 탐색 (Region Proposal)
  2. 각 박스를 잘라서 CNN에 넣고 무슨 객체인지 분류
  > 정확하지만 너무 느려서 산업용 실시간 검사에 부적합.
- **1-Stage (예: YOLO)**:
  1. (You Only Look Once) 이미지를 NxN 격자(Grid)로 나눕니다.
  2. 한 방의 CNN 연산으로 **각 격자마다 박스의 (x,y,w,h) 좌표와 (클래스별 확률)을 동시에 예측**하는 Matrix를 우수수 뱉어냅니다.
  > 속도 지상주의. 현대에는 알고리즘의 발달로 정확도도 2-Stage 모델을 능가합니다.

### 3.2 YOLOv8의 구조와 Anchor-Free 개념
본 프로젝트는 YOLO의 최신 안정성-성능 밸런스 챔피언인 Ultralytics YOLOv8을 지원합니다.

* **Anchor-Free의 도입**: 예전 YOLOv1~v5 모델들은 미리 정해둔 다양한 크기의 박스(Anchor Box) 형태를 기반으로 변형 비율을 계산했습니다(Anchor-based). 하지만 버블과 먼지는 형태 변형이 너무 다양해 수동 앵커 설정이 골치 아팠습니다. 
**YOLOv8은 Anchor-Free(Center-based)를 채택**하여 앵커 없이 객체의 픽셀 정중앙 시점에서의 객체 사방 경계거리(L,R,T,B)를 바로 회귀(Regression) 추론합니다.

* **CSP & C2f Module의 백본**: GPU 메모리 병목을 줄이고 그래디언트의 풍부한 흐름을 위해 레이어를 교차 분할 결합시키는 C2f 아키텍처를 가졌습니다. 추론 속도 최적화가 극에 달해 있습니다.

<br>

---

# 🚀 Part 4: 프로젝트 실제 파이프라인 (Project Implementation Roadmap)

자, 이제 이 지식들이 융합되어 프로젝트 C++ / Python 코드 내부에 어떻게 흘러가는지 보겠습니다. 
우리의 목표는 **"약한 대비 속의 바이알 병에서 미세한 기포/이물질을 초고속으로 검출+분류하는 것"** 입니다.

### 4.1 하이브리드 투-트랙(Two-Track) 검출 파이프라인 (`detection.py`)
이물질과 기포는 시각적 스펙트럼이 정반대이므로, 하나의 알고리즘으로 잡으려 하면 100% 한쪽 성능이 박살납니다. 그래서 두 개의 독립 파이프라인(Thread)으로 검출 후 `_merge_contours` 로 O(n) 속도로 좌표를 병합합니다.

#### 👉 Track A: 흑색 이물(Particle) 파이프라인 `detect_static()`
물질 고유의 색이 진하게 나타나는 직관적인 파이프라인.
1. `cv2.cvtColor(BGR2GRAY)`
2. `cv2.morphologyEx(MORPH_OPEN)` & `MORPH_CLOSE` (배경 반사광 잡음 제거)
3. `cv2.threshold()` 또는 `cv2.adaptiveThreshold` (검정색 성분 분리)
4. `cv2.findContours()`로 영역 도출.

#### 👉 Track B: 투명 기포(Bubble) 파이프라인 `detect_bubbles()`
테두리만 얇게 회색/흰색으로 빛의 굴절을 남기는 극히 흐린 파이프라인. (가장 복잡함)
1. **Background Subtraction**: `kernel=151`의 `MORPH_OPEN`으로 원본과 뺀 뒤 `cv2.absdiff` 절대차로 투명한 굴절광 양자 확보 (수학 모델: 빛의 Scattering 현상 캐치)
2. **CLAHE**: 굴절 희미한 빛의 Local Contrast 강제 극대화
3. **DoG Bandpass**: `cv2.GaussianBlur` 2채널 차집합 (반경이 너무 크거나 너무 작은 녀석 배제)
4. **MAD Adaptive Threshold**: $Thresh=Median+3.5 \times MAD$ 수식 계산 후 이진화 (진짜 윤곽선만 쟁취!)
5. `cv2.findContours()` 후, 기하학적 수식 $Circularity > 0.35$ 로 원형 아닌 것(즉 먼지가 빛난 것)들 소거.

### 4.2 인공지능 분류 및 통과 파이프라인 (`classification.py`)
두 개의 트랙에서 합쳐져 올라온 결함 의심 박스 좌표들(Contour List)이 넘어옵니다.

1. **RuleBased 모드 (초가벼움, DL 없이 동작)**
   - `cv2.contourArea`, `arcLength` 등으로 기하학 수학 수식 직접 평가.
   - 분류 속도는 무한대에 가까운 $\sim 0.1ms$ 소요.

2. **Deep Learning 모드 (초정밀 분류)**
   - 원본 컬러 이미지에서 Contour 좌표를 기반으로 사각형 크롭(Crop).
   - `cv2.resize(224x224)` 후 정규화(Normalization).
   - CPU 쓰레드가 2048개씩 잘라서 GPU에 넘기면 `ONNX Runtime(EfficientNet-B0)`이 CUDA 하드웨어 가속을 통해 병렬 추론.
   - 출력값 `Softmax` 확률벡터. 
   - 각 결함에 `Bubble`, `Particle`, `Noise_Dust` 딱지 부여.

### 4.3 통합 & YOLO 병행 전략 패러다임
위의 [1.전처리 도출 $\rightarrow$ 2.크롭 $\rightarrow$ 3.DL 분류]의 과정을 아예 대체하기 위해 `YOLODetector` 모듈이 존재합니다.
이 프로젝트의 진정한 강력함은 **기존의 고전 영상처리 파이프라인으로 잡아낸(자동 발견) 결함 데이터들을 어노테이션 삼아 `yolo_dataset.py`를 통해 자동으로 YOLO 학습 데이터셋으로 무한 증식 공장화** 시킬 수 있다는 점입니다.

1. 초반 장비 세팅: 조명 맞추고 Track A, Track B 파라미터 장인이 맞춰서 Rule base 구동
2. 일주일 가동 $\rightarrow$ 자동 스냅샷 수만 장 쌓임
3. 이 데이터를 그대로 Yolo 혹은 EfficientNet 폴더에 투척 후 `train()` 실행 
4. 완성된 강력한 `.onnx` / `.pt` 모델로 교체하고, 기존 복잡한 Track A/B를 모두 끄고 딥러닝 One-stage로 대동단결!

---

> 끝! 이 문서를 다 읽게 되면 컴퓨터 비전 신입 입문 수준을 훌쩍 뛰어넘어 현업 불량 검출 알고리즘의 메인 아키텍처 설계 사상을 통달하시게 됩니다. 추가로 알고 싶은 수학적/학술적 한계나 의문점이 있다면 언제든 말씀해 주세요.
