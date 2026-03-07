# 🫧 Bubble 검출 파이프라인 — 학술적 근거 및 참고 자료

> **작성일**: 2026. 03. 07  
> **목적**: 시연회에서 "버블 검출을 왜 이렇게 구현했는가?"에 대한 학술적 근거 제공  
> **대상**: 프로그램 시연회 발표 (2026. 03. 10 월요일)

---

## 📊 파이프라인 단계별 학술적 근거

### 1️⃣ Morph-Open 배경 평탄화 → 양극성 차이 계산

**기법**: Morphological Opening으로 배경을 추정하고, 원본에서 빼서 전경(버블)만 부각

**학술적 근거**:
- **Rolling Ball Background Subtraction** — 생물의학 영상 분석에서 표준 전처리 기법
  - Sternberg, S.R. (1983). *"Biomedical Image Processing"*, IEEE Computer, 16(1), 22-34
  - 원리: 큰 커널의 Opening으로 느리게 변하는 배경 조도를 추정 → 원본에서 빼면 국소적인 밝기 변화(=Bubble)만 남음
- **Top-Hat Transform** — `original - opening(original)` = White Top-Hat
  - Gonzalez, R.C. & Woods, R.E. (2018). *"Digital Image Processing"*, 4th Edition, Pearson (§9.6.5)
  - 배경보다 밝은 작은 객체를 추출하는 표준 기법으로, 조도 불균일 보정에 널리 사용

**왜 이 방식인가?**
> 바이알 검사에서 조명 불균일(vignetting)이 심하면 단순 threshold가 작동하지 않음.
> Morph-Open으로 배경을 먼저 빼면 조명 조건과 무관하게 안정적인 검출이 가능.

---

### 2️⃣ CLAHE (Contrast Limited Adaptive Histogram Equalization)

**기법**: 이미지를 타일로 나누어 국소적으로 대비를 극대화하되, 노이즈 과증폭을 제한

**핵심 논문**:
- **Zuiderveld, K. (1994)**. *"Contrast Limited Adaptive Histogram Equalization"*, Graphics Gems IV, Academic Press, pp. 474-485
  - CLAHE의 원본 논문. 타일별 히스토그램 clip → 재분배 → bilinear interpolation 알고리즘 제시
- **Pizer, S.M. et al. (1987)**. *"Adaptive Histogram Equalization and Its Variations"*, Computer Vision, Graphics, and Image Processing, 39(3), 355-368
  - AHE의 원본 논문. CLAHE는 이 AHE에 clip limit을 추가한 개선 버전

**산업 및 의료 검사 적용 증거 (References)**:
- **의료 영상 (X-ray, CT)**: 병변 구분을 위해 국소 대비를 높이는 목적으로 널리 연구 및 활용.
  - *예시 논문*: "Segmentation of Spine X-ray Images using CLAHE", "Brain Tumor Segmentation using CLAHE" 등 수많은 IEEE/SPIE 게재 논문에서 전처리 성능 향상 입증.
- **제약 바이알 이물 검사**: 유리 용기의 곡면 반사와 조명 변화 속에서 미세 이물/기포 에지를 강조하기 위해 적용.
  - *예시 문헌*: 다수의 "Automated Visual Inspection of Pharmaceutical Vials" 관련 자동 시각 검사 연구논문에서 미세 결함 시각화 전처리 단계로 CLAHE를 사용.
- 반도체 웨이퍼 및 초음파 결함 탐지 등 대비가 낮은 산업용 비전 시스템에 널리 쓰이며, OpenCV 공식 라이브러리에 `cv2.createCLAHE()`로 내장되어 시스템의 표준 베이스라인으로 채택됨.

**왜 이 방식인가?**
> 버블은 투명/반투명하여 주변 배경과의 대비가 매우 약함.
> 글로벌 히스토그램 균등화는 이미 밝은 영역의 노이즈까지 증폭시키지만,
> CLAHE는 clip limit으로 노이즈 증폭을 억제하면서 약한 버블 에지를 부각시킴.

---

### 3️⃣ 노이즈 제거 (Median / Bilateral Filter)

**기법**: 에지(경계)는 보존하면서 고주파 노이즈만 제거

**핵심 논문**:
- **Median Filter**:
  - Tukey, J.W. (1977). *"Exploratory Data Analysis"*, Addison-Wesley
  - 임펄스 노이즈(salt-and-pepper)에 특히 효과적. 에지 보존력이 Gaussian보다 우수
- **Bilateral Filter**:
  - Tomasi, C. & Manduchi, R. (1998). *"Bilateral Filtering for Gray and Color Images"*, Proc. IEEE ICCV, pp. 839-846
  - 공간적 거리 + 밝기 차이를 동시에 고려 → 에지를 유지하면서 평활화

**왜 이 방식인가? (왜 CLAHE 후 노이즈가 증폭되는가?)**
> **노이즈 증폭의 원인**: CLAHE는 밝기가 균일해 보이는 영역일지라도 픽셀 간의 미세한 차이를 강제로 벌려서 대비를 극대화합니다. 이 과정에서 **원래는 눈에 띄지 않던 미세한 배경 텍스처나 카메라 센서의 고주파 노이즈까지 함께 크게 강조(증폭)**되고 맙니다. (Limit 값으로 극단적 증폭은 막지만 노이즈 발생 자체를 없앨 수는 없습니다.)
> 
> 따라서 특징 추출(DoG 필터)을 적용하기 전에, 이렇게 증폭된 노이즈를 먼저 다듬어야 노이즈를 이물로 오인하는 거짓 양성(False Positive)을 막을 수 있습니다.
> 이때 일반적인 가우시안 블러를 쓰면 기포의 선명한 경계선까지 뭉개지게 되므로, **경계(에지)는 칼같이 유지하면서 배경 노이즈만 문질러서 없애주는 Bilateral Filter**가 이상적입니다.

---

### 4️⃣ DoG 밴드패스 필터 (Difference of Gaussians)

**기법**: 서로 다른 σ (시그마: Sigma) 값의 Gaussian blur 2개를 빼서, 특정 크기 범위의 특징만 통과시킴

> 💡 **용어 설명**: **σ (시그마, Sigma)**는 통계학에서 제안하는 기호로 '표준편차'라 읽지만, 영상 처리의 가우시안 블러에서는 **"블러 반경(얼마나 넓게 퍼지게 흐리게 만들 것인지)"**를 의미하는 핵심 결정값입니다.

**핵심 논문**:
- **Marr, D. & Hildreth, E. (1980)**. *"Theory of Edge Detection"*, Proc. Royal Society of London, B207, 187-217
  - DoG가 LoG(Laplacian of Gaussian)의 효율적 근사임을 증명한 원본 논문
- **Lindeberg, T. (1994)**. *"Scale-Space Theory in Computer Vision"*, Springer
  - Scale-Space 이론의 교과서. 멀티스케일에서 blob 검출의 수학적 기반 제시
  - DoG는 scale-normalized LoG의 근사로, blob의 특징적 스케일(characteristic scale)을 검출
- **Lowe, D. (2004)**. *"Distinctive Image Features from Scale-Invariant Keypoints"*, IJCV, 60(2), 91-110
  - SIFT 알고리즘에서 DoG를 핵심 feature detector로 사용. 가장 많이 인용된 DoG 활용 논문

**왜 이 방식인가?**
> 버블은 **특정 크기 범위**를 가진 원형 blob입니다. DoG의 두 시그마(σ) 값을 조절하면:
> - `σ_small`보다 작은 노이즈 → 너무 작아서 필터링됨
> - `σ_large`보다 큰 배경 변화 → 너무 커서 필터링됨
> - **`σ_small` ~ `σ_large` 사이 크기를 가진 버블만** 통과 → 밴드패스(특정 주파수/크기 대역 통과) 효과를 냄.
>
> 이것이 Particle(불규칙 형상)과 Bubble(특정 크기의 원형)을 구분하는 핵심 메커니즘.

---

### 5️⃣ MAD 적응형 임계 이진화

**기법**: Median Absolute Deviation으로 배경 노이즈를 robust하게 추정 → 적응형 threshold 설정

**핵심 논문/교재**:
- **Huber, P.J. (1981)**. *"Robust Statistics"*, Wiley
  - Robust 통계의 교과서. MAD가 표준편차보다 이상치(outlier)에 강인한 이유를 수학적으로 증명
- **Rousseeuw, P.J. & Croux, C. (1993)**. *"Alternatives to the Median Absolute Deviation"*, Journal of the American Statistical Association, 88(424), 1273-1283
  - MAD의 breakdown point = 50% (데이터의 절반이 이상치여도 올바른 추정 가능)
  - 표준편차의 breakdown point = 0% (이상치 하나에도 크게 영향 받음)

**적용 원리**:
```
threshold = median(pixel_values) + k × MAD(pixel_values)
```
- `k`를 조절하여 감도 설정 (k=3이면 약 99.7% 신뢰구간에 해당)
- MAD = `median(|x_i - median(x)|)` — 이상치에 강인한 분산 추정

**왜 이 방식인가?**
> 일반 Otsu나 고정 threshold는 이미지마다 밝기 분포가 다르면 실패함.
> MAD 기반 threshold는 **DoG 출력의 배경 노이즈 수준을 자동으로 추정**하여,
> 배경 밝기가 바뀌어도 항상 "배경 + k×노이즈" 이상인 픽셀만 검출함.
> 이것이 다양한 조명/시료 조건에서 안정적으로 동작하는 비결.

---

### 6️⃣ 형태학 정리 (Close → Open)

**기법**: Close(구멍 메우기) → Open(작은 노이즈 제거)

**교과서 참조**:
- Gonzalez, R.C. & Woods, R.E. (2018). *"Digital Image Processing"*, 4th Edition, §9.6
  - Closing: 작은 구멍과 끊어진 윤곽을 메움
  - Opening: 작은 노이즈 제거하면서 큰 객체 형상 보존
- Serra, J. (1982). *"Image Analysis and Mathematical Morphology"*, Academic Press
  - 수학적 형태학의 원본 교과서

**왜 Close → Open 순서인가?**
> 1. **Close 먼저**: DoG 출력에서 버블 내부의 작은 구멍(하이라이트 반사 등)을 메워서 하나의 연결된 영역으로 만듦
> 2. **Open 다음**: Close로 인해 이웃 노이즈와 합쳐진 작은 돌기를 다시 제거

---

### 7️⃣ 형상 필터 (크기, 원형도, 솔리디티, 종횡비)

**기법**: Contour의 기하학적 특성으로 버블 여부를 최종 판정

**핵심 지표 및 근거**:

| 지표 | 수식 | 버블 특성 | 참조 |
|:---|:---|:---|:---|
| **Circularity** | `4π × Area / Perimeter²` | ≈ 1.0 (원에 가까움) | ISO 9276-6 |
| **Solidity** | `Area / ConvexHullArea` | ≈ 1.0 (볼록에 가까움) | Russ, J.C. *"The Image Processing Handbook"* |
| **Aspect Ratio** | `width / height` | ≈ 1.0 (정사각형에 가까움) | OpenCV contour analysis docs |
| **Min/Max Area** | `cv2.contourArea()` | 크기 범위 제한 | 도메인 지식 |

**학술 참조**:
- **ISO 9276-6:2008** — *"Representation of results of particle size analysis — Part 6: Descriptive and quantitative representation of particle shape and morphology"*
  - 원형도(circularity), 종횡비(aspect ratio) 등 형상 기술자의 국제 표준 정의
- Russ, J.C. (2016). *"The Image Processing Handbook"*, 7th Edition, CRC Press
  - 형상 기술자를 이용한 입자 분류의 표준 교재

**왜 이 방식인가?**
> 버블은 물리적으로 **표면장력에 의해 구형**에 가까운 형상을 가짐.
> 따라서 Circularity ≈ 1, Solidity ≈ 1, Aspect Ratio ≈ 1인 contour만 필터링하면
> 불규칙한 형상의 Particle/Noise와 자연스럽게 분리됨.

---

## 📚 종합: 파이프라인 전체의 설계 철학

```
입력 → [배경 제거] → [대비 강화] → [노이즈 제거] → [특징 추출] → [적응형 이진화] → [형태학 정리] → [형상 필터]
         Top-Hat       CLAHE      Bilateral       DoG           MAD          Close→Open    Circularity
```

이 파이프라인의 설계는 아래 원칙을 따릅니다:

| 원칙 | 적용 | 근거 |
|:---|:---|:---|
| **Scale-Space 이론** | DoG로 특정 크기의 blob만 선택 | Lindeberg (1994), Lowe (2004) |
| **Robust Statistics** | MAD로 조명 변화에 강인한 threshold | Huber (1981), Rousseeuw (1993) |
| **적응형 처리** | CLAHE + MAD → 이미지마다 자동 조절 | Zuiderveld (1994) |
| **형태학적 분석** | 버블의 물리적 특성(구형) 활용 | ISO 9276-6, Serra (1982) |
| **산업 검사 표준** | USP <790> 규약에 맞는 이물 검출 | USP General Chapter <790>, <1790> |

---

## 🏭 산업 표준 참조 (제약 바이알 검사)

| 규격 | 내용 |
|:---|:---|
| **USP <790>** | *"Visible Particulates in Injections"* — 주사제 가시 이물 검사 기준 |
| **USP <1790>** | *"Visual Inspection of Injections"* — 육안/자동 검사 방법론 가이드 |
| **USP <788>** | *"Particulate Matter in Injections"* — 비가시 미립자 검사 기준 |
| **21 CFR 211.65** | FDA cGMP — 장비는 적절한 검사 기능을 가져야 함 |

---

## 📖 발표에서 활용하는 방법

### 질문: "왜 단순 threshold 대신 이렇게 복잡한 파이프라인을 사용하나요?"

> **답변 포인트**:
> 1. 바이알 내 버블은 **반투명**하여 배경과의 대비가 매우 낮습니다 → 단순 threshold 실패
> 2. 조명 불균일이 항상 존재합니다 → **Top-Hat(배경 제거) + CLAHE(적응형 대비 강화)** 필요
> 3. 버블은 **특정 크기 범위의 원형 blob**입니다 → **DoG 밴드패스 필터**가 이 특성에 최적
> 4. 조건이 바뀌어도 작동해야 합니다 → **MAD 기반 적응형 threshold**로 자동 조절
> 5. 각 기법은 독자적인 논문이 있는 확립된 기술(CLAHE: 1994, DoG/Scale-Space: 1980-2004, MAD: 1981)입니다

### 질문: "딥러닝 대신 이 방식을 쓰는 이유는?"

> **답변 포인트**:
> 1. 버블 학습 데이터를 충분히 확보하기 어렵습니다 (반투명, 다양한 크기)
> 2. 본 파이프라인은 **Rule-based 1차 검출 + 딥러닝 2차 분류** 하이브리드 구조입니다
> 3. Rule-based 검출로 후보를 빠르게 추출 → AI가 최종 판정 → **속도와 정확도 모두 확보**
> 4. Rule-based 검출은 **해석 가능(explainable)**하여 검증이 용이합니다 (GMP 감사 대응)

---

## 📑 참고 문헌 목록 (한눈에)

1. Sternberg, S.R. (1983). *"Biomedical Image Processing"*, IEEE Computer, 16(1), 22-34
2. Gonzalez, R.C. & Woods, R.E. (2018). *"Digital Image Processing"*, 4th Ed., Pearson
3. Zuiderveld, K. (1994). *"Contrast Limited Adaptive Histogram Equalization"*, Graphics Gems IV
4. Pizer, S.M. et al. (1987). *"Adaptive Histogram Equalization and Its Variations"*, CVGIP, 39(3)
5. Tomasi, C. & Manduchi, R. (1998). *"Bilateral Filtering for Gray and Color Images"*, IEEE ICCV
6. Marr, D. & Hildreth, E. (1980). *"Theory of Edge Detection"*, Proc. Royal Society B207
7. Lindeberg, T. (1994). *"Scale-Space Theory in Computer Vision"*, Springer
8. Lowe, D. (2004). *"Distinctive Image Features from Scale-Invariant Keypoints"*, IJCV, 60(2)
9. Huber, P.J. (1981). *"Robust Statistics"*, Wiley
10. Rousseeuw, P.J. & Croux, C. (1993). *"Alternatives to the MAD"*, JASA, 88(424)
11. Serra, J. (1982). *"Image Analysis and Mathematical Morphology"*, Academic Press
12. Russ, J.C. (2016). *"The Image Processing Handbook"*, 7th Ed., CRC Press
13. ISO 9276-6:2008 — Particle shape and morphology representation
14. USP General Chapters <788>, <790>, <1790> — Particulate matter in injections
