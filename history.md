# 📋 Vial Foreign Body Inspection System — 개발 히스토리

> ※ 이 문서는 "무슨 기능을 왜 넣었는지" 나중에 다시 봐도 이해할 수 있도록 요약과 배경 설명을 붙여 두었습니다.

---

## 📅 2026-02-05 (목) — 초기 개발

**개발자**: AI Assistant

---

### [UI 개선 - Settings] Threshold / Min Area 슬라이더에 숫자가 보이게

**상황**: Threshold(이진화 임계값), Min Area(최소 면적) 슬라이더를 움직여도 현재 값이 안 보여서 불편함.

**한 일**:
- 슬라이더 위에 현재 값을 보여주는 레이블(QLabel) 추가
- 슬라이더를 움직일 때마다(valueChanged) 그 숫자를 갱신
- 파란색 굵은 글씨로 보이게 스타일 지정

---

### [환경 설정 - 의존성] PyQt6 없다고 할 때

**상황**: 프로그램 실행 시 PyQt6를 찾을 수 없다는 에러 발생.

**한 일**:
- requirements.txt에 PyQt6가 들어 있는지 확인
- 없으면 터미널에서 `pip install PyQt6` 실행하면 됨을 안내

---

### [UI 개선 - Result 및 Defect View] 검출 목록 클릭 시 강조 + 선택한 것만 크게 보기

**상황**: 검사 결과(Result) 리스트에서 항목을 눌러도 어느 Defect인지 화면에서 구분이 잘 안 됨.

**한 일**:
- Result 리스트: 한 번에 하나만 선택되도록(SingleSelection), 선택된 행은 파란 배경/흰 글씨로 강조
- Defect View: 오른쪽에 260×260 영역에 선택한 Defect 부분만 잘라서 확대 표시
- 리스트 선택이 바뀔 때마다 Defect View가 자동으로 갱신

---

### [UI 개선 - 이미지 로딩] 드래그로 이미지 넣기 + 창 크기 고정 + Ctrl+휠 줌

**상황**: 이미지를 끌어다 넣고 싶고, 큰 이미지를 불러와도 창이 넓어지지 않았으면 함.

**한 일**:
- 드래그 앤 드롭으로 .png, .jpg, .jpeg, .bmp 파일 로드
- 이미지 뷰를 QScrollArea로 감싸서, 큰 이미지는 스크롤만 되고 창 크기는 그대로
- Ctrl + 마우스휠: 확대/축소 (0.2배 ~ 6배)

---

### [버그 수정 - Drag & Drop] 드래그할 때 금지 표시 나오는 문제

**상황**: 이미지를 창 위로 끌고 가면 커서에 금지 아이콘이 떠서 드롭이 안 되는 것처럼 보임.

**한 일**:
- 메인 창·스크롤·이미지 라벨·컨트롤 패널·스플리터 등에 `setAcceptDrops(True)` 설정
- 드롭 시 복사 동작으로 처리하라고 `setDropAction(CopyAction)` 명시
- 여러 위젯에 이벤트 필터를 걸어서 메인 창의 드래그 처리 로직이 이벤트를 받도록 연결

---

### [성능 최적화 - 확대/축소] 줌 할 때 버벅이고 흐릿해지는 문제

**상황**: 확대/축소를 하면 화면이 버벅이거나 이미지가 흐릿해짐.

**한 일**:
- 원본 이미지 전체를 메모리에 두고, 화면에 그릴 때만 "보이는 구간"을 잘라서 뷰 크기에 맞게 리사이즈
- 축소: INTER_AREA / 확대: INTER_CUBIC → 이후 INTER_NEAREST로 변경 (픽셀 블록)

---

### [UI 개선 - 패닝] 확대한 상태에서 화면 쭉쭉 밀기

**상황**: 많이 확대해 두면 다른 부분을 보려면 마우스로 화면을 이동하고 싶음.

**한 일**:
- 뷰의 "중심 좌표"(view_cx, view_cy)를 두고, 마우스로 드래그한 만큼 이동
- 이미지 밖으로 나가지 않도록 경계에서 멈추게 처리

---

### [버그 수정 - Import 에러] main_window.py만 F5로 실행하면 모듈을 못 찾을 때

**한 일**:
- main_window.py 맨 위에서 프로젝트 루트 폴더 경로를 계산해서 `sys.path`에 등록
- 또는, 항상 프로젝트 루트에서 `python src/main.py`로 실행 권장

---

### [버그 수정 - 프로그램 다운] Start Inspection 누르면 프로그램이 죽을 때

**한 일**:
- Defect View에 넣을 이미지(ROI)를 만들 때 경계 벗어나는 인덱스·0 나누기 방지
- `np.ascontiguousarray()`로 연속 메모리 보장
- `render_main_view()`에도 동일 안전장치 추가

---

### [버그 수정 - Result 선택 유지] 다른 Defect 클릭해도 선택이 도로 돌아가는 문제

**한 일**:
- 매 프레임마다 뷰 중심·줌을 리셋하지 않고, 새 이미지 로드 시에만 초기화
- Defect View는 원본 프레임에서 해당 Defect 영역만 잘라서 표시

---

### [UI 개선 - Result 리스트 안정화] 스크롤바 움직이면 선택이 풀리는 문제

**한 일**:
- 리스트를 매번 비우고 다시 채우지 않고, `setText()`로 갱신
- 갱신 중 `blockSignals(True)`로 시그널 잠금

---

### [UI 개선 - Defect 선택 하이라이트] Result/화면 클릭으로 Defect 짚어 보기

**한 일**:
- display_base_frame에 빨간색으로 그린 뒤, 선택된 contour만 노란색으로 하이라이트
- 이미지 뷰 클릭 시 `cv2.pointPolygonTest`로 해당 contour 찾아서 선택
- 클릭/드래그 구분: 5픽셀 미만 이동 = 클릭

---

### [성능 개선 - 멀티스레드 검사 + MainView 표시 옵션]

**한 일**:
- 검사와 그리기를 백그라운드 스레드(DetectionWorker, QThread)에서 실행
- "MainView에 표시" 체크박스: 켜면 Defect 사각형·라벨을 그림, 끄면 스킵

---

### [UI 개선 - 확대 배율 확장 및 1:1 줌]

**한 일**:
- 1:1 = 모니터 1픽셀 = 이미지 1픽셀. 확대 상한을 1:1의 32배까지
- 마우스 휠 버튼 더블클릭으로 1:1 토글

---

### [UI 개선 - 확대 시 픽셀 선명도] ImageJ 스타일

**한 일**:
- 확대 시 보간법을 `INTER_NEAREST`로 변경 → 픽셀 단위로 네모나게 표시

---

### [UI 개선 - Position/GrayValue]

**한 일**:
- `mapFromGlobal(globalPosition())` 사용으로 창 크기와 무관하게 정상 동작

---

### [기능 추가 - AVI/동영상 로딩 및 실시간 검사]

**한 일**:
- FileCamera에 동영상 확장자(.avi, .mp4, .mov 등) 지원
- 마지막 프레임 도달 시 0번 프레임으로 돌려서 무한 루프 재생
- 동영상: 33ms(≈30fps) 간격 / 단일 이미지: 100ms 간격

---

### [기능 추가 - 검사 ROI 설정]

**한 일**:
- "검사 ROI 설정" 버튼으로 마우스 사각형 드래그 → ROI 영역 저장
- 검사 시 ROI만 크롭 후 검사, 좌표 오프셋 보정
- "검사 ROI표시" 체크박스로 초록 사각형 표시

---

### ⭐ [버그 수정 - ROI 검사 시 MainView 멈춤] — 심각한 버그

**증상**: ROI 설정 후 Start Inspection 시 메인 화면만 멈춤.

**원인 1 — 스레드 데드락**: `_worker_busy`와 `isRunning()` 타이밍 불일치
- **해결**: `isRunning()` 체크 제거. `_on_worker_finished()`에서만 플래그 관리

**원인 2 — Contour 자료형 (핵심)**: ROI 오프셋을 더하면 float64가 되고, OpenCV가 int32를 기대하여 에러 발생
- **해결**: `.astype(np.int32)` 강제 변환

**원인 3 — 에러 시 화면 미갱신**: except 블록에서 `render_main_view()` 미호출
- **해결**: except에서도 `current_display_frame` 세팅 + `render_main_view()` 호출

---

### [성능 최적화 - Classification 검사 속도]

**한 일**:
- `classify_batch()` 추가: 모든 contour의 ROI를 한 번에 tensor로 묶어서 모델에 배치 추론
- GPU 사용 시 특히 효과적

---

### [기능 개선 - 이미지 로드 시 1회 검사]

**한 일**:
- 단일 이미지(bmp, jpg 등) 시 검사 1회 후 자동 정지
- 동영상은 기존대로 연속 검사

---

### [기능 개선 - pth 경로 및 모델 드래그앤드롭]

**한 일**:
- 프로젝트 루트 기준 경로 자동 설정
- `.pth` 파일 드래그앤드롭으로 모델 로드

---

### [기능 추가 - Basler 카메라 연동 강화]

**한 일**:
- `basler_camera.py`: acA1920-25gm 우선 선택, open/close, is_connected, get/set_parameters_dict
- Connect/Disconnect 버튼, RealTime View 체크(연속 grab / 1회 grab)
- Basler 설정 다이얼로그: 노출·게인 편집, JSON 저장/로드

---

### [개선 - Basler 연결 실패 시 원인 표시]

**한 일**:
- `_last_error`에 원인 저장, QMessageBox로 상세 안내 표시
- pypylon 미설치, 디바이스 미발견, GigE IP 설정 등 원인별 메시지

---

### [환경 - Basler Pylon 설치]

- Pylon 26.01 설치 후 소스 코드 변경 불필요. pypylon이 런타임 자동 탐색

---

### [기능 추가 - exe 빌드 (PyInstaller)]

**한 일**:
- `build_exe.py` 추가. `dist\ForeignBodyInsp\ForeignBodyInsp.exe` 생성
- exe 폴더에 모델 파일 두면 자동 로드
- torch DLL 검색 경로 문제 해결 (os.add_dll_directory)
- CPU 전용 환경(.venv_cpu)에서 빌드 권장

---

### [기능 추가 - run.bat]

- 더블클릭으로 실행. `.venv`가 있으면 그 Python, 없으면 시스템 python 사용

---

### [개선 - Ctrl+Shift+R 재시작 방식 변경]

**한 일**:
- `os.execv()` → `subprocess.Popen()`으로 변경. Windows에서 DETACHED_PROCESS로 독립 실행

---

### [개선 - 모델 로드 실패 시 상세 에러 표시]

**한 일**:
- `load_model()` 반환값을 `(bool, str|None)` 튜플로 변경
- QMessageBox에 실제 에러 내용 표시

---

## 📅 2026-02-05 (목) — 설정 저장 / 파라미터 / UI 개선

---

### [설정 저장 - path_config.json] ClassificationData 폴더 경로 영구 저장

**한 일**:
- exe/프로젝트 루트에 `path_config.json` 생성
- 프로그램 시작 시 path_config.json → QSettings → 기본값 순서로 읽기

---

### [기능 추가 - RuleBase 파라미터 다이얼로그]

**한 일**:
- `rule_params_dialog.py` 생성. 설정 → RuleBase 파라미터... 메뉴로 접근
- 각 파라미터를 QSpinBox/QDoubleSpinBox로 편집. JSON 저장/로드

---

### [개선 - Bubble 라벨 통합] Small/Big/BackGround Bubble → Bubble 하나로

**한 일**:
- `BUBBLE_LABEL = "Bubble"` 단일 상수로 통일
- LEGACY_BUBBLE_LABELS/FOLDERS는 하위 호환용으로만 유지

---

### [버그 수정 - 동영상 슬라이더 역방향 이동]

**한 일**:
- `_video_slider_just_sought` 플래그로 사용자 시크 직후 1프레임 동기화 스킵

---

### [기능 추가 - 어노테이션 표시 토글]

**한 일**:
- AnnotationCanvas에 `_show_annotations` 속성. "어노테이션 표기" 체크박스로 토글

---

### [버그 수정 - 어노테이션 저장 시 BMP 생성]

**한 일**:
- JSON만 저장하도록 재작성. BMP 파일 생성 제거

---

### [개선 - 분류 이미지 크기 확대] 64→128

**한 일**:
- `CLASSIFICATION_INPUT_SIZE = 128` 상수 도입. 학습/추론/ROI 추출 모두 통일

---

### [기능 변경 - Main 창 어노테이션 Brush 방식]

**한 일**:
- Rect 드래그 → Brush 그리기로 변경
- mask에서 contour 추출 → 128×128 크롭 → 라벨 선택 → 저장

---

### [기능 추가 - Pause/Resume/Stop 버튼]

**한 일**:
- Pause/Resume(Space), Stop/Play 버튼+아이콘 추가
- 탭 전환 시 타이머 정지/재개

---

### [기능 추가 - Results 라벨 필터]

**한 일**:
- Results 그룹에 Particle/Noise/Bubble 체크박스 추가
- 체크한 라벨만 리스트에 표시

---

## 📅 2026-02-05 (목) — YOLO 통합 및 Bubble 검출 강화

---

### [아키텍처 전환 - YOLO 통합 검출+분류]

**상황**: 기존 2단계(threshold→분류)에서 Bubble은 threshold에서 잡히지 않는 구조적 한계.

**신규 파일**:

| 파일 | 역할 |
|:---|:---|
| `src/core/yolo_detector.py` | ultralytics YOLOv8 래퍼 (load_model, detect, train) |
| `src/core/yolo_dataset.py` | YOLO 데이터셋 관리 (폴더 생성, data.yaml, 좌표 변환) |
| `src/ui/yolo_annotation_tab.py` | YOLO 어노테이션+학습 통합 탭 UI |

**기존 파일 수정**:
- `main_window.py`: 검사 모드 콤보박스, YOLO 모델 로드 버튼, `.pt` 드래그앤드롭
- `requirements.txt`: ultralytics 추가

---

### [검출 강화 - 고급 Bubble 검출 알고리즘]

**상황**: Bubble은 반투명하고 약한 에지만 있어 threshold로는 놓치는 경우가 많음.

**`detection.py` 완전 재작성**:
1. CLAHE 대비 강화
2. Hough Circle Transform
3. LoG (Laplacian of Gaussian) 멀티스케일 Blob 검출
4. Canny Edge + 원형도 필터
5. Bright-core 필터 (중심-테두리 밝기 차이)
6. NMS 중복 제거

**`rule_params_dialog.py` 재작성**:
- 탭 1 "분류 조건" / 탭 2 "Bubble 검출" (19개 파라미터)
- JSON 저장/로드 포맷 변경

---

## 📊 주요 기능 요약

| 카테고리 | 기능 |
|:---|:---|
| **입력** | Basler 카메라 (Connect/Disconnect, RealTime, 설정 다이얼로그), 파일/동영상 로드 (드래그앤드롭) |
| **검사 모드** | Threshold+분류 (기본), YOLO 통합 검출, Bubble 검출 (다중 기법 파이프라인) |
| **학습** | Classification 탭 (Brush → 128×128 ROI → CNN), YOLO 탭 (Bbox → YOLO 학습) |
| **결과** | Result 리스트 (라벨 필터), MainView (선택 강조), Defect View (확대) |
| **조작** | Ctrl+휠 줌, 드래그 패닝, 1:1 줌 토글, Space 일시정지 |
| **설정 파일** | path_config.json, rule_params.json, basler_settings.json |

---

## 📅 2026-03-05 (수) — Basler 카메라 녹화 기능

---

### 🕐 오후 — [기능 추가 - Basler Grab 이미지 저장]

**한 일**:
- Basler 카메라 설정 옆에 **"Grab" 버튼** 추가
- 클릭 시 현재 프레임을 "Basler Grab Image" 폴더에 `YYYYMMDD_HHMMSS.bmp`로 저장

---

### 🕐 저녁 — [기능 추가 - Basler 비디오 녹화]

**한 일**:
- **"Video Record" 체크박스** 추가
- 체크 후 Start Inspection → 녹화 시작, Stop → 녹화 종료
- "Basler Video File" 폴더에 `YYYYMMDD_HHMMSS.avi`로 저장
- 실시간 프레임을 VideoWriter로 기록

---

## 📅 2026-03-06 (목) — 문서 정비 / 빌드 개선 / UI 개선

---

### 🕐 12:00 — [문서] 전체 검사 파이프라인 도표 작성

**파일**: `doc/Inspection_Pipeline_Flowchart.md`

**한 일**:
- Mermaid 플로우차트로 전체 검사 흐름 시각화
- 텍스트 기반 플로우차트에 `vscode://file/` 링크 추가 → Ctrl+Click으로 해당 소스 코드 라인으로 이동
- 총 25개 이상의 소스 코드 위치 링크 설정

---

### 🕐 12:30 — [문서 수정] 플로우차트 소스 링크 검증 및 오류 수정

**한 일**:
- 전체 소스 코드 대조하여 잘못된 라인 링크 3곳 수정:
  - Basler `grab_frame()` 링크: `:20` → `:101`
  - 형태학 정리 링크: `:230` → `:227`
  - RuleBasedClassifier 링크: `:30` → `:29`
- Mermaid 그래프 내 잘못된 분류 설명 수정:
  - `Circularity ≥ 0.7 → Bubble` 분기 **삭제** (실제 코드에 없음)
  - Rule-based 분류 로직을 **Contrast 기반**으로 수정 (실제 코드와 일치)

**오류 원인**: 소스 코드를 끝까지 정독하지 않고, 변수명에서 로직을 유추하여 작성했음

---

### 🕐 13:42 — [빌드 개선] build_exe.py에 onnx 제외 옵션 추가

**파일**: `build_exe.py`

**한 일**:
- `--exclude-module=onnx`, `--exclude-module=onnx.reference` 추가
- PyInstaller가 onnx.reference C++ 확장 분석 시 크래시하는 문제 해결

---

### 🕐 13:45 — [빌드 개선] 빌드 시 ClassificationData 백업/복원

**파일**: `build_exe.py`

**한 일**:
- `PRESERVE_ITEMS` 리스트로 보존할 폴더/파일 정의
- 빌드 전: `dist/ForeignBodyInsp/` 안의 보존 대상을 임시 폴더로 백업
- 빌드 후: 임시 폴더에서 다시 복원
- 보존 대상: `ClassificationData/`, `path_config.json`, `classification_model.onnx`, `.pth`

---

### 🕐 14:03 — [UI 개선] Classification 탭 이미지 영역 최대화

**파일**: `src/ui/classification_tab.py`

**한 일**:
- 이미지 미리보기 영역을 `stretch=1`로 최대 확장
- 하단 파일 정보 텍스트 크기: 11px → 27px (2.5배)
- 파일 정보 영역 높이: 무제한 → `fixedHeight=90px`로 축소

---

### 🕐 15:58 — [버그 해결] 관리자 권한 실행 시 드래그 앤 드롭 차단 (UIPI)

**파일**: `src/main.py`

**증상**: 파일 탐색기에서 앱으로 드래그하면 금지 커서(🚫) 표시

**원인**: VS Code가 관리자 권한으로 실행 → Python 앱도 관리자 권한 → Windows UIPI가 일반 권한 탐색기로부터의 드래그를 차단

**한 일**:
- `_allow_drag_drop_as_admin()` 함수 추가
- Win32 API `ChangeWindowMessageFilter`로 WM_DROPFILES, WM_COPYDATA, WM_COPYGLOBALDATA 메시지 허용
- 앱 시작 시(`main()`) 자동 호출
- 관리자 권한에서도 드래그 앤 드롭 정상 동작

---

### 🕐 16:00 — [문서] BugReport.md 작성

**파일**: `doc/BugReport.md`

**한 일**:
- 오늘 발견/해결된 3개 버그를 체계적으로 정리 (증상, 원인, 해결, 수정 파일)

---

## 📊 주요 기능 요약 (최신)

| 카테고리 | 기능 |
|:---|:---|
| **입력** | Basler 카메라 (Connect/Disconnect, RealTime, 설정, **Grab 저장**, **Video 녹화**), 파일/동영상 로드 |
| **검사 모드** | Threshold+분류, YOLO 통합 검출, Bubble 검출 (다중 기법) |
| **학습** | Classification 탭 (CNN), YOLO 탭 (YOLOv8) |
| **결과** | Result 리스트 (라벨 필터), MainView (선택 강조), Defect View (확대) |
| **빌드** | `build_exe.py` (PyInstaller, onnx 제외, **데이터 백업/복원**) |
| **보안** | **UIPI 우회** (관리자 권한에서도 드래그 앤 드롭 가능) |
| **설정 파일** | path_config.json, rule_params.json, basler_settings.json |
| **문서** | Inspection_Pipeline_Flowchart.md, **BugReport.md**, **history.md** |
