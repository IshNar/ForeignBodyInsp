# ForeignBodyInsp (Vial Foreign Body Inspection System)

## 📌 프로젝트 개요
**ForeignBodyInsp**는 바이알(Vial) 내의 이물질 및 기포를 검사하기 위해 개발된 딥러닝 + 룰 기반 머신비전 검사 시스템입니다. 산업용 카메라(Basler)와 연동되어 실시간으로 이미지를 취득하고, 룰 기반 알고리즘과 딥러닝 기반의 분류(Classification) 및 객체 탐지(YOLO) 기술을 융합하여 정밀한 불량 검출을 수행합니다.

사용자 친화적인 GUI(PyQt6)를 제공하며, 실시간 검사뿐만 아니라 튜닝 기능(알고리즘 파라미터 조절, 딥러닝 이미지 어노테이션 및 학습)도 포함되어 있는 종합 비전 솔루션입니다.

---

## 🏗 시스템 구조 (Architecture)

프로젝트는 크게 비전 처리 로직을 담은 `core`, 하드웨어 연동을 담당하는 `hardware`, 사용자 인터페이스를 담당하는 `ui` 폴더 구조로 나뉘어 있습니다.

### 1. 주요 폴더 구성 (src/)
* **`core/` (비전 검사 알고리즘 및 딥러닝 엔진)**
  * `detection.py`: OpenCV 기반 전통적인 룰 기반(Rule-based) 이물/기포 탐지 알고리즘. (파라미터 조절 가능)
  * `classification.py`: PyTorch/ONNX 기반 이미지 분류(Classification) 모델 (EfficientNet-B0 활용, Rule-based 분류와 병행).
  * `yolo_detector.py` / `yolo_dataset.py`: YOLO를 활용한 딥러닝 기반 객체 탐지 알고리즘 연동 및 데이터 처리.
* **`hardware/` (카메라 및 장비 연동)**
  * `camera_interface.py`: 카메라 연결 추상화 클래스
  * `basler_camera.py`: Basler 산업용 카메라 제어 및 실시간 영상 취득
  * `file_camera.py`: 카메라가 없는 환경에서도 저장된 이미지/동영상 파일을 읽어와 카메라처럼 시뮬레이션할 수 있는 파일 기반 가상 카메라
* **`ui/` (사용자 인터페이스 (PyQt6 기반))**
  * `main_window.py`: 실시간 검사 결과 및 카메라 뷰를 확인할 수 있는 메인 대시보드. User / Maker / Viewer 모드 지원 (Viewer 시 좌측 Debug/Defect 패널 접기, Main View 확대)
  * `rule_params_dialog.py`: 룰 기반 이미지 프로세싱(전처리, 이진화 기준 등) 파라미터를 실시간으로 튜닝할 수 있는 대화상자
  * `classification_tab.py` / `yolo_annotation_tab.py`: 딥러닝 모델 학습을 위한 데이터 라벨링 및 학습 관리 탭
  * `basler_settings_dialog.py`: Basler 카메라 노출 시간, Gain 등을 설정

---

## 🚀 주요 기능 (Key Features)

1. **실시간 이물 검사 (Live Inspection)**
   - Basler 하드웨어 카메라 또는 로컬 파일 폴더에서 영상을 불러와 실시간으로 **룰 기반 탐지(1차)** + **딥러닝(2차)** 복합 검사를 수행합니다.
2. **다중 알고리즘 융합**
   - 룰기반 파라미터(이진화 임계값 크기, 형태 등)를 세밀하게 조정 가능하며, 룰 규칙만으로 구분이 어려운 애매한 불량(기포 vs 이물)에 대해선 AI(PyTorch Classification / YOLO)가 최종 판정합니다.
3. **학습 데이터 수집 및 학습 탭 내장**
   - 별도의 툴 없이도 프로그램 내부에서 이미지를 양품/불량(기포, 이물 등)으로 분류(라벨링)하고 즉각적인 모델 재학습 파이프라인으로 연결할 수 있는 탭이 구현되어 있습니다.
4. **Standalone Exe 실행**
   - `build_exe.py`와 PyInstaller를 사용하여 파이썬이나 개발 환경이 없는 PC에서도 즉시 실행 가능한 형태로 1-Click 빌드가 가능합니다.

---

## 💻 실행 및 설치 안내

프로젝트 루트 폴더 혹은 개발 문서에서 안내하고 있습니다.
* **Python 소스 실행**: 가상환경 세팅 이후 `run.bat` 실행 및 vscode 환경에서 `src/main.py` 실행
* **배포판 실행**: `dist/ForeignBodyInsp/ForeignBodyInsp.exe` 실행 (모든 의존성 포함)

> **💡 개발 환경 참고**
> 이 프로젝트는 `requirements.txt`에 명시된 PyTorch, OpenCV, PyQt6, Pypylon (Basler SDK) 기반으로 구축되어 있습니다.
> 딥러닝 관련 부분은 GPU를 지원하기 위해 CUDA 환경 위에서 돌아가도록 설계되었습니다.
