# OpenVINO CPU/GPU/NPU 가이드

이 문서는 `ForeignBodyInsp` 프로젝트에서 `Level 5`(OpenVINO EP) 분류 모델 실행 시, 각 장치(하드웨어)가 어떤 의미인지와 동작 방식을 정리합니다.

---

## 1. 개요

`src/core/classification.py`의 `DeepLearningClassifier`에서
- `optimization_level == 5`이면 OpenVINO EP를 사용하려 시도
- ONNX 모델(`.onnx`)만 OpenVINO로 실행 가능
- 실패하면 `optimization_level`을 4로 폴백 및 OpenVINO 세션 해제

### 1.1 OpenVINO EP 디바이스 옵션
- `CPU` : CPU 전용 엔진. 안정적이고 모든 시스템 지원.
- `GPU` : GPU(D3D12/Level Zero 기반). 인텔 iGPU와 NUC GPU, 일부 외장 GPU 지원.
- `NPU` : NPU/Myriad/비전 가속기. Intel Myriad 또는 VPU 같은 저전력 모듈.
- `AUTO`: 사용 가능한 디바이스 중 우선순위로 GPU > NPU > CPU 선택.


## 2. 코드 흐름

### 2.1 모델 로드 (`load_model`)
- `.onnx`:
  - 기본 providers: `CUDAExecutionProvider`, `CPUExecutionProvider` 로 로딩
  - `self.ort_session` 설정, `_device`는 `CUDA` 또는 `CPU`
  - `optimization_level == 5`이면 `_ensure_openvino_session()` 호출

- `.pth`:
  - torch 모델 로드 후 `self.model`에 적재
  - `self.ort_session = None`
  - `optimization_level >= 2`이면 torch.compile 적용, (GPU일 때)

### 2.2 OpenVINO 세션 활성화 (`_ensure_openvino_session`)
- `self.model_path`가 `.onnx` 아니면 실패 → level 4 폴백
- 이미 `self.ort_session` 없으면 바로 return
- OpenVINO EP 로딩 시도
  - **`device_type=<CPU/GPU/NPU/AUTO>` 옵션 전달**
  - 성공 시 `self._ort_session_openvino` 설정
  - `device == AUTO`이면 `_infer_openvino_actual_device()`로 실제 기기 추론
  - 실패 시 level 4 폴백 + 세션 정리

### 2.3 장치 상태 확인 UI (`get_device_display`)
- OpenVINO 로드 상태:
  - `OpenVINO (GPU)`, `OpenVINO (NPU)`, `OpenVINO (CPU)` 또는 `OpenVINO (AUTO)`
- ONNX 로드 상태:
  - `GPU (ONNX)` 또는 `CPU (ONNX)`
- PyTorch 로드 상태:
  - `GPU (PyTorch)` 또는 `CPU (PyTorch)`


## 3. MainWindow UI 동기화

`src/ui/main_window.py`에서:
- `lbl_dl_device`가 `Use Classification` 바로 옆에 UI로 표시
- `dl_model` 로드 성공/실패, `use_dl` 토글 시 자동 갱신
- Rule Params 다이얼로그에서 Level5/Device 변경 후 확인 시 갱신


## 4. 예시 시나리오

1. 상태: `Level 5` + `OpenVINO` + `NPU`
   - `dl_classifier._openvino_device = "NPU"`
   - `dl_classifier._ort_session_openvino != None`
   - `lbl_dl_device` → `NPU (OpenVINO)`

2. 상태: `Level 5` + `OpenVINO` + `AUTO` + NPU 없음 + GPU 있음
   - `_infer_openvino_actual_device()`는 `GPU` 반환
   - `lbl_dl_device` → `GPU (OpenVINO)`

3. 상태: `Level 5` + OpenVINO EP 불가 (onxxruntime-openvino 없거나 설치 실패)
   - `optimization_level`→ `4`, `_clear_openvino_session()`
   - `lbl_dl_device` → `CPU (ONNX)`


## 5. 중요 체크리스트

- `requirements-openvino.txt` 패키지 설치 필요
  - `onnxruntime-openvino` (또는 최신 엔진)
- `.onnx` 모델 사용 권장
- OpenVINO 하드웨어 지원 여부는 OS/버전/드라이버에 민감함


---

## 6. 향후 개선 아이디어

- OpenVINO 상태를 MainWindow에 로그/툴팁으로 추가 표시 (추론 시작 시 실제 선택된 디바이스 정보)
- 실패 시 (Level 5 → 4 폴백) 사용자 팝업/로그를 더 명확히
- `openvino_device_actual`을 상태 JSON에 기록하여 재시작 시 동일 장치 재사용
- OpenVINO `AUTO` 기기 우선순위 정책을 환경변수로 설정 가능하게 변경
