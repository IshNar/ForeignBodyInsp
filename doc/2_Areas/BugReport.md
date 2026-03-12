# 🐛 Vial Foreign Body Inspection System — 버그 리포트 및 해결 방법

> ※ 나중에 다시 봐도 이해할 수 있도록, "무슨 문제였는지 → 왜 생겼는지 → 뭘 하면 해결되는지" 순서로 적었습니다.
> ※ 중간에 시도했던 수정은 빼고, 최종적으로 해결한 방법만 적었습니다.

---

## BUG-001: Drag & Drop 시 금지 아이콘(빨간 동그라미) 표시

**무슨 문제**: 이미지를 프로그램 창 위로 끌고 가면 커서에 "금지" 모양이 떠서, 드롭이 안 되는 것처럼 보임 (실제로는 되기도 함).

**왜 생김**: Qt에서 드래그를 "받을 수 있다"고 알려 주는 위젯이 한정돼 있거나, 드롭 동작 종류를 명시하지 않으면 OS가 금지 커서를 보여 줄 수 있음.

**해결**:
- 드래그를 받을 모든 위젯(메인 창, 탭, 스크롤 영역, 이미지 라벨, 컨트롤 패널, 스플리터 등)에 `setAcceptDrops(True)`를 설정해서 "여기는 드롭 받음"이라고 알려 줌.
- dropEvent에서 `setDropAction(Qt.DropAction.CopyAction)`를 호출해서 "복사로 받는다"고 명시.
- 자식 위젯들이 이벤트를 부모(메인 창)로 넘기도록 `installEventFilter`로 메인 창이 드래그/드롭을 처리하게 연결.

> ※ PC/OS에 따라 완전히 사라지지 않을 수 있어서, 우선순위는 낮게 둔 상태.

---

## BUG-002: main_window.py 직접 실행(F5) 시 ModuleNotFoundError

**무슨 문제**: main_window.py 파일만 골라서 F5로 실행하면 "No module named 'src'" 같은 식으로 모듈을 찾을 수 없다고 함.

**왜 생김**: Python은 실행한 스크립트가 있는 폴더를 기준으로 import 경로를 찾는데, main_window.py는 src/ui 안에 있어서 프로젝트 루트가 경로에 없음. 그래서 "src"를 패키지로 인식하지 못함.

**해결** (둘 중 하나):
- main_window.py 맨 위에서 프로젝트 루트 폴더 경로를 계산한 뒤 `sys.path`에 넣어 줌. 그러면 `import src...`가 동작함.
- 또는, 실행은 항상 프로젝트 루트에서 `python src/main.py`로 하는 것을 권장. (이렇게 하면 프로젝트 루트가 경로에 들어가서 src를 찾을 수 있음.)

---

## BUG-003: Start Inspection 버튼 클릭 시 프로그램 크래시(다운)

**무슨 문제**: 검사 시작 버튼을 누르면 프로그램이 갑자기 종료됨.

**왜 생김**: Defect View에 넣을 이미지를 만들 때, 잘라낸 영역(ROI)이 이미지 밖으로 나가거나 크기가 0이 되는 경우가 있음. 그 상태로 나누기나 배열 접근을 하면 예외 발생. 또 Qt의 QImage는 "연속된 메모리"에 있는 데이터를 기대하는데, NumPy 배열이 연속이 아니면 잘못된 결과나 크래시가 날 수 있음.

**해결**:
- Defect View용 이미지를 만드는 `update_defect_view()`에서, ROI 좌표가 이미지 범위 안인지, 너비·높이가 0이 아닌지 검사. 0으로 나누는 연산이 없도록 함.
- QImage에 넘기기 전에 `np.ascontiguousarray()`로 한 번 감싸서 "연속된 메모리"로 만듦.
- `render_main_view()`에서도 비슷하게 경계·연속성 검사 추가.

---

## BUG-004: Result 리스트에서 다른 항목 클릭 시 선택이 원래대로 되돌아감

**무슨 문제**: Result 리스트에서 다른 Defect를 클릭해도, 곧바로 이전에 선택돼 있던 항목으로 선택이 돌아감.

**왜 생김**: 매 프레임마다 `update_frame()`이 돌면서 검사 결과로 리스트를 갱신할 때, 뷰의 중심 좌표(view_cx, view_cy)나 줌을 함께 리셋하고 있었음. 그 타이밍에 "선택이 바뀌었다"는 이벤트가 다시 발생하거나, 화면이 갱신되면서 사용자가 선택한 것이 덮어씌워짐.

**해결**:
- view center와 zoom은 "새 이미지를 로드했을 때" 또는 "카메라를 바꿨을 때"만 리셋하고, 매 프레임마다는 리셋하지 않음.
- Defect View는 원본 프레임(`original_frame_full`)에서 해당 Defect 영역을 잘라서 보여 주도록 함. (검출 결과만이 아니라 원본 픽셀 기준으로 보여 줌.)

---

## BUG-005: Result 리스트에서 스크롤바 드래그 시 선택이 풀림

**무슨 문제**: Result 리스트에서 스크롤바만 움직여도 선택된 항목이 해제되거나, 다른 항목이 선택된 것처럼 바뀜.

**왜 생김**: 검사 결과가 올 때마다 리스트를 `clear()`한 뒤 처음부터 다시 `addItem()`으로 채우고 있었음. 그 과정에서 리스트가 한 번 비워지고 다시 채워지면서 "선택" 상태가 초기화되거나, 항목 순서가 잠깐 바뀌는 것처럼 보여서 selection 이벤트가 엉뚱하게 발생함.

**해결**:
- 리스트를 비우지 않고, 기존 항목이 있으면 `setText()`만 해서 내용만 갱신. 검출 개수가 바뀔 때만 `addItem()`으로 추가하거나 `takeItem()`으로 제거.
- 리스트를 갱신하는 동안에는 "선택이 바뀌었다"는 시그널이 나가면 안 되므로, `blockSignals(True)`로 막았다가 갱신이 끝나면 `blockSignals(False)`로 다시 켜 줌.

---

## ⭐ BUG-006: 검사 ROI 설정 후 Start Inspection 시 MainView 멈춤 (핵심)

**무슨 문제**: "검사 ROI 설정"으로 영역을 정한 뒤 Start Inspection을 켜면, 디버그 뷰는 계속 갱신되는데 메인 이미지 뷰(MainView)만 멈춰서 더 이상 안 움직임.

### 해결 1 — 스레드와 플래그 타이밍(데드락)

**왜 생김**: "지금 검사 중인가?"를 나타내는 `_worker_busy`와, 백그라운드 스레드(QThread)가 "완전히 끝났는가"를 나타내는 `isRunning()`을 둘 다 사용하고 있었음. 검사가 끝나면 `result_ready` 시그널로 `_worker_busy`를 False로 바꾸는데, 그 시점에 스레드 내부 정리는 아직 안 끝나서 `isRunning()`이 True인 상태. 그 다음 타이머가 "_worker_busy가 False니까 다음 검사 시작"하고 `run_detection()`을 부르는데, `run_detection()` 안에서는 "isRunning()이 True면 아직 이전 검사 중"이라서 아무 것도 안 하고 return. 그 결과 "검사는 시작도 안 했는데 _worker_busy는 True로 세팅된 상태"가 되어, 다음 프레임부터는 "아직 검사 중"으로만 인식되어 영원히 멈춤.

**해결**:
- `run_detection()` 안의 `isRunning()` 체크를 제거. "검사가 끝났다"는 판단은 QThread의 `finished` 시그널이 날아올 때만 하도록 하고, 그 시그널에 연결된 `_on_worker_finished()`에서만 `_worker_busy = False`로 설정. `_on_detection_result()`에서는 `_worker_busy`를 건드리지 않음.
- Worker의 `run()` 안에 try-except를 넣어서, 예외가 나도 빈 결과라도 시그널로 보내서 메인 쪽이 "검사 끝남"을 인식하게 함.

### 해결 2 — contour 좌표 자료형 (가장 중요)

**왜 생김**: ROI 영역만 잘라서 검사하면, 검출된 contour(윤곽) 좌표는 "ROI 안에서의 좌표"임. 이걸 전체 이미지에 그리려면 (rx, ry) 오프셋을 더해야 함. 그런데 contour는 정수 배열(int32), 오프셋은 실수라서, 더한 결과가 float64 배열이 됨. OpenCV의 `drawContours`, `boundingRect`, `minAreaRect` 등은 보통 정수(int32) 좌표를 기대하는데 float64가 들어가면 에러가 남. 그 에러가 except로 잡히면서 `render_main_view()`가 호출되지 않고, `current_display_frame`도 안 바뀌어서 화면이 "멈춘 것처럼" 보임.

**해결**:
- 오프셋을 더한 뒤 반드시 정수 배열로 바꿔서 넘김:
```python
contours = [(cnt + np.array([[[rx, ry]]])).astype(np.int32) for cnt in contours]
```

### 해결 3 — 예외가 나도 화면은 갱신

**왜 생김**: 위와 같은 에러(또는 다른 예외)가 나면 except 블록에서 로그만 찍고 끝나서, `current_display_frame`이 갱신되지 않고 `render_main_view()`도 호출되지 않음. 그래서 사용자 입장에서는 "화면이 멈췄다"고 보임.

**해결**:
- `_on_detection_result()`의 except 블록에서도, `full_frame` 또는 `raw_frame`으로 `current_display_frame`·`display_base_frame`을 설정한 뒤 `render_main_view()`를 한 번 호출. 이렇게 하면 에러가 나도 "마지막으로 받은 프레임"이라도 화면에 그려져서 멈춘 것처럼 보이지 않음.

---

## BUG-007: PyTorch(torch) ModuleNotFoundError

**무슨 문제**: 프로그램 실행 시 "No module named 'torch'" 에러가 남.

**왜 생김**: 이 프로젝트는 가상환경(.venv)에 torch를 설치해 두고 쓰는데, 실행할 때는 그게 아닌 "시스템 Python"이나 다른 환경의 Python으로 실행된 경우. 그 환경에는 torch가 없음.

**해결** (둘 중 하나):
- IDE에서 "인터프리터"를 이 프로젝트의 `.venv` 쪽 Python으로 선택해서 실행.
- 또는 터미널에서 `.venv`를 활성화한 뒤 (`.\venv\Scripts\activate`) `python src/main.py`로 실행.
- VSCode/Cursor 사용 시 `.vscode/launch.json`에서 python 경로를 `.venv/Scripts/python.exe`로 지정해 두면 F5 실행 시 해당 환경이 사용됨.

---

## BUG-008: Basler 연결 시 "Camera Error"만 표시되어 원인 불명

**무슨 문제**: Connect Basler Camera를 눌렀을 때 상태창에 "Camera Error"만 나오고, 왜 연결이 안 됐는지 알 수 없음.

**왜 생김**: 연결 실패 시 코드에서 False만 반환하고, "pypylon이 없어서인지, 카메라를 못 찾아서인지, 타임아웃인지" 같은 구체적인 이유는 사용자에게 보여 주지 않고 있었음.

**해결**:
- `BaslerCamera.open()`이 실패할 때, 원인을 문자열로 `_last_error`에 저장하고, `get_last_error()`로 조회할 수 있게 함.
  - pypylon이 없으면: "pypylon이 설치되어 있지 않습니다" + pip install 안내.
  - `EnumerateDevices()` 결과가 없으면: "Basler 카메라를 찾을 수 없습니다" + GigE 카메라는 IP/서브넷 설정, Pylon IP Configurator, Pylon SDK 설치 여부 안내.
  - 그 외 예외: `str(e)` 저장.
- 메인 창에서 연결 실패 시, `QMessageBox.warning()`으로 `get_last_error()` 내용과 "pip install pypylon, Pylon SDK 설치, GigE 설정 확인" 안내를 띄움. 이렇게 하면 사용자가 "지금 이 PC에서는 왜 안 되는지"를 보고 다음에 무엇을 해야 할지 알 수 있음.

---

## ⭐ BUG-009: exe로 빌드하면 모델 로드 실패 — [WinError 1114] c10.dll 초기화 오류 (핵심)

**무슨 문제**: PyInstaller로 exe를 만든 뒤 실행하면, 딥러닝 모델(.pth)을 로드할 때 아래 에러가 뜸:

```
[WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다.
Error loading "...\dist\ForeignBodyInsp\_internal\torch\lib\c10.dll" or one of its dependencies.
```

Python(venv)으로 직접 실행하면 문제없이 모델이 로드되는데, exe에서만 이 에러가 남. GPU torch든 CPU 전용 torch든 관계없이 exe에서 동일하게 발생함.

**왜 생김**:
- PyInstaller로 exe를 만들면, torch의 DLL 파일들(c10.dll, torch_cpu.dll 등)이 `dist\ForeignBodyInsp\_internal\torch\lib\` 폴더 안에 복사됨.
- 그런데 exe가 실행될 때, Windows의 DLL 검색 경로에 이 폴더가 등록되어 있지 않음. Python으로 직접 실행할 때는 venv 안의 torch가 자기 lib 폴더를 알아서 등록하지만, exe 번들 안에서는 그 자동 등록이 동작하지 않음.
- 그래서 c10.dll을 찾긴 찾는데, c10.dll이 의존하는 다른 DLL들(같은 폴더에 있는)을 Windows가 못 찾아서 "DLL 초기화 루틴을 실행할 수 없습니다"라는 에러가 남.
- **핵심**: 모델 파일(.pth) 문제가 아니라, torch 자체의 DLL 로딩 문제.

**해결**:
- `src/main.py`의 **맨 앞**(torch가 import되기 전)에서, exe 실행일 때 torch DLL이 들어 있는 폴더를 Windows DLL 검색 경로에 미리 등록함.
- 두 가지 방법을 동시에 사용:
  1. `os.add_dll_directory(_internal\torch\lib)` — Python 3.8+ Windows 전용 함수. 이걸 호출하면 Windows가 이 폴더에서도 DLL을 찾게 됨.
  2. `os.environ["PATH"]`에 같은 폴더를 앞에 추가 — 일부 라이브러리가 PATH를 참조하므로.
- `_internal\torch\lib`, `_internal\torch\bin`, `_internal` 세 폴더를 모두 등록함 (torch 버전에 따라 DLL 위치가 다를 수 있어서).
- 이 코드는 반드시 `from src.ui.main_window import MainWindow` 같은 torch를 간접적으로라도 import하는 줄보다 **위에** 있어야 함. (classification.py → torch import 하므로, main_window → classification이 연쇄 import됨.)

**실제 코드** (`src/main.py` 상단):
```python
if getattr(sys, "frozen", False):
    _internal = os.path.join(_root, "_internal")
    _torch_lib = os.path.join(_internal, "torch", "lib")
    _torch_bin = os.path.join(_internal, "torch", "bin")
    for _dll_dir in [_torch_lib, _torch_bin, _internal]:
        if os.path.isdir(_dll_dir):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(_dll_dir)
                except OSError:
                    pass
            os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")
```

**참고**:
- GPU 버전 torch로 exe를 빌드하면 CUDA DLL도 비슷한 문제가 생길 수 있어서, exe 빌드는 CPU 전용 torch(`.venv_cpu`)로 하는 것을 권장.
  - CPU 전용 torch 설치: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- GPU가 필요한 환경에서는 Python(venv) + F5/run.bat으로 실행하는 것이 안정적.

---

## BUG-010: Ctrl+Shift+R 재시작 시 새 창이 뜨지 않고 그냥 꺼짐

**무슨 문제**: VS Code / Cursor에서 F5로 실행한 상태에서 Ctrl+Shift+R(재시작)을 누르면, 새 창이 뜨지 않고 프로그램이 그냥 종료됨.

**왜 생김**: 재시작이 `os.execv()`로 "현재 프로세스를 다른 실행으로 바꿔치기"하는 방식이었음. Python을 직접 실행했을 때는 이게 동작하지만, F5(디버거가 붙은 상태)로 실행하면 디버거가 "프로세스가 갑자기 다른 프로그램으로 교체됨"을 감지하고 연결을 끊어서, 결과적으로 프로세스가 종료되고 새 창도 안 뜸.

**해결**:
- `os.execv()` 대신 `subprocess.Popen()`으로 "새 프로세스를 띄우고, 지금 앱은 정상 종료"하는 방식으로 변경.
- Windows에서는 `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` 플래그로 디버거/부모와 분리된 프로세스 실행.
- 새 프로세스를 띄운 뒤 `QApplication.instance().quit()`로 현재 창을 정상 종료.
- 이렇게 하면 F5 / `run.bat` / exe 어디서 실행하든, Ctrl+Shift+R 시 "지금 창 닫힘 → 새 창 뜸" 형태로 재시작됨.

---

## BUG-011: 단일 이미지 로드 후 Start Inspection 두 번째부터 동작하지 않음

**무슨 문제**: bmp/jpg 같은 단일 이미지를 로드한 뒤 Start Inspection을 누르면 첫 번째는 잘 동작하는데, 두 번째 클릭부터는 이미지를 다시 로드하지 않는 한 아무 일도 안 일어남. (동영상에서는 문제 없음.)

**왜 생김**: 첫 번째 검사가 끝나면 `_on_detection_result()`에서 "단일 이미지이므로 1회만 검사" 로직이 타이머를 정지(`timer.stop()`)시키고, `is_inspecting = False`, 버튼도 "Start Inspection"으로 되돌림. 여기까지는 의도대로임. 문제는 두 번째 Start Inspection을 눌렀을 때: `toggle_inspection()`이 `is_inspecting = True`만 세팅하고 끝남. 타이머는 이미 꺼져 있어서 `update_frame()`이 불리지 않음. 그래서 검사가 아예 시작되지 않음.

**해결**:
- `toggle_inspection()`에서 "단일 이미지"(FileCamera + `is_video() == False`)일 때에도 `QTimer.singleShot(0, self.update_frame)`으로 1회 검사를 직접 실행하도록 추가.
- 이렇게 하면 타이머가 꺼져 있어도, Start Inspection을 누를 때마다 `update_frame()`이 한 번씩 불려서 검사가 실행됨. 검사 결과가 나오면 기존대로 자동 정지.

---

## BUG-012: exe에서 모델 로드 실패 메시지가 원인 불명

**무슨 문제**: exe에서 "모델 로드" 버튼을 누르거나 .pth를 드래그앤드롭할 때, "모델 로드에 실패했습니다."만 뜨고 왜 실패했는지 알 수 없음.

**왜 생김**: `load_model()`이 성공/실패를 bool(True/False)로만 반환하고, 실제 에러 메시지(`str(e)`)를 UI까지 전달하지 않고 있었음. 콘솔(print)에는 찍혔지만, exe(`--windowed`)에서는 콘솔이 안 보이므로 사용자는 원인을 알 수 없었음.

**해결**:
- `DeepLearningClassifier.load_model()`의 반환값을 `(bool, str|None)` 튜플로 변경. 실패 시 에러 메시지 문자열 반환.
- `ParticleClassifier.load_dl_model()`도 같은 형태로 전달.
- main_window의 `_on_load_model()`과 드래그앤드롭 처리에서, 실패 시 QMessageBox에 실제 에러 메시지를 표시하도록 변경.
- 파일 존재 여부도 먼저 체크해서 "파일이 없습니다: {경로}" 메시지를 보여 줌.

---

## BUG-013: PyInstaller 빌드 시 onnx.reference 크래시 (2026-03-06 추가)

| 항목 | 내용 |
|:---|:---|
| **발생일** | 2026-03-06 |
| **심각도** | 🔴 High |
| **상태** | ✅ 해결됨 |
| **영향 범위** | 빌드 프로세스 (exe 생성 불가) |

**무슨 문제**: `python build_exe.py` 실행 시 **SubprocessDiedError** 발생. PyInstaller 분석 단계에서 `onnx.reference` 모듈 import 시 크래시.

**왜 생김**:
- `onnx` 패키지는 `ultralytics` 의존성으로 자동 설치됨
- 우리 코드는 `onnx` 패키지를 **직접 사용하지 않음**:
  - 추론: `onnxruntime` 사용 (`import onnxruntime as ort`)
  - ONNX 변환: `torch.onnx.export()` (PyTorch 내장 함수)
- PyInstaller가 `onnx` 패키지 전체를 분석하면서 `onnx.reference` 내부의 C++ 확장 로드에 실패

**해결**: `build_exe.py`에 아래 옵션 추가:
```python
"--exclude-module=onnx",
"--exclude-module=onnx.reference",
```

---

## BUG-014: 빌드 시 ClassificationData 폴더 삭제 (2026-03-06 추가)

| 항목 | 내용 |
|:---|:---|
| **발생일** | 2026-03-06 |
| **심각도** | 🟡 Medium |
| **상태** | ✅ 해결됨 |
| **영향 범위** | 학습 데이터 손실 |

**무슨 문제**: `python build_exe.py` 실행 시 `dist/ForeignBodyInsp/` 폴더가 통째로 재생성됨. 그 안에 저장해둔 `ClassificationData/` (학습 이미지) 및 `classification_model.onnx` 등이 삭제됨.

**왜 생김**: PyInstaller의 `--onedir` 모드는 빌드 시 기존 출력 폴더를 삭제 후 재생성. `ClassificationData`가 `dist/ForeignBodyInsp/` 안에 있으면 함께 삭제됨.

**해결**: `build_exe.py`에 **백업 → 빌드 → 복원** 로직 추가:
```python
PRESERVE_ITEMS = [
    "ClassificationData",         # 학습 데이터 폴더
    "path_config.json",           # 경로 설정 파일
    "classification_model.onnx",  # 학습된 모델
    "classification_model.pth",   # 학습된 모델 (PyTorch)
]
```
1. 빌드 전: `dist/ForeignBodyInsp/` 안의 보존 대상을 임시 폴더로 백업
2. PyInstaller 빌드 실행
3. 빌드 후: 임시 폴더에서 다시 `dist/ForeignBodyInsp/`로 복원

---

## BUG-015: 관리자 권한 실행 시 드래그 앤 드롭 차단 — UIPI (2026-03-06 추가)

| 항목 | 내용 |
|:---|:---|
| **발생일** | 2026-03-06 |
| **심각도** | 🟡 Medium |
| **상태** | ⚠️ 코드 우회 불가 — 일반 권한 실행으로만 해결 |
| **영향 범위** | 전체 앱 (Main 창 + Classification 탭) |

**무슨 문제**: 파일 탐색기에서 앱으로 이미지를 드래그하면 **금지 커서**(🚫) 표시. Classification 탭뿐만 아니라 **Main 창**에서도 동일 현상. 이미지 로드 버튼 등 다른 기능은 정상 동작.

**왜 생김**: **Windows UIPI (User Interface Privilege Isolation)** 정책
```
파일 탐색기 (일반 권한)  →  드래그  →  Python 앱 (관리자 권한)
                                          ❌ Windows가 차단
```
- VS Code를 **관리자 권한**으로 실행한 상태에서 Python 앱을 실행하면, 앱도 관리자 권한으로 실행됨
- Windows는 **일반 권한 프로세스 → 관리자 권한 프로세스** 간의 드래그 앤 드롭을 보안상 차단
- 이것은 앱 코드 결함이 아닌 **OS 보안 정책**

**코드로 우회할 수 없는 이유**:
- Win32 API의 `ChangeWindowMessageFilter`로 `WM_DROPFILES` 메시지를 허용하는 방법이 있으나, 이는 **레거시 Win32 드래그 앤 드롭**에만 동작
- Qt(PyQt6)는 **OLE 기반 드래그 앤 드롭**(COM IDropTarget)을 사용하며, OLE 드래그 앤 드롭의 UIPI 제한은 `ChangeWindowMessageFilter`로 **우회할 수 없음**

| 드래그 앤 드롭 방식 | UIPI 코드 우회 | 비고 |
|:---|:---|:---|
| 레거시 Win32 (`WM_DROPFILES`) | ✅ 가능 | `ChangeWindowMessageFilter` |
| **Qt/OLE 기반 (COM IDropTarget)** | ❌ **불가** | Qt가 사용하는 방식 |

**해결**: VS Code를 **일반 권한(관리자가 아닌 상태)**으로 실행
- VS Code 종료 후, "관리자 권한으로 실행"을 선택하지 **않고** 일반 실행
- VS Code 타이틀바에 "[관리자]"가 표시되지 않는지 확인

**참고**:
- **Windows Vista 이후** UIPI가 도입됨
- Linux/macOS에서는 이 문제가 발생하지 않음

---

## 2026-02-05 — Classification 사용 시 검사 시간 변동·메모리 이슈

| 항목 | 내용 |
|:---|:---|
| **발생일** | 2026-02-05 |
| **심각도** | 🟡 Medium |
| **상태** | ✅ 완화됨 |
| **영향 범위** | Classification(ONNX) 사용 + 동영상 재생 검사 |

**무슨 문제**:
- 동영상 틀어놓고 반복 검사 시 사이클마다 검사 시간이 미세하게 늘어나는 것처럼 보임.
- Classification 사용 시 180ms~390ms까지 시간 편차가 큼(같은 동영상, 다른 프레임). 영상 멈추면 상대적으로 안정.

**원인**:
- **메모리**: DetectionWorker가 프레임 버퍼를 해제하지 않음, 매 프레임 `ThreadPoolExecutor` 재생성, 결과 처리 시 프레임 `.copy()` 과다 → GC·할당 부담 증가.
- **시간 변동**: 추론 도중 Python GC 실행으로 간헐적 스파이크; ONNX CUDA workspace 첫 사용 시 할당 지연; ROI 추출 시 매 청크 대용량 배열 할당.

**해결**:
- Worker `run()` 끝에서 `_frame_bgr`/`_full_frame_bgr` 해제, 검출용 ThreadPoolExecutor 1개 캐싱, 결과 처리 시 불필요한 `.copy()` 제거(참조 공유).
- ONNX 로드 후 warmup 1회, CPU pipeline 구간에서 `gc.disable()`/`gc.enable()` 구간 격리, 50프레임마다 `gc.collect()`+`torch.cuda.empty_cache()`.
- ROI 버퍼 풀 2개 도입, `_extract_rois_chunk(..., out=)`로 유효 ROI만 앞쪽에 채워 재사용.

**수정 파일**: `src/core/classification.py`, `src/ui/main_window.py`

