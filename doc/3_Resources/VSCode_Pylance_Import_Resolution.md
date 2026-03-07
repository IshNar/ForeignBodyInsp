# 🔧 VS Code Pylance Import 빨간 밑줄 해결 가이드

> 코드는 정상 실행되는데 IDE에서 `import` 구문에 **빨간/노란 밑줄**이 뜨는 경우의 원인과 해결법.

---

## 🔴 증상

소스 코드에서 아래와 같은 import에 빨간 밑줄(경고)이 표시됨:

```python
from src.core.detection import ForeignBodyDetector       # ❌ 빨간 밑줄
from src.core.classification import RuleBasedClassifier   # ❌ 빨간 밑줄
from src.hardware.basler_camera import BaslerCamera        # ❌ 빨간 밑줄
```

하지만 **F5로 실행하면 정상 동작**함.

---

## 🤔 원인

**런타임(실행)과 IDE(정적 분석)가 import 경로를 찾는 방식이 다르기 때문.**

| | 런타임 (F5 실행) | IDE (Pylance 분석) |
|:---|:---|:---|
| **import 경로** | `sys.path`에 프로젝트 루트가 추가됨 | `sys.path` 조작을 알 수 없음 |
| **동작** | `from src.core.xxx` 정상 인식 | `src`를 찾을 수 없다고 빨간 밑줄 |
| **시점** | 코드가 실행될 때 | 파일을 열자마자 (실행 전) |

### 왜 런타임에서는 되는가?

`main.py` 상단에서 프로젝트 루트를 `sys.path`에 등록하는 코드가 있기 때문:

```python
# main.py 상단
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
```

또는 `launch.json`에서 `"cwd": "${workspaceFolder}"`로 설정되어 있기 때문.

### 왜 IDE에서는 안 되는가?

Pylance(정적 분석기)는 **코드를 실행하지 않고** 텍스트만 읽으며 분석함.
따라서 `sys.path.insert()`를 통한 런타임 경로 추가를 전혀 인식하지 못함.

---

## ✅ 해결: `settings.json`에 한 줄 추가

`.vscode/settings.json`에 `python.analysis.extraPaths`를 추가:

```json
{
    "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
    "python.terminal.activateEnvInSelectedTerminal": true,
    "python.analysis.extraPaths": ["."]
}
```

- `"."` = **프로젝트 루트 폴더**를 Pylance의 import 검색 경로에 추가
- 이렇게 하면 Pylance가 `from src.core.xxx`를 프로젝트 루트 기준으로 찾을 수 있음

### 적용 후

```
Ctrl+Shift+P → "Reload Window" (또는 VS Code 재시작)
```

→ 빨간 밑줄이 모두 사라짐 ✅

---

## 📌 그 외 빨간 밑줄이 뜨는 경우

### 1. 외부 패키지 import에 밑줄 (`pypylon`, `torch` 등)

```python
from pypylon import pylon      # ❌ 밑줄
import torch                    # ❌ 밑줄
```

**원인**: VS Code가 선택한 Python 인터프리터에 해당 패키지가 설치되어 있지 않음.

**해결**:
1. `Ctrl+Shift+P` → "Python: Select Interpreter" 
2. 프로젝트의 `.venv` (가상환경) Python을 선택
3. 해당 가상환경에 패키지가 설치되어 있는지 확인:
   ```bash
   .venv\Scripts\pip list | findstr pypylon
   ```

### 2. 상대 import에 밑줄

```python
from .camera_interface import CameraSource    # ❌ 밑줄
```

**원인**: Pylance가 패키지 구조를 인식하지 못할 때 발생.

**해결**: 위의 `extraPaths` 설정이 적용되면 자동으로 해결됨.

---

## 💡 참고: extraPaths vs PYTHONPATH

| 방법 | 적용 범위 | 용도 |
|:---|:---|:---|
| `python.analysis.extraPaths` | **IDE 분석 전용** | Pylance 빨간 밑줄 해결 |
| `PYTHONPATH` 환경 변수 | IDE + 런타임 모두 | 시스템 전체 경로 설정 |
| `sys.path.insert()` (코드) | **런타임 전용** | 코드 내 동적 경로 추가 |

> 프로젝트 설정에서는 `extraPaths`만으로 충분합니다.  
> `PYTHONPATH`는 시스템 환경변수를 건드려야 해서 프로젝트 이동 시 번거롭습니다.
