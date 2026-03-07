# 🔗 VS Code 마크다운 소스코드 링크 전략 가이드

> 프로젝트 문서(.md)에서 소스코드 특정 라인으로 이동하는 링크를 걸 때,  
> **프로젝트 이동이나 md 파일 이동에도 깨지지 않는 방법**을 정리합니다.

---

## 📊 경로 방식 비교

| 방식 | 예시 | 프로젝트 폴더 이동 | md 파일 이동 | 판정 |
|:---|:---|:---|:---|:---|
| **절대경로 (vscode://)** | `vscode://file/D:/Project/src/main.py:10` | ❌ 깨짐 | ✅ 무관 | ❌ |
| **상대경로 (../)** | `../../src/main.py#L10` | ✅ 안전 | ❌ 깨짐 | ⚠️ |
| **워크스페이스 루트 기준 (/)** | `/src/main.py#L10` | ✅ 안전 | ✅ 안전 | ✅ **권장** |

---

## ✅ 권장: 워크스페이스 루트 기준 경로 (`/`로 시작)

### 사용법

```markdown
[함수 보기](/src/core/detection.py#L63)
```

- **`/`로 시작** → VS Code가 **워크스페이스(프로젝트) 루트 폴더**에서 경로를 찾음
- **`#L63`** → 해당 파일의 **63번째 줄**로 이동
- `Ctrl+Click`으로 바로 소스코드 해당 라인에 도달

### 왜 이게 최선인가?

```
프로젝트 루트/
├── src/
│   └── core/
│       └── detection.py     ← 항상 /src/core/detection.py
├── doc/
│   ├── 1_Projects/
│   ├── 2_Areas/
│   │   └── Pipeline.md      ← 여기서 링크해도 OK
│   └── 3_Resources/
│       └── Guide.md          ← 여기로 옮겨도 OK
```

- md 파일이 `doc/` 안 **어디에 있든** `/src/...`는 항상 프로젝트 루트의 `src/`를 가리킴
- 프로젝트 폴더를 `D:\` → `E:\`로 **통째로 옮겨도** 상대적이므로 깨지지 않음

---

## ❌ 피해야 할 방식

### 1. vscode:// 프로토콜 절대경로

```markdown
<!-- ❌ 드라이브 경로가 하드코딩됨 -->
[코드](vscode://file/D:/PythonProjects/MyProject/src/main.py:10)
```

- 프로젝트 폴더 이름을 바꾸거나, 다른 PC에서 열면 **즉시 깨짐**
- 협업 시 각 개발자의 경로가 다르면 아무도 못 씀

### 2. 깊은 상대경로 (`../../..`)

```markdown
<!-- ❌ md 파일이 이동하면 ../의 깊이가 달라져서 깨짐 -->
[코드](../../../src/core/detection.py#L63)
```

- md를 `doc/` 바로 밑에 놓을 때와, `doc/2_Areas/sub/` 안에 놓을 때 `../` 개수가 달라짐
- 문서 정리(폴더 이동) 할 때마다 링크를 전부 수정해야 함

---

## 🔧 라인 번호 표기법

VS Code 마크다운에서 특정 라인으로 이동하려면 `#L{번호}` fragment를 사용합니다.

```markdown
<!-- 파일만 열기 -->
[detection.py 열기](/src/core/detection.py)

<!-- 63번째 줄로 이동 -->
[detect_static() 함수](/src/core/detection.py#L63)
```

> ⚠️ **주의**: `#L63` 형식은 VS Code 마크다운 프리뷰의 `Ctrl+Click`과
> GitHub 렌더링에서 지원됩니다. `:63:1` 형식(line:col)은 `vscode://` 프로토콜
> 전용이므로 일반 마크다운 링크에서는 사용하지 않습니다.

---

## 📝 Mermaid 플로우차트에서의 사용

Mermaid의 `click` 디렉티브에서도 동일한 형식을 사용할 수 있습니다:

````markdown
```mermaid
flowchart TD
    A["검출 엔진"] --> B["분류 엔진"]
    
    click A "/src/core/detection.py#L63" "detect_static() 열기"
    click B "/src/core/classification.py#L254" "classify_batch() 열기"
```
````

> ⚠️ Mermaid click은 렌더러에 따라 동작이 다를 수 있습니다.
> VS Code 마크다운 프리뷰에서는 제한적이지만, 
> 텍스트 섹션의 `[링크](/src/...)` 형태는 확실하게 동작합니다.

---

## 🔄 기존 절대경로를 일괄 변환하는 방법 (PowerShell)

만약 기존 md 파일에 `vscode://file/...` 형태의 절대경로가 남아있다면,
아래 PowerShell 스크립트로 한 번에 변환할 수 있습니다:

```powershell
$file = "대상_파일.md"
$content = Get-Content $file -Raw -Encoding UTF8

# 1) vscode:// 절대경로 → 워크스페이스 루트 기준
$content = $content -replace 'vscode://file/[A-Za-z]:/[^/]+/[^/]+/', '/'

# 2) .py:line:col) → .py#Lline)  (마크다운 링크)
$content = $content -replace '\.py:(\d+):(\d+)\)', '.py#L$1)'

# 3) .py:line" → .py#Lline"  (Mermaid click)
$content = $content -replace '\.py:(\d+)"', '.py#L$1"'

$content | Set-Content $file -NoNewline -Encoding UTF8
```

> **정규식 설명**:
> - `vscode://file/[A-Za-z]:/[^/]+/[^/]+/` → 드라이브 문자 + 2단계 폴더까지 제거
> - `\.py:(\d+):(\d+)\)` → `.py:63:1)` 패턴에서 line만 추출
> - `\.py:(\d+)"` → `.py:63"` 패턴에서 line만 추출

---

## 💡 요약

```markdown
<!-- ✅ 이렇게 쓰세요 -->
[detect_static()](/src/core/detection.py#L63)

<!-- ❌ 이렇게 쓰지 마세요 -->
[detect_static()](vscode://file/D:/Projects/ForeignBodyInsp/src/core/detection.py:63:1)
[detect_static()](../../src/core/detection.py#L63)
```

**규칙 하나만 기억하기**: `/src/...#L줄번호` 형식으로 쓰면 끝!
