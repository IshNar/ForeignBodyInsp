# ForeignBodyInsp
Vial Foreign Body Inspection Program use Rulebased + classification

---

## 실행 방법 (VS Code 없이 켜는 방법)

**1) 더블클릭으로 실행 (Python만 설치돼 있으면 됨)**  
- 프로젝트 폴더에 **`run.bat`** 있음. 이걸 더블클릭하면 프로그램이 실행됩니다.  
- `.venv` 가상환경이 있으면 그걸 쓰고, 없으면 시스템 `python`을 씁니다.

**2) exe로 한 번 빌드해 두기 (Python 설치 없이 다른 PC에서도 실행)**  
- 아래 "exe 빌드"를 한 번 해 두면 **`dist\ForeignBodyInsp\ForeignBodyInsp.exe`** 가 생깁니다.  
- 이 **폴더 전체**(exe + dll 등)를 복사해서 다른 PC에 옮겨도 실행 가능합니다.

**3) VS Code / Cursor에서**  
- F5 (launch.json → `src/main.py` 실행)

**정리:** 기본 상태에서는 exe 파일은 없고, **run.bat 더블클릭** 또는 **exe 빌드** 중 하나를 쓰면 VS Code 없이 실행할 수 있습니다.

---

## exe 빌드 (선택)

1. `pip install pyinstaller` 로 PyInstaller 설치
2. 프로젝트 폴더에서  
   `python build_exe.py`  
   실행
3. 빌드가 끝나면 **`dist\ForeignBodyInsp\`** 폴더 안에  
   **`ForeignBodyInsp.exe`** 가 생성됩니다.  
   (같은 폴더의 dll 등과 함께 두고 실행)

4. **exe로 실행할 때** 딥러닝 모델·설정 파일은 **exe가 있는 폴더**를 기준으로 찾습니다.  
   - `classification_model.pth` → exe와 같은 폴더에 두면 "모델 로드" 또는 드래그앤드롭으로 로드 가능  
   - `basler_settings.json` → 같은 폴더에 두면 Basler 설정 다이얼로그에서 "파일에서 로드" 가능

---

## 재시작 (Ctrl+Shift+R)

- 도움말 메뉴 → "재시작 (업데이트 적용)" 또는 **Ctrl+Shift+R**
- F5(디버거)로 실행해도 재시작 시 **새 창이 뜨도록** 수정해 두었습니다. (이전에는 재시작하면 창만 꺼졌음)
