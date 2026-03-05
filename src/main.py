import sys
import os

# 프로젝트 루트를 sys.path에 (일반 실행 / exe 빌드 둘 다 대응)
if getattr(sys, "frozen", False):
    # PyInstaller exe: exe 폴더를 루트로, 작업 디렉터리도 여기로
    _root = os.path.dirname(sys.executable)
    try:
        os.chdir(_root)
    except Exception:
        pass

    # ★ exe에서 torch DLL(c10.dll 등)을 찾을 수 있도록,
    #    _internal/torch/lib 폴더를 DLL 검색 경로에 미리 등록.
    #    이 코드는 반드시 torch가 import 되기 **전에** 실행돼야 함.
    _internal = os.path.join(_root, "_internal")
    _torch_lib = os.path.join(_internal, "torch", "lib")
    _torch_bin = os.path.join(_internal, "torch", "bin")
    for _dll_dir in [_torch_lib, _torch_bin, _internal]:
        if os.path.isdir(_dll_dir):
            # Python 3.8+ Windows: add_dll_directory로 DLL 검색 경로 추가
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(_dll_dir)
                except OSError:
                    pass
            # PATH 환경변수에도 추가 (일부 라이브러리가 PATH를 참조)
            os.environ["PATH"] = _dll_dir + os.pathsep + os.environ.get("PATH", "")
else:
    _root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _root not in sys.path:
    sys.path.insert(0, os.path.abspath(_root))

from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)        
    app.setApplicationName("ForeignBodyInsp")
    app.setOrganizationName("ForeignBodyInsp")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
