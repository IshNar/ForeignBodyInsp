"""
exe 빌드 스크립트 (PyInstaller 사용).
실행: python build_exe.py
빌드 결과: dist 폴더에 ForeignBodyInsp.exe 생성 (한 폴더에 모아두는 방식)
"""
import subprocess
import sys
import os

def main():
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller가 없습니다. 설치 후 다시 실행하세요:")
        print("  pip install pyinstaller")
        sys.exit(1)

    root = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(root, "src", "main.py")

    src_dir = os.path.join(root, "src")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=ForeignBodyInsp",
        "--windowed",           # 콘솔 창 안 띄움
        "--onedir",             # 한 폴더에 exe + dll 등 (실행이 안정적). --onefile은 단일 exe지만 느릴 수 있음
        "--paths", root,        # import src 할 수 있도록
        "--add-data", src_dir + os.pathsep + "src",  # src 패키지 폴더 번들에 포함
        "--hidden-import=PyQt6",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--collect-all=torch",      # exe에서 torch 로드 실패 방지
        "--collect-all=torchvision",
        "--collect-all=pypylon",
        main_py,
    ]

    print("빌드 중... (완료 후 dist/ForeignBodyInsp 폴더에 exe가 생성됩니다)")
    subprocess.run(cmd, cwd=root)
    print("\n완료. 실행 파일: dist\\ForeignBodyInsp\\ForeignBodyInsp.exe")

if __name__ == "__main__":
    main()
