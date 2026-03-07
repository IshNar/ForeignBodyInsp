"""
exe 빌드 스크립트 (PyInstaller 사용).
실행: python build_exe.py
빌드 결과: dist 폴더에 ForeignBodyInsp.exe 생성 (한 폴더에 모아두는 방식)
"""
import subprocess
import sys
import os
import shutil
import tempfile

# ── 빌드 시 보존할 폴더/파일 목록 ──
# PyInstaller가 dist 폴더를 덮어쓸 때 삭제되지 않도록 백업 → 복원
PRESERVE_ITEMS = [
    "ClassificationData",       # 학습 데이터 폴더
    "path_config.json",         # 경로 설정 파일
    "classification_model.onnx",  # 학습된 모델 파일
    "classification_model.pth",   # 학습된 모델 파일 (PyTorch)
]


def _backup_preserved(dist_app_dir: str, backup_dir: str):
    """dist 내 보존 대상을 임시 폴더로 백업."""
    count = 0
    for item in PRESERVE_ITEMS:
        src = os.path.join(dist_app_dir, item)
        dst = os.path.join(backup_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
            count += 1
            print(f"  [백업] 폴더: {item} ({sum(1 for _, _, f in os.walk(src) for _ in f)}개 파일)")
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
            count += 1
            size_kb = os.path.getsize(src) / 1024
            print(f"  [백업] 파일: {item} ({size_kb:.1f} KB)")
    return count


def _restore_preserved(backup_dir: str, dist_app_dir: str):
    """임시 폴더에서 보존 대상을 dist로 복원."""
    count = 0
    for item in PRESERVE_ITEMS:
        src = os.path.join(backup_dir, item)
        dst = os.path.join(dist_app_dir, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            count += 1
            print(f"  [복원] 폴더: {item}")
        elif os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
            print(f"  [복원] 파일: {item}")
    return count


def main():
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller가 없습니다. 설치 후 다시 실행하세요:")
        print("  pip install pyinstaller")
        sys.exit(1)

    root = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(root, "src", "main.py")
    dist_app_dir = os.path.join(root, "dist", "ForeignBodyInsp")

    # ── 1. 보존 대상 백업 ──
    backup_dir = None
    backed_up = 0
    if os.path.isdir(dist_app_dir):
        backup_dir = tempfile.mkdtemp(prefix="fbi_backup_")
        print(f"\n── 사용자 데이터 백업 ({backup_dir}) ──")
        backed_up = _backup_preserved(dist_app_dir, backup_dir)
        if backed_up > 0:
            print(f"  총 {backed_up}개 항목 백업 완료\n")
        else:
            print("  보존할 항목 없음\n")

    # ── 2. PyInstaller 빌드 ──
    src_dir = os.path.join(root, "src")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=ForeignBodyInsp",
        "--windowed",           # 콘솔 창 안 띄움
        "--onedir",             # 한 폴더에 exe + dll 등 (실행이 안정적)
        "--paths", root,        # import src 할 수 있도록
        "--add-data", src_dir + os.pathsep + "src",  # src 패키지 폴더 번들에 포함
        "--hidden-import=PyQt6",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--collect-all=torch",      # exe에서 torch 로드 실패 방지
        "--collect-all=torchvision",
        "--collect-all=pypylon",
        # onnx 패키지는 ultralytics 의존성으로 설치되었으나 우리 코드가 직접 사용하지 않음
        # (추론: onnxruntime, ONNX 변환: torch.onnx.export = torch 내장)
        # PyInstaller가 onnx.reference 분석 시 C++ 확장 로드 실패로 크래시하므로 제외
        "--exclude-module=onnx",
        "--exclude-module=onnx.reference",
        main_py,
    ]

    print("빌드 중... (완료 후 dist/ForeignBodyInsp 폴더에 exe가 생성됩니다)")
    subprocess.run(cmd, cwd=root)

    # ── 3. 보존 대상 복원 ──
    if backup_dir and backed_up > 0:
        print(f"\n── 사용자 데이터 복원 ──")
        restored = _restore_preserved(backup_dir, dist_app_dir)
        print(f"  총 {restored}개 항목 복원 완료")
        # 임시 백업 폴더 삭제
        shutil.rmtree(backup_dir, ignore_errors=True)

    print("\n완료. 실행 파일: dist\\ForeignBodyInsp\\ForeignBodyInsp.exe")

if __name__ == "__main__":
    main()

