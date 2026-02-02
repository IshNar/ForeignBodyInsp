import sys
import os

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
