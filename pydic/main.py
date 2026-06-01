#!/usr/bin/env python3
"""PyDIC — Digital Image Correlation. Entry point."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtGui import QFont
from src.ui.wizard import Wizard

def main():
    # QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    app.setApplicationName("PyDIC")
    app.setApplicationVersion("2.0.0")
    app.setFont(QFont("Inter, Segoe UI, Helvetica Neue, Arial", 11))
    w = Wizard()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
