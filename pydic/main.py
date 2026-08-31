#!/usr/bin/env python3
"""PyDIC. Digital Image Correlation. Entry point."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtGui import QFont
from src.ui.crash_guard import install as install_crash_guard
from src.ui.wizard import Wizard

def main():
    # QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    app.setApplicationName("PyDIC")
    app.setApplicationVersion("2.0.0")
    app.setFont(QFont("Inter, Segoe UI, Helvetica Neue, Arial", 11))
    w = Wizard()
    # A slot that raises would otherwise abort the process outright,
    # closing the window mid-analysis with nothing written.
    install_crash_guard(w)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
