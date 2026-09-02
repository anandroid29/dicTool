#!/usr/bin/env python3
"""PyDIC — Digital Image Correlation. Entry point."""
import sys
from pathlib import Path

# Permit ``python src/pydic/__main__.py`` from a source checkout. Installed entry
# points already put the source root on sys.path.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtGui import QFont
from pydic.ui.wizard import Wizard

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
