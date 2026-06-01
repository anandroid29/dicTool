"""
components.py
-------------
Reusable UI components styled with PyDIC's global theme tokens.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor, QColor
from PyQt6.QtWidgets import QPushButton, QGraphicsDropShadowEffect, QWidget, QHBoxLayout, QLabel, QFrame

# Import your raw color tokens
from src.ui.theme import (
    C_ACCENT, C_ACCENT_G, C_ACCENT_D,
    C_SURFACE, C_BORDER, C_TEXT2, C_TEXT3
)


class FooterButton(QPushButton):
    """A highly visible, modern Call-To-Action button."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)

        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumHeight(38)

        self.setStyleSheet(f"""
            FooterButton {{
                background-color: {C_ACCENT};
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 28px;
                border-radius: 6px;
                border: 1px solid {C_ACCENT_D};
            }}
            FooterButton:hover {{
                background-color: {C_ACCENT_G};
                border: 1px solid #ffffff;
                color: #ffffff;
            }}
            FooterButton:pressed {{
                background-color: {C_ACCENT_D};
            }}
            FooterButton:disabled {{
                background-color: {C_SURFACE};
                color: {C_TEXT3};
                border: 1px solid {C_BORDER};
            }}
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)


class WizardStepper(QWidget):
    """A modern, breadcrumb-style step indicator for the wizard."""

    def __init__(self, steps: list[str], current_index: int = 0, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.steps = steps

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(0)

        self._labels = []
        self._lines = []

        for i, step_name in enumerate(steps):
            lbl = QLabel(f"{i + 1}  {step_name}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._labels.append(lbl)
            layout.addWidget(lbl)

            if i < len(steps) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFixedWidth(40)
                self._lines.append(line)
                layout.addWidget(line)

        self.set_step(current_index)

    def set_step(self, current_index: int) -> None:
        for i, lbl in enumerate(self._labels):
            if i == current_index:
                lbl.setStyleSheet(f"""
                    background-color: {C_ACCENT}; color: #ffffff;
                    font-weight: bold; font-size: 13px;
                    border-radius: 14px; padding: 6px 18px;
                """)
            elif i < current_index:
                lbl.setStyleSheet(f"color: {C_TEXT2}; font-weight: bold; font-size: 13px; padding: 6px 18px;")
            else:
                lbl.setStyleSheet(f"color: {C_TEXT3}; font-weight: bold; font-size: 13px; padding: 6px 18px;")

        for i, line in enumerate(self._lines):
            if i < current_index:
                line.setStyleSheet(f"border-top: 2px solid {C_ACCENT}; margin-top: 2px;")
            else:
                line.setStyleSheet(f"border-top: 2px solid {C_BORDER}; margin-top: 2px;")