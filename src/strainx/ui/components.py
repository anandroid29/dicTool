"""
components.py
-------------
Reusable UI components styled with strainX's global theme tokens.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor, QColor, QFont, QLinearGradient, QPainter
from PyQt6.QtWidgets import (
    QPushButton, QGraphicsDropShadowEffect, QWidget, QHBoxLayout, QLabel,
    QFrame, QSizePolicy,
)

from strainx.ui import render

# Import your raw color tokens
from strainx.ui.theme import (
    C_ACCENT, C_ACCENT_G, C_ACCENT_D,
    C_SURFACE, C_BORDER, C_TEXT2, C_TEXT3, C_WARNING,
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
                border-radius:3px;
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


class ResultColorBar(QWidget):
    """Scientific colour scale with explicit out-of-range indicators."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self._colors = [(8, 17, 29)] * 2
        self._vmin = self._vmax = 0.0
        self._unit = ""
        self._below = self._above = self._total = 0

    def update_bar(self, vmin, vmax, unit, colors,
                   below: int = 0, above: int = 0, total: int = 0):
        self._vmin, self._vmax, self._unit = vmin, vmax, unit
        self._colors = colors
        self._below, self._above, self._total = below, above, total
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, right, thickness = 6, 6, 12
        cap = 7
        has_low_cap = self._below > 0
        has_high_cap = self._above > 0
        width = (self.width() - left - right -
                 (cap if has_low_cap else 0) -
                 (cap if has_high_cap else 0))
        x0 = left + (cap if has_low_cap else 0)

        gradient = QLinearGradient(x0, 0, x0 + width, 0)
        for index, (red, green, blue) in enumerate(self._colors):
            gradient.setColorAt(
                index / max(len(self._colors) - 1, 1),
                QColor(red, green, blue))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(x0, 4, width, thickness, 3, 3)

        if has_low_cap:
            painter.setBrush(QColor(*render.UNDER_RANGE_RGB))
            painter.drawRect(left, 4, cap, thickness)
        if has_high_cap:
            painter.setBrush(QColor(*render.OVER_RANGE_RGB))
            painter.drawRect(x0 + width, 4, cap, thickness)

        painter.setPen(QColor(C_TEXT2))
        painter.setFont(QFont("Fira Code, Consolas, monospace", 9))
        minimum, maximum = f"{self._vmin:.4g}", f"{self._vmax:.4g}"
        baseline = 4 + thickness + 13
        metrics = painter.fontMetrics()
        painter.drawText(left, baseline, minimum)
        painter.drawText(
            self.width() - right - metrics.horizontalAdvance(maximum),
            baseline, maximum)

        if (has_low_cap or has_high_cap) and self._total:
            percentage = 100.0 * (self._below + self._above) / self._total
            note = f"{percentage:.2g}% outside range"
            painter.setFont(QFont("Fira Code, Consolas, monospace", 8))
            painter.setPen(QColor(C_WARNING))
            note_metrics = painter.fontMetrics()
            painter.drawText(
                (self.width() - note_metrics.horizontalAdvance(note)) // 2,
                baseline + 12, note)
        painter.end()


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
                    border-radius:3px; padding: 6px 18px;
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
