"""Entry screen for selecting the kind of strainX workflow."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from strainx.ui.theme import (
    C_ACCENT, C_BG, C_BORDER, C_CARD, C_SURFACE, C_TEXT, C_TEXT2, C_TEXT3,
)


class _ModeCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, eyebrow: str, title: str, description: str,
                 action: str, *, enabled: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("modeCard")
        self._enabled = enabled
        self.setMinimumSize(245, 220)
        self.setMaximumWidth(340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._normal = (
            f"QFrame#modeCard{{background:{C_CARD};border:1px solid {C_BORDER};"
            "border-radius:5px;}")
        self._hover = (
            f"QFrame#modeCard{{background:#31353a;border:1px solid {C_ACCENT};"
            "border-radius:5px;}")
        self.setStyleSheet(self._normal)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)
        tag = QLabel(eyebrow.upper())
        tag.setStyleSheet(
            f"color:{C_TEXT3};font-size:9px;font-weight:700;letter-spacing:1px;"
            "background:transparent;border:none;")
        layout.addWidget(tag)
        heading = QLabel(title)
        heading.setWordWrap(True)
        heading.setStyleSheet(
            f"color:{C_TEXT};font-size:20px;font-weight:750;"
            "background:transparent;border:none;")
        layout.addWidget(heading)
        copy = QLabel(description)
        copy.setWordWrap(True)
        copy.setStyleSheet(
            f"color:{C_TEXT2};font-size:11px;line-height:1.4;"
            "background:transparent;border:none;")
        layout.addWidget(copy)
        layout.addStretch()
        footer = QLabel(action.upper())
        footer.setStyleSheet(
            f"color:{C_ACCENT if enabled else C_TEXT3};font-size:10px;"
            "font-weight:750;letter-spacing:0.8px;background:transparent;"
            "border:none;")
        layout.addWidget(footer)
        if not enabled:
            self.setToolTip("This mode is reserved in the UI and is not connected yet.")

    def enterEvent(self, event):
        if self._enabled:
            self.setStyleSheet(self._hover)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self._normal)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self._enabled and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ModePage(QWidget):
    def __init__(self, wizard):
        super().__init__()
        self._wizard = wizard
        self.setStyleSheet(f"background:{C_BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(64, 42, 64, 54)
        root.setSpacing(18)

        eyebrow = QLabel("STRAINX")
        eyebrow.setStyleSheet(
            f"color:{C_ACCENT};font-size:10px;font-weight:700;letter-spacing:1.8px;")
        root.addWidget(eyebrow)
        title = QLabel("How do you want to analyse your data?")
        title.setStyleSheet(f"color:{C_TEXT};font-size:32px;font-weight:750;")
        root.addWidget(title)
        subtitle = QLabel(
            "Choose the type of analysis that best fits your data and goals.")
        subtitle.setStyleSheet(f"color:{C_TEXT2};font-size:13px;")
        root.addWidget(subtitle)
        root.addSpacing(16)

        cards = QHBoxLayout()
        cards.setSpacing(16)
        single = _ModeCard(
            "One dataset", "Single analysis mode",
            "Measure displacement and strain in one image sequence using a "
            "single set of analysis parameters.", "Start single analysis")
        single.clicked.connect(lambda: wizard.select_mode("single"))
        cards.addWidget(single)
        parametric = _ModeCard(
            "Parameter study", "Parametric analysis",
            "Test a range of analysis settings on one dataset and compare "
            "the results side by side.",
            "Start parametric analysis")
        parametric.clicked.connect(lambda: wizard.select_mode("parametric"))
        cards.addWidget(parametric)
        bulk = _ModeCard(
            "Multiple datasets", "Bulk analysis",
            "Apply the same analysis settings to several image sequences.",
            "Coming soon", enabled=False)
        cards.addWidget(bulk)
        saved = _ModeCard(
            "Saved results", "Load HDF5",
            "Continue exploring results from a previous single or parametric "
            "analysis.", "Choose HDF5")
        saved.clicked.connect(wizard.open_hdf5)
        cards.addWidget(saved)
        root.addLayout(cards)
        root.addStretch()

        note = QLabel(
            "Bulk analysis is not available yet.")
        note.setStyleSheet(
            f"color:{C_TEXT3};font-size:10px;background:{C_SURFACE};"
            f"border:1px solid {C_BORDER};border-radius:3px;padding:9px;")
        root.addWidget(note)
