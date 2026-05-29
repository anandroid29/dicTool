"""
wizard.py — PyDIC wizard controller.

Hosts all 5 pages in a QStackedWidget.  A persistent top navigation bar
shows step progress.  Each page gets a reference to the wizard so it can
access the shared DICAnalysis instance and trigger page transitions.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy,
)

from src.core.analysis import DICAnalysis
from src.ui.theme import STYLESHEET
from src.ui.pages.welcome_page   import WelcomePage
from src.ui.pages.roi_page       import ROIPage
from src.ui.pages.params_page    import ParamsPage
from src.ui.pages.analysis_page  import AnalysisPage
from src.ui.pages.results_page   import ResultsPage


# ---------------------------------------------------------------------------
# Step indicator bar
# ---------------------------------------------------------------------------

_STEPS = ["Import", "ROI", "Parameters", "Analysis", "Results"]

_C_BG      = "#08111d"
_C_SURFACE = "#0c1930"
_C_BORDER  = "#1e3a5a"
_C_ACCENT  = "#3b82f6"
_C_TEXT    = "#e2e8f0"
_C_TEXT2   = "#94a3b8"
_C_TEXT3   = "#334155"
_C_DONE    = "#10b981"


class _StepBar(QWidget):
    """Horizontal step indicator: ①  Import ──── ② ROI ──── …"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setStyleSheet(f"background:{_C_SURFACE}; border-bottom:1px solid {_C_BORDER};")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Left spacer
        lay.addStretch(1)

        self._labels: list[QLabel] = []
        for i, step in enumerate(_STEPS):
            # Connector line (not before first)
            if i > 0:
                line = QLabel("────")
                line.setStyleSheet(f"color:{_C_TEXT3}; font-size:11px;")
                lay.addWidget(line)

            # Step pill
            lbl = QLabel(f"  {i+1}  {step}  ")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedHeight(26)
            lbl.setStyleSheet(self._inactive_style())
            lay.addWidget(lbl)
            self._labels.append(lbl)

        lay.addStretch(1)

        self.set_step(0)

    def set_step(self, idx: int) -> None:
        for i, lbl in enumerate(self._labels):
            if i < idx:
                lbl.setStyleSheet(self._done_style())
            elif i == idx:
                lbl.setStyleSheet(self._active_style())
            else:
                lbl.setStyleSheet(self._inactive_style())

    @staticmethod
    def _active_style() -> str:
        return (
            f"background:{_C_ACCENT}; color:#ffffff; "
            f"border-radius:5px; font-size:11px; font-weight:700;"
        )

    @staticmethod
    def _done_style() -> str:
        return (
            f"background:{_C_DONE}; color:#ffffff; "
            f"border-radius:5px; font-size:11px; font-weight:600;"
        )

    @staticmethod
    def _inactive_style() -> str:
        return (
            f"background:transparent; color:{_C_TEXT3}; "
            f"border-radius:5px; font-size:11px;"
        )


# ---------------------------------------------------------------------------
# Wizard main window
# ---------------------------------------------------------------------------

class Wizard(QMainWindow):
    """
    Top-level window.  Manages navigation between the 5 DIC workflow steps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyDIC — Digital Image Correlation")
        self.setMinimumSize(1200, 720)
        self.resize(1440, 900)

        self.analysis = DICAnalysis()
        self.seed_xy  = None        # optional (x,y) seed override

        self.setStyleSheet(STYLESHEET)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)

        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Step indicator
        self._step_bar = _StepBar()
        root.addWidget(self._step_bar)

        # Stacked pages
        self._stack = QStackedWidget()
        self._welcome  = WelcomePage(self)
        self._roi      = ROIPage(self)
        self._params   = ParamsPage(self)
        self._analysis = AnalysisPage(self)
        self._results  = ResultsPage(self)

        for page in (self._welcome, self._roi, self._params,
                     self._analysis, self._results):
            self._stack.addWidget(page)

        root.addWidget(self._stack, 1)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go(self, idx: int, page_with_enter=None) -> None:
        self._stack.setCurrentIndex(idx)
        self._step_bar.set_step(idx)
        if page_with_enter is not None and hasattr(page_with_enter, "on_enter"):
            page_with_enter.on_enter()

    def go_welcome(self) -> None:
        self._go(0, self._welcome)

    def go_roi(self) -> None:
        self._go(1, self._roi)

    def go_params(self) -> None:
        self._go(2, self._params)

    def go_analysis(self) -> None:
        self._go(3, self._analysis)

    def go_results(self) -> None:
        self._go(4, self._results)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self.analysis.cancel()
        event.accept()
