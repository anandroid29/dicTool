from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy,
)

from src.core.analysis import DICAnalysis
from src.ui.theme import STYLESHEET
from src.ui.components import WizardStepper
from src.ui.pages.welcome_page import WelcomePage
from src.ui.pages.roi_page import ROIPage
from src.ui.pages.params_page import ParamsPage
from src.ui.pages.analysis_page import AnalysisPage
from src.ui.pages.results_page import ResultsPage

_STEPS = ["Import", "ROI", "Parameters", "Analysis", "Results"]


class Wizard(QMainWindow):
    """
    Top-level window. Manages navigation between the 5 DIC workflow steps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyDIC — Digital Image Correlation")
        self.setMinimumSize(1200, 720)
        self.resize(1440, 900)

        self.analysis = DICAnalysis()
        self.seed_xy = None
        self.use_gpu = False

        self.setStyleSheet(STYLESHEET)
        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)

        root = QVBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._step_bar = WizardStepper(_STEPS, current_index=0)
        root.addWidget(self._step_bar)

        self._stack = QStackedWidget()
        self._welcome = WelcomePage(self)
        self._roi = ROIPage(self)
        self._params = ParamsPage(self)
        self._analysis = AnalysisPage(self)
        self._results = ResultsPage(self)

        for page in (self._welcome, self._roi, self._params,
                     self._analysis, self._results):
            self._stack.addWidget(page)

        root.addWidget(self._stack, 1)

    def _go(self, idx: int, page_with_enter=None) -> None:
        self._stack.setCurrentIndex(idx)
        self._step_bar.set_step(idx)
        if page_with_enter is not None and hasattr(page_with_enter, "on_enter"):
            QTimer.singleShot(50, page_with_enter.on_enter)

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

    def closeEvent(self, event) -> None:
        self.analysis.cancel()
        event.accept()