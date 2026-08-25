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
from src.ui.pages.dynamic_roi_page import DynamicROIPage
from src.ui.pages.params_page import ParamsPage
from src.ui.pages.analysis_page import AnalysisPage
from src.ui.pages.results_page import ResultsPage

# Dynamic ROI sits between the static ROI and the numeric parameters: it is a
# masking decision, so it belongs with the other masking step rather than as a
# bare dropdown among the solver settings.
_STEPS = ["Import", "ROI", "Dynamic ROI", "Parameters", "Analysis", "Results"]


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
        self.use_gpu = bool(getattr(self.analysis, "prefer_gpu", True))

        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        # Settings repairs used to be a console print the user never saw, while
        # the stored value had been quietly producing bad results. Surface them.
        QTimer.singleShot(400, self._show_settings_notices)

    def _show_settings_notices(self) -> None:
        notices = getattr(self.analysis, "settings_notices", None)
        if not notices:
            return
        from PyQt6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Saved settings adjusted")
        box.setText("Some saved settings could not be used as stored and have "
                    "been corrected:")
        box.setInformativeText("\n\n".join(f"•  {n}" for n in notices))
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()
        notices.clear()

    def dynamic_roi_enabled(self) -> bool:
        return getattr(self.analysis.params, "dynamic_roi", "None") not in ("None", None, "")

    def go_after_roi(self) -> None:
        """Forward from the ROI step, skipping dynamic ROI when it is off."""
        if self.dynamic_roi_enabled():
            self.go_dynamic_roi()
        else:
            self.go_params()

    def go_before_params(self) -> None:
        """Back from the parameters step, skipping dynamic ROI when it is off."""
        if self.dynamic_roi_enabled():
            self.go_dynamic_roi()
        else:
            self.go_roi()

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
        self._dynroi = DynamicROIPage(self)
        self._params = ParamsPage(self)
        self._analysis = AnalysisPage(self)
        self._results = ResultsPage(self)

        for page in (self._welcome, self._roi, self._dynroi, self._params,
                     self._analysis, self._results):
            self._stack.addWidget(page)

        root.addWidget(self._stack, 1)

    def _go(self, idx: int, page_with_enter=None,
            run_leave_hook: bool = True) -> None:
        outgoing = self._stack.currentWidget()
        if outgoing is not None and outgoing is not page_with_enter:
            on_leave = getattr(outgoing, "on_leave", None)
            if run_leave_hook and callable(on_leave):
                on_leave()
            canvas = getattr(outgoing, "_canvas", None)
            release = getattr(canvas, "release_display_buffers", None)
            if callable(release):
                # The analysis model retains authoritative state; hidden pages
                # do not retain recreatable full-resolution arrays or pixmaps.
                release()

        # Clear the incoming page's stale visuals BEFORE it becomes visible.
        #
        # on_enter() is deferred by a timer because some of its work needs final
        # widget geometry (fit-to-window, colourbar sizing). But setCurrentIndex
        # shows the page immediately, so between the two the user gets at least
        # one painted frame of the PREVIOUS run's contents -- the remnant flash.
        # on_before_show() is the cheap, synchronous half: blank the canvas and
        # any stale labels now, then let on_enter() repopulate.
        if page_with_enter is not None and hasattr(page_with_enter, "on_before_show"):
            page_with_enter.on_before_show()

        self._stack.setCurrentIndex(idx)
        self._step_bar.set_step(idx)

        # Populate the incoming page before control returns to Qt's event loop.
        # The old 50 ms timer let the newly selected page paint once with stale
        # data, then visibly jump to the current session -- especially obvious
        # when navigating Back. Layout activation gives fit-to-window code its
        # final geometry without introducing that extra painted frame.
        page = self._stack.widget(idx)
        if page is not None and page.layout() is not None:
            page.layout().activate()
        if page_with_enter is not None and hasattr(page_with_enter, "on_enter"):
            page_with_enter.on_enter()

    def new_session(self) -> None:
        """
        Full reset. "New Session" previously only navigated back to the welcome
        page, leaving the finished DICAnalysis and every page's rendered state
        in place -- which is the other half of the remnant problem.
        """
        try:
            self.analysis.cancel()
        except Exception:
            pass
        self.analysis = DICAnalysis()
        self.seed_xy = None
        self.use_gpu = bool(getattr(self.analysis, "prefer_gpu", True))
        # These are concrete lifecycle operations, not optional plugin hooks.
        # Calling them explicitly makes ownership visible to readers and static
        # analysis, which previously misreported all three methods as dead.
        cleanup_actions = (
            self._results.reset_markers,
            self._dynroi.reset_page,
            self._analysis.on_before_show,
            self._dynroi.on_before_show,
            self._results.on_before_show,
        )
        for cleanup in cleanup_actions:
            try:
                cleanup()
            except Exception:
                pass
        # The analysis object was deliberately replaced above. Do not let an
        # outgoing page commit stale controls into the new session.
        self._go(0, self._welcome, run_leave_hook=False)

    def go_welcome(self) -> None:
        self._go(0, self._welcome)

    def go_roi(self) -> None:
        self._go(1, self._roi)

    def go_dynamic_roi(self) -> None:
        self._go(2, self._dynroi)

    def go_params(self) -> None:
        self._go(3, self._params)

    def go_analysis(self) -> None:
        self._go(4, self._analysis)

    def go_results(self) -> None:
        self._go(5, self._results)

    def closeEvent(self, event) -> None:
        self.analysis.cancel()
        # Give the native analysis worker a chance to observe cancellation and
        # finish before allowing its signal bridge/widget tree to be destroyed.
        if not self._analysis.shutdown():
            event.ignore()
            return
        event.accept()
