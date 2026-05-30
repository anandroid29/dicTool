"""
main_window.py
==============
"""

from __future__ import annotations

import traceback
from typing import Optional

import numpy as np
from PyQt6.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QWidget,
)

from src.core.analysis import DICAnalysis
from src.ui.image_canvas import ImageCanvas, ROITool
from src.ui.param_panel import AnalysisWorker, ParamPanel
from src.ui.results_panel import ResultsPanel
from src.ui.theme import STYLESHEET as DARK_STYLESHEET


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyDIC — Digital Image Correlation")
        self.setMinimumSize(1280, 760)
        self.resize(1600, 960)

        self._analysis = DICAnalysis()
        self._worker: Optional[AnalysisWorker] = None
        self._thread: Optional[QThread] = None

        self._build_ui()
        self._build_menu()
        self._build_status_bar()
        self._wire_signals()

        # Apply dark stylesheet
        self.setStyleSheet(DARK_STYLESHEET)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Splitter ───────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(
            "QSplitter::handle { background:#30363d; }"
        )
        self.setCentralWidget(splitter)

        # Left: param panel
        self._param_panel = ParamPanel(self._analysis)
        self._param_panel.setMinimumWidth(280)
        self._param_panel.setMaximumWidth(360)
        splitter.addWidget(self._param_panel)

        # Centre: image canvas
        self._canvas = ImageCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        splitter.addWidget(self._canvas)

        # Right: results panel
        self._results_panel = ResultsPanel(self._analysis)
        self._results_panel.setMinimumWidth(260)
        self._results_panel.setMaximumWidth(380)
        splitter.addWidget(self._results_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 960, 320])

    def _build_menu(self) -> None:
        menu = self.menuBar()
        menu.setStyleSheet(
            "QMenuBar { background:#0d1117; color:#e6edf3; "
            "border-bottom:1px solid #21262d; font-size:12px; } "
            "QMenuBar::item:selected { background:#2f81f7; border-radius:4px; } "
            "QMenu { background:#161b22; color:#e6edf3; border:1px solid #30363d; } "
            "QMenu::item:selected { background:#2f81f7; } "
            "QMenu::separator { background:#21262d; height:1px; margin:4px 0; }"
        )

        # File
        file_menu = menu.addMenu("&File")

        act_new = QAction("&New Session", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._new_session)
        file_menu.addAction(act_new)

        file_menu.addSeparator()

        act_export_csv = QAction("Export CSV…", self)
        act_export_csv.triggered.connect(self._results_panel._export_csv)
        file_menu.addAction(act_export_csv)

        act_export_h5 = QAction("Export HDF5…", self)
        act_export_h5.triggered.connect(self._results_panel._export_hdf5)
        file_menu.addAction(act_export_h5)

        file_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(QApplication.quit)
        file_menu.addAction(act_quit)

        # View
        view_menu = menu.addMenu("&View")

        act_zoom_fit = QAction("Zoom to &Fit", self)
        act_zoom_fit.setShortcut("Ctrl+0")
        act_zoom_fit.triggered.connect(self._canvas.zoom_fit)
        view_menu.addAction(act_zoom_fit)

        act_zoom_1 = QAction("&Actual Size (100%)", self)
        act_zoom_1.setShortcut("Ctrl+1")
        act_zoom_1.triggered.connect(lambda: self._canvas.set_zoom(1.0))
        view_menu.addAction(act_zoom_1)

        view_menu.addSeparator()

        act_clear_roi = QAction("&Clear ROI", self)
        act_clear_roi.setShortcut("Escape")
        act_clear_roi.triggered.connect(self._canvas.clear_roi)
        view_menu.addAction(act_clear_roi)

        # Analysis
        analysis_menu = menu.addMenu("&Analysis")

        act_run = QAction("&Run DIC", self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(self._param_panel._on_run)
        analysis_menu.addAction(act_run)

        act_cancel = QAction("&Cancel", self)
        act_cancel.setShortcut("Ctrl+.")
        act_cancel.triggered.connect(self._param_panel._on_cancel)
        analysis_menu.addAction(act_cancel)

        # Help
        help_menu = menu.addMenu("&Help")

        act_about = QAction("&About PyDIC", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        sb.setStyleSheet(
            "QStatusBar { background:#0d1117; color:#8b949e; "
            "border-top:1px solid #21262d; font-size:11px; }"
        )
        self.setStatusBar(sb)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        sb.addWidget(self._status_lbl)

        self._status_progress = QProgressBar()
        self._status_progress.setRange(0, 1000)
        self._status_progress.setValue(0)
        self._status_progress.setFixedWidth(180)
        self._status_progress.setVisible(False)
        self._status_progress.setStyleSheet(
            "QProgressBar { background:#21262d; border:1px solid #30363d; "
            "border-radius:4px; height:8px; text-align:center; color:transparent; } "
            "QProgressBar::chunk { background:#2f81f7; border-radius:4px; }"
        )
        sb.addPermanentWidget(self._status_progress)

        self._cursor_lbl = QLabel("")
        self._cursor_lbl.setStyleSheet(
            "color:#8b949e; font-size:11px; "
            "font-family:'Fira Code','Consolas',monospace;"
        )
        sb.addPermanentWidget(self._cursor_lbl)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        pp = self._param_panel
        cv = self._canvas
        rp = self._results_panel

        # Param panel → canvas
        pp.ref_loaded.connect(self._on_ref_loaded)
        pp.def_loaded.connect(self._on_def_loaded)
        pp.tool_changed.connect(cv.set_roi_tool)
        pp.clear_roi.connect(cv.clear_roi)
        pp.run_requested.connect(self._start_analysis)
        pp.frame_changed.connect(self._on_param_frame_changed)        # Canvas → param panel
        cv.roi_changed.connect(self._on_roi_changed)
        cv.seed_placed.connect(self._on_seed_placed)
        cv.cursor_moved.connect(self._on_cursor_moved)

        # Results panel → canvas
        rp.field_changed.connect(self._update_overlay)
        rp.frame_changed.connect(self._on_results_frame_changed)
        rp.colormap_changed.connect(self._on_cmap_changed)

    # ------------------------------------------------------------------
    # Slots: image loading
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_ref_loaded(self, path: str) -> None:
        img = self._analysis.reference_image
        if img is not None:
            self._canvas.set_base_image(img)
            self._canvas.zoom_fit()
            self._status_lbl.setText(
                f"Reference loaded: {_basename(path)}  "
                f"({img.shape[1]}×{img.shape[0]} px)"
            )

    @pyqtSlot(list)
    def _on_def_loaded(self, paths: list) -> None:
        n = len(self._analysis.deformed_paths)
        self._status_lbl.setText(f"{n} deformed image(s) loaded.")

    # ------------------------------------------------------------------
    # Slots: ROI
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def _on_roi_changed(self, mask: np.ndarray) -> None:
        """Canvas finished drawing an ROI; push it to the analysis."""
        self._analysis.set_roi_mask(mask)
        self._param_panel.set_roi_status(mask)
        n = int(mask.sum())
        self._status_lbl.setText(f"ROI updated: {n:,} pixels selected.")

    @pyqtSlot(int, int)
    def _on_seed_placed(self, x: int, y: int) -> None:
        self._analysis.params.seed_x = x
        self._analysis.params.seed_y = y
        self._param_panel.set_seed_display(x, y)
        self._status_lbl.setText(f"Seed point: ({x}, {y})")

    # ------------------------------------------------------------------
    # Slots: analysis
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _start_analysis(self) -> None:
        if self._analysis.reference_image is None:
            QMessageBox.warning(self, "PyDIC", "Please load a reference image first.")
            return
        if not self._analysis.deformed_paths:
            QMessageBox.warning(self, "PyDIC",
                                "Please add at least one deformed image.")
            return
        if self._analysis.roi_mask is None:
            reply = QMessageBox.question(
                self, "PyDIC",
                "No ROI defined. Use the full image as ROI?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._analysis.set_full_roi()
                self._param_panel.set_roi_status(self._analysis.roi_mask)
            else:
                return

        self._param_panel.start_analysis()
        self._status_progress.setVisible(True)
        self._status_progress.setValue(0)
        self._status_lbl.setText("Running DIC…")

        # Spin up worker thread
        self._thread = QThread()
        self._worker = AnalysisWorker(self._analysis)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    @pyqtSlot(float, str)
    def _on_progress(self, frac: float, msg: str) -> None:
        self._param_panel.update_progress(frac, msg)
        self._status_progress.setValue(int(frac * 1000))
        self._status_lbl.setText(msg)

    @pyqtSlot()
    def _on_analysis_finished(self) -> None:
        self._param_panel.finish_analysis()
        self._status_progress.setVisible(False)
        n = len(self._analysis.results)
        self._status_lbl.setText(f"Analysis complete — {n} frame(s) processed.")
        self._results_panel.refresh_after_analysis()
        self._update_overlay(self._results_panel.current_field_key())

    @pyqtSlot(str)
    def _on_analysis_error(self, msg: str) -> None:
        self._param_panel.finish_analysis()
        self._status_progress.setVisible(False)
        self._status_lbl.setText("Analysis failed.")
        QMessageBox.critical(self, "Analysis Error", msg)

    # ------------------------------------------------------------------
    # Slots: results display
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_results_frame_changed(self, index: int) -> None:
        self._update_overlay(self._results_panel.current_field_key(), index)

    @pyqtSlot(int)
    def _on_param_frame_changed(self, index: int) -> None:
        """Sync deformed image preview in canvas."""
        if index < len(self._analysis.deformed_paths):
            import cv2
            try:
                img = cv2.imread(self._analysis.deformed_paths[index],
                                 cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self._canvas.set_base_image(img.astype(np.float64) / 255.0)
            except Exception:
                pass

    @pyqtSlot(str, object, object)
    def _on_cmap_changed(self, cmap: str, vmin, vmax) -> None:
        self._update_overlay(
            self._results_panel.current_field_key(),
            frame_index=None,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    def _update_overlay(
        self,
        field_key: str,
        frame_index: Optional[int] = None,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        if not self._analysis.results:
            return

        if frame_index is None:
            frame_index = min(
                self._results_panel._current_frame,
                len(self._analysis.results) - 1,
            )

        result = self._analysis.results[frame_index]

        arr = getattr(result, field_key, None)

        if arr is None:
            return

        if cmap is None:
            cmap = self._results_panel.current_cmap()

        # Auto range if not provided
        if vmin is None or vmax is None:
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                return
            vmin = float(valid.min())
            vmax = float(valid.max())
            if self._results_panel.is_symmetric():
                lim = max(abs(vmin), abs(vmax))
                vmin, vmax = -lim, lim

        if vmin == vmax:
            vmax = vmin + 1e-12

        n_colors = self._results_panel.current_cmap_n()

        try:
            import matplotlib.cm as cm
            import matplotlib.colors as mc
            norm = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
            cmap_obj = cm.get_cmap(cmap, n_colors)
            rgba = cmap_obj(norm(arr), bytes=True)  # H×W×4 uint8
            # Set alpha=0 for NaN/invalid pixels
            invalid = ~np.isfinite(arr)
            rgba[invalid, 3] = 0
            self._canvas.set_result_overlay_rgba(rgba)
        except Exception as exc:
            self._status_lbl.setText(f"Overlay error: {exc}")

    # ------------------------------------------------------------------
    # Slots: canvas cursor
    # ------------------------------------------------------------------

    @pyqtSlot(int, int, float)
    def _on_cursor_moved(self, px: int, py: int, value: float) -> None:
        if np.isfinite(value):
            self._cursor_lbl.setText(f"({px}, {py})  val={value:.5g}")
        else:
            self._cursor_lbl.setText(f"({px}, {py})")

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def _new_session(self) -> None:
        reply = QMessageBox.question(
            self, "New Session",
            "Start a new session? All current results will be cleared.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._analysis = DICAnalysis()
            self._param_panel._analysis = self._analysis
            self._results_panel._analysis = self._analysis
            self._canvas.clear_roi()
            self._canvas.set_result_overlay_rgba(None)
            self._param_panel.step_indicator.set_step(0)
            self._status_lbl.setText("New session started.")

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About PyDIC",
            "<h3>PyDIC — Digital Image Correlation</h3>"
            "<p>A Python reimplementation of the Ncorr DIC algorithm "
            "(Blaber, Adair &amp; Antoniou, 2015) with a modern, "
            "professional user interface.</p>"
            "<p><b>Core algorithms:</b><br>"
            "• Biquintic B-spline sub-pixel interpolation<br>"
            "• Inverse Compositional Gauss-Newton (IC-GN) optimiser<br>"
            "• Reliability-Guided DIC (RG-DIC) propagation<br>"
            "• Green–Lagrangian strain computation</p>"
            "<p><b>Stack:</b> Python · NumPy · SciPy · OpenCV · PyQt6 · Matplotlib</p>"
            "<p style='color:#8b949e; font-size:11px;'>"
            "Reference: Blaber J, Adair B, Antoniou A (2015). "
            "Ncorr: Open-Source 2D Digital Image Correlation Matlab Software. "
            "<i>Experimental Mechanics</i>, 55(6), 1105–1122.</p>",
        )

    # ------------------------------------------------------------------
    # Close handler
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._thread and self._thread.isRunning():
            self._analysis.cancel()
            self._thread.quit()
            self._thread.wait(3000)
        event.accept()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _basename(path: str) -> str:
    import os
    return os.path.basename(path)
