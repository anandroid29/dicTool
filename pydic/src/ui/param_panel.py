"""
param_panel.py
--------------
Left-hand workflow + parameter panel.

Includes:
  • Step indicator (Images → ROI → Parameters → Analyse → Results)
  • Image file list (reference + deformed)
  • ROI tool selector
  • DIC parameter spinboxes
  • Run / Cancel button
  • Per-pair progress
"""

from __future__ import annotations

import os
from typing import Optional, List

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QListWidget, QListWidgetItem, QScrollArea, QFrame,
    QProgressBar, QFileDialog, QSizePolicy, QButtonGroup,
    QToolButton,
)

from .image_canvas import ROITool


# ---------------------------------------------------------------------------
# Step indicator widget
# ---------------------------------------------------------------------------

STEP_LABELS = ["① Images", "② ROI", "③ Parameters", "④ Analyse", "⑤ Results"]

class StepIndicator(QWidget):
    """Horizontal step indicator bar."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._labels: List[QLabel] = []
        for i, text in enumerate(STEP_LABELS):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setContentsMargins(4, 8, 4, 8)
            self._labels.append(lbl)
            layout.addWidget(lbl)
            if i < len(STEP_LABELS) - 1:
                arrow = QLabel("›")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                arrow.setStyleSheet("color: #30363d; font-size: 16px;")
                layout.addWidget(arrow)

        self.set_step(0)

    def set_step(self, step: int) -> None:
        for i, lbl in enumerate(self._labels):
            if i < step:
                lbl.setStyleSheet("color:#3fb950; font-weight:600; font-size:12px;")
            elif i == step:
                lbl.setStyleSheet(
                    "color:#e6edf3; font-weight:700; font-size:12px; "
                    "background:#1f6feb22; border-radius:4px;")
            else:
                lbl.setStyleSheet("color:#484f58; font-size:12px;")


# ---------------------------------------------------------------------------
# Worker thread for analysis
# ---------------------------------------------------------------------------

class AnalysisWorker(QObject):
    progress  = pyqtSignal(float, str)
    finished  = pyqtSignal()
    error     = pyqtSignal(str)

    def __init__(self, analysis) -> None:
        super().__init__()
        self._analysis = analysis

    def run(self) -> None:
        try:
            self._analysis.run(progress_cb=self._on_progress)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _on_progress(self, frac: float, msg: str) -> None:
        self.progress.emit(frac, msg)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class ParamPanel(QWidget):
    """
    Left-side panel: workflow steps, file management, ROI tools, DIC
    parameters, and the Run button.
    """

    # Signals to main window
    ref_loaded      = pyqtSignal(str)          # reference image path
    def_loaded      = pyqtSignal(list)         # list of deformed paths
    tool_changed    = pyqtSignal(ROITool)
    clear_roi       = pyqtSignal()
    run_requested   = pyqtSignal()
    cancel_requested = pyqtSignal()
    frame_changed   = pyqtSignal(int)          # index into deformed list

    def __init__(self, analysis, parent=None) -> None:
        super().__init__(parent)
        self._analysis = analysis
        self._worker: Optional[AnalysisWorker] = None
        self._thread: Optional[QThread]        = None
        self._running = False

        self.setFixedWidth(310)
        self.setMinimumHeight(400)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Step indicator
        self.step_indicator = StepIndicator()
        root.addWidget(self.step_indicator)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(0, 4, 0, 4)
        self._content_layout.setSpacing(8)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # Build sections
        self._build_images_section()
        self._build_roi_section()
        self._build_params_section()
        self._build_run_section()
        self._content_layout.addStretch(1)

        # Status / progress at bottom
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        root.addWidget(self._progress)

        self._status_lbl = QLabel("Ready.")
        self._status_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_images_section(self) -> None:
        grp = QGroupBox("IMAGES")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # ── Video import banner ────────────────────────────────────────────
        vid_row = QHBoxLayout()
        vid_lbl = QLabel("🎬  Have a video file?")
        vid_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        vid_row.addWidget(vid_lbl, 1)
        btn_vid = QPushButton("Import Video…")
        btn_vid.setFixedWidth(110)
        btn_vid.setToolTip(
            "Open a video file (MP4, AVI, MOV …) and extract frames "
            "automatically as PNG images for DIC analysis."
        )
        btn_vid.setStyleSheet(
            "QPushButton { background:#1f4e8c; color:#58a6ff; "
            "border:1px solid #2f81f7; border-radius:5px; "
            "font-size:11px; font-weight:600; padding:4px 8px; } "
            "QPushButton:hover { background:#2f81f7; color:#fff; } "
            "QPushButton:pressed { background:#1f6feb; }"
        )
        btn_vid.clicked.connect(self._import_video)
        vid_row.addWidget(btn_vid)
        lay.addLayout(vid_row)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("color:#21262d; margin:2px 0;")
        lay.addWidget(div)

        # ── Reference ────────────────────────────────────────────────────────
        ref_row = QHBoxLayout()
        self._ref_label = QLabel("No reference image")
        self._ref_label.setStyleSheet("color:#8b949e; font-size:11px;")
        self._ref_label.setWordWrap(True)
        ref_row.addWidget(self._ref_label, 1)
        btn_ref = QPushButton("Browse…")
        btn_ref.setFixedWidth(70)
        btn_ref.clicked.connect(self._pick_reference)
        ref_row.addWidget(btn_ref)
        lay.addLayout(ref_row)

        # ── Deformed list ────────────────────────────────────────────────
        self._def_list = QListWidget()
        self._def_list.setFixedHeight(100)
        self._def_list.currentRowChanged.connect(self.frame_changed)
        lay.addWidget(self._def_list)

        def_btns = QHBoxLayout()
        btn_add = QPushButton("Add Images…")
        btn_add.clicked.connect(self._pick_deformed)
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(self._clear_deformed)
        def_btns.addWidget(btn_add)
        def_btns.addWidget(btn_clr)
        lay.addLayout(def_btns)

        self._content_layout.addWidget(grp)

    def _build_roi_section(self) -> None:
        grp = QGroupBox("REGION OF INTEREST")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        hint = QLabel("Draw a ROI on the image.  Double-click to close a polygon.")
        hint.setStyleSheet("color:#8b949e; font-size:11px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        # Tool buttons
        tool_row = QHBoxLayout()
        tool_row.setSpacing(4)
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        tools = [
            ("⬠ Polygon",   ROITool.POLYGON),
            ("▭ Rectangle", ROITool.RECTANGLE),
            ("○ Circle",    ROITool.CIRCLE),
            ("✕ Erase",     ROITool.ERASE),
        ]
        for label, tool in tools:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, t=tool: self._select_tool(t))
            self._tool_group.addButton(btn)
            tool_row.addWidget(btn)
        lay.addLayout(tool_row)

        btn_row = QHBoxLayout()
        btn_clear = QPushButton("Clear ROI")
        btn_clear.clicked.connect(self.clear_roi)
        btn_full  = QPushButton("Full Image")
        btn_full.clicked.connect(self._use_full_roi)
        btn_row.addWidget(btn_clear)
        btn_row.addWidget(btn_full)
        lay.addLayout(btn_row)

        self._roi_status = QLabel("No ROI defined.")
        self._roi_status.setStyleSheet("color:#8b949e; font-size:11px;")
        lay.addWidget(self._roi_status)

        self._content_layout.addWidget(grp)

    def _build_params_section(self) -> None:
        grp = QGroupBox("DIC PARAMETERS")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        params_def = [
            ("Subset radius",  "px",  "subset_radius",  int,   5,  200,  20,
             "Radius of the circular correlation window"),
            ("Subset spacing", "px",  "subset_spacing", int,   1,   50,   5,
             "Centre-to-centre grid step between subsets"),
            ("Strain window",  "px",  "strain_window",  int,   2,   80,  10,
             "Half-width of the least-squares strain fitting window"),
            ("Max iterations", "",    "max_iter",       int,   5,  500,  50,
             "Maximum IC-GN iterations per subset"),
            ("Search radius",  "px",  "search_radius",  int,   5,  200,  30,
             "NCC search half-extent (seed point only)"),
        ]

        self._spinboxes = {}
        for label, unit, attr, typ, lo, hi, default, tip in params_def:
            row = QHBoxLayout()
            lbl = QLabel(label + (":" if not unit else f" ({unit}):"))
            lbl.setStyleSheet("color:#c9d1d9; font-size:12px;")
            lbl.setFixedWidth(130)
            lbl.setToolTip(tip)
            row.addWidget(lbl)
            if typ == int:
                sb = QSpinBox()
                sb.setRange(lo, hi)
                sb.setValue(default)
            else:
                sb = QDoubleSpinBox()
                sb.setRange(lo, hi)
                sb.setValue(default)
                sb.setDecimals(6)
            sb.setToolTip(tip)
            sb.valueChanged.connect(lambda val, a=attr: self._sync_param(a, val))
            row.addWidget(sb)
            self._spinboxes[attr] = sb
            lay.addLayout(row)

        # Convergence tolerance
        row = QHBoxLayout()
        lbl = QLabel("Conv. tolerance:")
        lbl.setStyleSheet("color:#c9d1d9; font-size:12px;")
        lbl.setFixedWidth(130)
        row.addWidget(lbl)
        self._tol_box = QDoubleSpinBox()
        self._tol_box.setRange(1e-8, 1e-1)
        self._tol_box.setValue(1e-4)
        self._tol_box.setDecimals(8)
        self._tol_box.valueChanged.connect(lambda v: self._sync_param("conv_tol", v))
        row.addWidget(self._tol_box)
        lay.addLayout(row)

        # Correlation cutoff
        row2 = QHBoxLayout()
        lbl2 = QLabel("Corr. cutoff:")
        lbl2.setStyleSheet("color:#c9d1d9; font-size:12px;")
        lbl2.setFixedWidth(130)
        row2.addWidget(lbl2)
        self._corr_box = QDoubleSpinBox()
        self._corr_box.setRange(0.01, 2.0)
        self._corr_box.setValue(0.8)
        self._corr_box.setDecimals(3)
        self._corr_box.valueChanged.connect(lambda v: self._sync_param("corr_cutoff", v))
        row2.addWidget(self._corr_box)
        lay.addLayout(row2)

        self._content_layout.addWidget(grp)

    def _build_run_section(self) -> None:
        grp = QGroupBox("ANALYSIS")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # Seed point info
        self._seed_label = QLabel("Seed: auto (ROI centroid)")
        self._seed_label.setStyleSheet("color:#8b949e; font-size:11px;")
        lay.addWidget(self._seed_label)

        # Right-click hint
        hint = QLabel("Right-click on image to set a custom seed point.")
        hint.setStyleSheet("color:#484f58; font-size:10px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self._run_btn = QPushButton("▶  Run Analysis")
        self._run_btn.setObjectName("run_btn")
        self._run_btn.setFixedHeight(40)
        self._run_btn.clicked.connect(self._on_run)
        lay.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("■  Cancel")
        self._cancel_btn.setObjectName("cancel_btn")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        lay.addWidget(self._cancel_btn)

        self._content_layout.addWidget(grp)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _import_video(self) -> None:
        """Open the video importer dialog and load extracted frames."""
        from .video_importer import VideoImporterDialog
        dlg = VideoImporterDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return

        paths = dlg.extracted_paths
        ref_idx = dlg.reference_index

        if not paths:
            return

        # The reference is the frame at ref_idx; deformed = the rest
        ref_path = paths[ref_idx]
        def_paths = [p for i, p in enumerate(paths) if i != ref_idx]

        # Set reference
        self._analysis.set_reference(ref_path)
        self._ref_label.setText(
            f"[video] frame {ref_idx:06d} — {os.path.basename(ref_path)}"
        )
        self._ref_label.setStyleSheet("color:#3fb950; font-size:11px;")
        self.ref_loaded.emit(ref_path)
        self.step_indicator.set_step(1)

        # Set deformed
        self._def_list.clear()
        self._analysis.clear_deformed()
        for p in def_paths:
            self._analysis.add_deformed(p)
            item = __import__("PyQt6.QtWidgets", fromlist=["QListWidgetItem"]).QListWidgetItem(
                os.path.basename(p)
            )
            item.setData(__import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.ItemDataRole.UserRole, p)
            self._def_list.addItem(item)

        self.def_loaded.emit(def_paths)

    def _pick_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Reference Image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.mat);;All Files (*)"
        )
        if path:
            self._ref_label.setText(os.path.basename(path))
            self._ref_label.setStyleSheet("color:#e6edf3; font-size:11px;")
            self.ref_loaded.emit(path)
            self.step_indicator.set_step(1)

    def _pick_deformed(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Deformed Images", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if paths:
            paths.sort()
            for p in paths:
                item = QListWidgetItem(os.path.basename(p))
                item.setData(Qt.ItemDataRole.UserRole, p)
                self._def_list.addItem(item)
            self.def_loaded.emit(paths)

    def _clear_deformed(self) -> None:
        self._def_list.clear()
        self._analysis.clear_deformed()

    def _select_tool(self, tool: ROITool) -> None:
        self.tool_changed.emit(tool)

    def _use_full_roi(self) -> None:
        self._analysis.set_full_roi()
        mask = self._analysis.roi_mask
        if mask is not None:
            n = int(mask.sum())
            self._roi_status.setText(f"Full image ROI ({mask.shape[1]}×{mask.shape[0]} px)")
            self.clear_roi.emit()  # signal canvas to reset drawing state

    def _sync_param(self, attr: str, val) -> None:
        setattr(self._analysis.params, attr, val)

    def _on_run(self) -> None:
        if self._running:
            return
        self.run_requested.emit()

    def _on_cancel(self) -> None:
        self._analysis.cancel()
        self._cancel_btn.setVisible(False)
        self._status_lbl.setText("Cancelling…")

    # ------------------------------------------------------------------
    # Public update methods called by main window
    # ------------------------------------------------------------------

    def set_seed_display(self, x: int, y: int) -> None:
        self._seed_label.setText(f"Seed: ({x}, {y})")
        self._seed_label.setStyleSheet("color:#2f81f7; font-size:11px;")

    def set_roi_status(self, mask: Optional[np.ndarray]) -> None:
        if mask is None:
            self._roi_status.setText("No ROI defined.")
            self._roi_status.setStyleSheet("color:#8b949e; font-size:11px;")
        else:
            n = int(mask.sum())
            pct = 100.0 * n / mask.size
            self._roi_status.setText(f"ROI: {n:,} pixels ({pct:.1f}%)")
            self._roi_status.setStyleSheet("color:#3fb950; font-size:11px;")
        self.step_indicator.set_step(2)

    def start_analysis(self) -> None:
        self._running = True
        self._run_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._progress.setValue(0)
        self.step_indicator.set_step(3)

    def finish_analysis(self) -> None:
        self._running = False
        self._run_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._progress.setValue(1000)
        self.step_indicator.set_step(4)

    def update_progress(self, frac: float, msg: str) -> None:
        self._progress.setValue(int(frac * 1000))
        self._status_lbl.setText(msg)
