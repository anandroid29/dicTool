"""
roi_page.py — Step 2: Define region of interest on full-screen canvas.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QToolButton, QButtonGroup,
    QSizePolicy, QFileDialog, QMessageBox,
)
from src.ui.components import FooterButton

if TYPE_CHECKING:
    from src.ui.wizard import Wizard

from src.ui.image_canvas import ImageCanvas, ROITool

_C_BG      = "#08111d"
_C_SURFACE = "#0e1c2e"
_C_CARD    = "#132035"
_C_BORDER  = "#1e3a5a"
_C_ACCENT  = "#3b82f6"
_C_TEXT    = "#e2e8f0"
_C_TEXT2   = "#94a3b8"
_C_SUCCESS = "#10b981"


def _tool_btn(icon: str, tip: str) -> QToolButton:
    btn = QToolButton()
    btn.setText(icon)
    btn.setToolTip(tip)
    btn.setCheckable(True)
    btn.setFixedSize(44, 44)
    btn.setStyleSheet(
        f"QToolButton {{ background:{_C_CARD}; color:{_C_TEXT2}; "
        f"border:1px solid {_C_BORDER}; border-radius:8px; font-size:18px; }} "
        f"QToolButton:hover {{ background:{_C_BORDER}; color:{_C_TEXT}; }} "
        f"QToolButton:checked {{ background:{_C_ACCENT}; color:#fff; "
        f"border-color:{_C_ACCENT}; }} "
    )
    return btn


class ROIPage(QWidget):
    """Step 2 — draw ROI on the full reference image."""

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────
        top = QWidget()
        top.setFixedHeight(52)
        top.setStyleSheet(f"background:{_C_SURFACE}; border-bottom:1px solid {_C_BORDER};")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(20, 0, 20, 0)
        top_lay.setSpacing(16)

        back = QPushButton("← Back")
        back.setFixedWidth(90)
        back.clicked.connect(self._wizard.go_welcome)
        top_lay.addWidget(back)

        title = QLabel("Step 2  ·  Define Region of Interest")
        title.setStyleSheet(f"color:{_C_TEXT}; font-size:13px; font-weight:600;")
        top_lay.addWidget(title)

        top_lay.addStretch()

        self._roi_lbl = QLabel("No ROI drawn")
        self._roi_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        top_lay.addWidget(self._roi_lbl)

        root.addWidget(top)

        # ── Main area (tools + canvas) ────────────────────────────────
        main = QWidget()
        main_lay = QHBoxLayout(main)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # Left toolbar
        toolbar = QWidget()
        toolbar.setFixedWidth(64)
        toolbar.setStyleSheet(f"background:{_C_SURFACE}; border-right:1px solid {_C_BORDER};")
        tb_lay = QVBoxLayout(toolbar)
        tb_lay.setContentsMargins(10, 16, 10, 16)
        tb_lay.setSpacing(8)
        tb_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        tools = [
            ("▱", "Rectangle ROI", ROITool.RECTANGLE),
            ("○", "Ellipse / Circle ROI", ROITool.CIRCLE),
            ("⬡", "Polygon ROI (click to place, dbl-click to close)", ROITool.POLYGON),
            ("✕", "Erase from ROI", ROITool.ERASE),
        ]
        self._tool_btns = {}
        for icon, tip, tool in tools:
            btn = _tool_btn(icon, tip)
            self._tool_group.addButton(btn)
            btn.clicked.connect(lambda checked, t=tool: self._canvas.set_roi_tool(t))
            tb_lay.addWidget(btn)
            self._tool_btns[tool] = btn

        tb_lay.addSpacing(12)

        # Seed
        seed_lbl = QLabel("Seed")
        seed_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:9px; text-align:center;")
        seed_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tb_lay.addWidget(seed_lbl)

        self._seed_btn = _tool_btn("⦿", "Place seed point for RG-DIC propagation")
        self._seed_btn.clicked.connect(
            lambda: self._canvas.set_roi_tool(ROITool.NONE)  # canvas handles seed on click
        )
        tb_lay.addWidget(self._seed_btn)

        tb_lay.addStretch()

        # Clear button
        clr_btn = QPushButton("⟳")
        clr_btn.setToolTip("Clear ROI")
        clr_btn.setFixedSize(44, 36)
        clr_btn.clicked.connect(lambda: self._canvas.clear_roi())
        clr_btn.setStyleSheet(
            f"QPushButton {{ background:{_C_CARD}; color:{_C_TEXT2}; "
            f"border:1px solid {_C_BORDER}; border-radius:8px; font-size:16px; }} "
            f"QPushButton:hover {{ background:{_C_BORDER}; color:{_C_TEXT}; }}"
        )
        tb_lay.addWidget(clr_btn)

        main_lay.addWidget(toolbar)

        # Canvas
        self._canvas = ImageCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        self._canvas.roi_changed.connect(self._on_roi_changed)
        self._canvas.seed_placed.connect(self._on_seed_placed)
        main_lay.addWidget(self._canvas, 1)

        root.addWidget(main, 1)

        # ── Footer ────────────────────────────────────────────────────
        footer = QWidget()
        footer.setFixedHeight(58)
        footer.setStyleSheet(f"background:{_C_SURFACE}; border-top:1px solid {_C_BORDER};")
        foot_lay = QHBoxLayout(footer)
        foot_lay.setContentsMargins(20, 0, 20, 0)

        full_btn = QPushButton("Use Full Image")
        full_btn.setFixedWidth(130)
        full_btn.clicked.connect(self._use_full)
        foot_lay.addWidget(full_btn)

        load_btn = QPushButton("📁  Load ROI from File")
        load_btn.setFixedWidth(180)
        load_btn.setToolTip(
            "Load a pre-defined ROI mask from:\n"
            "  • PNG / TIF / JPG image  (white = ROI)\n"
            "  • NumPy .npy array\n"
            "  • Ncorr .mat / .h5 file"
        )
        load_btn.clicked.connect(self._load_roi_from_file)
        foot_lay.addWidget(load_btn)

        foot_lay.addStretch()

        self._seed_status = QLabel("No seed — will default to ROI centroid")
        self._seed_status.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        foot_lay.addWidget(self._seed_status)

        foot_lay.addSpacing(24)

        self._next_btn = FooterButton("Parameters  →")
        self._next_btn.setProperty("class", "accent")
        self._next_btn.setFixedHeight(36)
        self._next_btn.setMinimumWidth(150)
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._wizard.go_params)
        foot_lay.addWidget(self._next_btn)

        root.addWidget(footer)

    # ------------------------------------------------------------------
    def on_enter(self) -> None:
        """Called by wizard when this page becomes visible."""
        img = self._wizard.analysis.reference_image
        if img is not None:
            self._canvas.set_image(img)
            self._canvas.zoom_fit()

    def _on_roi_changed(self, mask: np.ndarray) -> None:
        self._wizard.analysis.set_roi_mask(mask)
        n = int(mask.sum())
        self._roi_lbl.setText(f"ROI: {n:,} px selected")
        self._roi_lbl.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")
        self._next_btn.setEnabled(n > 0)

    def _on_seed_placed(self, x: int, y: int) -> None:
        self._wizard.seed_xy = (x, y)
        self._seed_status.setText(f"Seed: ({x}, {y})")
        self._seed_status.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")

    def _use_full(self) -> None:
        ana = self._wizard.analysis
        ana.set_full_roi()
        # Show full-image mask on canvas
        if ana.roi_mask is not None:
            self._canvas._roi_mask = ana.roi_mask.copy()
            self._canvas.update()
            self._roi_lbl.setText(f"ROI: {int(ana.roi_mask.sum()):,} px (full image)")
            self._roi_lbl.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")
            self._next_btn.setEnabled(True)

    def _load_roi_from_file(self) -> None:
        """Load a pre-defined ROI mask from a file (image, npy, or Ncorr MAT)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROI Mask",
            "",
            "ROI files (*.png *.tif *.tiff *.jpg *.bmp *.npy *.mat *.h5 *.hdf5);;"
            "Image files (*.png *.tif *.tiff *.jpg *.bmp);;"
            "NumPy (*.npy);;"
            "Ncorr MAT / HDF5 (*.mat *.h5 *.hdf5);;"
            "All Files (*)"
        )
        if not path:
            return
        try:
            ana = self._wizard.analysis
            ana.set_roi_from_file(path)
            mask = ana.roi_mask
            if mask is not None:
                self._canvas._roi_mask = mask.copy()
                self._canvas.update()
                n = int(mask.sum())
                self._roi_lbl.setText(f"ROI: {n:,} px  (loaded from file)")
                self._roi_lbl.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")
                self._next_btn.setEnabled(n > 0)
        except Exception as exc:
            QMessageBox.critical(
                self, "ROI Load Error",
                f"Could not load ROI from:\n{path}\n\n{exc}"
            )
