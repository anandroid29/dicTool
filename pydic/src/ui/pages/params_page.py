"""
params_page.py — Step 3: DIC parameters with live preview.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox,
    QFrame, QGridLayout, QSizePolicy,
)

from src.ui.components import FooterButton

if TYPE_CHECKING:
    from src.ui.wizard import Wizard

from src.ui.image_canvas import ImageCanvas

_C_SURFACE = "#0e1c2e"
_C_CARD    = "#132035"
_C_BORDER  = "#1e3a5a"
_C_ACCENT  = "#3b82f6"
_C_TEXT    = "#e2e8f0"
_C_TEXT2   = "#94a3b8"
_C_TEXT3   = "#475569"
_C_SUCCESS = "#10b981"


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{_C_TEXT3}; font-size:9px; font-weight:700; "
        f"text-transform:uppercase; letter-spacing:0.8px;"
    )
    return lbl


def _param_row(label: str, tooltip: str, widget: QWidget, unit: str = "") -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(12)
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:12px;")
    lbl.setToolTip(tooltip)
    lbl.setFixedWidth(140)
    row.addWidget(lbl)
    row.addWidget(widget)
    if unit:
        u = QLabel(unit)
        u.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        row.addWidget(u)
    row.addStretch()
    return row


class ParamsPage(QWidget):
    """Step 3 — set DIC parameters."""

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

        back = QPushButton("← Back")
        back.setFixedWidth(90)
        back.clicked.connect(self._wizard.go_roi)
        top_lay.addWidget(back)

        title = QLabel("Step 3  ·  Analysis Parameters")
        title.setStyleSheet(f"color:{_C_TEXT}; font-size:13px; font-weight:600;")
        top_lay.addWidget(title)
        top_lay.addStretch()

        root.addWidget(top)

        # ── Body ──────────────────────────────────────────────────────
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        # Left: canvas preview
        self._canvas = ImageCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        body_lay.addWidget(self._canvas, 1)

        # Right: parameters panel
        right = QWidget()
        right.setFixedWidth(340)
        right.setStyleSheet(f"background:{_C_SURFACE}; border-left:1px solid {_C_BORDER};")
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(24, 28, 24, 28)
        right_lay.setSpacing(20)

        # -- Subset --
        right_lay.addWidget(_section_label("Subset"))

        def spin(lo, hi, val, step=1):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setFixedWidth(80)
            return s

        def dspin(lo, hi, val, step=0.05, dec=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(dec)
            s.setFixedWidth(80)
            return s

        params = self._wizard.analysis.params

        self._sp_radius = spin(5, 200, params.subset_radius, 1)
        self._sp_radius.valueChanged.connect(self._on_param_changed)
        right_lay.addLayout(_param_row(
            "Subset radius", "Half-size of the correlation window in pixels.\n"
            "Larger = more robust but less spatial resolution.",
            self._sp_radius, "px"))

        self._sp_spacing = spin(1, 50, params.subset_spacing, 1)
        self._sp_spacing.valueChanged.connect(self._on_param_changed)
        right_lay.addLayout(_param_row(
            "Grid spacing", "Distance between subset centres.\n"
            "Smaller = denser result grid, longer analysis time.",
            self._sp_spacing, "px"))

        right_lay.addWidget(self._separator())
        right_lay.addWidget(_section_label("Strain"))

        self._sp_strain = spin(3, 100, params.strain_window, 1)
        right_lay.addLayout(_param_row(
            "Strain window", "Neighbourhood (in subset units) used for\n"
            "the least-squares plane fit when computing strains.",
            self._sp_strain, "subsets"))

        right_lay.addWidget(self._separator())
        right_lay.addWidget(_section_label("Optimizer"))

        # --- PASTE THIS NEW BLOCK RIGHT HERE ---
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setStyleSheet(
            "background: #132035; color: #94a3b8; border: 1px solid #1e3a5a; "
            "padding: 6px; border-radius: 4px; margin-top: 10px;"
        )
        reset_btn.clicked.connect(self._reset_defaults)
        right_lay.addWidget(reset_btn)
        # ---------------------------------------

        self._sp_maxiter = spin(5, 200, params.max_iter, 5)
        right_lay.addLayout(_param_row(
            "Max iterations", "Maximum IC-GN iterations per subset.",
            self._sp_maxiter, ""))

        self._sp_tol = dspin(1e-8, 0.01, params.conv_tol, 1e-7, 8)
        right_lay.addLayout(_param_row(
            "Convergence tol", "Weighted-norm convergence threshold.",
            self._sp_tol, ""))

        self._sp_cutoff = dspin(0.0, 4.0, params.corr_cutoff, 0.05, 2)
        right_lay.addLayout(_param_row(
            "Correlation cutoff", "Discard subsets with ZNSSD above this\n"
            "(lower = stricter; 0.8 is a good starting value).",
            self._sp_cutoff, ""))

        right_lay.addWidget(self._separator())
        right_lay.addWidget(_section_label("Search"))

        self._sp_search = spin(5, 500, params.search_radius, 10)
        right_lay.addLayout(_param_row(
            "NCC search radius", "Integer-pixel initial-guess search radius.",
            self._sp_search, "px"))

        right_lay.addStretch()

        # Grid preview label
        self._grid_lbl = QLabel("")
        self._grid_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        self._grid_lbl.setWordWrap(True)
        right_lay.addWidget(self._grid_lbl)

        body_lay.addWidget(right)
        root.addWidget(body, 1)

        # ── Footer ────────────────────────────────────────────────────
        footer = QWidget()
        footer.setFixedHeight(58)
        footer.setStyleSheet(f"background:{_C_SURFACE}; border-top:1px solid {_C_BORDER};")
        foot_lay = QHBoxLayout(footer)
        foot_lay.setContentsMargins(20, 0, 20, 0)
        foot_lay.addStretch()

        self._run_btn = FooterButton("▶  Run Analysis")
        self._run_btn.setProperty("class", "run")
        self._run_btn.setFixedHeight(38)
        self._run_btn.setMinimumWidth(160)
        self._run_btn.clicked.connect(self._start_analysis)
        foot_lay.addWidget(self._run_btn)

        root.addWidget(footer)

    # ------------------------------------------------------------------
    def on_enter(self) -> None:
        img = self._wizard.analysis.reference_image
        if img is not None:
            # Only set the image if it's new to preserve pan/zoom
            if self._canvas._image_arr is not img:
                self._canvas.set_image(img)
                self._canvas.zoom_fit()

        # Restore Mask
        mask = self._wizard.analysis.roi_mask
        if mask is not None:
            self._canvas.set_roi_mask(mask)

        # Restore Seed
        if getattr(self._wizard, "seed_xy", None) is not None:
            self._canvas.set_seed_xy(self._wizard.seed_xy)

        # Trigger parameter refresh to draw the subset radius
        self._on_param_changed()

    def _on_param_changed(self) -> None:
        p = self._wizard.analysis.params
        p.subset_radius  = self._sp_radius.value()
        p.subset_spacing = self._sp_spacing.value()
        p.strain_window  = self._sp_strain.value()
        p.max_iter       = self._sp_maxiter.value()
        p.conv_tol       = self._sp_tol.value()
        p.corr_cutoff    = self._sp_cutoff.value()
        p.search_radius  = self._sp_search.value()

        if hasattr(self._canvas, "set_subset_radius"):
            self._canvas.set_subset_radius(p.subset_radius)

        # Count estimated subsets
        img = self._wizard.analysis.reference_image
        mask = self._wizard.analysis.roi_mask
        if img is not None and mask is not None:
            H, W = img.shape
            r, s = p.subset_radius, p.subset_spacing
            ys = np.arange(r, H - r, s)
            xs = np.arange(r, W - r, s)
            cnt = sum(1 for y in ys for x in xs
                      if y < mask.shape[0] and x < mask.shape[1] and mask[y, x])
            self._grid_lbl.setText(
                f"≈ {cnt:,} subsets will be analysed\n"
                f"({W}×{H} image, {s} px spacing)"
            )

    def _start_analysis(self) -> None:
        self._wizard.go_analysis()

    def _separator(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
        return f

    def _reset_defaults(self) -> None:
        from src.core.rg_dic import DICParams
        from PyQt6.QtWidgets import QMessageBox

        # 1. Reset core settings and overwrite JSON
        self._wizard.analysis.params = DICParams()
        self._wizard.analysis.save_settings()
        p = self._wizard.analysis.params

        # 2. Update all spinboxes silently
        for sb, val in [
            (self._sp_radius, p.subset_radius),
            (self._sp_spacing, p.subset_spacing),
            (self._sp_strain, p.strain_window),
            (self._sp_maxiter, p.max_iter),
            (self._sp_search, p.search_radius),
            (self._sp_tol, p.conv_tol),
            (self._sp_cutoff, p.corr_cutoff)
        ]:
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)

        self._on_param_changed()  # Update the subset counter text

        QMessageBox.information(
            self, "Defaults Reset",
            f"Parameters reset to defaults.\n\n"
            f"Radius: {p.subset_radius}\n"
            f"Spacing: {p.subset_spacing}\n"
            f"Strain Window: {p.strain_window}"
        )
