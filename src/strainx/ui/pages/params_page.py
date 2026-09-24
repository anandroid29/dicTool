"""
params_page.py — Step 4: DIC parameters with live preview.
"""
from __future__ import annotations
import importlib.util
from typing import TYPE_CHECKING
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QDoubleSpinBox,
    QFrame, QGridLayout, QSizePolicy, QCheckBox, QComboBox, QScrollArea,
    QStackedWidget,
)

from strainx.ui.components import FooterButton

if TYPE_CHECKING:
    from strainx.ui.wizard import Wizard

from strainx.ui.image_canvas import ImageCanvas

# Palette comes from the single source of truth in theme.py. These were
# duplicated literals, which is why re-theming previously left pages behind.
from strainx.ui.theme import C_ACCENT, C_BORDER, C_CARD, C_SUCCESS, C_SURFACE, C_TEXT, C_TEXT2, C_TEXT3

_C_ACCENT = C_ACCENT
_C_BORDER = C_BORDER
_C_CARD = C_CARD
_C_SUCCESS = C_SUCCESS
_C_SURFACE = C_SURFACE
_C_TEXT = C_TEXT
_C_TEXT2 = C_TEXT2
_C_TEXT3 = C_TEXT3


from strainx.core.cuda_native import native_cuda_library_present

# Library presence is a filesystem-only UI check. The analysis worker performs
# authoritative driver/device validation when CUDA execution is chosen.
_HAS_GPU = native_cuda_library_present()



def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{_C_TEXT3}; font-size:9px; font-weight:700; "
        f"text-transform:uppercase; letter-spacing:0.8px;"
    )
    return lbl


# Panel and control widths. The label column previously got 140 px against a
# 340 px fixed panel (292 px usable after margins), which left names like
# "Correlation cutoff" elided and every spin box pinned at its 80 px minimum.
_PANEL_W  = 620
_LABEL_W  = 145
_FIELD_W  = 104
_UNIT_W   = 50


def _param_row(label: str, tooltip: str, widget: QWidget, unit: str = "") -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(10)
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:12px;")
    lbl.setToolTip(tooltip)
    lbl.setFixedWidth(_LABEL_W)
    row.addWidget(lbl)
    # Also put the tooltip on the control: hovering the thing you are about to
    # change is the natural gesture, and the label-only tooltip was easy to miss.
    widget.setToolTip(tooltip)
    row.addWidget(widget)
    u = QLabel(unit)
    u.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
    u.setFixedWidth(_UNIT_W)
    row.addWidget(u)
    row.addStretch()
    return row


class _RangeEditor(QWidget):
    """Compact inclusive From / To / Step integer-range editor."""

    changed = pyqtSignal()

    def __init__(self, minimum: int, maximum: int,
                 start: int, stop: int, step: int = 1, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(7)
        layout.setVerticalSpacing(2)
        self._spins = []
        for column, (caption, value) in enumerate(
                (("FROM", start), ("TO", stop), ("STEP", step))):
            label = QLabel(caption)
            label.setStyleSheet(
                f"color:{_C_TEXT3};font-size:9px;font-weight:700;")
            layout.addWidget(label, 0, column)
            spin = QSpinBox()
            spin.setRange(minimum if column < 2 else 1, maximum)
            spin.setValue(value)
            spin.setFixedWidth(84)
            spin.valueChanged.connect(self._emit_normalized)
            layout.addWidget(spin, 1, column)
            self._spins.append(spin)

    def _emit_normalized(self) -> None:
        start, stop, _step = self.values()
        if stop < start:
            sender = self.sender()
            target = self._spins[0] if sender is self._spins[1] else self._spins[1]
            target.blockSignals(True)
            target.setValue(stop if sender is self._spins[1] else start)
            target.blockSignals(False)
        self.changed.emit()

    def values(self) -> tuple[int, int, int]:
        return tuple(spin.value() for spin in self._spins)

    def set_values(self, values: tuple[int, int, int]) -> None:
        for spin, value in zip(self._spins, values):
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)

    def expanded(self) -> tuple[int, ...]:
        start, stop, step = self.values()
        return tuple(range(start, stop + 1, max(1, step)))


class _ModeParameterField(QStackedWidget):
    """Single-value field in normal mode, range field in parametric mode."""

    def __init__(self, single: QWidget, range_editor: _RangeEditor, parent=None):
        super().__init__(parent)
        single_host = QWidget()
        layout = QHBoxLayout(single_host)
        layout.setContentsMargins(0, 9, 0, 0)
        layout.addWidget(single)
        layout.addStretch()
        self.addWidget(single_host)
        self.addWidget(range_editor)
        self.setFixedWidth(270)
        self.setFixedHeight(50)


class ParamsPage(QWidget):
    """Step 4 — set DIC parameters."""

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._preview_roi_mask = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────
        top = QWidget()
        top.setObjectName("paramsTopBar")
        top.setFixedHeight(52)
        top.setStyleSheet(
            f"QWidget#paramsTopBar{{background:{_C_SURFACE};"
            f"border-bottom:1px solid {_C_BORDER};}}")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(20, 0, 20, 0)

        back = QPushButton("← Back")
        back.setFixedWidth(90)
        back.clicked.connect(self._wizard.go_before_params)
        top_lay.addWidget(back)

        title = QLabel("Step 4  ·  Analysis parameters")
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
        right.setObjectName("paramsSidePanel")
        right.setMinimumWidth(_PANEL_W - 4)
        right.setStyleSheet(
            f"QWidget#paramsSidePanel {{ background:{_C_SURFACE}; "
            f"border-left:1px solid {_C_BORDER}; }}")
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(22, 24, 22, 24)
        right_lay.setSpacing(16)

        # -- Subset --
        right_lay.addWidget(_section_label("Subset"))

        def spin(lo, hi, val, step=1):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setFixedWidth(_FIELD_W)
            return s

        def dspin(lo, hi, val, step=0.05, dec=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(dec)
            s.setFixedWidth(_FIELD_W)
            return s

        params = self._wizard.analysis.params

        self._sp_radius = spin(5, 200, params.subset_radius, 1)
        self._sp_radius.valueChanged.connect(self._on_param_changed)
        self._range_radius = _RangeEditor(5, 200, 5, 12, 1)
        self._range_radius.changed.connect(self._on_param_changed)
        self._field_radius = _ModeParameterField(
            self._sp_radius, self._range_radius)
        right_lay.addLayout(_param_row(
            "Subset radius", "Half-size of the correlation window in pixels.\n"
            "Larger = more robust but less spatial resolution.",
            self._field_radius, "px"))

        self._sp_spacing = spin(1, 50, params.subset_spacing, 1)
        self._sp_spacing.valueChanged.connect(self._on_param_changed)
        self._range_spacing = _RangeEditor(1, 50, 1, 5, 1)
        self._range_spacing.changed.connect(self._on_param_changed)
        self._field_spacing = _ModeParameterField(
            self._sp_spacing, self._range_spacing)
        right_lay.addLayout(_param_row(
            "Grid spacing", "Distance between subset centres.\n"
            "Smaller = denser result grid, longer analysis time.",
            self._field_spacing, "px"))

        # Dynamic ROI now has its own step, where the threshold and the manual
        # include/exclude regions can be seen against the reference frame. Keep
        # a read-only summary here so the setting is still discoverable from the
        # page where the rest of the analysis is configured.
        self._dyn_btn = QPushButton("Edit…")
        self._dyn_btn.setFixedWidth(_FIELD_W + _UNIT_W)
        self._dyn_btn.setFixedHeight(28)
        self._dyn_btn.clicked.connect(self._wizard.go_dynamic_roi)
        right_lay.addLayout(_param_row(
            "Dynamic ROI", "Per-frame texture masking, configured in Step 3.\n"
            "Useful for cutting experiments where material is removed.",
            self._dyn_btn, ""))
        self._dyn_lbl = QLabel("")
        self._dyn_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        self._dyn_lbl.setWordWrap(True)
        right_lay.addWidget(self._dyn_lbl)

        right_lay.addWidget(self._separator())
        right_lay.addWidget(_section_label("Strain"))

        # Units are PIXELS, not subsets -- the label used to say "subsets",
        # which invites values far too small to work. The plane fit only sees
        # correlation grid points, spaced `subset spacing` apart, so a window of
        # w px contains [2*floor(w/spacing) + 1]^2 nominal samples; fewer than
        # 3 points per axis cannot supply the six valid samples needed by the
        # plane fit and can leave every strain field empty.
        self._sp_strain = spin(3, 200, params.strain_window, 1)
        self._sp_strain.valueChanged.connect(self._on_param_changed)
        # Some platform styles defer valueChanged until the editor commits.
        # Validation should still follow valid text while it is being typed.
        self._sp_strain.lineEdit().textEdited.connect(
            self._on_strain_text_edited)
        self._range_strain = _RangeEditor(1, 200, 1, 15, 1)
        self._range_strain.changed.connect(self._on_param_changed)
        self._field_strain = _ModeParameterField(
            self._sp_strain, self._range_strain)
        right_lay.addLayout(_param_row(
            "Cached strain windows", "A small set of half-widths in PIXELS used for\n"
            "the least-squares plane fit when computing strains.\n"
            "Effective strain rate and accumulated effective strain are cached\n"
            "during the sweep so the results viewer never fits them on demand.\n"
            "Must cover at least 3 grid points across, i.e. keep the radius\n"
            "at least equal to the subset spacing, or strains may be empty.",
            self._field_strain, "px"))

        self._sp_temporal = spin(1, 10000, 1, 1)
        self._range_temporal = _RangeEditor(1, 10000, 1, 1, 1)
        self._range_temporal.changed.connect(self._on_param_changed)
        self._field_temporal = _ModeParameterField(
            self._sp_temporal, self._range_temporal)
        temporal_row = _param_row(
            "Temporal averaging", "Frame intervals in the motion-aware temporal window. "
            "A span of 1 uses each frame directly.", self._field_temporal,
            "frames")
        self._temporal_row_widgets = [
            temporal_row.itemAt(i).widget()
            for i in range(temporal_row.count())
            if temporal_row.itemAt(i).widget() is not None]
        right_lay.addLayout(temporal_row)
        self._precompute_temporal = QCheckBox(
            "Precompute temporal spans during analysis")
        self._precompute_temporal.setToolTip(
            "Compute and save every selected temporal span, strain window, "
            "and frame before opening results. A resumed sweep reuses completed "
            "correlation and temporal checkpoints.")
        self._precompute_temporal.toggled.connect(
            self._on_precompute_temporal_toggled)
        right_lay.addWidget(self._precompute_temporal)
        self._disk_lbl = QLabel("Estimated output storage: waiting for image and ROI")
        self._disk_lbl.setWordWrap(True)
        self._disk_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:10px;")
        self._disk_lbl.setToolTip(
            "The range estimates compressed correlation data, strain caches, "
            "and the temporal HDF5 cache if every requested span/window is "
            "opened. Temporal results are saved on first access, so actual "
            "disk use grows as you inspect them.")
        right_lay.addWidget(self._disk_lbl)

        # A strain window too small for the grid spacing silently produced an
        # entirely empty strain field. The solver clamps it, but the clamp was
        # only a console print -- say it here, where the value is being chosen.
        self._strain_warn = QLabel("")
        self._strain_warn.setWordWrap(True)
        self._strain_warn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._strain_warn.setMinimumHeight(46)
        self._strain_warn.setStyleSheet("color:#c2954f; font-size:11px;")
        self._strain_warn.setVisible(False)
        right_lay.addWidget(self._strain_warn)

        right_lay.addWidget(self._separator())
        right_lay.addWidget(_section_label("Optimizer"))

        # --- PASTE THIS NEW BLOCK RIGHT HERE ---
        reset_btn = QPushButton("Reset to defaults")
        reset_btn.setStyleSheet(
            "background: #282b2e; color: #94a3b8; border: 1px solid #3c4247; "
            "padding: 6px; border-radius:3px;"
        )
        reset_btn.setFixedHeight(32)
        reset_btn.clicked.connect(self._reset_defaults)
        right_lay.addWidget(reset_btn)
        # ---------------------------------------

        self._sp_maxiter = spin(5, 200, params.max_iter, 5)
        self._sp_maxiter.valueChanged.connect(self._on_param_changed)
        right_lay.addLayout(_param_row(
            "Max iterations", "Maximum IC-GN iterations per subset.",
            self._sp_maxiter, ""))

        # Lower bound is 1e-4, not 1e-8: this is subset-edge motion in PIXELS,
        # so anything smaller can never be reached and every subset just burns
        # all max_iter iterations before being given up on.
        self._sp_tol = dspin(1e-4, 0.1, params.conv_tol, 1e-4, 5)
        self._sp_tol.valueChanged.connect(self._on_param_changed)
        right_lay.addLayout(_param_row(
            "Convergence tol", "Stop IC-GN once an iteration moves the subset\n"
            "edge by less than this many pixels.",
            self._sp_tol, "px"))

        self._sp_cutoff = dspin(0.0, 4.0, params.corr_cutoff, 0.05, 2)
        self._sp_cutoff.valueChanged.connect(self._on_param_changed)
        right_lay.addLayout(_param_row(
            "Correlation cutoff", "Discard subsets with ZNSSD above this\n"
            "(lower = stricter; 0.8 is a good starting value).",
            self._sp_cutoff, ""))

        self._cb_hole_recovery = QComboBox()
        self._cb_hole_recovery.addItem("Off", "off")
        self._cb_hole_recovery.addItem("Neighbour-seeded retry", "neighbour")
        self._cb_hole_recovery.addItem("NCC-seeded retry", "ncc")
        selected_recovery = str(getattr(
            params, "hole_recovery", "neighbour")).lower()
        selected_index = self._cb_hole_recovery.findData(selected_recovery)
        self._cb_hole_recovery.setCurrentIndex(max(0, selected_index))
        self._cb_hole_recovery.setFixedWidth(196)
        self._cb_hole_recovery.currentIndexChanged.connect(
            self._on_param_changed)
        right_lay.addLayout(_param_row(
            "Hole recovery",
            "Optional second correlation pass for failed subsets.\n"
            "Neighbour retry extrapolates a reliable adjacent affine result;\n"
            "NCC retry performs a fresh integer-pixel search. Both then run\n"
            "IC-GN and must pass the normal correlation cutoff—no values are\n"
            "invented or interpolated into the result.",
            self._cb_hole_recovery, ""))

        self._sp_hole_passes = spin(
            1, 20, int(getattr(params, "hole_recovery_passes", 3)), 1)
        self._sp_hole_passes.valueChanged.connect(self._on_param_changed)
        right_lay.addLayout(_param_row(
            "Recovery passes",
            "Maximum second-pass attempts. Recovery stops early as soon as a\n"
            "pass produces no additional accepted correlations.",
            self._sp_hole_passes, ""))

        right_lay.addWidget(self._separator())
        right_lay.addWidget(_section_label("Shape function"))

        self._cb_order = QComboBox()
        self._cb_order.addItem("1st order affine (6 parameters)", 1)
        self._cb_order.addItem("2nd order quadratic (12 parameters)", 2)
        self._cb_order.setCurrentIndex(
            1 if int(getattr(params, "shape_order", 1)) >= 2 else 0)
        self._cb_order.setFixedWidth(196)
        self._cb_order.currentIndexChanged.connect(self._on_order_changed)
        right_lay.addLayout(_param_row(
            "Shape order",
            "1st order models uniform stretch/shear inside the subset.\n"
            "2nd order also models curvature, reducing systematic error where\n"
            "the strain gradient is high -- but the six extra parameters\n"
            "amplify noise. Only worth it on a well-textured pattern.",
            self._cb_order, ""))

        self._order_note = QLabel("")
        self._order_note.setWordWrap(True)
        self._order_note.setStyleSheet(
            f"color:{_C_TEXT2}; font-size:10px; background:{_C_CARD};"
            f" border:1px solid {_C_BORDER}; border-radius:3px; padding:6px;")
        self._order_note.setVisible(False)
        right_lay.addWidget(self._order_note)

        self._order_probe_btn = QPushButton("Measure on these images")
        self._order_probe_btn.setFixedHeight(26)
        self._order_probe_btn.setToolTip(
            "Estimate the extra random error 2nd order would add on this\n"
            "reference image, and the strain curvature needed to justify it.")
        self._order_probe_btn.clicked.connect(self._probe_shape_order)
        self._order_probe_btn.setVisible(False)
        right_lay.addWidget(self._order_probe_btn)

        right_lay.addWidget(self._separator())
        right_lay.addWidget(_section_label("Search"))

        self._sp_search = spin(5, 500, params.search_radius, 10)
        self._sp_search.valueChanged.connect(self._on_param_changed)
        right_lay.addLayout(_param_row(
            "NCC search radius", "Integer-pixel initial-guess search radius.",
            self._sp_search, "px"))

        right_lay.addStretch()

        # Grid preview label
        self._grid_lbl = QLabel("")
        self._grid_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        self._grid_lbl.setWordWrap(True)
        right_lay.addWidget(self._grid_lbl)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setFixedWidth(_PANEL_W)
        right_scroll.setStyleSheet(
            f"QScrollArea{{background:{_C_SURFACE}; border-left:1px solid {_C_BORDER};}}"
            f"QScrollBar:vertical{{background:{_C_SURFACE}; width:10px;}}"
            f"QScrollBar::handle:vertical{{background:{_C_BORDER}; border-radius:5px; min-height:28px;}}")
        right_scroll.setWidget(right)
        body_lay.addWidget(right_scroll)
        root.addWidget(body, 1)

        # ── Footer ────────────────────────────────────────────────────
        footer = QWidget()
        footer.setObjectName("paramsFooter")
        footer.setFixedHeight(58)
        footer.setStyleSheet(
            f"QWidget#paramsFooter{{background:{_C_SURFACE};"
            f"border-top:1px solid {_C_BORDER};}}")
        foot_lay = QHBoxLayout(footer)
        foot_lay.setContentsMargins(20, 0, 20, 0)
        foot_lay.addStretch()

        # ── GPU Acceleration Toggle ───────────────────────────────────
        self._gpu_chk = QPushButton("Use native C++/CUDA acceleration")
        self._gpu_chk.setCheckable(True)
        self._gpu_chk.toggled.connect(lambda *_: self._update_order_note())
        self._gpu_chk.setFixedHeight(32)
        self._gpu_chk.setCursor(Qt.CursorShape.PointingHandCursor)

        if _HAS_GPU:
            self._gpu_chk.setChecked(bool(getattr(self._wizard, "use_gpu", True)))
            self._gpu_chk.setStyleSheet(
                f"QPushButton:checked {{ background: {_C_ACCENT}; color: #ffffff; border: none; border-radius:3px; padding: 0 12px; }}"
                f"QPushButton:!checked {{ background: transparent; color: {_C_TEXT2}; border: 1px solid {_C_BORDER}; border-radius:3px; padding: 0 12px; }}"
            )
            self._gpu_chk.setToolTip(
                "The native CUDA runtime is built. The NVIDIA device and "
                "driver are checked when analysis starts.")
        else:
            self._gpu_chk.setChecked(False)
            self._gpu_chk.setEnabled(False)
            self._gpu_chk.setStyleSheet(
                f"background: transparent; color: {_C_BORDER}; border: 1px solid {_C_BORDER}; border-radius:3px; padding: 0 12px;"
            )
            self._gpu_chk.setToolTip(
                "Native CUDA library not built. Run scripts/build_cuda.py "
                "from the repository root.")

        foot_lay.addWidget(self._gpu_chk)

        self._run_btn = FooterButton("▶  Run Analysis")
        self._run_btn.setProperty("class", "run")
        self._run_btn.setFixedHeight(38)
        self._run_btn.setMinimumWidth(160)
        self._run_btn.clicked.connect(self._on_run_clicked)
        foot_lay.addWidget(self._run_btn)

        root.addWidget(footer)

    # ------------------------------------------------------------------
        # Initial sync now that every widget exists (the note references _gpu_chk).
        self._update_order_note()

    def on_enter(self) -> None:
        parametric = getattr(self._wizard, "analysis_mode", None) == "parametric"
        for field in (self._field_radius, self._field_spacing,
                      self._field_strain, self._field_temporal):
            field.setCurrentIndex(1 if parametric else 0)
        self._field_temporal.setVisible(parametric)
        self._precompute_temporal.setVisible(parametric)
        self._precompute_temporal.setChecked(bool(getattr(
            self._wizard, "parametric_precompute_temporal", False)))
        for widget in self._temporal_row_widgets:
            widget.setVisible(parametric)
        self._run_btn.setText(
            "▶  Run Parameter Sweep" if parametric else "▶  Run Analysis")
        self._sync_controls_from_model()
        img = self._wizard.analysis.strain_reference_image()
        if img is not None:
            # Only set the image if it's new to preserve pan/zoom
            if self._canvas._image_arr is not img:
                self._canvas.set_image(img)
                self._canvas.zoom_fit()

        # Show the mask the first interval will actually analyse. When dynamic
        # ROI is enabled this is the calibrated reference mask from Step 3,
        # rather than the larger static boundary behind it.
        mask = self._wizard.analysis.reference_analysis_mask()
        self._preview_roi_mask = None if mask is None else mask.copy()
        if mask is not None:
            self._canvas.set_roi_mask(mask)
        else:
            self._canvas.clear_roi()

        # Restore Seed
        if getattr(self._wizard, "seed_xy", None) is not None:
            self._canvas.set_seed_xy(self._wizard.seed_xy)

        # Trigger parameter refresh to draw the subset radius
        self._on_param_changed()

    def on_leave(self) -> None:
        self._preview_roi_mask = None

    def _sync_controls_from_model(self) -> None:
        """Restore cached/model values without firing partial updates.

        This page survives a New Session while DICAnalysis is replaced. Without
        this pull, its old widgets immediately overwrote the new analysis and
        made correctly loaded settings appear not to be cached.
        """
        p = self._wizard.analysis.params
        ranges = getattr(self._wizard, "parametric_ranges", {})
        self._range_radius.set_values(tuple(ranges.get(
            "subset_radius", (5, 12, 1))))
        self._range_spacing.set_values(tuple(ranges.get(
            "subset_spacing", (1, 5, 1))))
        self._range_strain.set_values(tuple(ranges.get(
            "strain_window", (1, 15, 1))))
        self._range_temporal.set_values(tuple(ranges.get(
            "temporal_span", (1, 1, 1))))
        pairs = (
            (self._sp_radius, p.subset_radius),
            (self._sp_spacing, p.subset_spacing),
            (self._sp_strain, p.strain_window),
            (self._sp_maxiter, p.max_iter),
            (self._sp_tol, p.conv_tol),
            (self._sp_cutoff, p.corr_cutoff),
            (self._sp_search, p.search_radius),
            (self._sp_hole_passes,
             int(getattr(p, "hole_recovery_passes", 3))),
        )
        for widget, value in pairs:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._cb_order.blockSignals(True)
        self._cb_order.setCurrentIndex(
            1 if int(getattr(p, "shape_order", 1)) >= 2 else 0)
        self._cb_order.blockSignals(False)
        self._cb_hole_recovery.blockSignals(True)
        recovery_index = self._cb_hole_recovery.findData(str(getattr(
            p, "hole_recovery", "neighbour")).lower())
        self._cb_hole_recovery.setCurrentIndex(max(0, recovery_index))
        self._cb_hole_recovery.blockSignals(False)
        if self._gpu_chk.isEnabled():
            self._gpu_chk.blockSignals(True)
            self._gpu_chk.setChecked(bool(getattr(
                self._wizard.analysis, "prefer_gpu", True)))
            self._gpu_chk.blockSignals(False)
            self._wizard.use_gpu = self._gpu_chk.isChecked()
        self._update_order_note()

    def _on_precompute_temporal_toggled(self, checked: bool) -> None:
        self._wizard.parametric_precompute_temporal = bool(checked)
        self._on_param_changed()

    def _on_param_changed(self) -> None:
        p = self._wizard.analysis.params
        old = (p.subset_radius, p.subset_spacing, p.strain_window,
               p.max_iter, p.conv_tol, p.corr_cutoff, p.search_radius,
               int(getattr(p, "shape_order", 1)),
               str(getattr(p, "hole_recovery", "neighbour")),
               int(getattr(p, "hole_recovery_passes", 3)))
        if getattr(self._wizard, "analysis_mode", None) == "parametric":
            ranges = {
                "subset_radius": self._range_radius.values(),
                "subset_spacing": self._range_spacing.values(),
                "strain_window": self._range_strain.values(),
                "temporal_span": self._range_temporal.values(),
            }
            self._wizard.parametric_ranges = ranges
            self._precompute_temporal.setEnabled(
                any(span > 1 for span in self._range_temporal.expanded()))
            # The first case drives ROI/grid preview; the sweep runner consumes
            # every inclusive range stored above.
            p.subset_radius = ranges["subset_radius"][0]
            p.subset_spacing = ranges["subset_spacing"][0]
            p.strain_window = ranges["strain_window"][0]
        else:
            p.subset_radius  = self._sp_radius.value()
            p.subset_spacing = self._sp_spacing.value()
            p.strain_window  = self._sp_strain.value()
        p.max_iter       = self._sp_maxiter.value()
        p.conv_tol       = self._sp_tol.value()
        p.corr_cutoff    = self._sp_cutoff.value()
        p.search_radius  = self._sp_search.value()
        p.shape_order    = int(self._cb_order.currentData() or 1)
        p.hole_recovery  = str(
            self._cb_hole_recovery.currentData() or "off")
        p.hole_recovery_passes = self._sp_hole_passes.value()
        self._sp_hole_passes.setEnabled(p.hole_recovery != "off")
        new = (p.subset_radius, p.subset_spacing, p.strain_window,
               p.max_iter, p.conv_tol, p.corr_cutoff, p.search_radius,
               p.shape_order, p.hole_recovery, p.hole_recovery_passes)
        if old != new:
            # Results belong to the parameter set that produced them. A visible
            # edit must not leave old contours looking current.
            self._wizard.analysis.results.clear()

        self._update_strain_warning()

        method = getattr(p, "dynamic_roi", "None")
        if method in ("None", None):
            self._dyn_lbl.setText("Off. Every pixel of the ROI is analysed in every frame.")
        else:
            thr = getattr(p, "dynamic_roi_threshold", None)
            thr_txt = "automatic threshold" if thr is None else f"threshold {thr*100:.0f}%"
            a = self._wizard.analysis
            n_inc = int(a.dynamic_include_mask.sum()) if a.dynamic_include_mask is not None else 0
            n_exc = int(a.dynamic_exclude_mask.sum()) if a.dynamic_exclude_mask is not None else 0
            extra = ""
            if n_inc or n_exc:
                extra = f", {n_inc:,} px forced in / {n_exc:,} px forced out"
            self._dyn_lbl.setText(f"{method}, {thr_txt}{extra}.")

        if hasattr(self._canvas, "set_subset_radius"):
            self._canvas.set_subset_radius(p.subset_radius)

        # Count estimated subsets / correlations.
        img = self._wizard.analysis.strain_reference_image()
        mask = self._preview_roi_mask
        if img is not None and mask is not None:
            H, W = img.shape
            if getattr(self._wizard, "analysis_mode", None) == "parametric":
                radii = self._range_radius.expanded()
                spacings = self._range_spacing.expanded()
                windows = self._range_strain.expanded()
                counts = []
                for radius in radii:
                    for spacing in spacings:
                        ys = np.arange(radius, H - radius, spacing)
                        xs = np.arange(radius, W - radius, spacing)
                        counts.append(int(mask[np.ix_(ys, xs)].sum()))
                correlations = len(radii) * len(spacings)
                logical = correlations * len(windows)
                total = sum(counts)
                self._grid_lbl.setText(
                    f"{correlations:,} reusable correlations · "
                    f"{logical * len(self._range_temporal.expanded()):,} logical cases\n"
                    f"≈ {total:,} subset centres per frame across the sweep · "
                    f"{len(windows)} strain windows and "
                    f"{len(self._range_temporal.expanded())} temporal spans")
                # HDF5 compression varies with image texture. Estimate stored
                # float/index payload and expose a range rather than a false
                # byte-exact promise.
                frame_count = max(1, len(getattr(
                    self._wizard.analysis, "def_paths", ())))
                centre_count = max(0, total)
                strain_outputs = len(windows) * 2
                raw_bytes = (centre_count * frame_count * 16 +
                             centre_count * frame_count *
                             strain_outputs * 4)
                temporal_spans = tuple(self._range_temporal.expanded())
                # Temporal outputs are now persisted as compact HDF5 sidecars
                # on first access. This is the upper estimate if every
                # requested span/window/frame is eventually opened.
                temporal_variants = (len(windows) *
                                     sum(span > 1 for span in temporal_spans))
                raw_bytes += (centre_count * frame_count *
                              temporal_variants * 32)
                low, high = raw_bytes * 0.35, raw_bytes * 0.85
                def human_size(value):
                    for unit in ("B", "KB", "MB", "GB", "TB"):
                        if value < 1024 or unit == "TB":
                            return f"{value:.1f} {unit}"
                        value /= 1024
                self._disk_lbl.setText(
                    f"Estimated storage: {human_size(low)}–{human_size(high)} "
                    + ("(includes precomputed temporal cache)"
                       if self._precompute_temporal.isChecked() else
                       "(upper range if all temporal results are viewed)"))
                self._disk_lbl.setToolTip(
                    "Compressed estimate for correlation data, strain caches, "
                    "and all requested temporal spans/windows. "
                    + ("Temporal results will be saved during analysis."
                       if self._precompute_temporal.isChecked() else
                       "Temporal results are saved as they are opened."))
                return
            r, s = p.subset_radius, p.subset_spacing
            ys = np.arange(r, H - r, s)
            xs = np.arange(r, W - r, s)
            cnt = sum(1 for y in ys for x in xs
                      if y < mask.shape[0] and x < mask.shape[1] and mask[y, x])
            self._grid_lbl.setText(
                f"≈ {cnt:,} subsets will be analysed\n"
                f"({W}×{H} image, {s} px spacing)"
            )
            frame_count = max(1, len(getattr(
                self._wizard.analysis, "def_paths", ())))
            # Raw single-analysis fields are stored as compact float32 arrays;
            # use a conservative 16 bytes per retained subset/frame.
            estimated = cnt * frame_count * 16
            self._disk_lbl.setText(
                f"Estimated output storage: approximately "
                f"{estimated / (1024 ** 2):.1f} MB before compression")

    def _on_strain_text_edited(self, text: str) -> None:
        """Update validation while the operator is still typing."""
        if getattr(self._wizard, "analysis_mode", None) == "parametric":
            return
        try:
            value = int(text.strip())
        except ValueError:
            return
        if not (self._sp_strain.minimum() <= value <= self._sp_strain.maximum()):
            return
        p = self._wizard.analysis.params
        if p.strain_window != value:
            p.strain_window = value
            self._wizard.analysis.results.clear()
        self._update_strain_warning(value)

    def _update_strain_warning(self, window: int | None = None) -> None:
        p = self._wizard.analysis.params
        if getattr(self._wizard, "analysis_mode", None) == "parametric":
            windows = self._range_strain.expanded()
            spacings = self._range_spacing.expanded()
            clamped = sum(1 for spacing in spacings for value in windows
                          if value < spacing)
            if clamped:
                self._strain_warn.setText(
                    f"ℹ  {clamped} grid/strain-window pairings request fewer "
                    "than 3 grid points across. They remain listed separately, "
                    "but use the grid spacing as their effective fit radius.")
                self._strain_warn.setVisible(True)
            else:
                self._strain_warn.setVisible(False)
            return
        sw = int(p.strain_window if window is None else window)
        eff = p.effective_strain_window(warn=False, window=sw)
        if eff != sw:
            n_pts = p.strain_points_per_axis(sw)
            point_word = "point" if n_pts == 1 else "points"
            self._strain_warn.setText(
                f"⚠  {sw} px spans only {n_pts} grid {point_word} at "
                f"{p.subset_spacing} px spacing: too few to fit a strain plane. "
                f"{eff} px will be used instead.")
            self._strain_warn.setVisible(True)
            self._strain_warn.updateGeometry()
        else:
            self._strain_warn.setVisible(False)

    def _separator(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
        return f

    def _on_order_changed(self) -> None:
        self._update_order_note()
        self._on_param_changed()

    def _update_order_note(self) -> None:
        second = int(self._cb_order.currentData() or 1) >= 2
        self._order_note.setVisible(second)
        self._order_probe_btn.setVisible(second)
        if not second:
            return
        msgs = []
        # The GPU wavefront solver has no 12-parameter path; it would silently
        # keep running first order, so say so rather than let it look applied.
        if self._gpu_chk.isChecked():
            msgs.append("<b style='color:#c2954f;'>GPU solver supports only "
                        "1st order.</b> Select 1st order or turn off GPU "
                        "acceleration before running.")
        msgs.append("2nd order lowers systematic error where strain curves "
                    "inside a subset, but roughly doubles random error on a "
                    "weak pattern. Measure before choosing it.")
        self._order_note.setText("<br><br>".join(msgs))

    def _probe_shape_order(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        img = self._wizard.analysis.strain_reference_image()
        if img is None:
            QMessageBox.information(self, "No reference image",
                                    "Load a reference image first.")
            return
        try:
            from strainx.core.shape_order import shape_order_report
            mask = (self._preview_roi_mask if self._preview_roi_mask is not None
                    else self._wizard.analysis.roi_mask)
            r = shape_order_report(img, mask,
                                   radius=self._sp_radius.value(), verbose=False)
        except Exception as e:
            QMessageBox.warning(self, "Could not measure", str(e))
            return
        rad = r["radius"]
        pen = r["penalty_px"]
        need = max(0.0, 2.0 * pen / (rad ** 2))
        QMessageBox.information(
            self, "Measured cost of 2nd order",
            f"Image noise:            {r['noise_sigma']:.2f} grey levels (8-bit)\n"
            f"Mean gradient in ROI:   {r['mean_gradient']:.2f} grey/px (8-bit)\n"
            f"Usable samples:         {r['valid_samples_order1']} / "
            f"{r['valid_samples_order2']} (1st / 2nd order)\n\n"
            f"Predicted random error\n"
            f"   1st order:           {r['sigma_u_order1']:.4f} px\n"
            f"   2nd order:           {r['sigma_u_order2']:.4f} px\n"
            f"   extra cost:          {pen:+.4f} px\n\n"
            f"2nd order pays off only where the in-subset curvature exceeds\n"
            f"about uxx = {need:.2e} /px at this subset radius ({rad} px).\n\n"
            f"If your shear zone curvature is below that, stay on 1st order.")

    def _reset_defaults(self) -> None:
        from strainx.core.rg_dic import DICParams
        from PyQt6.QtWidgets import QMessageBox

        # 1. Reset core settings and overwrite JSON
        self._wizard.analysis.params = DICParams()
        self._wizard.analysis.save_settings()
        p = self._wizard.analysis.params
        if getattr(self._wizard, "analysis_mode", None) == "parametric":
            self._wizard.parametric_ranges = {
                "subset_radius": (5, 12, 1),
                "subset_spacing": (1, 5, 1),
                "strain_window": (1, 15, 1),
                "temporal_span": (1, 1, 1),
            }
            self._wizard.parametric_precompute_temporal = False
            self._precompute_temporal.setChecked(False)
            self._range_radius.set_values((5, 12, 1))
            self._range_spacing.set_values((1, 5, 1))
            self._range_strain.set_values((1, 15, 1))
            self._range_temporal.set_values((1, 1, 1))

        # 2. Update all spinboxes silently
        for sb, val in [
            (self._sp_radius, p.subset_radius),
            (self._sp_spacing, p.subset_spacing),
            (self._sp_strain, p.strain_window),
            (self._sp_maxiter, p.max_iter),
            (self._sp_search, p.search_radius),
            (self._sp_tol, p.conv_tol),
            (self._sp_cutoff, p.corr_cutoff),
            (self._sp_hole_passes, p.hole_recovery_passes),
        ]:
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)

        self._cb_order.blockSignals(True)
        self._cb_order.setCurrentIndex(1 if int(getattr(p, 'shape_order', 1)) >= 2 else 0)
        self._cb_order.blockSignals(False)
        self._cb_hole_recovery.blockSignals(True)
        recovery_index = self._cb_hole_recovery.findData(
            getattr(p, "hole_recovery", "neighbour"))
        self._cb_hole_recovery.setCurrentIndex(max(0, recovery_index))
        self._cb_hole_recovery.blockSignals(False)
        self._update_order_note()

        self._on_param_changed()  # Update the subset counter and dynamic-ROI summary

        if getattr(self._wizard, "analysis_mode", None) == "parametric":
            message = ("Parametric ranges restored.\n\n"
                       "Radius: 5 to 12 px\nGrid spacing: 1 to 5 px\n"
                       "Cached effective-strain windows: every px from 1 to 15")
        else:
            message = (f"Parameters reset to defaults.\n\n"
                       f"Radius: {p.subset_radius}\n"
                       f"Spacing: {p.subset_spacing}\n"
                       f"Strain Window: {p.strain_window}")
        QMessageBox.information(self, "Defaults restored", message)

    def _on_run_clicked(self) -> None:
        """Save the GPU preference to the wizard, then proceed to the analysis screen."""
        origin = self._wizard.analysis.strain_origin_mask
        if origin is None or not np.any(origin):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Strain origin required",
                "Draw the strain-origin line inside the ROI. Material "
                "crossing it starts at zero accumulated strain.")
            self._wizard.go_roi()
            return
        # Commit every visible control even if it was edited by keyboard and has
        # not yet emitted editingFinished.
        self._on_param_changed()
        if hasattr(self, '_gpu_chk'):
            self._wizard.use_gpu = self._gpu_chk.isChecked()
        else:
            self._wizard.use_gpu = False

        if (self._wizard.use_gpu and
                int(getattr(self._wizard.analysis.params,
                            "shape_order", 1)) != 1):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Unsupported CUDA shape function",
                "Native CUDA currently supports first-order affine DIC only. "
                "Select 1st order, or turn off GPU acceleration to run the "
                "second-order CPU solver.")
            return

        self._wizard.analysis.save_settings()
        if getattr(self._wizard, "analysis_mode", None) == "parametric":
            from pathlib import Path
            from PyQt6.QtWidgets import QFileDialog
            previous = getattr(self._wizard, "parametric_output_directory", "")
            if not previous:
                previous = str(Path(self._wizard.analysis.ref_path).parent)
            output = QFileDialog.getExistingDirectory(
                self, "Select a separate folder for parametric HDF5 results",
                previous)
            if not output:
                return
            self._wizard.parametric_output_directory = output
        self._wizard.go_analysis()
