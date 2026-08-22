"""
dynamic_roi_page.py — Step 3: tune the dynamic (per-frame) ROI.

The dynamic ROI decides, for every frame, which pixels still carry enough
texture to correlate. Previously the method was a bare dropdown on the
parameters page and the threshold was chosen automatically by Otsu with no way
to see or influence the result -- so when it cut into material the operator
wanted kept (or kept background the operator wanted gone), there was nothing to
do about it.

This page shows the resulting mask on the selected zero-strain frame while you change the
threshold, and lets you force regions in or out by hand. Include beats exclude
beats the texture metric, so an operator decision is never quietly reinterpreted
frame to frame.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSlider, QCheckBox, QFrame, QSizePolicy, QButtonGroup, QDoubleSpinBox,
)

from src.ui.components import FooterButton
from src.ui.image_canvas import ImageCanvas, ROITool

if TYPE_CHECKING:
    from src.ui.wizard import Wizard

_C_SURFACE = "#0e1c2e"
_C_CARD    = "#132035"
_C_BORDER  = "#1e3a5a"
_C_ACCENT  = "#3b82f6"
_C_TEXT    = "#e2e8f0"
_C_TEXT2   = "#94a3b8"
_C_TEXT3   = "#475569"
_C_INCLUDE = (16, 185, 129)    # green
_C_EXCLUDE = (239, 68, 68)     # red

_PANEL_W = 400
_METHODS = ["None", "Contrast", "Edge Detection", "Hybrid"]

_CHOICE_STYLE = f"""
QPushButton {{
    background: {_C_CARD}; color: {_C_TEXT2}; border: 1px solid {_C_BORDER};
    border-radius: 6px;
}}
QPushButton:hover {{ border-color: {_C_ACCENT}; color: {_C_TEXT}; }}
QPushButton:checked {{
    background: {_C_ACCENT}; color: #ffffff; border-color: {_C_ACCENT};
    font-weight: 700;
}}
"""


def _section(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;")
    return lbl


class DynamicROIPage(QWidget):
    """Step 3 — dynamic ROI method, threshold and manual overrides."""

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        # Reference-frame boolean overrides, edited one channel at a time.
        self._include: Optional[np.ndarray] = None
        self._exclude: Optional[np.ndarray] = None
        self._channel = "include"          # which one the canvas is editing
        self._tool = ROITool.RECTANGLE      # active drawing geometry
        self._updating = False             # guard against feedback loops
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self._refresh)
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        top = QWidget()
        top.setFixedHeight(52)
        top.setStyleSheet(f"background:{_C_SURFACE}; border-bottom:1px solid {_C_BORDER};")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(20, 0, 20, 0)
        back = QPushButton("← Back")
        back.setFixedWidth(90)
        back.clicked.connect(self._wizard.go_roi)
        top_lay.addWidget(back)
        title = QLabel("Step 3  ·  Dynamic ROI")
        title.setStyleSheet(f"color:{_C_TEXT}; font-size:13px; font-weight:600;")
        top_lay.addWidget(title)
        top_lay.addStretch()
        self._reset_view_btn = QPushButton("Reset Zoom")
        self._reset_view_btn.setFixedHeight(30)
        self._reset_view_btn.setToolTip("Fit the reference image back into the viewport.")
        self._reset_view_btn.clicked.connect(self._canvas_fit_image)
        top_lay.addWidget(self._reset_view_btn)
        root.addWidget(top)

        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        self._canvas = ImageCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.seed_enabled = False
        self._canvas.roi_changed.connect(self._on_canvas_roi)
        body_lay.addWidget(self._canvas, 1)

        right = QWidget()
        right.setFixedWidth(_PANEL_W)
        right.setStyleSheet(f"background:{_C_SURFACE}; border-left:1px solid {_C_BORDER};")
        lay = QVBoxLayout(right)
        lay.setContentsMargins(22, 24, 22, 24)
        lay.setSpacing(14)

        # -- method --
        lay.addWidget(_section("METHOD"))
        self._cb_method = QComboBox()
        self._cb_method.addItems(_METHODS)
        self._cb_method.setToolTip(
            "Contrast  — local intensity standard deviation.\n"
            "Edge Detection — Sobel gradient magnitude.\n"
            "Hybrid — both, each normalised before being summed.\n\n"
            "The threshold is calibrated once on the selected zero-strain frame and then\n"
            "held fixed, so 'enough texture' means the same thing in frame 500\n"
            "as in frame 1.")
        self._cb_method.currentTextChanged.connect(self._on_method_changed)
        lay.addWidget(self._cb_method)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        lay.addWidget(self._hint)

        lay.addWidget(self._sep())

        # -- threshold --
        lay.addWidget(_section("TEXTURE THRESHOLD"))
        self._auto_chk = QCheckBox("Automatic (Otsu)")
        self._auto_chk.setChecked(True)
        self._auto_chk.setToolTip(
            "Pick the threshold automatically from the selected zero-strain frame.\n"
            "Uncheck to set it by hand.")
        self._auto_chk.toggled.connect(self._on_auto_toggled)
        lay.addWidget(self._auto_chk)

        trow = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        self._slider.valueChanged.connect(self._on_threshold_changed)
        trow.addWidget(self._slider, 1)
        self._thr_lbl = QLabel("—")
        self._thr_lbl.setFixedWidth(46)
        self._thr_lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:12px;")
        trow.addWidget(self._thr_lbl)
        lay.addLayout(trow)

        arow = QHBoxLayout()
        a_lbl = QLabel("Min region size")
        a_lbl.setFixedWidth(168)
        a_lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:12px;")
        a_lbl.setToolTip(
            "Discard connected regions smaller than this fraction of the\n"
            "largest one. Keeps speckle noise out without throwing away the\n"
            "chip or the workpiece, which are separate bodies.")
        arow.addWidget(a_lbl)
        self._area_spin = QDoubleSpinBox()
        self._area_spin.setRange(0.0, 1.0)
        self._area_spin.setSingleStep(0.01)
        self._area_spin.setDecimals(3)
        self._area_spin.setValue(0.02)
        self._area_spin.setFixedWidth(104)
        self._area_spin.valueChanged.connect(self._on_threshold_changed)
        arow.addWidget(self._area_spin)
        arow.addStretch()
        lay.addLayout(arow)

        self._fill_chk = QCheckBox("Fill enclosed holes")
        self._fill_chk.setChecked(True)
        self._fill_chk.setToolTip(
            "Keep a region the texture metric rejected when it is completely\n"
            "surrounded by kept material — a glare spot or washed-out patch\n"
            "inside the specimen is a local dropout, not a gap in the material.\n\n"
            "No size limit: a hole is filled because it is enclosed, not because\n"
            "it is small. Use Exclude to override it where a gap is genuine.")
        self._fill_chk.toggled.connect(self._on_threshold_changed)
        lay.addWidget(self._fill_chk)

        lay.addWidget(self._sep())

        # -- manual overrides --
        lay.addWidget(_section("MANUAL OVERRIDES"))
        ov_hint = QLabel("Draw a region, then it is forced in or out for every "
                         "frame. Include wins over exclude.")
        ov_hint.setWordWrap(True)
        ov_hint.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        lay.addWidget(ov_hint)

        crow = QHBoxLayout()
        self._btn_inc = QPushButton("Include")
        self._btn_exc = QPushButton("Exclude")
        self._channel_group = QButtonGroup(self)
        self._channel_group.setExclusive(True)
        for b, chan in ((self._btn_inc, "include"), (self._btn_exc, "exclude")):
            b.setCheckable(True)
            b.setFixedHeight(30)
            b.setStyleSheet(_CHOICE_STYLE)
            self._channel_group.addButton(b)
            b.clicked.connect(lambda _c, ch=chan: self._set_channel(ch))
            crow.addWidget(b)
        self._btn_inc.setChecked(True)
        lay.addLayout(crow)

        trow2 = QHBoxLayout()
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons = {}
        for label, tool in (("Rect", ROITool.RECTANGLE),
                            ("Poly", ROITool.POLYGON),
                            ("Circle", ROITool.CIRCLE)):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setFixedHeight(28)
            b.setStyleSheet(_CHOICE_STYLE)
            self._tool_group.addButton(b)
            self._tool_buttons[tool] = b
            b.clicked.connect(lambda _c, t=tool: self._set_tool(t))
            trow2.addWidget(b)
        self._tool_buttons[ROITool.RECTANGLE].setChecked(True)
        lay.addLayout(trow2)

        crow2 = QHBoxLayout()
        b_clr = QPushButton("Clear this channel")
        b_clr.setFixedHeight(28)
        b_clr.clicked.connect(self._clear_channel)
        crow2.addWidget(b_clr)
        b_clr_all = QPushButton("Clear both")
        b_clr_all.setFixedHeight(28)
        b_clr_all.clicked.connect(self._clear_all)
        crow2.addWidget(b_clr_all)
        lay.addLayout(crow2)

        lay.addWidget(self._sep())
        self._stats = QLabel("")
        self._stats.setWordWrap(True)
        self._stats.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        lay.addWidget(self._stats)

        lay.addStretch()
        cont = FooterButton("Continue  →")
        cont.clicked.connect(self._on_continue)
        lay.addWidget(cont)

        body_lay.addWidget(right)
        root.addWidget(body, 1)

    def _canvas_fit_image(self) -> None:
        if self._canvas._image_arr is not None:
            self._canvas.fit_image()

    def _sep(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"color:{_C_BORDER};")
        return f

    # ------------------------------------------------------------- lifecycle
    def on_before_show(self) -> None:
        self._canvas.clear_result_overlay()

    def on_enter(self) -> None:
        a = self._wizard.analysis
        ref = a.strain_reference_image()
        if ref is None:
            return
        self._updating = True
        self._canvas.set_image(ref)
        self._canvas.fit_image()

        if self._include is None or self._include.shape != ref.shape:
            stored = a.dynamic_include_mask
            self._include = (stored.copy() if stored is not None and
                             stored.shape == ref.shape else
                             np.zeros(ref.shape, bool))
        if self._exclude is None or self._exclude.shape != ref.shape:
            stored = a.dynamic_exclude_mask
            self._exclude = (stored.copy() if stored is not None and
                             stored.shape == ref.shape else
                             np.zeros(ref.shape, bool))

        self._cb_method.setCurrentText(getattr(a.params, "dynamic_roi", "None"))
        thr = getattr(a.params, "dynamic_roi_threshold", None)
        self._auto_chk.setChecked(thr is None)
        if thr is not None:
            self._slider.setValue(int(round(float(thr) * 100)))
        self._area_spin.setValue(float(getattr(a.params, "dynamic_roi_min_area_frac", 0.02)))
        self._fill_chk.setChecked(bool(getattr(a.params, "dynamic_roi_fill_holes", True)))
        self._updating = False

        self._set_channel(self._channel)
        self._refresh()

    def on_leave(self) -> None:
        """Commit controls, then release full-resolution editor work masks."""
        self._refresh_timer.stop()
        self._commit_to_analysis()
        self._include = None
        self._exclude = None

    # ---------------------------------------------------------------- events
    def _on_method_changed(self, _t: str) -> None:
        if not self._updating:
            self._refresh()

    def _on_auto_toggled(self, on: bool) -> None:
        self._slider.setEnabled(not on)
        if not self._updating:
            self._refresh()

    def _on_threshold_changed(self, _v) -> None:
        if not self._updating:
            # Threshold/area controls emit continuously while dragged. Coalesce
            # those events so a large texture mask is computed only for the
            # latest visible value.
            self._refresh_timer.start()

    def _set_channel(self, chan: str) -> None:
        """Swap which override the canvas is painting into."""
        cur = self._canvas.roi_mask
        if cur is not None and self._channel in ("include", "exclude"):
            target = self._include if self._channel == "include" else self._exclude
            if target is not None and cur.shape == target.shape:
                target |= cur
        self._channel = chan
        self._canvas.clear_roi()
        other = self._include if chan == "include" else self._exclude
        if other is not None and other.any():
            self._canvas.set_roi_mask(other.copy())
        r, g, b = _C_INCLUDE if chan == "include" else _C_EXCLUDE
        from PyQt6.QtGui import QColor
        self._canvas.set_roi_colors(
            QColor(r, g, b, 70), QColor(r, g, b, 210))
        self._canvas.set_tool(self._tool)
        self._btn_inc.setChecked(chan == "include")
        self._btn_exc.setChecked(chan == "exclude")
        self._refresh()

    def _set_tool(self, tool: ROITool) -> None:
        """Select a drawing geometry and keep its button visibly active."""
        self._tool = tool
        self._canvas.set_tool(tool)
        for candidate, button in self._tool_buttons.items():
            button.setChecked(candidate == tool)

    def _on_canvas_roi(self, mask) -> None:
        if self._updating or mask is None:
            return
        target = self._include if self._channel == "include" else self._exclude
        if target is not None and mask.shape == target.shape:
            target[:] = mask
        self._refresh()

    def _clear_channel(self) -> None:
        target = self._include if self._channel == "include" else self._exclude
        if target is not None:
            target[:] = False
        self._canvas.clear_roi()
        self._refresh()

    def _clear_all(self) -> None:
        for m in (self._include, self._exclude):
            if m is not None:
                m[:] = False
        self._canvas.clear_roi()
        self._refresh()

    # --------------------------------------------------------------- preview
    def _commit_to_analysis(self) -> None:
        a = self._wizard.analysis
        old_params = (
            getattr(a.params, "dynamic_roi", "None"),
            getattr(a.params, "dynamic_roi_threshold", None),
            getattr(a.params, "dynamic_roi_min_area_frac", 0.02),
            getattr(a.params, "dynamic_roi_fill_holes", True),
        )
        old_inc = a.dynamic_include_mask
        old_exc = a.dynamic_exclude_mask
        a.params.dynamic_roi = self._cb_method.currentText()
        a.params.dynamic_roi_threshold = (
            None if self._auto_chk.isChecked() else self._slider.value() / 100.0)
        a.params.dynamic_roi_min_area_frac = float(self._area_spin.value())
        a.params.dynamic_roi_fill_holes = bool(self._fill_chk.isChecked())

        # Overrides only mean anything inside the static ROI, so clip them to it
        # rather than storing regions the solver will discard.
        static = a.roi_mask
        inc, exc = self._include, self._exclude
        if static is not None:
            if inc is not None and inc.shape == static.shape:
                inc = inc & static
            if exc is not None and exc.shape == static.shape:
                exc = exc & static
        a.dynamic_include_mask = inc.copy() if inc is not None and inc.any() else None
        a.dynamic_exclude_mask = exc.copy() if exc is not None and exc.any() else None
        new_params = (
            a.params.dynamic_roi, a.params.dynamic_roi_threshold,
            a.params.dynamic_roi_min_area_frac,
            a.params.dynamic_roi_fill_holes,
        )
        masks_changed = not (
            (old_inc is None and a.dynamic_include_mask is None or
             old_inc is not None and a.dynamic_include_mask is not None and
             np.array_equal(old_inc, a.dynamic_include_mask)) and
            (old_exc is None and a.dynamic_exclude_mask is None or
             old_exc is not None and a.dynamic_exclude_mask is not None and
             np.array_equal(old_exc, a.dynamic_exclude_mask))
        )
        if old_params != new_params or masks_changed:
            a.results.clear()

    def _refresh(self) -> None:
        a = self._wizard.analysis
        ref = a.strain_reference_image()
        method = self._cb_method.currentText()
        self._commit_to_analysis()

        if ref is None:
            return

        if method == "None":
            self._hint.setText("No dynamic masking — every pixel of the static ROI "
                               "is analysed in every frame.")
            self._canvas.clear_result_overlay()
            self._stats.setText("")
            self._thr_lbl.setText("—")
            return

        self._hint.setText(
            "Green = kept. Grey = outside the ROI you drew, never analysed. "
            "Regions with too little texture are dropped for that frame; a point "
            "that comes back is picked up again.")

        roi = a.make_dynamic_roi()
        roi.calibrate(ref)
        auto = roi.auto_threshold_normalised()
        if self._auto_chk.isChecked() and auto is not None:
            self._updating = True
            self._slider.setValue(int(round(auto * 100)))
            self._updating = False
        self._thr_lbl.setText(f"{self._slider.value()}%")

        # This editor is the one place where static ROI, overrides and texture
        # mask are all in the same (reference-image) coordinate system.
        m = roi.mask(ref, reference_frame=True)
        if m is None:
            self._canvas.clear_result_overlay()
            return

        static = a.roi_mask

        # This is a categorical mask, so a byte of palette indices is enough.
        # Building a four-channel RGBA frame made every slider tick allocate and
        # copy four times more data than necessary.
        overlay = np.zeros(m.shape, np.uint8)
        # Everything outside the static ROI is dimmed, so it is obvious that the
        # dynamic mask only ever refines the region already drawn.
        if static is not None:
            overlay[~static] = 1
        overlay[m] = 2
        inside = static if static is not None else np.ones_like(m)
        if self._exclude is not None and self._exclude.any():
            overlay[self._exclude & inside] = 3
        if self._include is not None and self._include.any():
            overlay[self._include & inside] = 4
        self._canvas.set_result_overlay_indexed(overlay, [
            QColor(0, 0, 0, 0),
            QColor(15, 23, 42, 130),
            QColor(16, 185, 129, 70),
            QColor(239, 68, 68, 90),
            QColor(16, 185, 129, 130),
        ])
        denom = int(static.sum()) if static is not None else m.size
        kept = int((m & static).sum()) if static is not None else int(m.sum())
        pct = 100.0 * kept / max(1, denom)
        auto_txt = f"auto {auto*100:.0f}%" if auto is not None else "auto n/a"
        n_inc = int((self._include & inside).sum()) if self._include is not None else 0
        n_exc = int((self._exclude & inside).sum()) if self._exclude is not None else 0
        self._stats.setText(
            f"Keeps {pct:.1f}% of the static ROI on the zero-strain frame  ({auto_txt}).\n"
            f"Overrides: {n_inc:,} px forced in, {n_exc:,} px forced out.")

    def _on_continue(self) -> None:
        self._refresh_timer.stop()
        self._set_channel(self._channel)   # flush any in-progress drawing
        self._commit_to_analysis()
        try:
            self._wizard.analysis.save_settings()
        except Exception:
            pass
        self._wizard.go_params()

    def reset_page(self) -> None:
        self._include = None
        self._exclude = None
        self._canvas.clear_roi()
        self._canvas.clear_result_overlay()
