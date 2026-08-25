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
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
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
        self._frame_overrides: dict[int, dict[str, object]] = {}
        self._future_overrides: dict[int, dict[str, object]] = {}
        self._edit_scope = "global"
        # Global and exact-frame editing are independent workspaces.  Keeping
        # their selections separately prevents (for example) choosing Circle
        # for one frame from changing a previously selected global Rect tool.
        self._scope_channels = {"global": "include", "frame": "include"}
        self._scope_tools = {
            "global": ROITool.RECTANGLE,
            "frame": ROITool.RECTANGLE,
        }
        self._channel = self._scope_channels["global"]
        self._tool = self._scope_tools["global"]
        self._preview_frame = 0
        self._updating = False             # guard against feedback loops
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self._refresh)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_next_frame)
        self._build_ui()
        self._install_shortcuts()
        self._sync_scope_controls()

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

        # Frame-specific controls stay above the image, like a video editor's
        # active-clip toolbar. They always operate on exactly the displayed frame.
        frame_bar = QWidget()
        frame_bar.setFixedHeight(58)
        frame_bar.setStyleSheet(
            f"background:{_C_CARD}; border-bottom:1px solid {_C_BORDER};")
        fbar = QHBoxLayout(frame_bar)
        fbar.setContentsMargins(14, 8, 14, 8)
        fbar.setSpacing(7)
        self._frame_override_lbl = QPushButton("FRAME 0 OVERRIDE")
        self._frame_override_lbl.setCheckable(True)
        self._frame_override_lbl.setStyleSheet(
            _CHOICE_STYLE +
            "QPushButton { padding: 0 9px; font-size:10px; font-weight:700; }")
        self._frame_override_lbl.setToolTip(
            "Switch the canvas and toolbar to edits that apply only to this frame.")
        self._frame_override_lbl.clicked.connect(
            lambda: self._activate_scope("frame"))
        fbar.addWidget(self._frame_override_lbl)
        self._frame_replace_chk = QCheckBox("Replace base")
        self._frame_replace_chk.setToolTip(
            "Use this frame's Include drawing as the complete Dynamic ROI.\n"
            "Without this option, frame Include/Exclude modify the global base.")
        self._frame_replace_chk.toggled.connect(self._on_frame_replace_toggled)
        fbar.addWidget(self._frame_replace_chk)

        self._frame_btn_inc = QPushButton("Include")
        self._frame_btn_exc = QPushButton("Exclude")
        self._frame_channel_group = QButtonGroup(self)
        self._frame_channel_group.setExclusive(True)
        for button, channel in ((self._frame_btn_inc, "include"),
                                (self._frame_btn_exc, "exclude")):
            button.setCheckable(True)
            button.setFixedHeight(30)
            button.setStyleSheet(_CHOICE_STYLE)
            self._frame_channel_group.addButton(button)
            button.clicked.connect(
                lambda _c, ch=channel: self._set_channel(ch, "frame"))
            fbar.addWidget(button)

        self._frame_tool_group = QButtonGroup(self)
        self._frame_tool_group.setExclusive(True)
        self._frame_tool_buttons = {}
        for label, tool in (("Rect", ROITool.RECTANGLE),
                            ("Poly", ROITool.POLYGON),
                            ("Circle", ROITool.CIRCLE),
                            ("Erase", ROITool.ERASE)):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setFixedHeight(30)
            button.setStyleSheet(_CHOICE_STYLE)
            self._frame_tool_group.addButton(button)
            self._frame_tool_buttons[tool] = button
            button.clicked.connect(
                lambda _c, t=tool: self._set_tool(t, "frame"))
            fbar.addWidget(button)
        self._frame_tool_buttons[ROITool.RECTANGLE].setChecked(True)

        fbar.addSpacing(8)
        self._frame_thr_chk = QCheckBox("Threshold override")
        self._frame_thr_chk.toggled.connect(self._on_frame_threshold_toggled)
        fbar.addWidget(self._frame_thr_chk)
        self._frame_thr_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_thr_slider.setRange(0, 100)
        self._frame_thr_slider.setValue(50)
        self._frame_thr_slider.setFixedWidth(120)
        self._frame_thr_slider.setEnabled(False)
        self._frame_thr_slider.valueChanged.connect(
            self._on_frame_threshold_changed)
        fbar.addWidget(self._frame_thr_slider)
        self._frame_thr_lbl = QLabel("base")
        self._frame_thr_lbl.setFixedWidth(42)
        fbar.addWidget(self._frame_thr_lbl)
        fbar.addStretch()
        self._copy_prev_btn = QPushButton("Copy prev")
        self._copy_prev_btn.setFixedHeight(30)
        self._copy_prev_btn.clicked.connect(self._copy_previous_override)
        fbar.addWidget(self._copy_prev_btn)
        self._set_future_btn = QPushButton("Set → future")
        self._set_future_btn.setFixedHeight(30)
        self._set_future_btn.setToolTip(
            "Use this frame's effective override as the default beginning with "
            "the next frame. Exact-frame edits still win.")
        self._set_future_btn.clicked.connect(self._set_current_as_future_default)
        fbar.addWidget(self._set_future_btn)
        self._clear_future_btn = QPushButton("Clear future")
        self._clear_future_btn.setFixedHeight(30)
        self._clear_future_btn.setToolTip(
            "Keep defaults through this frame, then stop inheriting them from "
            "the next frame onward. Exact-frame edits remain.")
        self._clear_future_btn.clicked.connect(self._clear_future_defaults)
        fbar.addWidget(self._clear_future_btn)
        self._clear_frame_btn = QPushButton("Clear frame")
        self._clear_frame_btn.setFixedHeight(30)
        self._clear_frame_btn.clicked.connect(self._clear_frame_override)
        fbar.addWidget(self._clear_frame_btn)
        root.addWidget(frame_bar)

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
        lay.addWidget(_section("GLOBAL BASE OVERRIDES"))
        self._global_mode_btn = QPushButton("EDIT GLOBAL BASE")
        self._global_mode_btn.setCheckable(True)
        self._global_mode_btn.setStyleSheet(_CHOICE_STYLE)
        self._global_mode_btn.setFixedHeight(30)
        self._global_mode_btn.setToolTip(
            "Switch the canvas and toolbar to overrides applied to every frame.")
        self._global_mode_btn.clicked.connect(
            lambda: self._activate_scope("global"))
        lay.addWidget(self._global_mode_btn)
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
            b.clicked.connect(lambda _c, ch=chan: self._set_channel(ch, "global"))
            crow.addWidget(b)
        self._btn_inc.setChecked(True)
        lay.addLayout(crow)

        trow2 = QHBoxLayout()
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons = {}
        for label, tool in (("Rect", ROITool.RECTANGLE),
                            ("Poly", ROITool.POLYGON),
                            ("Circle", ROITool.CIRCLE),
                            ("Erase", ROITool.ERASE)):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setFixedHeight(28)
            b.setStyleSheet(_CHOICE_STYLE)
            self._tool_group.addButton(b)
            self._tool_buttons[tool] = b
            b.clicked.connect(lambda _c, t=tool: self._set_tool(t, "global"))
            trow2.addWidget(b)
        self._tool_buttons[ROITool.RECTANGLE].setChecked(True)
        lay.addLayout(trow2)

        crow2 = QHBoxLayout()
        self._clear_channel_btn = QPushButton("Clear this channel")
        self._clear_channel_btn.setFixedHeight(28)
        self._clear_channel_btn.clicked.connect(self._clear_channel)
        crow2.addWidget(self._clear_channel_btn)
        self._clear_all_btn = QPushButton("Clear both")
        self._clear_all_btn.setFixedHeight(28)
        self._clear_all_btn.clicked.connect(self._clear_all)
        crow2.addWidget(self._clear_all_btn)
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

        # Familiar video-player transport along the bottom.
        transport = QWidget()
        transport.setFixedHeight(60)
        transport.setStyleSheet(
            f"background:{_C_SURFACE}; border-top:1px solid {_C_BORDER};")
        transport_lay = QHBoxLayout(transport)
        transport_lay.setContentsMargins(18, 8, 18, 8)
        transport_lay.setSpacing(9)
        self._prev_btn = QPushButton("|◀")
        self._play_btn = QPushButton("▶")
        self._next_btn = QPushButton("▶|")
        for button in (self._prev_btn, self._play_btn, self._next_btn):
            button.setFixedSize(42, 32)
            transport_lay.addWidget(button)
        self._prev_btn.clicked.connect(lambda: self._step_frame(-1))
        self._play_btn.clicked.connect(self._toggle_playback)
        self._next_btn.clicked.connect(lambda: self._step_frame(1))
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.valueChanged.connect(self._on_frame_changed)
        transport_lay.addWidget(self._frame_slider, 1)
        self._frame_lbl = QLabel("Frame 0 / 0")
        self._frame_lbl.setFixedWidth(105)
        self._frame_lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:11px;")
        transport_lay.addWidget(self._frame_lbl)
        root.addWidget(transport)

    def _install_shortcuts(self) -> None:
        """Install page-scoped editor shortcuts and expose them in tooltips."""
        self._shortcuts: dict[str, QShortcut] = {}

        def add(name: str, keys: str, callback, widgets=()) -> None:
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts[name] = shortcut
            readable = QKeySequence(keys).toString(
                QKeySequence.SequenceFormat.NativeText)
            for widget in widgets:
                existing = widget.toolTip().strip()
                suffix = f"Shortcut: {readable}"
                widget.setToolTip(
                    f"{existing}\n\n{suffix}" if existing else suffix)

        def frame_action(callback) -> None:
            self._activate_scope("frame")
            callback()

        add("previous_frame", "Left", lambda: self._step_frame(-1),
            (self._prev_btn,))
        add("next_frame", "Right", lambda: self._step_frame(1),
            (self._next_btn,))
        add("play_pause", "Space", self._toggle_playback,
            (self._play_btn,))
        add("copy_previous", "C",
            lambda: frame_action(self._copy_previous_override),
            (self._copy_prev_btn,))
        add("set_future", "Ctrl+F",
            lambda: frame_action(self._set_current_as_future_default),
            (self._set_future_btn,))
        add("clear_future", "Ctrl+Shift+F",
            lambda: frame_action(self._clear_future_defaults),
            (self._clear_future_btn,))
        add("clear_frame", "Delete",
            lambda: frame_action(self._clear_frame_override),
            (self._clear_frame_btn,))

        add("global_mode", "G", lambda: self._activate_scope("global"),
            (self._global_mode_btn,))
        add("frame_mode", "F", lambda: self._activate_scope("frame"),
            (self._frame_override_lbl,))
        add("include", "I", lambda: self._set_channel("include"),
            (self._btn_inc, self._frame_btn_inc))
        add("exclude", "X", lambda: self._set_channel("exclude"),
            (self._btn_exc, self._frame_btn_exc))
        for name, keys, tool in (
                ("rectangle", "R", ROITool.RECTANGLE),
                ("polygon", "P", ROITool.POLYGON),
                ("circle", "O", ROITool.CIRCLE),
                ("erase", "E", ROITool.ERASE)):
            add(name, keys, lambda t=tool: self._set_tool(t),
                (self._tool_buttons[tool], self._frame_tool_buttons[tool]))
        add("reset_zoom", "Ctrl+0", self._canvas_fit_image,
            (self._reset_view_btn,))

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
        calibration = a.strain_reference_image()
        if calibration is None:
            return
        self._updating = True
        last_frame = len(getattr(a, "def_paths", []))
        self._frame_slider.setRange(0, last_frame)
        self._preview_frame = min(
            max(0, int(getattr(a, "strain_start_frame", 0))), last_frame)
        self._frame_slider.setValue(self._preview_frame)
        preview = a.frame_image(self._preview_frame)
        if preview is None:
            preview = calibration
        self._frame_lbl.setText(f"Frame {self._preview_frame} / {last_frame}")
        self._canvas.set_image(preview)
        self._canvas.fit_image()

        if self._include is None or self._include.shape != calibration.shape:
            stored = a.dynamic_include_mask
            self._include = (stored.copy() if stored is not None and
                             stored.shape == calibration.shape else
                             np.zeros(calibration.shape, bool))
        if self._exclude is None or self._exclude.shape != calibration.shape:
            stored = a.dynamic_exclude_mask
            self._exclude = (stored.copy() if stored is not None and
                             stored.shape == calibration.shape else
                             np.zeros(calibration.shape, bool))
        self._frame_overrides = {}
        for frame_index, entry in getattr(a, "dynamic_frame_overrides", {}).items():
            copied: dict[str, object] = {}
            if entry.get("threshold") is not None:
                copied["threshold"] = float(entry["threshold"])
            if "replace" in entry:
                copied["replace"] = bool(entry["replace"])
            for channel in ("include", "exclude"):
                mask = entry.get(channel)
                if mask is not None and np.asarray(mask).shape == calibration.shape:
                    copied[channel] = np.asarray(mask, dtype=bool).copy()
            if copied:
                self._frame_overrides[int(frame_index)] = copied
        self._future_overrides = {}
        for start_frame, entry in getattr(a, "dynamic_future_overrides", {}).items():
            copied: dict[str, object] = {}
            if bool(entry.get("reset", False)):
                copied["reset"] = True
            if entry.get("threshold") is not None:
                copied["threshold"] = float(entry["threshold"])
            if "replace" in entry:
                copied["replace"] = bool(entry["replace"])
            for channel in ("include", "exclude"):
                mask = entry.get(channel)
                if mask is not None and np.asarray(mask).shape == calibration.shape:
                    copied[channel] = np.asarray(mask, dtype=bool).copy()
            if copied:
                self._future_overrides[int(start_frame)] = copied

        self._cb_method.setCurrentText(getattr(a.params, "dynamic_roi", "None"))
        thr = getattr(a.params, "dynamic_roi_threshold", None)
        self._auto_chk.setChecked(thr is None)
        if thr is not None:
            self._slider.setValue(int(round(float(thr) * 100)))
        self._area_spin.setValue(float(getattr(a.params, "dynamic_roi_min_area_frac", 0.02)))
        self._fill_chk.setChecked(bool(getattr(a.params, "dynamic_roi_fill_holes", True)))
        self._updating = False

        self._load_frame_controls()
        self._activate_scope(self._edit_scope, force=True)
        self._refresh()

    def on_leave(self) -> None:
        """Commit controls, then release full-resolution editor work masks."""
        self._refresh_timer.stop()
        self._play_timer.stop()
        self._flush_active_channel()
        self._commit_to_analysis()
        self._include = None
        self._exclude = None
        self._frame_overrides = {}
        self._future_overrides = {}

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

    def _on_frame_changed(self, frame_index: int) -> None:
        """Preview the configured dynamic mask on any imported frame."""
        if not self._updating:
            self._flush_active_channel()
        self._preview_frame = int(frame_index)
        a = self._wizard.analysis
        last_frame = len(getattr(a, "def_paths", []))
        self._frame_lbl.setText(
            f"Frame {self._preview_frame} / {last_frame}")
        if self._updating:
            return
        image = a.frame_image(self._preview_frame)
        if image is not None:
            self._canvas.set_image(image, keep_view=True)
        self._load_frame_controls()
        self._load_active_channel()
        self._refresh()

    def _frame_entry(self, create: bool = False) -> Optional[dict[str, object]]:
        entry = self._frame_overrides.get(self._preview_frame)
        if entry is None and create:
            entry = {}
            self._frame_overrides[self._preview_frame] = entry
        return entry

    def _effective_local_override(self, frame_index: Optional[int] = None
                                  ) -> dict[str, object]:
        index = self._preview_frame if frame_index is None else int(frame_index)
        starts = [start for start in self._future_overrides if start <= index]
        merged = dict(self._future_overrides[max(starts)]) if starts else {}
        merged.update(self._frame_overrides.get(index, {}))
        return merged

    @staticmethod
    def _clone_override(entry: dict[str, object]) -> dict[str, object]:
        copied: dict[str, object] = {}
        if bool(entry.get("reset", False)):
            copied["reset"] = True
        if entry.get("threshold") is not None:
            copied["threshold"] = float(entry["threshold"])
        if "replace" in entry:
            copied["replace"] = bool(entry["replace"])
        for channel in ("include", "exclude"):
            mask = entry.get(channel)
            if mask is not None:
                copied[channel] = np.asarray(mask, dtype=bool).copy()
        return copied

    def _target_mask(self, create: bool = False) -> Optional[np.ndarray]:
        if self._edit_scope == "global":
            return self._include if self._channel == "include" else self._exclude
        entry = self._frame_entry(create)
        if entry is None:
            return None
        mask = entry.get(self._channel)
        if mask is None and create and self._canvas._image_arr is not None:
            mask = np.zeros(self._canvas._image_arr.shape, dtype=bool)
            entry[self._channel] = mask
        return mask

    def _flush_active_channel(self) -> None:
        current = self._canvas.roi_mask
        if current is None:
            return
        target = self._target_mask(create=True)
        if target is not None and target.shape == current.shape:
            target[:] = current

    def _load_active_channel(self) -> None:
        self._updating = True
        self._canvas.clear_roi()
        target = self._target_mask(create=False)
        if target is not None and target.any():
            self._canvas.set_roi_mask(target.copy())
        self._updating = False

    def _activate_scope(self, scope: str, force: bool = False) -> None:
        """Activate one editor and restore that editor's channel and tool."""
        if scope not in self._scope_channels:
            return
        self._play_timer.stop()
        self._play_btn.setText("▶")
        if not force and scope == self._edit_scope:
            self._sync_scope_controls()
            return
        self._flush_active_channel()
        self._edit_scope = scope
        self._channel = self._scope_channels[scope]
        self._tool = self._scope_tools[scope]
        self._load_active_channel()
        r, g, b = _C_INCLUDE if self._channel == "include" else _C_EXCLUDE
        self._canvas.set_roi_colors(
            QColor(r, g, b, 70), QColor(r, g, b, 210))
        self._canvas.set_tool(self._tool)
        self._sync_scope_controls()
        self._refresh()

    def _sync_scope_controls(self) -> None:
        """Grey the inactive editor while preserving both selections."""
        global_active = self._edit_scope == "global"
        frame_active = self._edit_scope == "frame"
        self._global_mode_btn.setChecked(global_active)
        self._frame_override_lbl.setChecked(frame_active)

        self._btn_inc.setChecked(self._scope_channels["global"] == "include")
        self._btn_exc.setChecked(self._scope_channels["global"] == "exclude")
        self._frame_btn_inc.setChecked(
            self._scope_channels["frame"] == "include")
        self._frame_btn_exc.setChecked(
            self._scope_channels["frame"] == "exclude")
        for candidate, button in self._tool_buttons.items():
            button.setChecked(self._scope_tools["global"] == candidate)
        for candidate, button in self._frame_tool_buttons.items():
            button.setChecked(self._scope_tools["frame"] == candidate)

        global_widgets = (
            self._btn_inc, self._btn_exc, *self._tool_buttons.values(),
            self._clear_channel_btn, self._clear_all_btn,
        )
        for widget in global_widgets:
            widget.setEnabled(global_active)

        frame_widgets = (
            self._frame_replace_chk, self._frame_btn_inc, self._frame_btn_exc,
            *self._frame_tool_buttons.values(), self._frame_thr_chk,
            self._clear_frame_btn,
        )
        for widget in frame_widgets:
            widget.setEnabled(frame_active)
        self._frame_thr_slider.setEnabled(
            frame_active and self._frame_thr_chk.isChecked())
        self._copy_prev_btn.setEnabled(frame_active and self._preview_frame > 0)
        self._set_future_btn.setEnabled(
            frame_active and
            self._preview_frame < self._frame_slider.maximum())
        self._clear_future_btn.setEnabled(
            frame_active and self._can_clear_future_defaults())

    def _can_clear_future_defaults(self) -> bool:
        """Whether a future value is scheduled or currently being inherited."""
        if self._preview_frame >= self._frame_slider.maximum():
            return False
        if any(start > self._preview_frame for start in self._future_overrides):
            return True
        starts = [
            start for start in self._future_overrides
            if start <= self._preview_frame]
        if not starts:
            return False
        return not bool(self._future_overrides[max(starts)].get("reset", False))

    def _set_channel(self, chan: str, scope: Optional[str] = None) -> None:
        """Swap the active editor's include/exclude channel."""
        target_scope = scope or self._edit_scope
        if target_scope != self._edit_scope:
            self._activate_scope(target_scope)
        self._play_timer.stop()
        self._play_btn.setText("▶")
        self._flush_active_channel()
        self._channel = chan
        self._scope_channels[self._edit_scope] = chan
        self._load_active_channel()
        r, g, b = _C_INCLUDE if chan == "include" else _C_EXCLUDE
        self._canvas.set_roi_colors(
            QColor(r, g, b, 70), QColor(r, g, b, 210))
        self._canvas.set_tool(self._tool)
        self._sync_scope_controls()
        self._refresh()

    def _set_tool(self, tool: ROITool, scope: Optional[str] = None) -> None:
        """Select a drawing geometry and keep its button visibly active."""
        if scope is not None and scope != self._edit_scope:
            self._activate_scope(scope)
        self._tool = tool
        self._scope_tools[self._edit_scope] = tool
        self._canvas.set_tool(tool)
        self._sync_scope_controls()

    def _on_canvas_roi(self, mask) -> None:
        if self._updating or mask is None:
            return
        target = self._target_mask(create=True)
        if target is not None and mask.shape == target.shape:
            target[:] = mask
        self._refresh()

    def _clear_channel(self) -> None:
        self._set_channel(self._channel, "global")
        target = self._target_mask(create=True)
        if target is not None:
            target[:] = False
        self._canvas.clear_roi()
        self._refresh()

    def _clear_all(self) -> None:
        self._set_channel(self._channel, "global")
        for m in (self._include, self._exclude):
            if m is not None:
                m[:] = False
        self._canvas.clear_roi()
        self._refresh()

    def _load_frame_controls(self) -> None:
        entry = self._frame_entry(False) or {}
        threshold = entry.get("threshold")
        inherited_starts = [
            start for start in self._future_overrides
            if start <= self._preview_frame]
        inherited_from = max(inherited_starts) if inherited_starts else None
        inherited_reset = bool(
            inherited_from is not None and
            self._future_overrides[inherited_from].get("reset", False))
        self._updating = True
        self._frame_override_lbl.setText(
            f"FRAME {self._preview_frame} OVERRIDE" +
            (f"  ·  {'CLEARED' if inherited_reset else 'DEFAULT'} FROM "
             f"{inherited_from}"
             if inherited_from is not None else ""))
        self._frame_thr_chk.setChecked(threshold is not None)
        self._frame_replace_chk.setChecked(bool(
            self._effective_local_override().get("replace", False)))
        if threshold is not None:
            self._frame_thr_slider.setValue(int(round(float(threshold) * 100)))
        self._frame_thr_slider.setEnabled(threshold is not None)
        self._frame_thr_lbl.setText(
            f"{int(round(float(threshold) * 100))}%" if threshold is not None
            else "base")
        self._copy_prev_btn.setEnabled(self._preview_frame > 0)
        self._set_future_btn.setEnabled(
            self._preview_frame < self._frame_slider.maximum())
        self._clear_future_btn.setEnabled(self._can_clear_future_defaults())
        self._updating = False
        self._sync_scope_controls()

    def _on_frame_replace_toggled(self, enabled: bool) -> None:
        if self._updating:
            return
        effective_replace = bool(
            self._effective_local_override().get("replace", False))
        entry = self._frame_entry(enabled or effective_replace)
        if enabled and entry is not None:
            entry["replace"] = True
        elif entry is not None:
            if effective_replace:
                entry["replace"] = False
            else:
                entry.pop("replace", None)
                self._drop_empty_frame_entry()
        self._refresh()

    def _on_frame_threshold_toggled(self, enabled: bool) -> None:
        self._frame_thr_slider.setEnabled(
            enabled and self._edit_scope == "frame")
        if self._updating:
            return
        entry = self._frame_entry(enabled)
        if enabled and entry is not None:
            entry["threshold"] = self._frame_thr_slider.value() / 100.0
        elif entry is not None:
            entry.pop("threshold", None)
            self._drop_empty_frame_entry()
        self._load_frame_controls()
        self._refresh()

    def _on_frame_threshold_changed(self, value: int) -> None:
        self._frame_thr_lbl.setText(f"{value}%")
        if self._updating or not self._frame_thr_chk.isChecked():
            return
        entry = self._frame_entry(True)
        entry["threshold"] = value / 100.0
        self._refresh_timer.start()

    def _drop_empty_frame_entry(self) -> None:
        entry = self._frame_entry(False)
        if entry is None:
            return
        for channel in ("include", "exclude"):
            mask = entry.get(channel)
            if mask is not None and not np.asarray(mask, dtype=bool).any():
                entry.pop(channel, None)
        if not entry:
            self._frame_overrides.pop(self._preview_frame, None)

    def _clear_frame_override(self) -> None:
        self._play_timer.stop()
        self._frame_overrides.pop(self._preview_frame, None)
        if self._edit_scope == "frame":
            self._canvas.clear_roi()
        self._load_frame_controls()
        self._refresh()

    def _copy_previous_override(self) -> None:
        if self._preview_frame <= 0:
            return
        self._flush_active_channel()
        previous = self._frame_overrides.get(self._preview_frame - 1)
        if not previous:
            return
        copied: dict[str, object] = {}
        if previous.get("threshold") is not None:
            copied["threshold"] = float(previous["threshold"])
        if "replace" in previous:
            copied["replace"] = bool(previous["replace"])
        for channel in ("include", "exclude"):
            mask = previous.get(channel)
            if mask is not None:
                copied[channel] = np.asarray(mask, dtype=bool).copy()
        self._frame_overrides[self._preview_frame] = copied
        self._load_frame_controls()
        if self._edit_scope == "frame":
            self._load_active_channel()
        self._refresh()

    def _set_current_as_future_default(self) -> None:
        """Store one compact inherited keyframe beginning at the next frame."""
        self._flush_active_channel()
        start = self._preview_frame + 1
        if start > self._frame_slider.maximum():
            return
        effective = self._effective_local_override(self._preview_frame)
        if not effective:
            return
        self._future_overrides[start] = self._clone_override(effective)
        self._load_frame_controls()
        self._commit_to_analysis()
        self._refresh()

    def _clear_future_defaults(self) -> None:
        """Stop future inheritance without rewriting already-previewed frames."""
        current = self._preview_frame
        active_starts = [
            start for start in self._future_overrides if start <= current]
        had_active_default = bool(
            active_starts and not self._future_overrides[
                max(active_starts)].get("reset", False))

        # Past keyframes are history: they must continue to describe all frames
        # up to and including the displayed one. Only scheduled future changes
        # are removed.
        self._future_overrides = {
            start: entry for start, entry in self._future_overrides.items()
            if start <= current
        }
        next_frame = current + 1
        if (had_active_default and
                next_frame <= self._frame_slider.maximum()):
            # A reset keyframe terminates the last inherited default while
            # preserving its effect on every preceding frame.
            self._future_overrides[next_frame] = {"reset": True}
        self._load_frame_controls()
        self._commit_to_analysis()
        self._refresh()

    def _step_frame(self, delta: int) -> None:
        self._play_timer.stop()
        self._play_btn.setText("▶")
        self._frame_slider.setValue(int(np.clip(
            self._preview_frame + int(delta),
            self._frame_slider.minimum(), self._frame_slider.maximum())))

    def _toggle_playback(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_btn.setText("▶")
            return
        fps = max(0.1, float(getattr(self._wizard.analysis, "fps", 10.0)))
        self._play_timer.start(int(np.clip(1000.0 / fps, 35, 400)))
        self._play_btn.setText("❚❚")

    def _play_next_frame(self) -> None:
        if self._preview_frame >= self._frame_slider.maximum():
            self._play_timer.stop()
            self._play_btn.setText("▶")
            return
        self._frame_slider.setValue(self._preview_frame + 1)

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
        old_frame_overrides = getattr(a, "dynamic_frame_overrides", {})
        old_future_overrides = getattr(a, "dynamic_future_overrides", {})
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
        cleaned_frame_overrides: dict[int, dict[str, object]] = {}
        for frame_index, entry in self._frame_overrides.items():
            cleaned: dict[str, object] = {}
            if entry.get("threshold") is not None:
                cleaned["threshold"] = float(np.clip(
                    entry["threshold"], 0.0, 1.0))
            if "replace" in entry:
                cleaned["replace"] = bool(entry["replace"])
            for channel in ("include", "exclude"):
                mask = entry.get(channel)
                if mask is not None:
                    arr = np.asarray(mask, dtype=bool)
                    if arr.any():
                        cleaned[channel] = arr.copy()
            if cleaned:
                cleaned_frame_overrides[int(frame_index)] = cleaned
        a.dynamic_frame_overrides = cleaned_frame_overrides
        cleaned_future_overrides: dict[int, dict[str, object]] = {}
        for start_frame, entry in self._future_overrides.items():
            cleaned = self._clone_override(entry)
            for channel in ("include", "exclude"):
                mask = cleaned.get(channel)
                if mask is not None and not np.asarray(mask, dtype=bool).any():
                    cleaned.pop(channel, None)
            if cleaned:
                cleaned_future_overrides[int(start_frame)] = cleaned
        a.dynamic_future_overrides = cleaned_future_overrides
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

        def overrides_equal(left, right) -> bool:
            if set(left) != set(right):
                return False
            for frame_index in left:
                a_entry, b_entry = left[frame_index], right[frame_index]
                if bool(a_entry.get("reset", False)) != bool(
                        b_entry.get("reset", False)):
                    return False
                if a_entry.get("threshold") != b_entry.get("threshold"):
                    return False
                if (("replace" in a_entry) != ("replace" in b_entry) or
                        bool(a_entry.get("replace", False)) != bool(
                            b_entry.get("replace", False))):
                    return False
                for channel in ("include", "exclude"):
                    a_mask, b_mask = a_entry.get(channel), b_entry.get(channel)
                    if a_mask is None or b_mask is None:
                        if a_mask is not None or b_mask is not None:
                            return False
                    elif not np.array_equal(a_mask, b_mask):
                        return False
            return True

        frame_overrides_changed = not overrides_equal(
            old_frame_overrides, a.dynamic_frame_overrides)
        future_overrides_changed = not overrides_equal(
            old_future_overrides, a.dynamic_future_overrides)
        if (old_params != new_params or masks_changed or
                frame_overrides_changed or future_overrides_changed):
            a.results.clear()

    def _refresh(self) -> None:
        a = self._wizard.analysis
        calibration = a.strain_reference_image()
        preview = a.frame_image(self._preview_frame)
        method = self._cb_method.currentText()
        self._commit_to_analysis()

        if calibration is None:
            return
        if preview is None:
            preview = calibration

        if method == "None":
            self._hint.setText("No dynamic masking — every pixel of the static ROI "
                               "is analysed in every frame.")
            self._canvas.clear_result_overlay()
            self._stats.setText("")
            self._thr_lbl.setText("—")
            return

        self._hint.setText(
            "Green = kept. Grey = outside the ROI you drew, never analysed. "
            "Regions with too little texture are dropped for that frame; their "
            "eligibility is reassessed on the next frame.")

        roi = a.make_dynamic_roi()
        roi.calibrate(calibration)
        auto = roi.auto_threshold_normalised()
        if self._auto_chk.isChecked() and auto is not None:
            self._updating = True
            self._slider.setValue(int(round(auto * 100)))
            self._updating = False
        self._thr_lbl.setText(f"{self._slider.value()}%")
        roi.set_threshold_normalised(
            a.dynamic_threshold_for_frame(self._preview_frame))

        # Static ROI and overrides are fixed image-space masks; the texture mask
        # is evaluated on whichever sequence frame the operator is previewing.
        m = roi.mask(preview, reference_frame=True)
        if m is None:
            self._canvas.clear_result_overlay()
            return
        m = a.apply_dynamic_frame_override(
            m, self._preview_frame, clip_static=False)

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
        frame_entry = self._effective_local_override(self._preview_frame)
        frame_exc = frame_entry.get("exclude")
        frame_inc = frame_entry.get("include")
        if frame_exc is not None:
            overlay[np.asarray(frame_exc, dtype=bool)] = 5
        if frame_inc is not None:
            overlay[np.asarray(frame_inc, dtype=bool)] = 6
        self._canvas.set_result_overlay_indexed(overlay, [
            QColor(0, 0, 0, 0),
            QColor(15, 23, 42, 130),
            QColor(16, 185, 129, 70),
            QColor(239, 68, 68, 90),
            QColor(16, 185, 129, 130),
            QColor(249, 115, 22, 155),
            QColor(34, 211, 238, 155),
        ])
        denom = int(static.sum()) if static is not None else m.size
        kept = int((m & static).sum()) if static is not None else int(m.sum())
        pct = 100.0 * kept / max(1, denom)
        auto_txt = f"auto {auto*100:.0f}%" if auto is not None else "auto n/a"
        n_inc = int((self._include & inside).sum()) if self._include is not None else 0
        n_exc = int((self._exclude & inside).sum()) if self._exclude is not None else 0
        n_frame_inc = (int(np.asarray(frame_inc, dtype=bool).sum())
                       if frame_inc is not None else 0)
        n_frame_exc = (int(np.asarray(frame_exc, dtype=bool).sum())
                       if frame_exc is not None else 0)
        frame_thr = frame_entry.get("threshold")
        frame_text = (f", threshold {float(frame_thr) * 100:.0f}%"
                      if frame_thr is not None else "")
        self._stats.setText(
            f"Keeps {pct:.1f}% of the static ROI on frame {self._preview_frame}  ({auto_txt}).\n"
            f"Global: {n_inc:,} in, {n_exc:,} out.  "
            f"Frame: {n_frame_inc:,} in, {n_frame_exc:,} out{frame_text}.")

    def _on_continue(self) -> None:
        self._refresh_timer.stop()
        self._play_timer.stop()
        self._flush_active_channel()
        self._commit_to_analysis()
        try:
            self._wizard.analysis.save_settings()
        except Exception:
            pass
        self._wizard.go_params()

    def reset_page(self) -> None:
        self._include = None
        self._exclude = None
        self._frame_overrides = {}
        self._future_overrides = {}
        self._play_timer.stop()
        self._canvas.clear_roi()
        self._canvas.clear_result_overlay()
