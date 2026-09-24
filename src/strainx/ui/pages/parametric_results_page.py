"""Interactive comparison viewer for lazily loaded parametric DIC caches."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget, QSplitter,
    QStackedWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from strainx.core.compact_field import CompactField, finite_values
from strainx.core.parametric import ParametricSweep
from strainx.core.stats import field_summary, robust_limits
from strainx.core.units import Calibration, LENGTH_UNIT_ORDER
from strainx.ui import render
from strainx.ui.components import ResultColorBar
from strainx.ui.image_canvas import ImageCanvas
from strainx.ui.parametric_plot_3d import ParametricPlot3D
from strainx.ui.pages.results_page import _interpolate_between_subset_centres
from strainx.ui.result_controls import (
    CMAPS, DEFAULT_CMAP, DEFAULT_COVERAGE_TEXT, FieldFamilySelector,
)
from strainx.ui.theme import (
    C_BG, C_BORDER, C_CARD, C_SURFACE, C_TEXT, C_TEXT2, C_TEXT3,
)


PARAMETRIC_FIELDS = {
    "u": ("Horizontal displacement u", "px"),
    "v": ("Vertical displacement v", "px"),
    "magnitude": ("Displacement magnitude", "px"),
    "Vx": ("Horizontal velocity Vx", "px/s"),
    "Vy": ("Vertical velocity Vy", "px/s"),
    "Veff": ("Velocity magnitude", "px/s"),
    "Eeff_rate": ("Equivalent strain rate", "s⁻¹"),
    "Eeff_gl": ("Accumulated equivalent strain", ""),
    "corr": ("ZNSSD correlation cost", ""),
    "valid": ("Valid coverage", ""),
}

PARAMETRIC_FIELD_GROUPS = {
    "Displacement": ["u", "v", "magnitude"],
    "Velocity": ["Vx", "Vy", "Veff"],
    "Strain rate": ["Eeff_rate"],
    "Strain": ["Eeff_gl"],
    "Quality": ["corr", "valid"],
}


def _bar_colors(cmap_name: str) -> list[tuple[int, int, int]]:
    cmap = render.get_cmap(cmap_name, 64)
    return [tuple(int(channel * 255) for channel in cmap(i / 63)[:3])
            for i in range(64)]


class _ComparisonPanel(QFrame):
    def __init__(self, case_index: int, parent=None):
        super().__init__(parent)
        self.setObjectName("comparisonPanel")
        self.case_index = case_index
        self.values = self.rendered_rgb = None
        self.title_text = self.unit = ""
        self.spacing = 1
        self.origin = 0
        self.limits = (0.0, 1.0)
        self.cmap_name = DEFAULT_CMAP
        self.setStyleSheet(
            f"QFrame#comparisonPanel{{background:{C_CARD};border:1px solid {C_BORDER};"
            "border-radius:4px;}QLabel{background:transparent;border:none;}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)
        self.heading = QLabel()
        self.heading.setStyleSheet(
            f"color:{C_TEXT};font-size:11px;font-weight:700;"
            "background:transparent;border:none;")
        layout.addWidget(self.heading)

        self.canvas = ImageCanvas()
        self.canvas.seed_enabled = False
        self.canvas.setMinimumSize(280, 210)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.canvas.cursor_moved.connect(self._show_probe)
        self.canvas.set_linked_view(True)
        layout.addWidget(self.canvas, 1)

        self.probe = QLabel("Hover to probe · wheel zoom · middle-drag pan")
        self.probe.setStyleSheet(
            f"color:{C_TEXT2};font-size:9px;background:transparent;border:none;"
            "font-family:'Cascadia Mono','Consolas',monospace;")
        layout.addWidget(self.probe)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.stats.setStyleSheet(
            f"color:{C_TEXT2};font-size:9px;background:transparent;border:none;"
            "font-family:'Cascadia Mono','Consolas',monospace;")
        layout.addWidget(self.stats)

    def update_content(self, *, title: str, values, background: np.ndarray,
                       overlay: np.ndarray, rgb: np.ndarray,
                       limits: tuple[float, float], cmap_name: str, unit: str,
                       coverage: float, corr: float, spacing: int = 1,
                       origin: int = 0) -> None:
        self.values, self.rendered_rgb = values, rgb
        self.title_text, self.unit = title, unit
        self.spacing, self.origin = int(spacing), int(origin)
        self.limits, self.cmap_name = limits, cmap_name
        self.heading.setText(title)
        self.heading.setToolTip(title)
        self.canvas.set_image(
            background.astype(np.float32) / 255.0,
            keep_view=self.canvas._image_arr is not None)
        self.canvas.set_result_overlay_rgba(overlay)
        summary = field_summary(values)
        if summary is None:
            self.stats.setText("No finite values on this frame.")
        else:
            suffix = f" {unit}" if unit else ""
            self.stats.setText(
                f"mean {summary['mean']:.5g}{suffix}   "
                f"median {summary['median']:.5g}{suffix}   "
                f"std {summary['std']:.4g}{suffix}\n"
                f"P1–P99 {summary['p_low']:.4g}…{summary['p_high']:.4g}   "
                f"points {summary['count']:,}   valid {coverage:.1%}   "
                f"ZNSSD {corr:.5g}")
        if self.canvas.probe_position is None:
            self.probe.setText("Hover to probe · wheel zoom · middle-drag pan")
        else:
            self._show_probe(*self.canvas.probe_position, float("nan"))
        self.canvas.probe_annotation = None

    def _show_probe(self, x: int, y: int, _value: float) -> None:
        if (self.values is None or x < 0 or y < 0 or
                y >= self.values.shape[0] or x >= self.values.shape[1]):
            self.probe.setText("Outside image")
            return
        value = _interpolate_between_subset_centres(
            self.values, x, y, self.spacing, self.origin)
        is_interpolated = bool(
            (x - self.origin) % self.spacing or
            (y - self.origin) % self.spacing)
        suffix = " (interpolated)" if value is not None and is_interpolated else ""
        shown = (f"{float(value):.6g} {self.unit}{suffix}"
                 if value is not None else "no valid sample")
        self.probe.setText(f"x={x}  y={y}  {shown}")
        self.canvas.probe_position = (x, y)


class ParametricResultsPage(QWidget):
    def __init__(self, wizard):
        super().__init__()
        self._wizard = wizard
        self._sweep = None
        self._calibration = Calibration()
        self._panels: dict[int, _ComparisonPanel] = {}
        self._linked_state = None
        self._page_index = 0
        self._loaded_key, self._loaded = None, []
        self._global_ranges: dict[tuple, tuple[float, float]] = {}
        self._image_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._compute_queue = queue.SimpleQueue()
        self._compute_progress_queue = queue.SimpleQueue()
        self._compute_thread = None
        self._compute_request = None
        self._compute_cancel = None
        self._compute_started = 0.0
        self._compute_display_second = -1
        self._compute_display_percent = -1
        self._compute_percent = 0
        self._compute_label = "Computing selected results"
        self._compute_poll = QTimer(self)
        self._compute_poll.setInterval(50)
        self._compute_poll.timeout.connect(self._poll_compute)
        self._play_timer = QTimer(self)
        self._play_timer.setSingleShot(True)
        self._play_timer.timeout.connect(self._advance)
        self._probe_plot_timer = QTimer(self)
        self._probe_plot_timer.setSingleShot(True)
        self._probe_plot_timer.setInterval(60)
        self._probe_plot_timer.timeout.connect(
            lambda: self._update_plot(self._loaded, self._selected()))
        self._plot_axes = None
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit_all)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        top = QWidget()
        top.setObjectName("parametricTopBar")
        top.setStyleSheet(
            f"QWidget#parametricTopBar{{background:{C_SURFACE};"
            f"border-bottom:1px solid {C_BORDER};}}")
        rows = QVBoxLayout(top)
        rows.setContentsMargins(14, 5, 14, 5)
        rows.setSpacing(4)
        select_row, display_row = QHBoxLayout(), QHBoxLayout()
        select_row.setSpacing(8)
        display_row.setSpacing(8)
        rows.addLayout(select_row)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background:{C_BORDER};max-height:1px;border:none;")
        rows.addWidget(separator)
        rows.addLayout(display_row)

        home = QPushButton("← New session")
        home.setFixedWidth(120)
        home.clicked.connect(self._wizard.new_session)
        select_row.addWidget(home)
        add = QPushButton("+ Add HDF5")
        add.clicked.connect(self._add_hdf5)
        select_row.addWidget(add)
        self._sidebar_toggle = QPushButton("Hide parameters")
        self._sidebar_toggle.setCheckable(True)
        self._sidebar_toggle.toggled.connect(self._set_sidebar_collapsed)
        select_row.addWidget(self._sidebar_toggle)
        self._plot_toggle = QPushButton("Hide graph")
        self._plot_toggle.setCheckable(True)
        self._plot_toggle.toggled.connect(self._set_plot_collapsed)
        select_row.addWidget(self._plot_toggle)
        select_row.addSpacing(10)
        self._field_selector = FieldFamilySelector(
            PARAMETRIC_FIELDS, PARAMETRIC_FIELD_GROUPS, "u")
        self._field_selector.field_changed.connect(self._field_changed)
        select_row.addWidget(self._field_selector)
        self._cat_combo = self._field_selector.category_combo
        self._field_btns = self._field_selector.buttons

        select_row.addWidget(QLabel("Columns:"))
        self._columns = QSpinBox()
        self._columns.setRange(1, 6)
        self._columns.setValue(2)
        self._columns.valueChanged.connect(self._columns_changed)
        select_row.addWidget(self._columns)
        self._fit_all_button = QPushButton("Fit all")
        self._fit_all_button.setToolTip("Centre and fit every visible panel.")
        self._fit_all_button.clicked.connect(self._fit_all)
        select_row.addWidget(self._fit_all_button)
        select_row.addWidget(QLabel("Panels:"))
        self._panel_mode = QComboBox()
        self._panel_mode.addItem("All", 0)
        self._panel_mode.addItem("4 per page", 4)
        self._panel_mode.addItem("6 per page", 6)
        self._panel_mode.addItem("9 per page", 9)
        self._panel_mode.setCurrentIndex(2)
        self._panel_mode.currentIndexChanged.connect(self._panel_mode_changed)
        select_row.addWidget(self._panel_mode)
        self._page_back = QPushButton("‹")
        self._page_back.setMinimumWidth(34)
        self._page_back.clicked.connect(lambda: self._change_page(-1))
        select_row.addWidget(self._page_back)
        self._page_text = QLabel("Page 1 / 1")
        select_row.addWidget(self._page_text)
        self._page_forward = QPushButton("›")
        self._page_forward.setMinimumWidth(34)
        self._page_forward.clicked.connect(lambda: self._change_page(1))
        select_row.addWidget(self._page_forward)
        select_row.addStretch()
        self._summary = QLabel()
        self._summary.setStyleSheet(f"color:{C_TEXT2};font-size:10px;")
        select_row.addWidget(self._summary)

        display_row.addWidget(QLabel("Colormap:"))
        self._cmap = QComboBox()
        self._cmap.addItems(CMAPS)
        self._cmap.setCurrentText(DEFAULT_CMAP)
        self._cmap.setFixedWidth(108)
        self._cmap.currentTextChanged.connect(self.refresh)
        display_row.addWidget(self._cmap)
        display_row.addWidget(QLabel("Scale:"))
        self._scale_group = QButtonGroup(self)
        self._frame_scale = QRadioButton("Frame")
        self._global_scale = QRadioButton("Sequence")
        self._manual_scale = QRadioButton("Range")
        self._frame_scale.setToolTip("One shared scale for all shown cases on this frame.")
        self._global_scale.setToolTip(
            "One shared scale across every frame of the selected cases. Computed on demand.")
        self._manual_scale.setToolTip("Use the limits entered here.")
        for button in (self._frame_scale, self._global_scale, self._manual_scale):
            self._scale_group.addButton(button)
            display_row.addWidget(button)
        self._frame_scale.setChecked(True)
        self._scale_group.buttonToggled.connect(self._scale_changed)

        self._range_min, self._range_max = QDoubleSpinBox(), QDoubleSpinBox()
        for box in (self._range_min, self._range_max):
            box.setDecimals(6)
            box.setRange(-1e12, 1e12)
            box.setFixedWidth(96)
            box.setEnabled(False)
            box.setKeyboardTracking(False)
            box.valueChanged.connect(self.refresh)
            display_row.addWidget(box)
        self._fit_range = QPushButton("Fit range")
        self._fit_range.setFixedWidth(72)
        self._fit_range.setEnabled(False)
        self._fit_range.setToolTip("Use this frame's shared limits as a fixed range.")
        self._fit_range.clicked.connect(self._fit_current_range)
        display_row.addWidget(self._fit_range)
        self._symmetric = QCheckBox("Sym")
        self._symmetric.setToolTip("Centre the shared colour scale on zero.")
        self._symmetric.stateChanged.connect(self.refresh)
        display_row.addWidget(self._symmetric)
        display_row.addWidget(QLabel("Coverage:"))
        self._coverage = QComboBox()
        for text, value in (("100%", 100.0), ("99.5%", 99.5), ("99%", 99.0),
                            ("98%", 98.0), ("95%", 95.0), ("90%", 90.0)):
            self._coverage.addItem(text, value)
        self._coverage.setCurrentText(DEFAULT_COVERAGE_TEXT)
        self._coverage.setFixedWidth(74)
        self._coverage.setToolTip(
            "Share of finite samples covered by automatic colour limits.")
        self._coverage.currentIndexChanged.connect(self.refresh)
        display_row.addWidget(self._coverage)
        self._flag_clipped = QCheckBox("Flag clipped")
        self._flag_clipped.setChecked(True)
        self._flag_clipped.setToolTip(
            "Show values outside the chosen range in cyan or magenta.")
        self._flag_clipped.stateChanged.connect(self.refresh)
        display_row.addWidget(self._flag_clipped)
        display_row.addStretch()
        export_grid = QPushButton("Export visible page…")
        export_grid.clicked.connect(self._export_grid)
        display_row.addWidget(export_grid)
        root.addWidget(top)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        self._sidebar = QWidget()
        self._sidebar.setObjectName("parametricSidebar")
        self._sidebar.setFixedWidth(245)
        self._sidebar.setStyleSheet(
            f"QWidget#parametricSidebar{{background:{C_SURFACE};"
            f"border-right:1px solid {C_BORDER};}}")
        side = QVBoxLayout(self._sidebar)
        side.setContentsMargins(10, 9, 10, 9)
        side.setSpacing(4)
        title = QLabel("PARAMETERS TO VARY")
        title.setStyleSheet(
            f"color:{C_TEXT3};font-size:9px;font-weight:700;letter-spacing:1px;")
        side.addWidget(title)
        self._cases = QListWidget()
        self._cases.itemChanged.connect(self._selection_changed)
        self._axis_x = QComboBox()
        for label, value in (("Subset radius r", "radius"),
                             ("Grid spacing", "grid"),
                             ("Strain window", "window"),
                             ("Temporal span", "temporal")):
            self._axis_x.addItem(label, value)
        self._axis_x.currentIndexChanged.connect(self._axes_changed)
        self._axis_y = QComboBox()
        self._axis_y.addItem("No second variable", None)
        for label, value in (("Grid spacing", "grid"),
                             ("Subset radius r", "radius"),
                             ("Strain window", "window"),
                             ("Temporal span", "temporal")):
            self._axis_y.addItem(label, value)
        self._axis_y.currentIndexChanged.connect(self._axes_changed)
        self._cases.setVisible(False)
        side.addWidget(QLabel("Graph x axis"))
        side.addWidget(self._axis_x)
        self._axis_x_range, self._axis_x_min, self._axis_x_max = (
            self._make_axis_range())
        side.addWidget(self._axis_x_range)
        self._axis_y_values_label = QLabel("Graph y axis (optional)")
        side.addWidget(self._axis_y_values_label)
        side.addWidget(self._axis_y)
        self._axis_y_range, self._axis_y_min, self._axis_y_max = (
            self._make_axis_range())
        side.addWidget(self._axis_y_range)
        side.addSpacing(3)
        fixed_heading = QLabel("FIXED VARIABLES")
        fixed_heading.setStyleSheet(
            f"color:{C_TEXT3};font-size:9px;font-weight:700;"
            "letter-spacing:1px;border:none;")
        side.addWidget(fixed_heading)
        self._fixed = {}
        self._fixed_rows = {}
        self._fixed_labels = {}
        for key, label in (("radius", "Subset radius"),
                           ("grid", "Grid spacing"),
                           ("window", "Strain window"),
                           ("temporal", "Temporal span")):
            row = QWidget()
            row.setStyleSheet("background:transparent;")
            row.setSizePolicy(QSizePolicy.Policy.Preferred,
                              QSizePolicy.Policy.Fixed)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)
            name = QLabel(label)
            name.setStyleSheet(
                f"color:{C_TEXT2};font-size:11px;"
                "background:transparent;border:none;")
            row_layout.addWidget(name, 1)
            combo = QComboBox()
            combo.setFixedWidth(96)
            combo.currentIndexChanged.connect(self._selection_changed)
            row_layout.addWidget(combo)
            side.addWidget(row)
            self._fixed[key] = combo
            self._fixed_rows[key] = row
            self._fixed_labels[key] = name
        self._summary_mode = QComboBox()
        self._summary_mode.addItem("Hover value", "probe")
        self._summary_mode.addItem("ROI mean", "mean")
        self._summary_mode.addItem("ROI median", "median")
        self._summary_mode.currentIndexChanged.connect(
            lambda *_: self._update_plot(self._loaded, self._selected()))
        side.addWidget(self._summary_mode)
        side.addWidget(self._cases, 0)
        side.addStretch(1)
        self._cache_note = QLabel(
            "Temporal results are saved when viewed.")
        self._cache_note.setWordWrap(True)
        self._cache_note.setSizePolicy(QSizePolicy.Policy.Preferred,
                                       QSizePolicy.Policy.Fixed)
        self._cache_note.setStyleSheet(
            f"color:{C_TEXT2};font-size:9px;"
            "background:transparent;border:none;")
        side.addWidget(self._cache_note)
        body_layout.addWidget(self._sidebar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f"background:{C_BG};border:none;")
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(14, 14, 14, 14)
        self._grid.setSpacing(12)
        self._empty_label = QLabel("No combinations match the selected variables.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(f"color:{C_TEXT2};font-size:13px;")
        self._scroll.setWidget(self._grid_host)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self._scroll)
        self._plot_figure = Figure(figsize=(7, 2.2))
        self._plot_figure.subplots_adjust(
            left=0.11, right=0.86, bottom=0.22, top=0.88)
        self._plot_canvas = FigureCanvasQTAgg(self._plot_figure)
        self._plot_3d = ParametricPlot3D()
        self._plot_stack = QStackedWidget()
        self._plot_stack.addWidget(self._plot_canvas)
        self._plot_stack.addWidget(self._plot_3d)
        self._probe_coordinate = None
        self._plot_stack.setMinimumHeight(150)
        self._splitter.addWidget(self._plot_stack)
        self._splitter.setSizes([650, 240])
        self._plot_sizes = [650, 240]
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._splitter, 1)
        self._colorbar_row = QWidget()
        colorbar_layout = QHBoxLayout(self._colorbar_row)
        colorbar_layout.setContentsMargins(14, 0, 14, 0)
        colorbar_layout.setSpacing(6)
        self._shared_colorbar = ResultColorBar()
        self._shared_colorbar.setToolTip(
            "Shared colour scale for the visible comparisons.")
        colorbar_layout.addWidget(self._shared_colorbar)
        self._colorbar_unit = QLabel()
        self._colorbar_unit.setMinimumWidth(38)
        colorbar_layout.addWidget(self._colorbar_unit)
        right_layout.addWidget(self._colorbar_row)
        body_layout.addWidget(right, 1)
        root.addWidget(body, 1)

        bottom = QWidget()
        bottom.setObjectName("parametricTransport")
        bottom.setStyleSheet(
            f"QWidget#parametricTransport{{background:{C_SURFACE};"
            f"border-top:1px solid {C_BORDER};}}")
        transport = QHBoxLayout(bottom)
        transport.setContentsMargins(16, 6, 16, 6)
        previous = QPushButton("Previous")
        previous.setFixedWidth(76)
        previous.clicked.connect(lambda: self._step_frame(-1))
        transport.addWidget(previous)
        self._play = QPushButton("Play")
        self._play.setCheckable(True)
        self._play.setFixedWidth(58)
        self._play.toggled.connect(self._toggle_play)
        transport.addWidget(self._play)
        following = QPushButton("Next")
        following.setFixedWidth(58)
        following.clicked.connect(lambda: self._step_frame(1))
        transport.addWidget(following)
        self._frame_text = QLabel("Frame 1 / 1")
        transport.addWidget(self._frame_text)
        self._frame = QSlider(Qt.Orientation.Horizontal)
        self._frame.valueChanged.connect(self._data_changed)
        transport.addWidget(self._frame, 1)
        transport.addWidget(QLabel("Playback:"))
        self._playback_rate = QComboBox()
        for label, fps in (("2 fps", 2), ("5 fps", 5), ("10 fps", 10),
                           ("20 fps", 20)):
            self._playback_rate.addItem(label, fps)
        self._playback_rate.setCurrentText("5 fps")
        self._playback_rate.setFixedWidth(78)
        transport.addWidget(self._playback_rate)
        transport.addSpacing(12)
        transport.addWidget(QLabel("1 px ="))
        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setDecimals(9)
        self._pixel_size.setRange(0.0, 1e9)
        self._pixel_size.setSpecialValueText("uncalibrated")
        self._pixel_size.setKeyboardTracking(False)
        self._pixel_size.setFixedWidth(116)
        self._pixel_size.setToolTip(
            "Physical size of one pixel. Changes display values only; "
            "the cached analysis remains in pixels.")
        self._pixel_size.valueChanged.connect(self._pixel_size_changed)
        transport.addWidget(self._pixel_size)
        self._length_unit = QComboBox()
        self._length_unit.addItems(LENGTH_UNIT_ORDER)
        self._length_unit.setFixedWidth(64)
        self._length_unit.currentTextChanged.connect(self._length_unit_changed)
        transport.addWidget(self._length_unit)
        root.addWidget(bottom)

    @property
    def current_field(self) -> str:
        return self._field_selector.field

    def on_enter(self) -> None:
        sweep = getattr(self._wizard, "parametric_sweep", None)
        if sweep is None:
            return
        self.set_sweep(sweep) if sweep is not self._sweep else self.refresh()

    def on_leave(self) -> None:
        self._play_timer.stop()
        self._probe_plot_timer.stop()
        self._play.blockSignals(True)
        self._play.setChecked(False)
        self._play.blockSignals(False)
        self._play.setText("Play")
        if self._compute_cancel is not None:
            self._compute_cancel[0] = True
        self._compute_request = None

    def set_sweep(self, sweep) -> None:
        if self._compute_cancel is not None:
            self._compute_cancel[0] = True
        self._sweep = sweep
        self._calibration = getattr(sweep, "calibration", None) or getattr(
            getattr(self._wizard, "analysis", None), "calibration", None
        ) or Calibration()
        self._pixel_size.blockSignals(True)
        self._length_unit.blockSignals(True)
        self._length_unit.setCurrentText(self._calibration.display_unit)
        self._pixel_size.setValue(
            self._calibration.pixel_size_in(self._calibration.display_unit) or 0.0)
        self._length_unit.blockSignals(False)
        self._pixel_size.blockSignals(False)
        self._compute_request = None
        self._loaded_key = None
        self._linked_state = None
        self._global_ranges.clear()
        self._image_cache.clear()
        self._cases.blockSignals(True)
        self._cases.clear()
        for index, case in enumerate(sweep.cases):
            item = QListWidgetItem(case.label)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if index < 4 else Qt.CheckState.Unchecked)
            item.setToolTip(
                f"Mean valid coverage: {case.mean_valid_fraction:.1%}\n"
                f"Median ZNSSD: {case.median_corr:.5g}\n{case.path}")
            self._cases.addItem(item)
        self._cases.blockSignals(False)
        self._fill_axis_range(self._axis_x_min, self._axis_x_max,
                              sweep, self._axis_x.currentData())
        self._fill_axis_range(self._axis_y_min, self._axis_y_max,
                              sweep, self._axis_y.currentData() or "grid")
        values_by_key = {
            "radius": sorted({case.subset_radius for case in sweep.cases}),
            "grid": sorted({case.grid_spacing for case in sweep.cases}),
            "window": sorted(sweep.strain_windows or
                             getattr(sweep, "requested_strain_windows", (1,))),
            "temporal": sorted(getattr(sweep, "temporal_spans", (1,))),
        }
        for key, combo in self._fixed.items():
            combo.blockSignals(True)
            combo.clear()
            for value in values_by_key[key]:
                combo.addItem(f"{value} {'frames' if key == 'temporal' else 'px'}",
                              int(value))
            combo.blockSignals(False)
        self._field_selector.set_available_fields(getattr(
            sweep, "available_fields", PARAMETRIC_FIELDS))
        if sweep.strain_windows:
            self._cache_note.setText(
                "Temporal results are saved when viewed.")
        else:
            self._cache_note.setText(
                "Correlation-only sweep. Resume analysis to build the "
                "strain cache without rerunning correlation.")
        self._frame.setRange(0, max(0, sweep.frame_count - 1))
        self._update_fixed_visibility()
        self.refresh()

    def _field_factor(self) -> tuple[float, str]:
        base_unit = PARAMETRIC_FIELDS[self.current_field][1]
        return self._calibration.factor_and_unit(self.current_field, base_unit)

    def _set_display_calibration(self, calibration: Calibration) -> None:
        old_factor, _ = self._field_factor()
        self._calibration = calibration
        if self._sweep is not None:
            self._sweep.calibration = calibration
            save = getattr(self._sweep, "save_display_calibration", None)
            if save is not None:
                try:
                    save()
                except OSError as exc:
                    self._pixel_size.setToolTip(
                        f"Display calibration is active but could not be saved: {exc}")
        new_factor, _ = self._field_factor()
        if self._manual_scale.isChecked() and old_factor > 0:
            ratio = new_factor / old_factor
            for box in (self._range_min, self._range_max):
                box.blockSignals(True)
                box.setValue(box.value() * ratio)
                box.blockSignals(False)
        self.refresh()

    def _pixel_size_changed(self, value: float) -> None:
        unit = self._length_unit.currentText()
        self._set_display_calibration(Calibration.from_pixel_size(value, unit))

    def _length_unit_changed(self, unit: str) -> None:
        calibration = Calibration(self._calibration.metres_per_pixel, unit)
        self._pixel_size.blockSignals(True)
        self._pixel_size.setValue(calibration.pixel_size_in(unit) or 0.0)
        self._pixel_size.blockSignals(False)
        self._set_display_calibration(calibration)

    def _make_axis_range(self) -> tuple[QWidget, QSpinBox, QSpinBox]:
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        boxes = []
        for label in ("Min", "Max"):
            column = QWidget()
            column.setStyleSheet("background:transparent;")
            column_layout = QVBoxLayout(column)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(2)
            caption = QLabel(label)
            caption.setStyleSheet(
                f"color:{C_TEXT3};font-size:9px;background:transparent;")
            column_layout.addWidget(caption)
            box = QSpinBox()
            box.setRange(0, 0)
            box.setToolTip("Inclusive range of computed parameter values.")
            column_layout.addWidget(box)
            layout.addWidget(column, 1)
            boxes.append(box)
        lower, upper = boxes
        lower.valueChanged.connect(
            lambda value: self._axis_range_changed(lower, upper, value))
        upper.valueChanged.connect(
            lambda value: self._axis_range_changed(upper, lower, value))
        return row, lower, upper

    @staticmethod
    def _fill_axis_range(lower: QSpinBox, upper: QSpinBox,
                         sweep, variable: str) -> None:
        values = {
            "radius": {case.subset_radius for case in sweep.cases},
            "grid": {case.grid_spacing for case in sweep.cases},
            "window": set(sweep.strain_windows or
                          getattr(sweep, "requested_strain_windows", (1,))),
            "temporal": set(getattr(sweep, "temporal_spans", (1,))),
        }[variable]
        first, last = min(values, default=0), max(values, default=0)
        suffix = " frames" if variable == "temporal" else " px"
        for box, value in ((lower, first), (upper, last)):
            box.blockSignals(True)
            box.setRange(int(first), int(last))
            box.setSuffix(suffix)
            box.setValue(int(value))
            box.setEnabled(bool(values))
            box.blockSignals(False)

    def _axis_range_changed(self, changed: QSpinBox,
                            other: QSpinBox, value: int) -> None:
        if ((changed is self._axis_x_min or changed is self._axis_y_min)
                and value > other.value()) or (
                (changed is self._axis_x_max or changed is self._axis_y_max)
                and value < other.value()):
            other.blockSignals(True)
            other.setValue(value)
            other.blockSignals(False)
        self._selection_changed()

    def _axes_changed(self, *_args) -> None:
        if self._sweep is None:
            return
        x_name = self._axis_x.currentData()
        y_name = self._axis_y.currentData()
        if x_name == y_name:
            self._axis_y.blockSignals(True)
            self._axis_y.setCurrentIndex(0)
            self._axis_y.blockSignals(False)
            y_name = None
        sender = self.sender()
        if x_name and sender is not self._axis_y:
            self._fill_axis_range(self._axis_x_min, self._axis_x_max,
                                  self._sweep, x_name)
        if y_name and sender is not self._axis_x:
            self._fill_axis_range(self._axis_y_min, self._axis_y_max,
                                  self._sweep, y_name)
        self._update_fixed_visibility()
        self._selection_changed()

    def _update_fixed_visibility(self) -> None:
        varied = {self._axis_x.currentData(), self._axis_y.currentData()}
        for key, row in self._fixed_rows.items():
            row.setVisible(key not in varied)
        show_y_values = self._axis_y.currentData() is not None
        self._axis_y_values_label.setVisible(show_y_values)
        self._axis_y_range.setVisible(show_y_values)

    def _columns_changed(self, *_args) -> None:
        self._reflow()
        self._request_fit_all()

    def _request_fit_all(self) -> None:
        self._fit_timer.start(0)

    def _fit_all(self) -> None:
        if self._sweep is None:
            return
        self._linked_state = (0.5, 0.5, 1.0)
        for key in self._visible_keys():
            panel = self._panels.get(key)
            if panel is not None:
                panel.canvas.set_view_state(self._linked_state)

    def _set_sidebar_collapsed(self, collapsed: bool) -> None:
        self._sidebar.setVisible(not collapsed)
        self._sidebar_toggle.setText(
            "Show parameters" if collapsed else "Hide parameters")
        self._request_fit_all()

    def _set_plot_collapsed(self, collapsed: bool) -> None:
        if collapsed:
            self._plot_sizes = self._splitter.sizes()
        self._plot_stack.setVisible(not collapsed)
        self._plot_toggle.setText("Show graph" if collapsed else "Hide graph")
        if not collapsed:
            self._splitter.setSizes(self._plot_sizes)
            self._update_plot(self._loaded, self._selected())
        self._request_fit_all()

    def _selected(self) -> list[tuple[int, int, int]]:
        if self._sweep is None:
            return []
        x_name = self._axis_x.currentData()
        y_name = self._axis_y.currentData()
        selected = {}
        for key, lower, upper in (
                (x_name, self._axis_x_min, self._axis_x_max),
                (y_name, self._axis_y_min, self._axis_y_max)):
            if key is not None:
                selected[key] = (lower.value(), upper.value())
        fixed = {key: combo.currentData() for key, combo in self._fixed.items()}
        windows = self._sweep.strain_windows or getattr(
            self._sweep, "requested_strain_windows", (1,))
        spans = getattr(self._sweep, "temporal_spans", (1,))
        combos = []
        for case_index, case in enumerate(self._sweep.cases):
            for window in windows:
                for span in spans:
                    current = {"radius": case.subset_radius,
                               "grid": case.grid_spacing,
                               "window": int(window),
                               "temporal": int(span)}
                    allowed = True
                    for axis in (x_name, y_name):
                        if axis is not None and not (
                                selected[axis][0] <= current[axis] <=
                                selected[axis][1]):
                            allowed = False
                    for variable, value in current.items():
                        if variable not in (x_name, y_name) and fixed[variable] is not None:
                            if value != int(fixed[variable]):
                                allowed = False
                    if allowed:
                        combos.append((case_index, int(window), int(span)))
        return combos

    def _check_all(self, checked: bool) -> None:
        self._cases.blockSignals(True)
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for index in range(self._cases.count()):
            self._cases.item(index).setCheckState(state)
        self._cases.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self, *_args) -> None:
        self._global_ranges.clear()
        self._data_changed()

    def _field_changed(self, _field: str) -> None:
        if (self.current_field == "corr" and any(
                key[2] > 1 for key in self._selected())):
            self._field_selector.set_field("u")
        self._data_changed()

    def _data_changed(self, *_args) -> None:
        if (self.current_field == "corr" and self._sweep is not None and
                any(span > 1 for _, _, span in self._selected())):
            self._field_selector.set_field("u", emit=False)
        self._loaded_key = None
        self.refresh()

    def _step_frame(self, delta: int) -> None:
        self._frame.setValue(max(
            self._frame.minimum(), min(self._frame.maximum(),
                                       self._frame.value() + delta)))

    def _toggle_play(self, playing: bool) -> None:
        self._play.setText("Pause" if playing else "Play")
        if playing:
            if self._frame.value() >= self._frame.maximum():
                self._frame.setValue(self._frame.minimum())
            if self._compute_thread is None:
                self._schedule_next_frame()
        else:
            self._play_timer.stop()
            if self._sweep is not None:
                self._update_plot(self._loaded, self._selected())

    def _advance(self) -> None:
        if not self._play.isChecked():
            return
        if self._frame.value() >= self._frame.maximum():
            self._play.setChecked(False)
            return
        self._frame.setValue(self._frame.value() + 1)
        if self._compute_thread is None:
            self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        if self._play.isChecked() and not self._play_timer.isActive():
            self._play_timer.start(
                max(20, int(1000 / int(self._playback_rate.currentData()))))

    def _scale_changed(self, _button, checked: bool) -> None:
        if not checked:
            return
        manual = self._manual_scale.isChecked()
        for widget in (self._range_min, self._range_max, self._fit_range):
            widget.setEnabled(manual)
        if manual and self._range_min.value() == self._range_max.value():
            self._fit_current_range()
        else:
            self.refresh()

    def _load_current(self, selected):
        key = (tuple(selected), self._frame.value(), self.current_field)
        if key != self._loaded_key:
            self._loaded = self._read_selected_frame(
                self._sweep, selected, key[1], key[2])
            self._loaded_key = key
        return self._loaded

    @staticmethod
    def _read_selected_frame(sweep, selected, frame: int, field: str,
                             cancel_flag=None, on_step=None, on_done=None):
        loaded = []
        for case, window, span in selected:
            if cancel_flag is not None and cancel_flag[0]:
                raise RuntimeError("Temporal calculation cancelled.")
            try:
                kwargs = {"temporal_span": span}
                if isinstance(sweep, ParametricSweep):
                    kwargs["cancel_flag"] = cancel_flag
                    if on_step is not None:
                        kwargs["progress_cb"] = lambda fraction, _message: on_step(fraction)
                values, path = sweep.frame_data(case, frame, field, window,
                                                **kwargs)
            except ValueError as exc:
                if "Insufficient temporal history" not in str(exc):
                    raise
                template, path = sweep.frame_data(case, 0, "u", window)
                values = CompactField.empty(template.shape)
            loaded.append(((case, window, span), values, path))
            if on_done is not None:
                on_done()
        return loaded

    @staticmethod
    def _calculate_sequence_limits(sweep, selected, field: str,
                                   coverage: float, cancel_flag=None,
                                   on_step=None, on_done=None
                                   ) -> tuple[float, float]:
        samples = []
        for case, window, span in selected:
            for frame in range(max(0, span - 1), sweep.frame_count):
                if cancel_flag is not None and cancel_flag[0]:
                    raise RuntimeError("Temporal calculation cancelled.")
                kwargs = {"temporal_span": span}
                if isinstance(sweep, ParametricSweep):
                    kwargs["cancel_flag"] = cancel_flag
                    if on_step is not None:
                        kwargs["progress_cb"] = lambda fraction, _message: on_step(fraction)
                values, _ = sweep.frame_data(case, frame, field, window,
                                             **kwargs)
                if on_done is not None:
                    on_done()
                finite = finite_values(values)
                if finite.size > 50_000:
                    finite = finite[::finite.size // 50_000 + 1]
                if finite.size:
                    samples.append(finite)
        limits = robust_limits(
            np.concatenate(samples), coverage) if samples else None
        return limits or (0.0, 1.0)

    def _request_compute(self, selected, *, need_global: bool) -> None:
        coverage = float(self._coverage.currentData())
        request = (self._sweep, tuple(selected), self._frame.value(),
                   self.current_field, coverage if need_global else None)
        self._compute_request = request
        self._summary.setText("Computing selected results…")
        if self._compute_thread is None:
            self._start_compute(request)
        elif request != self._running_request and self._compute_cancel is not None:
            self._compute_cancel[0] = True

    def _start_compute(self, request) -> None:
        sweep, selected, frame, field, coverage = request
        cancel_flag = [False]
        self._compute_cancel = cancel_flag
        self._running_request = request
        self._compute_started = time.monotonic()
        self._compute_display_second = -1
        self._compute_display_percent = -1
        self._compute_percent = 0
        self._compute_label = (
            "Evaluating temporal results" if any(
                span > 1 for _, _, span in selected) else
            "Computing sequence scale" if coverage is not None else
            "Loading plot data")
        total_jobs = len(selected) + (sum(
            max(0, sweep.frame_count - max(0, span - 1))
            for _, _, span in selected) if coverage is not None else 0)

        def work():
            completed = 0
            last_reported = -1.0

            def report(fraction):
                nonlocal last_reported
                overall = (completed + max(0.0, min(1.0, float(fraction)))) / max(1, total_jobs)
                if overall - last_reported >= 0.005 or overall >= 1.0:
                    self._compute_progress_queue.put((request, overall))
                    last_reported = overall

            def done():
                nonlocal completed
                completed += 1
                report(0.0)

            try:
                loaded = self._read_selected_frame(
                    sweep, selected, frame, field, cancel_flag, report, done)
                limits = (self._calculate_sequence_limits(
                    sweep, selected, field, coverage, cancel_flag, report, done)
                    if coverage is not None else None)
                report(0.0)
                self._compute_queue.put((request, loaded, limits, None))
            except Exception as exc:
                self._compute_queue.put((request, None, None, str(exc)))

        self._compute_thread = threading.Thread(
            target=work, name="strainX-temporal-view", daemon=True)
        self._compute_thread.start()
        self._compute_poll.start()

    def _poll_compute(self) -> None:
        while True:
            try:
                progress_request, fraction = self._compute_progress_queue.get_nowait()
            except queue.Empty:
                break
            if progress_request == self._running_request:
                self._compute_percent = max(
                    self._compute_percent, min(100, int(100 * fraction)))
        try:
            request, loaded, limits, error = self._compute_queue.get_nowait()
        except queue.Empty:
            elapsed = int(time.monotonic() - self._compute_started)
            if (elapsed != self._compute_display_second or
                    self._compute_percent != self._compute_display_percent):
                self._compute_display_second = elapsed
                self._compute_display_percent = self._compute_percent
                self._summary.setText(
                    f"{self._compute_label}… ~{self._compute_percent}% · "
                    f"{elapsed}s elapsed")
            return
        self._compute_thread = None
        self._compute_cancel = None
        if request != self._compute_request:
            if self._compute_request is None:
                self._compute_poll.stop()
            else:
                self._start_compute(self._compute_request)
            return
        self._compute_poll.stop()
        if error is not None:
            self._summary.setText(f"Could not compute selected results: {error}")
            self._play.setChecked(False)
            return
        sweep, selected, frame, field, coverage = request
        self._loaded = loaded
        self._loaded_key = (selected, frame, field)
        if coverage is not None:
            self._global_ranges[(selected, field, coverage)] = limits
        self.refresh()
        self._schedule_next_frame()

    def _frame_limits(self, loaded) -> tuple[float, float]:
        samples = [finite_values(values) for _, values, _ in loaded]
        samples = [values for values in samples if values.size]
        limits = robust_limits(
            np.concatenate(samples), float(self._coverage.currentData())) if samples else None
        return limits or (0.0, 1.0)

    def _sequence_limits(self, selected) -> tuple[float, float]:
        key = (tuple(selected), self.current_field,
               float(self._coverage.currentData()))
        if key in self._global_ranges:
            return self._global_ranges[key]
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._global_ranges[key] = self._calculate_sequence_limits(
                self._sweep, selected, self.current_field,
                float(self._coverage.currentData()))
            return self._global_ranges[key]
        finally:
            QApplication.restoreOverrideCursor()

    def _resolved_limits(self, loaded, selected) -> tuple[float, float]:
        frame_limits = self._frame_limits(loaded)
        factor, _ = self._field_factor()
        limits = ((self._range_min.value(), self._range_max.value())
                  if self._manual_scale.isChecked() else
                  self._sequence_limits(selected)
                  if self._global_scale.isChecked() else frame_limits)
        if not self._manual_scale.isChecked():
            limits = limits[0] * factor, limits[1] * factor
            if self.current_field in ("Eeff_rate", "Eeff_gl"):
                # These measures are nonnegative. Percentile-based lower
                # limits can otherwise mark valid near-zero strain as cyan.
                limits = 0.0, limits[1]
        if limits[0] > limits[1]:
            limits = limits[1], limits[0]
        if self._symmetric.isChecked():
            bound = max(abs(limits[0]), abs(limits[1]))
            limits = -bound, bound
        if limits[0] == limits[1]:
            pad = max(abs(limits[0]) * 1e-6, 1e-12)
            limits = limits[0] - pad, limits[1] + pad
        if not self._manual_scale.isChecked():
            for box, value in ((self._range_min, limits[0]),
                               (self._range_max, limits[1])):
                box.blockSignals(True)
                box.setValue(value)
                box.blockSignals(False)
        return limits

    def _fit_current_range(self) -> None:
        if self._sweep is None:
            return
        selected = self._selected()
        key = (tuple(selected), self._frame.value(), self.current_field)
        if any(span > 1 for _, _, span in selected) and key != self._loaded_key:
            self._request_compute(selected, need_global=False)
            return
        factor, _ = self._field_factor()
        native = self._frame_limits(self._load_current(selected))
        limits = native[0] * factor, native[1] * factor
        for box, value in ((self._range_min, limits[0]),
                           (self._range_max, limits[1])):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self.refresh()

    def _background(self, path: str, shape: tuple[int, int]) -> np.ndarray:
        cached = self._image_cache.get(path)
        if cached is not None:
            self._image_cache.move_to_end(path)
            return cached
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
        if image is None or image.shape != shape:
            image = np.zeros(shape, np.uint8)
        self._image_cache[path] = image
        if len(self._image_cache) > 12:
            self._image_cache.popitem(last=False)
        return image

    def _reflow(self, *_args) -> None:
        if self._sweep is None:
            return
        while self._grid.count():
            self._grid.takeAt(0)
        columns = self._columns.value()
        selected = self._selected()
        limit = int(self._panel_mode.currentData() or 0)
        pages = max(1, (len(selected) + limit - 1) // limit) if limit else 1
        self._page_index = max(0, min(self._page_index, pages - 1))
        self._page_text.setText(f"Page {self._page_index + 1} / {pages}")
        self._page_back.setEnabled(self._page_index > 0)
        self._page_forward.setEnabled(self._page_index + 1 < pages)
        visible_keys = (selected if not limit else
                        selected[self._page_index * limit:
                                 (self._page_index + 1) * limit])
        self._empty_label.setVisible(not selected)
        if not selected:
            self._grid.addWidget(self._empty_label, 0, 0)
        for position, key in enumerate(visible_keys):
            panel = self._panels.get(key)
            if panel is not None:
                panel.show()
                self._grid.addWidget(panel, position // columns, position % columns)
        visible = set(visible_keys)
        for index, panel in self._panels.items():
            panel.setVisible(index in visible)

    def refresh(self, *_args) -> None:
        if self._sweep is None:
            return
        selected = self._selected()
        if selected:
            minimum_frame = min(key[2] for key in selected) - 1
            if self._frame.minimum() != minimum_frame:
                self._frame.blockSignals(True)
                self._frame.setRange(minimum_frame, self._sweep.frame_count - 1)
                self._frame.blockSignals(False)
        frame = self._frame.value()
        self._frame_text.setText(f"Frame {frame + 1} / {self._sweep.frame_count}")
        self._summary.setText(f"{len(selected)} parameter combinations")
        if not selected:
            self._colorbar_row.hide()
            self._update_plot([], [])
            self._reflow()
            return

        loaded_key = (tuple(selected), frame, self.current_field)
        global_key = (tuple(selected), self.current_field,
                      float(self._coverage.currentData()))
        need_global = (self._global_scale.isChecked() and
                       global_key not in self._global_ranges)
        need_temporal = (any(span > 1 for _, _, span in selected) and
                         loaded_key != self._loaded_key)
        need_plot_load = len(selected) >= 8 and loaded_key != self._loaded_key
        if need_temporal or need_global or need_plot_load:
            self._request_compute(selected, need_global=need_global)
            return
        if self._compute_thread is not None:
            self._compute_request = None
            if self._compute_cancel is not None:
                self._compute_cancel[0] = True

        loaded = self._load_current(selected)
        limits = self._resolved_limits(loaded, selected)
        factor, unit = self._field_factor()
        mode = ("sequence" if self._global_scale.isChecked() else
                "fixed" if self._manual_scale.isChecked() else "frame")
        self._summary.setText(
            f"{len(selected)} parameter combinations  ·  shared {mode} "
            f"scale {limits[0]:.5g} to {limits[1]:.5g} {unit}")
        field = self.current_field
        cmap = self._cmap.currentText()
        is_strain = field in getattr(
            self._sweep, "DERIVED_FIELDS", self._sweep.STRAIN_FIELDS)
        visible_keys = self._visible_keys(selected)
        created_panel = False
        below = above = total = 0
        for key, values, image_path in loaded:
            if key not in visible_keys:
                continue
            values = values if factor == 1.0 else values * factor
            finite = finite_values(values)
            below += int(np.count_nonzero(finite < limits[0]))
            above += int(np.count_nonzero(finite > limits[1]))
            total += finite.size
            case_index, strain_window, temporal_span = key
            case = self._sweep.cases[case_index]
            background = self._background(image_path, values.shape)
            overlay = render.field_to_rgba(
                values, *limits, cmap, spacing=case.grid_spacing, alpha=205,
                mark_out_of_range=self._flag_clipped.isChecked())
            rgb = render.alpha_over(render.gray_to_rgb(background), overlay)
            title = case.label
            if temporal_span > 1:
                title += (f"  ·  temporal span {temporal_span} frames "
                          + ("· Insufficient temporal history"
                             if frame + 1 < temporal_span else
                             f"ending at frame {frame + 1}"))
            if is_strain:
                effective = max(strain_window, case.grid_spacing)
                title += (f"  ·  strain window {strain_window} px"
                          f" (effective {effective})")
            panel = self._panels.get(key)
            if panel is None:
                panel = _ComparisonPanel(key)
                panel.canvas.view_changed.connect(self._linked_view_changed)
                panel.canvas.cursor_moved.connect(
                    lambda x, y, _v, source=key:
                    self._linked_probe(source, x, y))
                self._panels[key] = panel
                created_panel = True
            panel.update_content(
                title=title, values=values, background=background,
                overlay=overlay, rgb=rgb, limits=limits, cmap_name=cmap,
                unit=unit, coverage=case.mean_valid_fraction,
                corr=case.median_corr, spacing=case.grid_spacing,
                origin=case.subset_radius)
            if self._linked_state is not None:
                panel.canvas.set_view_state(self._linked_state)
        self._shared_colorbar.update_bar(
            *limits, unit, _bar_colors(cmap), below, above, total)
        self._colorbar_unit.setText(unit)
        self._colorbar_row.show()
        self._update_plot(loaded, selected)
        self._reflow()
        if created_panel and self._linked_state is None:
            self._request_fit_all()

    def _visible_keys(self, selected=None):
        selected = self._selected() if selected is None else selected
        limit = int(self._panel_mode.currentData() or 0)
        if not limit:
            return set(selected)
        start = self._page_index * limit
        return set(selected[start:start + limit])

    def _panel_mode_changed(self, *_args) -> None:
        self._page_index = 0
        self.refresh()
        self._request_fit_all()

    def _change_page(self, delta: int) -> None:
        self._page_index = max(0, self._page_index + int(delta))
        self.refresh()
        self._request_fit_all()

    def _linked_view_changed(self, x: float, y: float, zoom: float) -> None:
        self._linked_state = (x, y, zoom)
        source = self.sender()
        for panel in self._panels.values():
            if panel.canvas is source:
                continue
            panel.canvas.set_view_state(self._linked_state)

    def _linked_probe(self, source: int, x: int, y: int) -> None:
        self._probe_coordinate = (int(x), int(y))
        visible = self._visible_keys()
        for index, panel in self._panels.items():
            if index == source or index not in visible or panel.values is None:
                continue
            panel._show_probe(x, y, float("nan"))
        if self._summary_mode.currentData() == "probe":
            self._probe_plot_timer.start()

    def _update_plot(self, loaded, selected) -> None:
        if self._sweep is None or self._plot_toggle.isChecked():
            return
        use_3d = self._axis_y.currentData() is not None
        factor, unit = self._field_factor()
        cases = {key: self._sweep.cases[key[0]] for key in selected}
        points = []
        for key, values, _path in loaded:
            if key not in cases:
                continue
            case = cases[key]
            mode = self._summary_mode.currentData()
            if mode == "probe":
                if self._probe_coordinate is None:
                    continue
                px, py = self._probe_coordinate
                if py < 0 or px < 0 or py >= values.shape[0] or px >= values.shape[1]:
                    continue
                scalar = _interpolate_between_subset_centres(
                    values, px, py, case.grid_spacing,
                    case.subset_radius)
                if scalar is None:
                    continue
            else:
                finite = finite_values(values)
                if not finite.size:
                    continue
                scalar = (float(np.mean(finite)) if mode == "mean" else
                          float(np.median(finite)))
            variables = {"radius": case.subset_radius,
                         "grid": case.grid_spacing,
                         "window": key[1], "temporal": key[2]}
            x = variables[self._axis_x.currentData()]
            y_key = self._axis_y.currentData()
            y = variables[y_key] if y_key is not None else None
            points.append((x, y, scalar * factor))
        field_label = PARAMETRIC_FIELDS[self.current_field][0]
        title = (f"{field_label} · "
                 f"{self._summary_mode.currentText()} · frame "
                 f"{self._frame.value() + 1}")
        value_label = field_label
        if unit:
            value_label += f" [{unit}]"
        if use_3d:
            self._plot_stack.setCurrentWidget(self._plot_3d)
            self._plot_3d.set_data(
                points, title, self._axis_x.currentText(),
                self._axis_y.currentText(), value_label,
                colors=self._shared_colorbar._colors,
                limits=(self._shared_colorbar._vmin,
                        self._shared_colorbar._vmax),
                mark_out_of_range=self._flag_clipped.isChecked())
            return

        self._plot_stack.setCurrentWidget(self._plot_canvas)
        if self._plot_axes is None:
            self._plot_axes = self._plot_figure.add_subplot(111)
        else:
            self._plot_axes.clear()
        axes = self._plot_axes
        if points:
            points.sort(key=lambda item: (item[0], item[1] or 0))
            grouped = {}
            for px, _py, value in points:
                grouped.setdefault(px, []).append(value)
            curve = [(px, float(np.mean(values)))
                     for px, values in sorted(grouped.items())]
            colors = [render.sample_colorbar(
                value, self._shared_colorbar._vmin,
                self._shared_colorbar._vmax,
                self._shared_colorbar._colors,
                self._flag_clipped.isChecked()) for _px, value in curve]
            plot_colors = [tuple(channel / 255 for channel in color)
                           for color in colors]
            axes.plot([p[0] for p in curve], [p[1] for p in curve],
                      color=plot_colors[len(plot_colors) // 2])
            axes.scatter([p[0] for p in curve], [p[1] for p in curve],
                         c=plot_colors, edgecolors="#e6e8ea",
                         linewidths=0.7, zorder=3)
            axes.set_ylabel(value_label)
            axes.set_xlabel(self._axis_x.currentText())
            axes.set_title(title)
        else:
            axes.text(0.5, 0.5, "No finite values for this selection",
                      ha="center", va="center", transform=axes.transAxes)
        self._plot_canvas.draw_idle()

    def _export_rgb(self, case_index: int) -> np.ndarray | None:
        panel = self._panels.get(case_index)
        if panel is None or panel.rendered_rgb is None:
            return None
        return render.draw_label(panel.rendered_rgb.copy(), panel.title_text)

    def _export_grid(self) -> None:
        images = [self._export_rgb(key) for key in self._visible_keys()]
        images = [image for image in images if image is not None]
        if not images:
            QMessageBox.information(self, "Nothing to export",
                                    "Select at least one parameter case.")
            return
        columns = min(self._columns.value(), len(images))
        rows = (len(images) + columns - 1) // columns
        aspect = images[0].shape[0] / max(images[0].shape[1], 1)
        mosaic = render.compose_grid(
            images, rows, columns, 640, max(360, int(640 * aspect)))
        first_panel = next(
            (self._panels[key] for key in self._visible_keys()
             if key in self._panels and self._panels[key].rendered_rgb is not None),
            None)
        if first_panel is not None:
            mosaic = np.pad(mosaic, ((0, 44), (0, 0), (0, 0)),
                            mode="constant", constant_values=28)
            finite = [finite_values(self._panels[key].values)
                      for key in self._visible_keys()
                      if key in self._panels and self._panels[key].values is not None]
            mosaic = render.draw_colorbar(
                mosaic, first_panel.cmap_name, *first_panel.limits,
                first_panel.unit,
                clipped_low=any(np.any(values < first_panel.limits[0])
                                for values in finite),
                clipped_high=any(np.any(values > first_panel.limits[1])
                                 for values in finite))
        default = self._sweep.cases[0].path.parent / (
            f"comparison_{self.current_field}_frame_{self._frame.value() + 1}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export comparison", str(default), "PNG image (*.png)")
        if path and not cv2.imwrite(path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)):
            QMessageBox.warning(self, "Export failed", f"Could not write {path}")

    def _add_hdf5(self) -> None:
        start = (str(self._sweep.cases[0].path.parent)
                 if self._sweep is not None else str(Path.home()))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add parametric HDF5 caches", start,
            "HDF5 files (*.h5 *.hdf5)")
        if not paths or self._sweep is None:
            return
        for path in paths:
            self._sweep.add_hdf5(path)
        self.set_sweep(self._sweep)
