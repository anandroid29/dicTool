"""
results_page.py — Step 6: Results viewer with correct frame synchronisation.

Critical fix: every time the temporal scrubber moves to frame N, we:
  1. Load the actual deformed image for frame N and set it as the canvas background
  2. Render the selected field (strain rate by default) as a semi-transparent overlay
  3. Update the colorbar and statistics panel

This is the correct behaviour — the canvas must show the deformed frame,
not the fixed reference image.
"""
from __future__ import annotations
import importlib.util
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional
import numpy as np
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSlot, QThread, pyqtSignal, QObject, QRunnable, QThreadPool,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialog,
    QSlider, QComboBox, QCheckBox, QFrame, QSizePolicy,
    QFileDialog, QMessageBox, QSpinBox, QToolButton, QProgressDialog, QProgressBar,
    QListWidget, QListWidgetItem, QGroupBox, QGridLayout, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QScrollArea, QMenu,
)

from src.core.stats import field_summary
from src.core.compact_field import finite_values, CompactField, CompactMask
from src.core.units import LENGTH_UNIT_ORDER
from src.ui import render
from src.ui.components import ResultColorBar
from src.ui.render import RangeSpec

if TYPE_CHECKING:
    from src.ui.wizard import Wizard

from src.ui.image_canvas import ImageCanvas, marker_color

# Palette comes from the single source of truth in theme.py. These were
# duplicated literals, which is why re-theming previously left pages behind.
from src.ui.theme import C_ACCENT, C_BG, C_BORDER, C_CARD, C_RAISED, C_SUCCESS, C_SURFACE, C_TEXT, C_TEXT2, C_TEXT3, C_WARNING

_C_ACCENT = C_ACCENT
_C_BG = C_BG
_C_BORDER = C_BORDER
_C_CARD = C_CARD
_C_RAISED = C_RAISED
_C_SUCCESS = C_SUCCESS
_C_SURFACE = C_SURFACE
_C_TEXT = C_TEXT
_C_TEXT2 = C_TEXT2
_C_TEXT3 = C_TEXT3
_C_WARN = C_WARNING



FIELDS = {
    # Displacement is always the current previous-frame -> current-frame
    # interval. u_inc/v_inc remain file/API aliases but are not duplicated here.
    "u": ("Instantaneous displacement u", "px"),
    "v": ("Instantaneous displacement v", "px"),
    "mag_inc": ("Instantaneous displacement magnitude", "px"),

    # 2. Velocities
    "Vx": ("Velocity Vx", "px/s"),
    "Vy": ("Velocity Vy", "px/s"),
    "Veff": ("Effective Velocity", "px/s"),

    # 3. Strain Rates
    "Exx_rate": ("Strain Rate  Ėxx", "s⁻¹"),
    "Exy_rate": ("Tensor Shear Strain Rate  Ėxy", "s⁻¹"),
    "Eyy_rate": ("Strain Rate  Ėyy", "s⁻¹"),
    "Eeff_rate": ("Effective Strain Rate", "s⁻¹"),

    # The user-facing strain measure is Green-Lagrange. Legacy infinitesimal
    # and engineering-shear arrays remain loadable/exportable for old sessions,
    # but no longer compete with the selected strain convention in the UI.
    "Exx_gl": ("Accumulated strain Exx", "dimensionless"),
    "Eyy_gl": ("Accumulated strain Eyy", "dimensionless"),
    "Exy_gl": ("Accumulated tensor shear strain Exy", "dimensionless"),
    "Eeff_gl": ("Accumulated equivalent strain magnitude", "dimensionless"),
}

# Field families, in the order they appear in the category dropdown. Only the
# members of the selected family get a button in the toolbar.
FIELD_GROUPS = {
    "Displacement": ["u", "v", "mag_inc"],
    "Velocity":     ["Vx", "Vy", "Veff"],
    "Strain rate":  ["Exx_rate", "Eyy_rate", "Exy_rate", "Eeff_rate"],
    "Strain":       ["Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"],
}

_ACCUMULATED_STRAIN_FIELDS = {
    "Exx_inf", "Eyy_inf", "Exy_inf", "Gxy_inf", "Eeff_inf",
    "Exx_gl", "Eyy_gl", "Exy_gl", "Gxy_gl", "Eeff_gl",
}

# Short button captions, now that the family is named by the dropdown.
_FIELD_SHORT = {
    "u": "u", "v": "v",
    "u_inc": "du", "v_inc": "dv", "mag_inc": "|d|",
    "Vx": "Vx", "Vy": "Vy", "Veff": "eff",
    "Exx_rate": "Ėxx", "Exy_rate": "Ėxy", "Eyy_rate": "Ėyy", "Eeff_rate": "Ėeff",
    "Exx_gl": "Exx", "Eyy_gl": "Eyy", "Exy_gl": "Exy", "Eeff_gl": "Eeq",
}


def _field_short(key: str) -> str:
    return _FIELD_SHORT.get(key, key)


def _group_of(field: str) -> str:
    for g, keys in FIELD_GROUPS.items():
        if field in keys:
            return g
    return next(iter(FIELD_GROUPS))


def _interpolate_between_subset_centres(
        arr: np.ndarray, x: int, y: int, spacing: int, origin: int
        ) -> Optional[float]:
    """Bilinearly interpolate a sparse regular-grid field at ``(x, y)``.

    Every contributing subset centre must contain a finite value.  This is
    deliberately stricter than nearest-neighbour filling: a failed correlation,
    dynamic-ROI hole, or specimen boundary must not be painted over with an
    apparently trustworthy number.
    """
    if arr is None or arr.ndim != 2:
        return None
    h, w = arr.shape
    if not (0 <= x < w and 0 <= y < h):
        return None

    s = max(1, int(spacing))
    o = int(origin)
    gx = (x - o) / s
    gy = (y - o) / s
    ix0, ix1 = int(np.floor(gx)), int(np.ceil(gx))
    iy0, iy1 = int(np.floor(gy)), int(np.ceil(gy))

    # A missing value at an actual subset centre is a failed/unsupported
    # measurement, not a gap that interpolation is allowed to conceal.
    if ix0 == ix1 and iy0 == iy1:
        return None

    x0, x1 = o + ix0 * s, o + ix1 * s
    y0, y1 = o + iy0 * s, o + iy1 * s
    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        return None

    x_terms = [(x0, 1.0)] if x0 == x1 else [
        (x0, (x1 - x) / s), (x1, (x - x0) / s)]
    y_terms = [(y0, 1.0)] if y0 == y1 else [
        (y0, (y1 - y) / s), (y1, (y - y0) / s)]

    value = 0.0
    for sy, wy in y_terms:
        for sx, wx in x_terms:
            sample = arr[sy, sx]
            if not np.isfinite(sample):
                return None
            value += float(sample) * wx * wy
    return value if np.isfinite(value) else None


# FEA-style rainbow ramps first: blue (low) through cyan/green/yellow to red
# (high) is the contour convention every ANSYS/Abaqus user reads instinctively.
# turbo is the default rather than jet -- same blue-to-red identity, but without
# jet's false banding at cyan/yellow, which invents contour edges that are not in
# the data. jet is kept immediately below for matching legacy figures exactly.
CMAPS = ["turbo","jet","rainbow","nipy_spectral",
         "RdBu_r","seismic","bwr","coolwarm",
         "viridis","inferno","magma","plasma","cividis",
         "hot","afmhot","gist_heat","copper","gray"]

DEFAULT_CMAP = "turbo"

# Default colour-scale coverage. Not 100%: DIC fields reliably contain a few
# subsets that converged onto noise, and a raw min/max scale hands the entire
# colourbar to them. 99% keeps essentially all real signal while ignoring the
# extreme 0.5% at each tail.
DEFAULT_COVERAGE_TEXT = "99%"


class ExportWorker(QThread):
    finished_export = pyqtSignal(bool, str)
    progress_export = pyqtSignal(int)

    def __init__(self, analysis, path, temporal_results=None,
                 temporal_pairs=None, temporal_metadata=None, parent=None):
        super().__init__(parent)
        self.analysis = analysis
        self.path = path
        self.temporal_results = temporal_results
        self.temporal_pairs = temporal_pairs
        self.temporal_metadata = dict(temporal_metadata or {})

    def run(self):
        try:
            def prog_cb(frac):
                self.progress_export.emit(int(frac * 100))
            self.analysis.export_hdf5(
                self.path, progress_cb=prog_cb,
                temporal_results=self.temporal_results,
                temporal_pairs=self.temporal_pairs,
                temporal_metadata=self.temporal_metadata)
            self.finished_export.emit(True, self.path)
        except Exception as e:
            self.finished_export.emit(False, str(e))


class _VideoWorker(QThread):
    """Renders and writes the export off the GUI thread.

    Safe to do here precisely because rendering is pure numpy/OpenCV -- no
    QPixmap or QWidget is touched, so there is no thread-affinity problem.
    """
    progress = pyqtSignal(int)
    done = pyqtSignal(bool, str)

    def __init__(self, analysis, spec, path, markers,
                 temporal_results=None, temporal_pairs=None, parent=None):
        super().__init__(parent)
        self._analysis = analysis
        self._spec = spec
        self._path = path
        self._markers = list(markers or [])
        self._temporal_results = temporal_results
        self._temporal_pairs = temporal_pairs
        self.cancel_flag = [False]

    def run(self):
        try:
            from src.ui.video_export import export_video
            out = export_video(
                self._analysis, self._spec, self._path, markers=self._markers,
                results=self._temporal_results, pairs=self._temporal_pairs,
                progress_cb=lambda f, _m: self.progress.emit(int(f * 100)),
                cancel_flag=self.cancel_flag)
            self.done.emit(True, out)
        except Exception as e:
            self.done.emit(False, str(e))


class _PairTaskSignals(QObject):
    done = pyqtSignal(int, int, object, str)
    progress = pyqtSignal(int, int, int, str)


class _PairTask(QRunnable):
    """One pair calculation; emits data only and never touches a QWidget."""

    def __init__(self, analysis, generation: int, index: int, pair,
                 strain_window: int, use_gpu: bool,
                 cancel_flag: list,
                 include_strain: bool = True,
                 include_rate: bool = True,
                 cache_path: Optional[str] = None) -> None:
        super().__init__()
        self.signals = _PairTaskSignals()
        self._analysis = analysis
        self._generation = int(generation)
        self._index = int(index)
        self._pair = pair
        self._strain_window = int(strain_window)
        self._use_gpu = bool(use_gpu)
        self._cancel_flag = cancel_flag
        self._include_strain = bool(include_strain)
        self._include_rate = bool(include_rate)
        self._cache_path = cache_path

    def run(self) -> None:
        try:
            a, b = self._pair
            result = self._analysis.pair_kinematics(
                a, b, strain_window=self._strain_window,
                use_gpu=self._use_gpu,
                progress_cb=lambda fraction, message: self.signals.progress.emit(
                    self._generation, self._index,
                    int(np.clip(fraction, 0.0, 1.0) * 100), message),
                cancel_flag=self._cancel_flag,
                include_strain=self._include_strain,
                include_rate=self._include_rate)
            if self._cancel_flag[0]:
                raise RuntimeError("Temporal calculation cancelled.")
            if self._cache_path:
                from src.core.temporal import save_temporal_result
                result = save_temporal_result(self._cache_path, result)
            self.signals.done.emit(
                self._generation, self._index, result, "")
        except Exception as exc:
            self.signals.done.emit(
                self._generation, self._index, None, str(exc))


class _TemporalHistorySignals(QObject):
    progress = pyqtSignal(int, int, str)
    item_ready = pyqtSignal(int, int)
    done = pyqtSignal(int, bool, str)


class _TemporalHistoryTask(QRunnable):
    """Derive rate and accumulated strain from averaged velocity frames."""

    def __init__(self, analysis, generation: int, pairs, strain_window: int,
                 use_gpu: bool, cancel_flag: list, store) -> None:
        super().__init__()
        self.signals = _TemporalHistorySignals()
        self._analysis = analysis
        self._generation = int(generation)
        self._pairs = list(pairs)
        self._strain_window = int(strain_window)
        self._use_gpu = bool(use_gpu)
        self._cancel_flag = cancel_flag
        self._store = store

    def run(self) -> None:
        try:
            from src.core.compact_field import CompactField
            from src.core.strain import compute_velocity_strains
            from src.core.strain_accum import StrainPathTracker
            from src.core.temporal import save_temporal_result

            shape = tuple(int(value) for value in self._store[0].u.shape)
            roi = np.asarray(self._analysis._roi_mask, dtype=bool)
            origin = getattr(self._analysis, "_strain_origin_mask", None)
            if origin is None or not np.any(origin):
                origin = roi
            tracker = StrainPathTracker(
                shape, origin, roi,
                self._analysis.params.subset_radius,
                self._analysis.params.subset_spacing)
            persistent = (
                "Exx_inf", "Eyy_inf", "Exy_inf", "Eeff_inf",
                "Exx_gl", "Eyy_gl", "Exy_gl")
            swept = {name: np.full(shape, np.nan, dtype=np.float32)
                     for name in persistent}
            encountered = np.zeros(shape, dtype=bool)

            def deposit(snapshot) -> None:
                new = roi & ~encountered
                for name in persistent:
                    new &= np.isfinite(np.asarray(snapshot[name]))
                if not new.any():
                    return
                for name in persistent:
                    swept[name][new] = np.asarray(snapshot[name])[new]
                encountered[new] = True

            def accumulated_fields():
                values = {
                    "Exx_gl": swept["Exx_gl"],
                    "Eyy_gl": swept["Eyy_gl"],
                    "Exy_gl": swept["Exy_gl"],
                    "Eeff_gl": swept["Eeff_inf"],
                }
                complete = np.logical_and.reduce(
                    [np.isfinite(values[name]) for name in values])
                indices = np.flatnonzero(complete.reshape(-1)).astype(
                    np.uint32, copy=False)
                return {
                    name: CompactField.from_dense(field, indices=indices)
                    for name, field in values.items()
                }

            total = max(1, len(self._pairs))
            previous_endpoint = None
            for index, (_start, endpoint) in enumerate(self._pairs):
                if self._cancel_flag[0]:
                    raise RuntimeError("Temporal calculation cancelled.")
                result = self._store[index]
                valid = np.asarray(result.valid, dtype=bool)
                rates = compute_velocity_strains(
                    np.asarray(result.Vx), np.asarray(result.Vy), valid,
                    self._strain_window,
                    self._analysis.params.subset_spacing,
                    use_gpu=self._use_gpu)
                for name in ("dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                             "Exx_rate", "Exy_rate", "Eyy_rate",
                             "Eeff_rate"):
                    setattr(result, name, np.asarray(rates[name], np.float32))
                result.Gxy_rate = np.asarray(
                    2.0 * rates["Exy_rate"], np.float32)

                # The first averaged frame spans its selected pair. Subsequent
                # sliding frames advance by their endpoint cadence (normally
                # one source frame); non-overlapping frames advance by a block.
                if previous_endpoint is None:
                    step_dt = float(result.elapsed)
                else:
                    endpoint_step = int(endpoint) - int(previous_endpoint)
                    step_dt = (endpoint_step / max(self._analysis.fps, 1e-9)
                               if endpoint_step > 0 else float(result.elapsed))
                step_u = np.asarray(result.Vx) * step_dt
                step_v = np.asarray(result.Vy) * step_dt
                tracker.seed(valid)
                deposit(tracker.snapshot())
                tracker.advance(
                    step_u, step_v,
                    np.asarray(rates["dVx_dx"]) * step_dt,
                    np.asarray(rates["dVx_dy"]) * step_dt,
                    np.asarray(rates["dVy_dx"]) * step_dt,
                    np.asarray(rates["dVy_dy"]) * step_dt)
                deposit(tracker.snapshot())
                state = accumulated_fields()
                for name in ("Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"):
                    setattr(result, name, state[name])
                result.Exx = result.Exx_gl
                result.Eyy = result.Eyy_gl
                result.Exy = result.Exy_gl
                result.Eeff = result.Eeff_gl
                save_temporal_result(self._store.path_for(index), result)
                previous_endpoint = endpoint
                self.signals.progress.emit(
                    self._generation, int(100 * (index + 1) / total),
                    f"Strain rates {index + 1}/{len(self._pairs)}")
                self.signals.item_ready.emit(self._generation, index)
            self.signals.done.emit(self._generation, True, "")
        except Exception as exc:
            self.signals.done.emit(self._generation, False, str(exc))


class ResultsPage(QWidget):
    """Step 6 — results viewer with correct frame-by-frame image updates."""

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._frame  = 0
        self._field  = "Eeff_rate"
        self._unit_scale_cache: dict = {}
        # Frame-pair sequence. Each selected interval owns one timeline item;
        # `_pair_avg` is retained as the current pair result for the existing
        # display/export paths, not as one average collapsed across all pairs.
        self._pair_mode = False
        self._pair_avg = None
        self._pair_list: list = []
        self._pair_cache: "OrderedDict[int, object]" = OrderedDict()
        self._PAIR_CACHE_MAX = 4
        self._pair_strain_window = int(
            getattr(getattr(self._wizard.analysis, "params", None),
                    "strain_window", 5))
        self._pair_generation = 0
        self._pair_cancel_flag = [False]
        self._pair_queue: "OrderedDict[int, None]" = OrderedDict()
        self._pair_active: set[tuple[int, int]] = set()
        self._pair_tasks: dict[tuple[int, int], _PairTask] = {}
        self._pair_fixed_ranges: dict[tuple, tuple[float, float]] = {}
        self._pair_sequence_mode = "custom"
        self._pair_store = None
        self._pair_store_dirs: dict[int, str] = {}
        self._pair_retained_generations: set[int] = set()
        self._pair_bulk_ready: set[int] = set()
        self._pair_rate_ready: set[int] = set()
        self._pair_rates_only = False
        self._pair_history_active = False
        self._pair_history_task = None
        self._pending_hdf5_path: Optional[str] = None
        self._pair_pool = QThreadPool(self)
        self._pair_pool.setMaxThreadCount(max(1, min(3, os.cpu_count() or 1)))
        self._single_frame_before_pairs = 0
        self._pair_frame_before_single = 0
        # Decoded background frames, keyed by path. Playback and scrubbing both
        # revisit frames constantly, and decoding a PNG per tick is what makes
        # play stutter -- the decode alone outlasts the frame interval at any
        # useful rate. Bounded so a long sequence cannot exhaust memory.
        self._img_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._IMG_CACHE_MAX = 48
        # Trajectories for (frame, trail, markers). Recomputing them walks the
        # whole displacement history per frame, which is the second thing that
        # makes playback with streaklines on unusable.
        self._traj_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._TRAJ_CACHE_MAX = 64
        self._play_timer = QTimer(self)
        self._play_timer.setSingleShot(True)
        self._play_timer.setInterval(200)
        self._play_timer.timeout.connect(self._advance)
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(35)
        self._scrub_timer.timeout.connect(self._render_scrubbed_frame)
        # Colour-scale mode. Radio buttons, not checkboxes: the three are
        # mutually exclusive, and as checkboxes they suggested combinations that
        # do not exist (ticking both "Static" and "Range" only ever meant
        # "Range"). Sym is a separate modifier and stays a checkbox.
        self._scale_group = QButtonGroup(self)
        self._scale_auto_rb = QRadioButton("Auto")
        self._scale_global_rb = QRadioButton("Global")
        self._scale_manual_rb = QRadioButton("Range")
        for rb, tip in (
            (self._scale_auto_rb, "Rescale to each frame's own min/max."),
            (self._scale_global_rb, "Fix the scale to the min/max across the whole sequence."),
            (self._scale_manual_rb, "Pin the scale to explicit limits, in the units\n"
                                    "shown on the colourbar."),
        ):
            rb.setToolTip(tip)
            self._scale_group.addButton(rb)
        # Fixed by default. Rescaling per frame makes a colour mean something
        # different on every frame, so a feature appears to pulse when only the
        # scale moved. Auto stays available for inspecting one frame closely.
        self._scale_global_rb.setChecked(True)
        self._scale_group.buttonToggled.connect(self._on_scale_mode_changed)
        self._build_ui()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top toolbar ───────────────────────────────────────────────
        # Two rows, split by what they control: the first selects WHAT is being
        # looked at, the second controls HOW it is drawn. As one row this held
        # roughly twenty widgets and simply ran off the edge of the window --
        # "New Session", "Streaklines" and "Flag clipped" were all truncated at
        # 1500 px, and a fixed-height QHBoxLayout gives no indication that
        # anything is missing.
        top = QWidget()
        top.setStyleSheet(f"background:{_C_SURFACE}; border-bottom:1px solid {_C_BORDER};")
        top_outer = QVBoxLayout(top)
        top_outer.setContentsMargins(14, 5, 14, 5)
        top_outer.setSpacing(4)

        top_lay = QHBoxLayout()        # row 1 — what is displayed
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(8)
        top_outer.addLayout(top_lay)

        row_sep = QFrame()
        row_sep.setFrameShape(QFrame.Shape.HLine)
        row_sep.setStyleSheet(f"background:{_C_BORDER}; max-height:1px; border:none;")
        top_outer.addWidget(row_sep)

        disp_lay = QHBoxLayout()       # row 2 — how it is drawn
        disp_lay.setContentsMargins(0, 0, 0, 0)
        disp_lay.setSpacing(8)
        top_outer.addLayout(disp_lay)

        new_btn = QPushButton("← New session")
        new_btn.setFixedWidth(120)
        new_btn.clicked.connect(self._wizard.new_session)
        top_lay.addWidget(new_btn)

        top_lay.addSpacing(12)

        # Field selector: a category dropdown plus a short row of buttons for
        # the members of that category. All 13 fields used to sit in this bar as
        # individual buttons, which together with the marker, trail and colormap
        # controls left the row overflowing.
        self._cat_combo = QComboBox()
        self._cat_combo.addItems(list(FIELD_GROUPS.keys()))
        self._cat_combo.setFixedWidth(132)
        self._cat_combo.setToolTip("Which family of results to display.")
        self._cat_combo.currentTextChanged.connect(self._select_category)
        top_lay.addWidget(self._cat_combo)
        top_lay.addSpacing(6)

        self._field_btns: dict[str, QToolButton] = {}
        for key, (label, _) in FIELDS.items():
            btn = QToolButton()
            btn.setText(_field_short(key))
            btn.setToolTip(label)
            btn.setCheckable(True)
            btn.setChecked(key == self._field)
            btn.clicked.connect(lambda c, k=key: self._select_field(k))
            top_lay.addWidget(btn)
            self._field_btns[key] = btn

        self._apply_tab_style()
        self._cat_combo.blockSignals(True)
        self._cat_combo.setCurrentText(_group_of(self._field))
        self._cat_combo.blockSignals(False)
        self._sync_field_buttons()

        # ─── PROMINENT STREAKLINES BLOCK (Just after Eff) ───
        top_lay.addSpacing(12)

        vsep1 = QFrame()
        vsep1.setFrameShape(QFrame.Shape.VLine)
        vsep1.setStyleSheet(f"background:{_C_BORDER}; max-width:1px;")
        top_lay.addWidget(vsep1)
        top_lay.addSpacing(12)

        self._streak_chk = QCheckBox("Streaklines")
        self._streak_chk.setToolTip(
            "Trace trajectories from markers placed on the image.")
        self._streak_chk.setStyleSheet(f"color:{_C_ACCENT}; font-size: 12px; font-weight: 800;")
        self._streak_chk.toggled.connect(self._on_streak_toggled)
        top_lay.addWidget(self._streak_chk)

        self._place_btn = QPushButton("＋ Place markers")
        self._place_btn.setCheckable(True)
        self._place_btn.setFixedHeight(28)
        self._place_btn.setToolTip(
            "Click the image to drop a marker · drag to move · right-click or Del to remove")
        self._place_btn.setStyleSheet(
            f"QPushButton{{background:{_C_CARD}; color:{_C_TEXT}; border:1px solid {_C_BORDER};"
            f" padding:3px 10px; border-radius:3px; font-size:11px;}}"
            f"QPushButton:checked{{background:{_C_ACCENT}; color:#ffffff; border:1px solid {_C_ACCENT};"
            f" font-weight:700;}}")
        self._place_btn.toggled.connect(self._on_place_toggled)
        top_lay.addWidget(self._place_btn)

        self._marker_count_lbl = QLabel("0 markers")
        self._marker_count_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        top_lay.addWidget(self._marker_count_lbl)

        self._clear_markers_btn = QPushButton("Clear")
        self._clear_markers_btn.setFixedHeight(28)
        self._clear_markers_btn.setToolTip("Remove all markers.")
        self._clear_markers_btn.setStyleSheet(
            f"background:{_C_CARD}; color:{_C_TEXT2}; border:1px solid {_C_BORDER};"
            f" padding:3px 10px; border-radius:3px; font-size:11px;")
        self._clear_markers_btn.clicked.connect(self._clear_markers)
        top_lay.addWidget(self._clear_markers_btn)

        self._trail_lbl = QLabel("Trail:")
        self._trail_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        top_lay.addWidget(self._trail_lbl)

        self._trail_combo = QComboBox()
        self._trail_combo.addItem("Full", 0)
        for n in (10, 25, 50, 100):
            self._trail_combo.addItem(f"{n} frames", n)
        self._trail_combo.setToolTip("How much trajectory history to draw.")
        self._trail_combo.setFixedWidth(96)
        self._trail_combo.currentIndexChanged.connect(self._refresh_overlay)
        top_lay.addWidget(self._trail_combo)

        for w in (self._place_btn, self._marker_count_lbl, self._clear_markers_btn,
                  self._trail_lbl, self._trail_combo):
            w.setVisible(False)

        top_lay.addSpacing(12)

        vsep2 = QFrame()
        vsep2.setFrameShape(QFrame.Shape.VLine)
        vsep2.setStyleSheet(f"background:{_C_BORDER}; max-width:1px;")
        top_lay.addWidget(vsep2)
        # ────────────────────────────────────────────────────

        top_lay.addStretch()

        # Colormap
        cmap_lbl = QLabel("Colormap:")
        cmap_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        disp_lay.addWidget(cmap_lbl)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(CMAPS)
        self._cmap_combo.setCurrentText(DEFAULT_CMAP)
        self._cmap_combo.setFixedWidth(100)
        self._cmap_combo.currentTextChanged.connect(self._refresh_overlay)
        disp_lay.addWidget(self._cmap_combo)

        self._sym_chk = QCheckBox("Sym")
        self._sym_chk.setToolTip("Centre the colour scale on zero.")
        self._sym_chk.stateChanged.connect(self._refresh_overlay)
        disp_lay.addWidget(self._sym_chk)

        scale_lbl = QLabel("Scale:")
        scale_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        disp_lay.addWidget(scale_lbl)
        for rb in (self._scale_auto_rb, self._scale_global_rb, self._scale_manual_rb):
            disp_lay.addWidget(rb)

        self._range_min_spin = QDoubleSpinBox()
        self._range_max_spin = QDoubleSpinBox()
        for sb, tip in ((self._range_min_spin, "Lower colour limit"),
                        (self._range_max_spin, "Upper colour limit")):
            sb.setDecimals(6)
            sb.setRange(-1e12, 1e12)
            sb.setFixedWidth(96)
            sb.setToolTip(tip)
            sb.setEnabled(False)
            sb.setKeyboardTracking(False)   # only fire on commit, not per digit
            sb.valueChanged.connect(self._refresh_overlay)
            disp_lay.addWidget(sb)

        self._range_fit_btn = QPushButton("Fit")
        self._range_fit_btn.setFixedWidth(40)
        self._range_fit_btn.setToolTip(
            "Set the limits from this frame, then keep them fixed.")
        self._range_fit_btn.setEnabled(False)
        self._range_fit_btn.clicked.connect(self._fit_range_to_frame)
        disp_lay.addWidget(self._range_fit_btn)

        # ── Robust scaling ────────────────────────────────────────────
        # A few subsets always converge onto noise at an edge or a dropout, and
        # their values can be orders of magnitude outside the real range. On a
        # raw min/max scale those pixels own the whole colourbar and the actual
        # field flattens to one colour, so the default trims the tails.
        cov_lbl = QLabel("Coverage:")
        cov_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        disp_lay.addWidget(cov_lbl)

        self._coverage_combo = QComboBox()
        for text, val in (("100%", 100.0), ("99.5%", 99.5), ("99%", 99.0),
                          ("98%", 98.0), ("95%", 95.0), ("90%", 90.0)):
            self._coverage_combo.addItem(text, val)
        self._coverage_combo.setCurrentText(DEFAULT_COVERAGE_TEXT)
        self._coverage_combo.setFixedWidth(74)
        self._coverage_combo.setToolTip(
            "Share of the data the colour scale spans.\n\n"
            "100% uses the true minimum and maximum, so a single bad\n"
            "correlation can flatten the map. 98% ignores the most extreme\n"
            "1% at each end.\n\n"
            "Affects the colours only. Statistics and exports report the\n"
            "true values."
        )
        self._coverage_combo.currentIndexChanged.connect(self._refresh_overlay)
        disp_lay.addWidget(self._coverage_combo)

        self._clip_chk = QCheckBox("Flag clipped")
        # Off by default. With marks on, every value past the trimmed limits is
        # painted magenta, so a normal hot spot flashes as if it were an error.
        # The colourbar reports the fraction outside the range either way.
        self._clip_chk.setChecked(False)
        self._clip_chk.setToolTip(
            "Draw values above the range in magenta and below it in cyan,\n"
            "so clipping is visible instead of blending into the end\n"
            "colours.")
        self._clip_chk.stateChanged.connect(self._refresh_overlay)
        disp_lay.addWidget(self._clip_chk)
        disp_lay.addStretch()

        root.addWidget(top)

        # ── Body: canvas + right sidebar ──────────────────────────────
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        # Canvas
        self._canvas = ImageCanvas()
        self._canvas.seed_enabled = False  # Disable seed placement here
        self._canvas.marker_requested.connect(self._on_marker_requested)
        self._canvas.markers_changed.connect(self._on_markers_changed)
        self._canvas.marker_selected.connect(self._on_marker_selected)
        self._canvas.cursor_moved.connect(self._on_cursor_moved)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        body_lay.addWidget(self._canvas, 1)

        # Right sidebar.
        #
        # The panel holds more than fits on a short screen -- markers and the
        # frame-pair controls both appear conditionally -- and a plain layout
        # answers that by squeezing widgets until they are unreadable and
        # clipping whatever is left. Scrolling instead keeps every control at
        # its natural size and reachable.
        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setFixedWidth(232)
        sidebar.setFrameShape(QFrame.Shape.NoFrame)
        sidebar.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sidebar.setStyleSheet(
            f"QScrollArea{{background:{_C_SURFACE};"
            f" border-left:1px solid {_C_BORDER};}}"
            f"QScrollBar:vertical{{background:{_C_SURFACE}; width:9px; margin:0;}}"
            f"QScrollBar::handle:vertical{{background:{_C_BORDER};"
            f" border-radius:4px; min-height:28px;}}"
            f"QScrollBar::handle:vertical:hover{{background:{_C_TEXT3};}}"
            f"QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0;}}"
            f"QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{{background:none;}}")

        sidebar_body = QWidget()
        sidebar_body.setStyleSheet(f"background:{_C_SURFACE};")
        sidebar.setWidget(sidebar_body)
        sb_lay = QVBoxLayout(sidebar_body)
        sb_lay.setContentsMargins(16, 18, 12, 18)
        sb_lay.setSpacing(12)

        # Stats
        stats_head_row = QHBoxLayout()
        stats_head_row.setContentsMargins(0, 0, 0, 0)
        stats_hdr = QLabel("STATISTICS")
        stats_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;"
        )
        stats_head_row.addWidget(stats_hdr)
        stats_head_row.addStretch()
        self._stats_unit_lbl = QLabel("")
        self._stats_unit_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._stats_unit_lbl.setStyleSheet(
            f"color:{_C_TEXT2}; font-size:9px; font-style:italic;")
        stats_head_row.addWidget(self._stats_unit_lbl)
        sb_lay.addLayout(stats_head_row)

        # Robust and non-robust statistics side by side. Mean/std are what
        # people expect; median and IQR are what survive a decorrelated subset.
        # Showing both means a disagreement between them is visible, and that
        # disagreement is the signal that the field contains outliers.
        self._stat_labels: dict[str, QLabel] = {}
        for stat, tip in (
            ("Mean",    "Arithmetic mean. One decorrelated subset can shift "
                        "it without bound."),
            ("Median",  "Middle value, unaffected by a minority of bad points.\n"
                        "Where it differs markedly from the mean, trust this one."),
            ("Std Dev", "Standard deviation. Like the mean, sensitive to outliers."),
            ("IQR",     "Interquartile range: the spread of the middle 50%.\n"
                        "A robust alternative to standard deviation."),
            ("P1–P99",  "1st to 99th percentile: the practical extremes,\n"
                        "with the most extreme 1% at each end set aside."),
            ("Min/Max", "True extremes, including any failed correlations.\n"
                        "Compare with P1–P99 to see how far the tails reach."),
            ("Points",  "Number of correlated subsets contributing to these\n"
                        "statistics."),
        ):
            row = QHBoxLayout()
            k_lbl = QLabel(stat + ":")
            k_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
            k_lbl.setFixedWidth(58)
            k_lbl.setToolTip(tip)
            row.addWidget(k_lbl)
            v_lbl = QLabel("—")
            v_lbl.setStyleSheet(
                f"color:{_C_TEXT}; font-size:11px; "
                f"font-family:'Fira Code','Cascadia Code',monospace;"
            )
            v_lbl.setToolTip(tip)
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(v_lbl, 1)
            sb_lay.addLayout(row)
            self._stat_labels[stat] = v_lbl

        # ── Value probe ───────────────────────────────────────────────
        # Reading a number off a colourbar is an estimate. This reports the
        # stored value at the pixel under the cursor, which is the only way to
        # get the actual number at a point without exporting the whole field.
        self._probe_lbl = QLabel("Hover the image to read a value")
        self._probe_lbl.setWordWrap(True)
        self._probe_lbl.setStyleSheet(
            f"color:{_C_TEXT2}; font-size:10px; background:{_C_CARD};"
            f" border:1px solid {_C_BORDER}; border-radius:3px; padding:6px;"
            f" font-family:'Fira Code','Cascadia Code',monospace;")
        sb_lay.addWidget(self._probe_lbl)

        sb_lay.addWidget(self._sep())

        # Colorbar
        cb_head_row = QHBoxLayout()
        cb_head_row.setContentsMargins(0, 0, 0, 0)
        cb_hdr = QLabel("COLORBAR")
        cb_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;"
        )
        cb_head_row.addWidget(cb_hdr)
        cb_head_row.addStretch()
        self._colorbar_unit_lbl = QLabel("")
        self._colorbar_unit_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._colorbar_unit_lbl.setStyleSheet(
            f"color:{_C_TEXT2}; font-size:9px; font-style:italic;")
        cb_head_row.addWidget(self._colorbar_unit_lbl)
        sb_lay.addLayout(cb_head_row)
        self._colorbar = ResultColorBar()
        sb_lay.addWidget(self._colorbar)

        sb_lay.addWidget(self._sep())

        # ── Marker panel (visible only while Streaklines is on) ──
        self._marker_panel = QWidget()
        mp_lay = QVBoxLayout(self._marker_panel)
        mp_lay.setContentsMargins(0, 0, 0, 0)
        mp_lay.setSpacing(6)

        mk_hdr = QLabel("TRAJECTORY MARKERS")
        mk_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:10px; font-weight:800; letter-spacing:1px;")
        mp_lay.addWidget(mk_hdr)

        self._marker_hint = QLabel(
            "Enable <b>Place markers</b> and click the image to add a point to track.")
        self._marker_hint.setWordWrap(True)
        self._marker_hint.setStyleSheet(
            f"color:{_C_TEXT2}; font-size:10px; background:{_C_CARD};"
            f" border:1px dashed {_C_BORDER}; border-radius:3px; padding:7px;")
        mp_lay.addWidget(self._marker_hint)

        self._marker_list = QListWidget()
        self._marker_list.setFixedHeight(132)
        self._marker_list.setStyleSheet(
            f"QListWidget{{background:{_C_CARD}; border:1px solid {_C_BORDER};"
            f" border-radius:3px; font-size:10px; padding:2px;}}"
            f"QListWidget::item{{padding:3px 2px;}}"
            f"QListWidget::item:selected{{background:{_C_RAISED}; border-left:2px solid {_C_ACCENT};}}")
        self._marker_list.currentRowChanged.connect(self._canvas.select_marker)
        mp_lay.addWidget(self._marker_list)

        self._del_marker_btn = QPushButton("Remove selected")
        self._del_marker_btn.setFixedHeight(26)
        self._del_marker_btn.setStyleSheet(
            f"background:{_C_CARD}; color:{_C_TEXT2}; border:1px solid {_C_BORDER};"
            f" border-radius:3px; font-size:10px;")
        self._del_marker_btn.clicked.connect(
            lambda: self._canvas.remove_marker(self._canvas.selected_marker))
        mp_lay.addWidget(self._del_marker_btn)

        self._marker_plot_btn = QPushButton("Plot against time…")
        self._marker_plot_btn.setFixedHeight(26)
        self._marker_plot_btn.setToolTip(
            "Plot any field at these markers over the sequence.\n"
            "Curves are named per marker and can be saved as an image or CSV.")
        self._marker_plot_btn.setStyleSheet(
            f"background:{_C_CARD}; color:{_C_ACCENT}; border:1px solid {_C_ACCENT};"
            f" border-radius:3px; font-size:10px; font-weight:600;")
        self._marker_plot_btn.clicked.connect(self._plot_marker_timeseries)
        mp_lay.addWidget(self._marker_plot_btn)

        self._marker_panel.setVisible(False)
        sb_lay.addWidget(self._marker_panel)

        # ── Smoothed frame-pair sequence ────────────────────────────
        pair_hdr = QLabel("FRAME PAIRS")
        pair_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;")
        sb_lay.addWidget(pair_hdr)

        self._pair_btn = QPushButton("Smoothed pair sequence…")
        self._pair_btn.setFixedHeight(30)
        self._pair_btn.setToolTip(
            "Build a playable sequence of longer frame intervals. Each one\n"
            "composes material motion and recomputes velocity, strain rate\n"
            "and Green-Lagrange strain.")
        self._pair_btn.clicked.connect(self._pair_button_clicked)
        sb_lay.addWidget(self._pair_btn)

        self._pair_banner = QLabel("")
        self._pair_banner.setWordWrap(True)
        self._pair_banner.setStyleSheet(
            f"color:{_C_SUCCESS}; font-size:10px; background:{_C_CARD};"
            f" border:1px solid {_C_SUCCESS}; border-radius:3px; padding:6px;")
        self._pair_banner.setVisible(False)
        sb_lay.addWidget(self._pair_banner)

        self._pair_exit_btn = QPushButton("← Back to single frames")
        self._pair_exit_btn.setFixedHeight(26)
        self._pair_exit_btn.setStyleSheet(
            f"background:{_C_CARD}; color:{_C_TEXT2}; border:1px solid {_C_BORDER};"
            f" border-radius:3px; font-size:10px;")
        self._pair_exit_btn.clicked.connect(self._clear_pair_average)
        self._pair_exit_btn.setVisible(False)
        sb_lay.addWidget(self._pair_exit_btn)

        self._pair_clear_btn = QPushButton("Clear saved temporal data")
        self._pair_clear_btn.setFixedHeight(26)
        self._pair_clear_btn.setStyleSheet(
            f"background:{_C_CARD}; color:{_C_WARN}; border:1px solid {_C_WARN};"
            f" border-radius:3px; font-size:10px;")
        self._pair_clear_btn.setToolTip(
            "Discard the temporal cache and pair list. This cannot be undone.\n"
            "Back to single frames keeps them.")
        self._pair_clear_btn.clicked.connect(self._discard_pair_data)
        self._pair_clear_btn.setVisible(False)
        sb_lay.addWidget(self._pair_clear_btn)

        sb_lay.addWidget(self._sep())

        # One button, one menu. Four stacked export buttons pushed the panel
        # past the window on a short screen, and they are used occasionally
        # rather than continuously, so they do not need to be permanently on
        # display.
        exp2_hdr = QLabel("EXPORT")
        exp2_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;")
        sb_lay.addWidget(exp2_hdr)

        self._export_menu = QMenu(self)
        self._act_export_csv = self._export_menu.addAction(
            "This frame, as CSV…", self._export_csv)
        self._act_export_hdf5 = self._export_menu.addAction(
            "All frames, as HDF5…", self._export_hdf5)
        self._act_export_video = self._export_menu.addAction(
            "Video or image sequence…", self._export_video)
        self._export_menu.addSeparator()
        self._act_export_marker_csv = self._export_menu.addAction(
            "Marker time series, as CSV…", self._export_marker_timeseries)
        self._act_plot_markers = self._export_menu.addAction(
            "Plot markers against time…", self._plot_marker_timeseries)

        self._export_btn = QToolButton()
        self._export_btn.setText("Export…")
        self._export_btn.setFixedHeight(30)
        self._export_btn.setMenu(self._export_menu)
        self._export_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup)
        self._export_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._export_btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        self._export_btn.setToolTip("Save results, or plot marker histories.")
        sb_lay.addWidget(self._export_btn)

        self._export_progress = QProgressBar()
        self._export_progress.setFixedHeight(24)
        self._export_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._export_progress.setStyleSheet(
            f"QProgressBar {{ background: {_C_CARD}; border: 1px solid {_C_BORDER}; border-radius:3px; color: {_C_TEXT}; }}"
            f"QProgressBar::chunk {{ background: {_C_ACCENT}; border-radius: 3px; }}"
        )
        self._export_progress.hide()
        sb_lay.addWidget(self._export_progress)

        sb_lay.addStretch()
        body_lay.addWidget(sidebar)
        root.addWidget(body, 1)

        # ── Bottom: temporal scrubber ─────────────────────────────────
        bottom = QWidget()
        bottom.setFixedHeight(62)
        bottom.setStyleSheet(
            f"background:{_C_SURFACE}; border-top:1px solid {_C_BORDER};"
        )
        bot_lay = QHBoxLayout(bottom)
        bot_lay.setContentsMargins(16, 0, 16, 0)
        bot_lay.setSpacing(10)

        # ── NEW: Reset Zoom Button ────────────────────────────────────
        self._reset_view_btn = QPushButton("Reset zoom")
        self._reset_view_btn.setFixedHeight(30)
        self._reset_view_btn.clicked.connect(self._canvas.fit_image)
        bot_lay.addWidget(self._reset_view_btn)
        # ──────────────────────────────────────────────────────────────

        self._prev_btn = self._nav_btn("◀", self._prev_frame)
        bot_lay.addWidget(self._prev_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        bot_lay.addWidget(self._slider, 1)

        self._next_btn = self._nav_btn("▶", self._next_frame)
        bot_lay.addWidget(self._next_btn)

        bot_lay.addSpacing(12)

        self._frame_lbl = QLabel("Frame — / —")
        self._frame_lbl.setStyleSheet(
            f"color:{_C_TEXT2}; font-size:11px; "
            f"font-family:'Fira Code','Cascadia Code',monospace; min-width:90px;"
        )
        bot_lay.addWidget(self._frame_lbl)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedWidth(76)
        self._play_btn.setFixedHeight(30)
        self._play_btn.clicked.connect(self._toggle_play)
        bot_lay.addWidget(self._play_btn)

        fps_lbl = QLabel("FPS:")
        fps_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        bot_lay.addWidget(fps_lbl)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(5)
        self._fps_spin.setFixedWidth(64)
        # The rate is read fresh on every tick by _advance, so changing it
        # mid-playback takes effect at the next frame without restarting.
        self._fps_spin.setToolTip(
            "Playback speed for review only. The capture rate used for\n"
            "velocity and strain rate is set separately.")
        bot_lay.addWidget(self._fps_spin)

        # ── Scale: pixel size and the unit to report in ───────────────
        # Placed here next to frame rate: together they are the two physical
        # calibrations (space and time) that turn pixel results into engineering
        # quantities, and both can be set after the fact without re-running.
        bot_lay.addSpacing(16)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"background:{_C_BORDER}; max-width:1px;")
        bot_lay.addWidget(sep)
        bot_lay.addSpacing(16)

        scale_lbl = QLabel("1 px =")
        scale_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        scale_lbl.setToolTip(
            "Physical size of one pixel. Displacement and velocity are then\n"
            "reported in real units. Strain and strain rate are ratios and\n"
            "are unaffected.")
        bot_lay.addWidget(scale_lbl)

        self._px_size_spin = QDoubleSpinBox()
        self._px_size_spin.setDecimals(6)
        self._px_size_spin.setRange(0.0, 1e9)
        self._px_size_spin.setValue(0.0)
        self._px_size_spin.setSpecialValueText("— none —")
        self._px_size_spin.setFixedWidth(104)
        self._px_size_spin.setToolTip("0 leaves results in pixels.")
        self._px_size_spin.valueChanged.connect(self._on_calibration_changed)
        bot_lay.addWidget(self._px_size_spin)

        self._unit_combo = QComboBox()
        self._unit_combo.addItems(LENGTH_UNIT_ORDER)
        self._unit_combo.setCurrentText("mm")
        self._unit_combo.setFixedWidth(66)
        self._unit_combo.setToolTip(
            "Unit for the pixel size. Results move up to the next unit "
            "once values reach 100.")
        self._unit_combo.currentTextChanged.connect(self._on_calibration_changed)
        bot_lay.addWidget(self._unit_combo)

        root.addWidget(bottom)

    def _on_calibration_changed(self, *_):
        if getattr(self, "_syncing_calibration", False):
            return
        from src.core.units import Calibration
        unit = self._unit_combo.currentText()
        val = float(self._px_size_spin.value())
        self._wizard.analysis.calibration = (
            Calibration.from_pixel_size(val, unit) if val > 0 else Calibration(None, unit))
        self._unit_scale_cache.clear()
        try:
            self._wizard.analysis.save_settings()
        except Exception:
            pass
        self._refresh_overlay()

    def _sync_calibration_controls(self) -> None:
        cal = self._wizard.analysis.calibration
        self._syncing_calibration = True
        self._unit_combo.setCurrentText(cal.display_unit)
        self._px_size_spin.setValue(cal.pixel_size_in(cal.display_unit) or 0.0)
        self._syncing_calibration = False

    # ------------------------------------------------------------------
    # Public API — called by wizard
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Trajectory markers
    # ------------------------------------------------------------------
    def _on_streak_toggled(self, on: bool) -> None:
        for w in (self._place_btn, self._marker_count_lbl, self._clear_markers_btn,
                  self._trail_lbl, self._trail_combo):
            w.setVisible(on)
        self._marker_panel.setVisible(on)
        if not on:
            self._place_btn.setChecked(False)
            self._canvas.set_marker_mode(False)
        self._refresh_overlay()

    def _on_place_toggled(self, on: bool) -> None:
        self._canvas.set_marker_mode(on)
        self._place_btn.setText("● Placing…" if on else "＋ Place markers")

    def _on_marker_requested(self, x: float, y: float) -> None:
        """
        Canvas click -> reference-frame marker.

        The canvas shows the DEFORMED frame, so a click at frame N is a current
        position. Markers are stored in reference coordinates, so map it back;
        otherwise a marker placed while scrubbed forward would follow the wrong
        material point.
        """
        analysis = self._wizard.analysis
        if not analysis.results:
            return
        mapped = analysis.reference_from_current(x, y, self._frame)
        if mapped is None:
            QMessageBox.information(self, "No data",
                                    "This frame has no valid correlation data.")
            return
        (rx, ry), resid = mapped
        if resid > 25.0:
            QMessageBox.information(
                self, "Outside analysed region",
                "That point is outside the analysed region for this frame.\n"
                f"Nearest tracked material point is {resid:.0f} px away.")
            return
        self._canvas.add_marker(rx, ry)

    def _on_markers_changed(self, pts) -> None:
        n = len(pts)
        self._marker_count_lbl.setText("1 marker" if n == 1 else f"{n} markers")
        self._rebuild_marker_list()
        self._refresh_overlay()

    def _clear_markers(self) -> None:
        self._canvas.clear_markers()

    def _on_marker_selected(self, i: int) -> None:
        if 0 <= i < self._marker_list.count() and self._marker_list.currentRow() != i:
            self._marker_list.blockSignals(True)
            self._marker_list.setCurrentRow(i)
            self._marker_list.blockSignals(False)
        self._del_marker_btn.setEnabled(i >= 0)

    def reset_markers(self) -> None:
        """Drop all marker state. Called when a new session starts."""
        self._traj_cache.clear()
        self._place_btn.setChecked(False)
        self._streak_chk.setChecked(False)
        self._trail_combo.setCurrentIndex(0)
        self._canvas.set_marker_mode(False)
        # clear_markers() emits markers_changed, whose slot immediately tries
        # to render trajectories. Wizard.new_session has already replaced the
        # analysis model at this point, so that re-entrant render mixes the old
        # Results canvas with an empty new model and can native-crash Qt. Reset
        # this page-owned state silently, as on_enter() already does.
        self._canvas.set_markers([])
        self._canvas.set_marker_draw_positions([])
        self._canvas.set_streaklines(None)
        self._marker_list.clear()
        self._del_marker_btn.setEnabled(False)
        self._marker_hint.setVisible(True)
        self._marker_count_lbl.setText("0 markers")

    def _rebuild_marker_list(self) -> None:
        self._marker_list.blockSignals(True)
        self._marker_list.clear()
        analysis = self._wizard.analysis
        pts = self._canvas.markers
        trail = self._trail_combo.currentData() or 0
        trajs = analysis.get_trajectories_from_seeds(pts, self._frame, trail) if pts else []
        for i, (x, y) in enumerate(pts):
            if i < len(trajs) and trajs[i]["lost_at"] is not None:
                status = f"lost @ frame {trajs[i]['lost_at']}"
            elif i < len(trajs):
                npts = len(trajs[i]["points"])
                status = f"{npts} pts"
            else:
                status = "—"
            it = QListWidgetItem(f"  {i+1}.   x={x:.0f}, y={y:.0f}    ·  {status}")
            it.setForeground(marker_color(i))
            self._marker_list.addItem(it)
        sel = self._canvas.selected_marker
        if 0 <= sel < self._marker_list.count():
            self._marker_list.setCurrentRow(sel)
        self._marker_list.blockSignals(False)
        self._marker_hint.setVisible(len(pts) == 0)

    def _render_trajectories(self, idx: int) -> None:
        analysis = self._wizard.analysis
        if not self._streak_chk.isChecked():
            self._canvas.set_streaklines(None)
            self._canvas.set_markers([])
            return
        pts = self._canvas.markers
        if not pts:
            self._canvas.set_marker_draw_positions([])
            self._canvas.set_streaklines(None)
            return
        trail = self._trail_combo.currentData() or 0
        trajs, draw_pts = self._cached_trajectories(pts, idx, trail)
        self._canvas.set_marker_draw_positions(draw_pts)
        self._canvas.set_streaklines(
            [t["points"] for t in trajs],
            colors=[marker_color(i) for i in range(len(trajs))],
            lost_flags=[t["lost_at"] is not None for t in trajs],
        )

    def on_before_show(self) -> None:
        """Cheap synchronous blanking so no stale frame is ever painted."""
        try:
            self._play_timer.stop()
            self._scrub_timer.stop()
            if self._play_btn.isChecked():
                self._play_btn.setChecked(False)
            self._invalidate_pair_jobs()
            self._canvas.clear_result_overlay()
            self._canvas.set_streaklines(None)
            self._canvas.set_markers([])
            self._canvas.set_marker_mode(False)
            self._marker_list.clear()
            self._place_btn.setChecked(False)
            self._slider.setValue(0)
            self._frame = 0
            for lbl in getattr(self, "_stat_labels", {}).values():
                lbl.setText("—")
        except Exception:
            pass

    def on_enter(self) -> None:
        """Refresh after analysis completes."""
        n = len(self._wizard.analysis.results)
        self._unit_scale_cache.clear()
        # Both caches are keyed by path and frame index, which a new run reuses
        # while their contents differ. Clearing here is what stops the previous
        # sequence's frames and trajectories being served for this one.
        self._invalidate_caches()
        # Markers are indexed against the previous run's displacement fields, so
        # they are meaningless for a new sequence.
        # Do this silently: clear_markers() emits markers_changed, whose slot
        # renders frame 0 re-entrantly while this page is only half initialised.
        # That nested QPixmap/render path caused Qt6Core fast-fail exits during
        # the Analysis -> Results transition.
        self._canvas.set_markers([])
        self._canvas.set_streaklines(None)
        self._marker_list.clear()
        # Same reasoning for a stored pair sequence: its frame indices and its
        # arrays belong to the run that produced them. Keeping it across a
        # re-analysis would silently show the previous run's numbers.
        self._pair_avg = None
        self._pair_mode = False
        self._pair_list = []
        self._pair_sequence_mode = "custom"
        self._pair_strain_window = int(
            getattr(getattr(self._wizard.analysis, "params", None),
                    "strain_window", 5))
        self._invalidate_pair_jobs()
        saved_temporal = getattr(self._wizard.analysis, "temporal_results", None)
        saved_pairs = list(getattr(
            self._wizard.analysis, "temporal_pairs", []) or [])
        if saved_temporal is not None and saved_pairs:
            metadata = dict(getattr(
                self._wizard.analysis, "temporal_metadata", {}) or {})
            self._pair_list = saved_pairs
            self._pair_sequence_mode = str(metadata.get("mode", "custom"))
            self._pair_strain_window = int(metadata.get(
                "strain_window", self._pair_strain_window))
            self._pair_mode = True
            if int(metadata.get("schema", 0)) >= 4:
                self._pair_store = saved_temporal
                self._pair_bulk_ready = set(range(len(saved_pairs)))
            else:
                # Older temporal files either reset finite strain at every
                # window or replayed frames outside the selected range.
                # Preserve the pair definition but regenerate schema-4 data.
                self._begin_pair_bulk()
        self._sync_pair_ui()
        self._sync_calibration_controls()
        timeline_n = len(self._pair_list) if self._pair_mode else n
        self._slider.setMaximum(max(0, timeline_n - 1))
        self._slider.setValue(0)
        self._frame = 0
        if self._pair_mode:
            self._show_pair_average()
        else:
            self._show_frame(0)

    # ------------------------------------------------------------------
    # Caches
    # ------------------------------------------------------------------
    # Scrubbing and playback revisit the same frames repeatedly. Both caches
    # are bounded and keyed by exactly what the result depends on, so a stale
    # entry cannot be served after the thing it was derived from changes.

    def _background_image(self, path: str):
        """Decoded frame for `path`, from cache when possible."""
        hit = self._img_cache.get(path)
        if hit is not None:
            self._img_cache.move_to_end(path)
            return hit
        try:
            from src.core.analysis import _load_image
            img = _load_image(path)
        except Exception as exc:
            print(f"Failed to load image {path}: {exc}")
            return None
        self._img_cache[path] = img
        while len(self._img_cache) > self._IMG_CACHE_MAX:
            self._img_cache.popitem(last=False)
        return img

    def _cached_trajectories(self, pts, idx: int, trail: int):
        """Marker trajectories, from cache when the inputs are unchanged.

        Keyed on the marker positions themselves, so moving, adding or removing
        a marker misses the cache rather than redrawing the previous paths.
        """
        key = (idx, trail, tuple((round(x, 3), round(y, 3)) for x, y in pts))
        hit = self._traj_cache.get(key)
        if hit is not None:
            self._traj_cache.move_to_end(key)
            return hit
        analysis = self._wizard.analysis
        trajs = analysis.get_trajectories_from_seeds(pts, idx, trail)
        value = (trajs, analysis.marker_positions(pts, idx))
        self._traj_cache[key] = value
        while len(self._traj_cache) > self._TRAJ_CACHE_MAX:
            self._traj_cache.popitem(last=False)
        return value

    def _invalidate_caches(self) -> None:
        """Drop cached frames and trajectories.

        Called whenever the underlying sequence changes. Trajectories also
        depend on the displacement fields, so a re-analysis must clear them
        even though the marker positions may be identical.
        """
        self._img_cache.clear()
        self._traj_cache.clear()
        self._pair_cache.clear()

    def _show_frame(self, idx: int) -> None:
        """
        Load the actual deformed image for frame `idx` and display it as
        the canvas background, then render the field overlay on top.
        """
        analysis = self._wizard.analysis
        n = len(analysis.results)
        if n == 0 or idx >= n:
            return

        # 1. ── Load and display the deformed image ──────────────────
        if idx < len(analysis.def_paths):
            img = self._background_image(analysis.def_paths[idx])

            if img is not None:
                keep = self._canvas._image_arr is not None
                # Dim only the pixmap; retain the compact float32 source for
                # cursor values without allocating a second full image.
                self._canvas.set_image(
                    img, keep_view=keep, display_scale=0.45)
            else:
                self._canvas.clear_result_overlay()

        # 2. ── Render field overlay ──────────────────────────────────
        result = analysis.results[idx]
        arr, _ = self._display_array(result)
        if arr is not None and finite_values(arr).size:
            # _apply_overlay internally checks if static_scale_chk is enabled
            self._apply_overlay(arr)
        else:
            self._canvas.set_result_overlay_rgba(None)

        # 3. ── Render marker trajectories ───────────────────────────
        self._render_trajectories(idx)

        # 4. ── Update sidebar ────────────────────────────────────────
        self._update_stats(result)
        self._frame_lbl.setText(f"Frame {idx + 1} / {n}")

    def _on_scale_mode_changed(self, *_):
        """Enable the manual boxes only in Range mode, then re-render once.

        buttonToggled fires twice per change (off for the old button, on for the
        new one), so act only on the checked signal or every switch renders the
        frame twice.
        """
        if len(_) >= 2 and _[1] is False:
            return
        manual = self._scale_manual_rb.isChecked()
        for w in (self._range_min_spin, self._range_max_spin, self._range_fit_btn):
            w.setEnabled(manual)
        if manual and self._range_min_spin.value() == self._range_max_spin.value():
            # Seed from what is on screen rather than making the user guess.
            self._fit_range_to_frame()
        else:
            self._refresh_overlay()

    def _fit_range_to_frame(self) -> None:
        """Load the current frame's actual limits into the manual boxes."""
        results = self._wizard.analysis.results
        if not results:
            return
        if self._pair_mode:
            result = self._pair_result_at(self._frame)
        else:
            idx = max(0, min(self._frame, len(results) - 1))
            result = results[idx]
        if result is None:
            return
        arr, _ = self._display_array(result)
        if arr is None:
            return
        vals = finite_values(arr)
        if not vals.size:
            return
        blocked = (self._range_min_spin.blockSignals(True),
                   self._range_max_spin.blockSignals(True))
        self._range_min_spin.setValue(float(vals.min()))
        self._range_max_spin.setValue(float(vals.max()))
        self._range_min_spin.blockSignals(False)
        self._range_max_spin.blockSignals(False)
        self._refresh_overlay()

    def _update_range_placeholders(self, vmin: float, vmax: float) -> None:
        """Keep the disabled boxes showing the range actually in use.

        Ticking Range then starts from what you were already looking at instead
        of from zero.
        """
        if self._scale_manual_rb.isChecked():
            return
        for sb, v in ((self._range_min_spin, vmin), (self._range_max_spin, vmax)):
            sb.blockSignals(True)
            sb.setValue(float(v))
            sb.blockSignals(False)

    def current_range_spec(self) -> "RangeSpec":
        """How the colour limits are currently chosen.

        Manual wins over the static global scale, which wins over per-frame
        auto. Manual limits are read in DISPLAY units because that is what the
        colourbar is labelled with and therefore what the user typed against.
        """
        mode = "auto"
        vmin = vmax = None
        if self._scale_manual_rb.isChecked():
            mode = "manual"
            vmin, vmax = self._range_min_spin.value(), self._range_max_spin.value()
        elif self._scale_global_rb.isChecked():
            mode = "global"
        return RangeSpec(mode=mode, vmin=vmin, vmax=vmax,
                         symmetric=self._sym_chk.isChecked(),
                         percentile=self.current_coverage())

    def current_coverage(self) -> float:
        """Central share of the data the colour scale must span, in percent."""
        val = self._coverage_combo.currentData()
        return 100.0 if val is None else float(val)

    def _apply_overlay(self, arr: np.ndarray) -> None:
        """Colour-map the selected field onto the canvas.

        The pixel work lives in ui/render.py so that the video exporter renders
        through exactly the same code and the two can never drift apart.
        """
        analysis = self._wizard.analysis
        spec = self.current_range_spec()

        # get_global_range works on the stored (pixel) arrays, but `arr` arrives
        # already converted, so the fixed scale has to be converted too or the
        # colourbar and the image disagree.
        global_rng = None
        if spec.mode == "global" and self._pair_avg is None:
            factor, _ = self._unit_factor()
            lo, hi = analysis.get_global_range(self._field, spec.percentile)
            global_rng = (lo * factor, hi * factor)
        elif spec.mode == "global":
            # Pair fields have different interval semantics from stored
            # single-frame fields, so their limits cannot use analysis' normal
            # global range. Establish the limits once from the first displayed
            # pair and keep them fixed for every subsequent pair. Previously we
            # silently changed Global back to Auto here, making both overlay and
            # colourbar jump during playback.
            factor, _ = self._unit_factor(result=self._pair_avg)
            key = (self._pair_generation, self._field,
                   float(spec.percentile), float(factor))
            seed_spec = RangeSpec(
                mode="auto", symmetric=False, percentile=spec.percentile)
            global_rng = self._pair_fixed_ranges.get(key)
            if global_rng is None:
                # Establish the range once and keep it: a scale that moves
                # between pairs makes a feature pulse when only the mapping
                # changed. Seed it from pairs sampled across the whole
                # sequence, because the first pair alone routinely
                # under-covers -- the shear zone intensifies later, and every
                # subsequent pair then sits past the top of the scale.
                global_rng = self._pair_sequence_range(seed_spec, factor)
                if global_rng is None:
                    global_rng = seed_spec.resolve(arr)
                if global_rng is not None:
                    self._pair_fixed_ranges[key] = global_rng

        rng = spec.resolve(arr, global_rng)
        if rng is None:
            self._canvas.set_result_overlay_rgba(None)
            return
        vmin, vmax = rng
        cmap_name = self._cmap_combo.currentText()

        try:
            rgba = render.field_to_rgba(
                arr, vmin, vmax, cmap_name,
                roi_mask=analysis.roi_mask,
                spacing=analysis.params.subset_spacing,
                mark_out_of_range=self._clip_chk.isChecked())
            self._canvas.set_result_overlay_rgba(rgba)

            # How much of the field the scale is actually showing. Reported
            # rather than inferred, so a trimmed scale is never mistaken for
            # the full data range.
            finite = finite_values(arr)
            below = int(np.count_nonzero(finite < vmin))
            above = int(np.count_nonzero(finite > vmax))
            self._update_colorbar(render.get_cmap(cmap_name, 256), vmin, vmax,
                                  below=below, above=above, total=finite.size)
            self._update_range_placeholders(vmin, vmax)
        except Exception as exc:
            print(f"Overlay error: {exc}")
            self._canvas.set_result_overlay_rgba(None)

    # ------------------------------------------------------------------
    # Physical units
    # ------------------------------------------------------------------
    # Results are computed and stored in pixels. Everything the user reads --
    # overlay range, colourbar, statistics -- goes through this one conversion
    # so the number and its unit can never disagree.

    def _unit_factor(self, result=None, native_arr=None) -> tuple:
        analysis = self._wizard.analysis
        cal = analysis.calibration
        base = FIELDS.get(self._field, ("", ""))[1]
        results = analysis.results
        lazy = bool(getattr(analysis, "hdf5_lazy", False))
        if result is None and lazy and results:
            result = (self._pair_avg if self._pair_avg is not None else
                      results[max(0, min(self._frame, len(results) - 1))])
        key = (
            self._field, cal.metres_per_pixel, cal.display_unit, len(results),
            id(result) if lazy else (id(results[0]) if results else None),
            None if lazy else (id(results[-1]) if results else None),
        )
        cached = self._unit_scale_cache.get(key)
        if cached is not None:
            return cached

        factor_unit = cal.factor_and_unit(self._field, base)
        if cal.calibrated and results:
            if lazy and result is not None:
                values = (np.asarray(native_arr) if native_arr is not None else
                          np.asarray(getattr(result, self._field, np.zeros(0))))
                finite = np.abs(values[np.isfinite(values)])
                magnitude = (float(np.percentile(finite, 99.0))
                             if finite.size else 0.0)
            else:
                lo, hi = analysis.get_global_range(self._field, 99.0)
                magnitude = max(abs(float(lo)), abs(float(hi)))
            factor_unit = cal.compact_factor_and_unit(
                self._field, magnitude, base)
        self._unit_scale_cache.clear()
        self._unit_scale_cache[key] = factor_unit
        return factor_unit

    def _display_array(self, result):
        """The selected field in display units, plus its unit label."""
        arr = getattr(result, self._field, None)
        if (arr is not None and
                bool(getattr(self._wizard.analysis, "hdf5_lazy", False)) and
                not isinstance(arr, np.ndarray)):
            arr = np.asarray(arr)
        factor, unit = self._unit_factor(result, arr)
        if arr is None or factor == 1.0:
            return arr, unit
        return arr * factor, unit

    #: Pairs sampled when establishing the fixed colour range. Enough to cover
    #: how the field evolves, few enough that entering pair mode stays instant.
    _PAIR_RANGE_SAMPLES = 12

    def _pair_sequence_range(self, seed_spec, factor: float):
        """
        Colour limits covering the pair sequence, from already-computed pairs.

        Only pairs present on disk are read, so this never triggers
        computation and never blocks: if nothing has been precomputed the
        caller falls back to the pair on screen.
        """
        store = self._pair_store
        if store is None:
            return None
        try:
            total = len(store)
        except Exception:
            return None
        if total <= 0:
            return None

        step = max(1, total // self._PAIR_RANGE_SAMPLES)
        lo = hi = None
        for index in range(0, total, step):
            try:
                if not store.has(index):
                    continue
                res = store[index]
            except Exception:
                continue
            arr = getattr(res, self._field, None)
            if arr is None:
                continue
            rng = seed_spec.resolve(np.asarray(arr) * factor)
            if rng is None:
                continue
            lo = rng[0] if lo is None else min(lo, rng[0])
            hi = rng[1] if hi is None else max(hi, rng[1])
        if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        return (float(lo), float(hi))

    def _update_colorbar(self, cmap, vmin, vmax,
                         below: int = 0, above: int = 0, total: int = 0):
        n_bar = 64
        bar_colors = []
        for i in range(n_bar):
            r, g, b, _ = cmap(i / (n_bar - 1))
            bar_colors.append((int(r * 255), int(g * 255), int(b * 255)))
        _, unit = self._unit_factor()
        self._colorbar_unit_lbl.setText(unit)
        self._colorbar.update_bar(vmin, vmax, unit, bar_colors,
                                  below=below, above=above, total=total)

    def _update_stats(self, result) -> None:
        arr, unit = self._display_array(result)
        self._stats_unit_lbl.setText(unit)
        summary = field_summary(arr) if arr is not None else None
        if summary is None:
            for v in self._stat_labels.values():
                v.setText("—")
            return

        # Units are section metadata, not part of every value. Keeping the
        # values numeric prevents the label/value collision seen for the long
        # word "dimensionless" and makes the rows easier to scan.
        self._stat_labels["Mean"].setText(f"{summary['mean']:.4g}")
        self._stat_labels["Median"].setText(f"{summary['median']:.4g}")
        self._stat_labels["Std Dev"].setText(f"{summary['std']:.4g}")
        self._stat_labels["IQR"].setText(f"{summary['iqr']:.4g}")
        self._stat_labels["P1–P99"].setText(
            f"{summary['p_low']:.3g} … {summary['p_high']:.3g}")
        self._stat_labels["Min/Max"].setText(
            f"{summary['minimum']:.3g} … {summary['maximum']:.3g}")
        self._stat_labels["Points"].setText(f"{summary['count']:,}")

        # Flag a mean the outliers have run away with. The threshold compares
        # the mean-median gap against the robust spread, so it fires on genuine
        # skew rather than on any field that merely is not symmetric.
        iqr = summary["iqr"]
        skewed = (np.isfinite(iqr) and iqr > 0 and
                  abs(summary["mean"] - summary["median"]) > 0.5 * iqr)
        self._stat_labels["Mean"].setStyleSheet(
            f"color:{_C_WARN if skewed else _C_TEXT}; font-size:11px; "
            f"font-family:'Fira Code','Cascadia Code',monospace;")
        self._stat_labels["Mean"].setToolTip(
            "Mean and median differ by more than half the interquartile "
            "range.\nOutliers are pulling the mean; prefer the median."
            if skewed else
            "Arithmetic mean. One decorrelated subset can shift it "
            "without bound.")

    # ------------------------------------------------------------------
    # Value probe
    # ------------------------------------------------------------------

    def _on_cursor_moved(self, x: int, y: int, _v: float) -> None:
        """Report the field value at the pixel under the cursor.

        Stored subset-centre values are reported exactly. Pixels between subset
        centres use bilinear interpolation and are visibly marked approximate;
        values are never inferred by inverting the rendered colour.
        """
        result = self._pair_avg
        if result is None:
            results = self._wizard.analysis.results
            if not results or not (0 <= self._frame < len(results)):
                return
            result = results[self._frame]

        arr, unit = self._display_array(result)
        if arr is None or arr.ndim != 2:
            return
        h, w = arr.shape
        if not (0 <= y < h and 0 <= x < w):
            self._probe_lbl.setText(f"x={x}  y={y}\noutside field")
            return

        analysis = self._wizard.analysis
        roi = analysis.roi_mask
        if roi is not None and roi.shape == arr.shape and not roi[y, x]:
            self._probe_lbl.setText(f"x={x}  y={y}\noutside ROI")
            return

        val = arr[y, x]
        short = _field_short(self._field)
        suffix = f" {unit}" if unit else ""
        if np.isfinite(val):
            self._probe_lbl.setText(f"x={x}  y={y}\n{short} = {val:.6g}{suffix}")
        else:
            params = analysis.params
            spacing = max(1, int(getattr(params, "subset_spacing", 1)))
            origin = int(getattr(params, "subset_radius", 0))
            estimate = _interpolate_between_subset_centres(
                arr, x, y, spacing, origin)
            if estimate is not None:
                self._probe_lbl.setText(
                    f"x={x}  y={y}\n{short} ≈ {estimate:.6g}{suffix}"
                    "\n(interpolated between subset centres)")
                return

            on_grid = ((x - origin) % spacing == 0 and
                       (y - origin) % spacing == 0)
            if on_grid:
                valid = getattr(result, "valid", None)
                failed = (valid is not None and valid.shape == arr.shape and
                          not bool(valid[y, x]))
                reason = ("not correlated here" if failed else
                          "no valid field value at this subset centre")
            else:
                reason = "insufficient surrounding subset data"
            self._probe_lbl.setText(
                f"x={x}  y={y}\n{short} = no data\n({reason})")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_slider(self, val: int) -> None:
        self._frame = val
        # Loading, colouring and uploading a full-resolution frame for every
        # intermediate slider event makes scrubbing lag behind the pointer.
        # Render the most recent position after a very short coalescing window.
        self._scrub_timer.start()

    def _render_scrubbed_frame(self) -> None:
        if self._pair_mode:
            self._show_pair_average()
        else:
            self._show_frame(self._frame)

    def _sync_field_buttons(self) -> None:
        """Show only the buttons belonging to the selected category."""
        visible = set(FIELD_GROUPS.get(self._cat_combo.currentText(), []))
        for k, btn in self._field_btns.items():
            btn.setVisible(k in visible)
            btn.setChecked(k == self._field)

    def _select_category(self, name: str) -> None:
        keys = FIELD_GROUPS.get(name, [])
        if keys and self._field not in keys:
            self._field = keys[0]
        self._sync_field_buttons()
        self._apply_tab_style()
        self._refresh_overlay()

    def _select_field(self, key: str) -> None:
        self._field = key
        grp = _group_of(key)
        if self._cat_combo.currentText() != grp:
            self._cat_combo.blockSignals(True)
            self._cat_combo.setCurrentText(grp)
            self._cat_combo.blockSignals(False)
        self._sync_field_buttons()
        self._apply_tab_style()
        self._refresh_overlay()

    def _refresh_overlay(self, *_) -> None:
        if self._pair_mode:
            self._show_pair_average()
        else:
            self._show_frame(self._frame)

    # ------------------------------------------------------------------
    # Smoothed frame-pair sequence
    # ------------------------------------------------------------------

    def _pair_button_clicked(self) -> None:
        if not self._pair_mode and self._pair_list:
            self._resume_pair_sequence()
        else:
            self._open_pair_dialog()

    def _resume_pair_sequence(self) -> None:
        """Return to the retained temporal timeline without recalculating it."""
        if not self._pair_list:
            self._open_pair_dialog()
            return
        self._single_frame_before_pairs = max(
            0, min(self._slider.value(),
                   max(0, len(self._wizard.analysis.results) - 1)))
        self._pair_mode = True
        index = max(0, min(self._pair_frame_before_single,
                           len(self._pair_list) - 1))
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, len(self._pair_list) - 1))
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._frame = index
        self._sync_pair_ui()
        self._show_pair_average()

    def _open_pair_dialog(self) -> None:
        analysis = self._wizard.analysis
        n = len(analysis.results)
        if n < 2:
            QMessageBox.information(
                self, "PyDIC",
                "Frame-pair averaging needs at least two analysed frames.")
            return

        from src.ui.pages.frame_pair_dialog import FramePairDialog
        dlg = FramePairDialog(
            n, analysis.fps, self._pair_list,
            strain_window=self._pair_strain_window,
            grid_spacing=analysis.params.subset_spacing,
            existing_mode=self._pair_sequence_mode,
            parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        pairs = dlg.pairs()
        if not pairs:
            return
        new_window = dlg.strain_window()
        sequence_mode = dlg.sequence_mode()
        if (pairs == self._pair_list and
                new_window == self._pair_strain_window and
                sequence_mode == self._pair_sequence_mode):
            self._sync_pair_ui()
            return
        self._pair_strain_window = new_window
        if not self._pair_mode:
            self._single_frame_before_pairs = max(
                0, min(self._slider.value(), len(analysis.results) - 1))
        self._pair_list = pairs
        self._pair_sequence_mode = sequence_mode
        self._pair_mode = True
        self._invalidate_pair_jobs()
        self._scrub_timer.stop()
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, len(pairs) - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._frame = 0

        self._sync_pair_ui()
        if self._pair_sequence_mode in ("sliding", "non_overlapping"):
            self._begin_pair_bulk()
        self._refresh_overlay()

    def _clear_pair_average(self) -> None:
        """Display single frames while retaining temporal data and workers."""
        if self._play_btn.isChecked():
            self._play_btn.setChecked(False)
            self._toggle_play(False)
        self._pair_frame_before_single = self._frame
        self._pair_mode = False
        self._pair_avg = None
        n = len(self._wizard.analysis.results)
        restored = max(0, min(self._single_frame_before_pairs, max(0, n - 1)))
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(restored)
        self._slider.blockSignals(False)
        self._frame = restored
        self._sync_pair_ui()
        self._refresh_overlay()

    def _discard_pair_data(self) -> None:
        """Explicit UI action that permanently drops the temporal cache."""
        if self._pair_mode:
            self._clear_pair_average()
        self._invalidate_pair_jobs()
        self._pair_list = []
        self._pair_sequence_mode = "custom"
        self._pair_frame_before_single = 0
        self._sync_pair_ui()

    def _sync_pair_ui(self) -> None:
        """Show pair-sequence state while keeping its timeline interactive."""
        active = self._pair_mode
        self._pair_banner.setVisible(active)
        self._pair_exit_btn.setVisible(active)
        self._pair_clear_btn.setVisible(bool(self._pair_list))
        self._scale_global_rb.setText("Pair-fixed" if active else "Global")
        self._scale_global_rb.setToolTip(
            "Keep the limits from the first pair so the sequence stays "
            "comparable."
            if active else
            "Fix the scale to the min/max across the whole sequence.")
        self._pair_btn.setText(
            "Edit pair sequence…" if active else
            ("Return to temporal sequence" if self._pair_list else
             "Smoothed pair sequence…"))

        if active:
            n = len(self._pair_list)
            label = ", ".join(f"{a + 1}→{b + 1}" for a, b in self._pair_list[:4])
            if n > 4:
                label += f", +{n - 4} more"
            spans = [abs(b - a) for a, b in self._pair_list]
            dt_ms = (float(np.mean(spans)) /
                     max(float(self._wizard.analysis.fps), 1e-9) * 1e3)
            text = (f"<b>Playing {n} smoothed pair{'s' if n != 1 else ''}</b>"
                    f"<br>{label}"
                    f"<br>Mean interval {dt_ms:.4g} ms"
                    f" · spatial strain radius {self._pair_strain_window} px")
            if len(set(spans)) > 1:
                text += ("<br><span style='color:" + C_WARNING + "'>"
                         "Pairs span different durations; compare velocity and "
                         "strain rate rather than raw displacement.</span>")
            text += ("<br>Rate is calculated from each smoothed interval. "
                     "Green–Lagrange strain retains the accumulated selected "
                     "history through the interval endpoint. Pair-fixed "
                     "colour limits stay constant during playback.")
            text += f"<br>Processing backend: {self._pair_backend_text()}."
            if self._pair_sequence_mode in ("sliding", "non_overlapping"):
                mode_label = ("sliding" if self._pair_sequence_mode == "sliding"
                              else "non-overlapping")
                if self._pair_rates_only:
                    rate_ready = len(self._pair_rate_ready)
                    phase = (f"calculated strain rates "
                             f"{len(self._pair_bulk_ready)}/{n}"
                             if self._pair_history_active else
                             f"velocity averages {rate_ready}/{n}; "
                             "strain rate hidden")
                    text += (f"<br>{mode_label.title()} preprocessing: "
                             f"{phase}.")
                else:
                    ready = len(self._pair_bulk_ready)
                    text += (f"<br>{mode_label.title()} preprocessing: "
                             f"{ready}/{n} ready for seeking and export.")
            self._pair_banner.setText(text)

        for w in (self._slider, self._play_btn, self._prev_btn, self._next_btn):
            w.setEnabled(True)

        for k, btn in self._field_btns.items():
            if k in _ACCUMULATED_STRAIN_FIELDS:
                btn.setEnabled(True)
                btn.setToolTip("")

    def _invalidate_pair_jobs(self) -> None:
        """Forget work whose sequence or spatial window is no longer current."""
        old_generation = self._pair_generation
        # Running QRunnables cannot be removed from QThreadPool. Give them a
        # cooperative stop token so New Session and pair edits return promptly
        # instead of leaving a full strain-history replay consuming the GUI's
        # process for minutes.
        self._pair_cancel_flag[0] = True
        self._pair_cancel_flag = [False]
        self._pair_generation += 1
        self._pair_queue.clear()
        self._pair_cache.clear()
        self._pair_fixed_ranges.clear()
        self._pair_avg = None
        self._pair_store = None
        self._pair_bulk_ready.clear()
        self._pair_rate_ready.clear()
        self._pair_rates_only = False
        self._pair_history_active = False
        self._pair_history_task = None
        self._cleanup_pair_store_dirs(exclude_generation=old_generation)

    def _cleanup_pair_store_dirs(self, exclude_generation: Optional[int] = None) -> None:
        """Remove retired exact temp directories once no worker can write them."""
        for generation, directory in list(self._pair_store_dirs.items()):
            if generation == self._pair_generation:
                continue
            if generation in self._pair_retained_generations:
                continue
            if generation == exclude_generation and any(
                    key[0] == generation for key in self._pair_active):
                continue
            if any(key[0] == generation for key in self._pair_active):
                continue
            shutil.rmtree(directory, ignore_errors=True)
            self._pair_store_dirs.pop(generation, None)

    def _begin_pair_bulk(self) -> None:
        """Queue every generated temporal pair and expose aggregate progress."""
        from src.core.temporal import TemporalResultSequence

        directory = tempfile.mkdtemp(prefix="pydic_temporal_")
        self._pair_store = TemporalResultSequence(directory, self._pair_list)
        self._pair_store_dirs[self._pair_generation] = directory
        self._pair_bulk_ready.clear()
        self._pair_rate_ready.clear()
        self._pair_rates_only = True
        self._pair_history_active = False
        self._export_progress.show()
        self._export_progress.setValue(0)
        self._export_progress.setFormat(
            f"Temporal 0/{len(self._pair_list)} · %p%")
        for index in range(len(self._pair_list)):
            self._request_pair(index, priority=(index == self._frame))
        self._sync_pair_ui()

    def _pair_use_gpu(self) -> bool:
        analysis = self._wizard.analysis
        available = bool(
            getattr(analysis, "prefer_gpu", False) and
            getattr(analysis, "last_backend", "cpu") == "gpu" and
            importlib.util.find_spec("cupy") is not None)
        if not available or not getattr(analysis, "results", None):
            return False

        # CuPy filters win only after transfer/launch overhead is amortised.
        # Use the same finite-support crop estimate as strain.py. A small ROI on
        # a large camera frame must not serialize the queue as a "GPU" job that
        # the strain routine will immediately send back to CPU.
        result = analysis.results[min(1, len(analysis.results) - 1)]
        shape = tuple(int(v) for v in result.u.shape)
        valid = getattr(result, "valid", None)
        if isinstance(valid, CompactMask):
            indices = valid.indices
        elif isinstance(result.u, CompactField):
            indices = result.u.indices
        else:
            mask = (np.asarray(valid, dtype=bool) if valid is not None else
                    (np.isfinite(result.u) & np.isfinite(result.v)))
            indices = np.flatnonzero(mask.reshape(-1))
        if not len(indices):
            return False
        ys = np.asarray(indices, dtype=np.int64) // shape[1]
        xs = np.asarray(indices, dtype=np.int64) % shape[1]
        pad = (int(self._pair_strain_window) +
               max(0, int(analysis.params.subset_spacing) // 2) + 1)
        height = min(shape[0], int(ys.max()) + pad + 1) - max(
            0, int(ys.min()) - pad)
        width = min(shape[1], int(xs.max()) + pad + 1) - max(
            0, int(xs.min()) - pad)
        return height * width >= 512 * 512

    def _pair_backend_text(self) -> str:
        if self._pair_use_gpu():
            return "CuPy GPU derivatives for large spatial fits"
        if importlib.util.find_spec("cupy") is None:
            return "multithreaded CPU (CuPy unavailable)"
        if not bool(getattr(self._wizard.analysis, "prefer_gpu", False)):
            return "multithreaded CPU (GPU disabled)"
        if getattr(self._wizard.analysis, "last_backend", "cpu") == "gpu":
            return "multithreaded CPU (CuPy available; ROI below GPU crossover)"
        return "multithreaded CPU (source analysis was CPU)"

    def _pair_worker_limit(self) -> int:
        analysis = self._wizard.analysis
        # A changed spatial window replays one shared accumulated-strain
        # history. Additional workers would only wait on that cache lock and
        # waste memory; after the replay, the queued pair products are cheap.
        if (not self._pair_rates_only and
                int(self._pair_strain_window) !=
                int(analysis.params.effective_strain_window())):
            return 1
        # h5py-backed lazy fields share one open file handle. They still run off
        # the GUI thread, but concurrent reads are deliberately serialized.
        if bool(getattr(analysis, "hdf5_lazy", False)) or self._pair_use_gpu():
            return 1
        return max(1, min(3, (os.cpu_count() or 2) // 2 or 1))

    def _request_pair(self, index: int, priority: bool = True) -> None:
        if not self._pair_list or (not self._pair_mode and self._pair_store is None):
            return
        index = max(0, min(int(index), len(self._pair_list) - 1))
        key = (self._pair_generation, index)
        if index in self._pair_cache or key in self._pair_active:
            return
        if self._pair_store is not None and self._pair_store.has(index):
            return
        if index in self._pair_queue:
            if priority:
                self._pair_queue.move_to_end(index, last=False)
        else:
            self._pair_queue[index] = None
            if priority:
                self._pair_queue.move_to_end(index, last=False)
        self._pump_pair_workers()

    def _pump_pair_workers(self) -> None:
        limit = self._pair_worker_limit()
        while self._pair_queue and len(self._pair_active) < limit:
            index, _ = self._pair_queue.popitem(last=False)
            if (not self._pair_list or
                    (not self._pair_mode and self._pair_store is None) or
                    not (0 <= index < len(self._pair_list))):
                continue
            generation = self._pair_generation
            key = (generation, index)
            if key in self._pair_active or index in self._pair_cache:
                continue
            task = _PairTask(
                self._wizard.analysis, generation, index,
                self._pair_list[index], self._pair_strain_window,
                self._pair_use_gpu(),
                self._pair_cancel_flag,
                include_strain=not self._pair_rates_only,
                include_rate=not self._pair_rates_only,
                cache_path=(self._pair_store.path_for(index)
                            if self._pair_store is not None else None))
            task.signals.done.connect(self._on_pair_ready)
            task.signals.progress.connect(self._on_pair_progress)
            self._pair_active.add(key)
            self._pair_tasks[key] = task
            self._pair_pool.start(task)

    @pyqtSlot(int, int, int, str)
    def _on_pair_progress(self, generation: int, _index: int,
                          percent: int, message: str) -> None:
        if generation != self._pair_generation or self._pair_store is None:
            return
        self._export_progress.show()
        self._export_progress.setValue(max(0, min(100, int(percent))))
        self._export_progress.setFormat(f"{message} · %p%")

    def _start_temporal_history_phase(self) -> None:
        if (self._pair_history_active or not self._pair_rates_only or
                self._pair_store is None or
                len(self._pair_rate_ready) != len(self._pair_list)):
            return
        self._pair_history_active = True
        self._export_progress.setValue(0)
        self._export_progress.setFormat("Strain rates 0/"
                                        f"{len(self._pair_list)} · %p%")
        self._pair_fixed_ranges.clear()
        task = _TemporalHistoryTask(
            self._wizard.analysis, self._pair_generation, self._pair_list,
            self._pair_strain_window, self._pair_use_gpu(),
            self._pair_cancel_flag, self._pair_store)
        task.signals.progress.connect(self._on_temporal_history_progress)
        task.signals.item_ready.connect(self._on_temporal_derived_item_ready)
        task.signals.done.connect(self._on_temporal_history_done)
        self._pair_history_task = task
        self._pair_pool.start(task)

    @pyqtSlot(int, int, str)
    def _on_temporal_history_progress(self, generation: int, percent: int,
                                      message: str) -> None:
        if generation != self._pair_generation:
            return
        self._export_progress.setValue(max(0, min(100, int(percent))))
        self._export_progress.setFormat(f"{message} · %p%")

    @pyqtSlot(int, int)
    def _on_temporal_derived_item_ready(self, generation: int,
                                        index: int) -> None:
        """Expose each newly calculated rate/strain frame immediately."""
        if generation != self._pair_generation:
            return
        self._pair_bulk_ready.add(int(index))
        self._pair_cache.pop(int(index), None)
        if self._pair_mode and int(index) == self._frame:
            result = self._pair_result_at(index)
            if result is not None:
                self._render_pair_result(index, result)
        self._sync_pair_ui()

    @pyqtSlot(int, bool, str)
    def _on_temporal_history_done(self, generation: int, ok: bool,
                                  error: str) -> None:
        if generation != self._pair_generation:
            return
        self._pair_history_active = False
        self._pair_history_task = None
        if not ok:
            self._export_progress.setFormat("Temporal strain rate failed")
            if self._pair_mode:
                self._frame_lbl.setText(
                    f"Temporal strain-rate failed: {error}")
            return
        self._pair_rates_only = False
        self._pair_bulk_ready = set(range(len(self._pair_list)))
        self._pair_cache.clear()
        self._export_progress.setValue(100)
        self._export_progress.setFormat("Temporal sequence ready")
        QTimer.singleShot(2500, self._export_progress.hide)
        strain_cache = getattr(
            self._wizard.analysis, "_temporal_strain_cache", None)
        if strain_cache is not None:
            strain_cache.clear()
        if self._pair_mode:
            self._show_pair_average()
        if self._pending_hdf5_path:
            path = self._pending_hdf5_path
            self._pending_hdf5_path = None
            self._start_hdf5_worker(path)
        self._sync_pair_ui()

    @pyqtSlot(int, int, object, str)
    def _on_pair_ready(self, generation: int, index: int,
                       result, error: str) -> None:
        key = (generation, index)
        self._pair_active.discard(key)
        self._pair_tasks.pop(key, None)
        self._cleanup_pair_store_dirs()
        if generation != self._pair_generation:
            self._pump_pair_workers()
            return

        if error:
            if self._pair_mode and index == self._frame:
                self._pair_avg = None
                self._frame_lbl.setText(
                    f"Pair {index + 1} failed: {error}")
                self._canvas.set_result_overlay_rgba(None)
                if self._play_btn.isChecked():
                    self._play_btn.setChecked(False)
                    self._toggle_play(False)
            if self._pair_store is not None:
                self._export_progress.setFormat("Temporal preprocessing failed")
            self._pump_pair_workers()
            return

        if self._pair_store is not None and self._pair_rates_only:
            self._pair_rate_ready.add(index)
            total = max(1, len(self._pair_list))
            ready = len(self._pair_rate_ready)
            self._export_progress.setValue(int(100 * ready / total))
            self._export_progress.setFormat(
                f"Velocity averages {ready}/{total} · %p%")
            self._sync_pair_ui()
        elif self._pair_store is not None:
            self._pair_bulk_ready.add(index)

        self._pair_cache[index] = result
        self._pair_cache.move_to_end(index)
        while len(self._pair_cache) > self._PAIR_CACHE_MAX:
            self._pair_cache.popitem(last=False)

        if self._pair_mode and index == self._frame:
            self._render_pair_result(index, result)
            self._prefetch_pair_neighbors(index)
            if self._play_btn.isChecked():
                self._play_timer.start(
                    int(1000.0 / max(1, self._fps_spin.value())))
        self._start_temporal_history_phase()
        self._pump_pair_workers()

    def _prefetch_pair_neighbors(self, index: int) -> None:
        n = len(self._pair_list)
        if n < 2:
            return
        self._request_pair((index + 1) % n, priority=False)
        if n > 2:
            self._request_pair((index - 1) % n, priority=False)

    def _pair_result_at(self, index: int):
        """Return a cached pair immediately or queue it without blocking Qt."""
        if not self._pair_mode or not self._pair_list:
            return None
        index = max(0, min(int(index), len(self._pair_list) - 1))
        hit = self._pair_cache.get(index)
        if hit is not None:
            self._pair_cache.move_to_end(index)
            return hit
        if self._pair_store is not None and self._pair_store.has(index):
            try:
                hit = self._pair_store[index]
                self._pair_cache[index] = hit
                self._pair_cache.move_to_end(index)
                while len(self._pair_cache) > self._PAIR_CACHE_MAX:
                    self._pair_cache.popitem(last=False)
                return hit
            except Exception as exc:
                self._frame_lbl.setText(f"Could not read temporal pair: {exc}")
        self._request_pair(index, priority=True)
        return None

    def _show_pair_average(self) -> None:
        """Render the current smoothed pair on its ending-frame image."""
        analysis = self._wizard.analysis
        res = self._pair_result_at(self._frame)
        if res is None:
            self._pair_avg = None
            a, b = self._pair_list[self._frame]
            img = (self._background_image(analysis.def_paths[b])
                   if b < len(analysis.def_paths) else analysis.reference_image)
            if img is not None:
                self._canvas.set_image(
                    img, keep_view=self._canvas._image_arr is not None,
                    display_scale=0.45)
            self._canvas.set_result_overlay_rgba(None)
            self._frame_lbl.setText(
                f"Calculating pair {self._frame + 1} / {len(self._pair_list)}"
                f"   ·   {a + 1}→{b + 1}…")
            return
        self._render_pair_result(self._frame, res)
        self._prefetch_pair_neighbors(self._frame)

    def _render_pair_result(self, index: int, res) -> None:
        """Render an already-computed pair; this always runs on Qt's thread."""
        analysis = self._wizard.analysis
        self._pair_avg = res

        a, b = self._pair_list[index]
        img = (self._background_image(analysis.def_paths[b])
               if b < len(analysis.def_paths) else analysis.reference_image)
        if img is not None:
            keep = self._canvas._image_arr is not None
            self._canvas.set_image(
                img, keep_view=keep, display_scale=0.45)

        arr, _ = self._display_array(res)
        if arr is not None and finite_values(arr).size:
            self._apply_overlay(arr)
        else:
            self._canvas.set_result_overlay_rgba(None)

        self._canvas.set_streaklines(None)
        self._canvas.set_markers([])

        self._update_stats(res)
        n = len(self._pair_list)
        self._frame_lbl.setText(
            f"Pair {index + 1} / {n}   ·   {a + 1}→{b + 1}")

    def _prev_frame(self) -> None:
        self._slider.setValue(max(0, self._slider.value() - 1))

    def _next_frame(self) -> None:
        self._slider.setValue(min(self._slider.maximum(), self._slider.value() + 1))

    def _advance(self) -> None:
        """Show the next frame, then schedule the one after it.

        Self-rescheduling rather than a repeating timer. A repeating timer keeps
        firing whether or not the previous frame finished drawing, so any frame
        that overruns the interval -- a large image, a cold cache, a busy
        machine -- leaves queued timeouts behind it. Those accumulate, the event
        loop never drains, and the window stops responding to the Stop button
        that would end it.
        //
        Measuring the render and deducting it from the next delay keeps the
        requested rate when frames are quick, and degrades to "as fast as this
        machine can draw" when they are not, instead of building a backlog.
        """
        if not self._play_btn.isChecked():
            return

        started = time.perf_counter()
        nxt = self._slider.value() + 1
        if nxt > self._slider.maximum():
            nxt = 0
        self._slider.setValue(nxt)
        # Slider changes are normally coalesced for interactive dragging. During
        # playback, render this tick synchronously so a costly pair calculation
        # cannot be skipped by the next timer tick repeatedly restarting the
        # scrub timer. The elapsed render time is deducted below.
        self._scrub_timer.stop()
        self._render_scrubbed_frame()

        # An uncached pair is now running on a worker. Do not queue timer ticks
        # while it is pending; _on_pair_ready resumes playback after rendering
        # the requested item. This prevents playback from racing ahead and
        # showing a result under the wrong slider position.
        if self._pair_mode and self._pair_avg is None:
            return

        target = 1000.0 / max(1, self._fps_spin.value())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # A floor of one tick keeps the event loop breathing even when the
        # render already exceeds the requested interval.
        self._play_timer.start(int(max(1.0, target - elapsed_ms)))

    def _toggle_play(self, checked: bool) -> None:
        if checked:
            self._play_btn.setText("⏹  Stop")
            if self._pair_mode and self._pair_result_at(self._frame) is None:
                self._show_pair_average()
            else:
                self._play_timer.start(
                    int(1000.0 / max(1, self._fps_spin.value())))
        else:
            self._play_btn.setText("▶  Play")
            self._play_timer.stop()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Export Directory")
        if not directory:
            return
        try:
            target = self._pair_avg if self._pair_avg is not None else self._frame
            self._wizard.analysis.export_csv(target, directory)
            QMessageBox.information(self, "Exported", f"CSV files saved to:\n{directory}")
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))

    def _export_marker_timeseries(self) -> None:
        """Write the per-frame history of every marker to one spreadsheet."""
        analysis = self._wizard.analysis
        pts = self._canvas.markers
        if not pts:
            QMessageBox.warning(
                self, "No markers",
                "Enable Place markers, then click the point you want to follow.")
            return
        if not analysis.results:
            QMessageBox.warning(self, "Nothing to export", "Run an analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Marker Time Series", "marker_timeseries.csv",
            "CSV files (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            labels = [f"M{i + 1}" for i in range(len(pts))]
            rows = analysis.export_marker_timeseries(pts, path, labels=labels)
            QMessageBox.information(
                self, "Exported",
                f"{rows} rows for {len(pts)} marker(s) written to:\n{path}\n\n"
                "One row per frame per marker. Filter on the marker column, "
                "then plot any field against time_s.")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _plot_marker_timeseries(self) -> None:
        """Open the temporal plot for the markers currently placed."""
        analysis = self._wizard.analysis
        pts = self._canvas.markers
        if not pts:
            QMessageBox.information(
                self, "No markers",
                "Enable Place markers, then click the points you want to plot.")
            return
        if not analysis.results:
            QMessageBox.information(self, "Nothing to plot", "Run an analysis first.")
            return
        try:
            from src.ui.pages.marker_plot_dialog import MarkerPlotDialog
        except Exception as exc:
            QMessageBox.warning(
                self, "Plotting unavailable",
                f"matplotlib is required to plot marker histories.\n\n{exc}")
            return
        labels = [f"M{i + 1}" for i in range(len(pts))]
        # Open on the field being viewed when that field is plottable, so the
        # plot answers the question the user already had on screen.
        from src.ui.pages.marker_plot_dialog import PLOTTABLE
        field = (self._field if any(self._field == n for n, _, _ in PLOTTABLE)
                 else "Eeff_rate")
        try:
            dlg = MarkerPlotDialog(analysis, pts, labels=labels,
                                   parent=self, field=field)
        except Exception as exc:
            QMessageBox.warning(self, "Could not build the plot", str(exc))
            return
        dlg.exec()

    def _export_video(self) -> None:
        analysis = self._wizard.analysis
        if not analysis.results:
            QMessageBox.warning(self, "Nothing to export", "Run an analysis first.")
            return

        from src.ui.pages.video_export_dialog import VideoExportDialog
        from src.ui.video_export import CODECS, export_video

        temporal_export = (
            self._pair_mode and
            self._pair_sequence_mode in ("sliding", "non_overlapping"))
        if temporal_export and (
                self._pair_store is None or
                len(self._pair_bulk_ready) != len(self._pair_list)):
            if self._pair_store is not None:
                for index in range(len(self._pair_list)):
                    if not self._pair_store.has(index):
                        self._request_pair(index, priority=False)
            QMessageBox.information(
                self, "Temporal preprocessing",
                "The temporal sequence is still being calculated. The progress "
                "bar reads Temporal sequence ready when averaged video "
                "export becomes available.")
            return
        export_count = (len(self._pair_list) if temporal_export
                        else len(analysis.results))

        dlg = VideoExportDialog(FIELDS, CMAPS, export_count,
                                self._field, self._cmap_combo.currentText(),
                                fps=float(self._fps_spin.value()), parent=self)
        # Carry the on-screen colour range into panel 1 so "export what I'm
        # looking at" is the default rather than something to reconstruct.
        rng = self.current_range_spec()
        if dlg._editors:
            e = dlg._editors[0]
            i = e.range_mode.findData(rng.mode)
            if i >= 0:
                e.range_mode.setCurrentIndex(i)
            if rng.vmin is not None:
                e.vmin.setValue(rng.vmin)
                e.vmax.setValue(rng.vmax)
            e.symmetric.setChecked(rng.symmetric)

        if dlg.exec() == 0:
            return
        spec = dlg.spec()

        # Coverage and clip-flagging have no per-panel control in the export
        # dialog, so they inherit the viewer's settings. Without this an export
        # would silently fall back to raw min/max and look nothing like the
        # frame the user was looking at when they hit Export.
        for panel in spec.panels:
            panel.range_spec.percentile = rng.percentile
            panel.mark_out_of_range = self._clip_chk.isChecked()

        ext = CODECS.get(spec.codec, (".mp4", None))[0]
        start = getattr(analysis, "last_hdf5_directory", os.path.expanduser("~"))
        default = os.path.join(start, "dic_view" + (".png" if spec.writes_sequence else ext))
        path, _ = QFileDialog.getSaveFileName(self, "Export video", default,
                                              f"*{ext}")
        if not path:
            return

        self._export_progress.show()
        self._export_progress.setValue(0)
        self._export_progress.setFormat("Rendering… %p%")
        markers = self._canvas.markers
        trail = self._trail_combo.currentData() or 0
        spec.trail = trail

        self._video_worker = _VideoWorker(
            analysis, spec, path, markers,
            temporal_results=(self._pair_store if temporal_export else None),
            temporal_pairs=(self._pair_list if temporal_export else None),
            parent=self)
        self._video_store_generation = (
            self._pair_generation if temporal_export else None)
        if self._video_store_generation is not None:
            self._pair_retained_generations.add(self._video_store_generation)
        self._video_worker.progress.connect(self._export_progress.setValue)
        self._video_worker.done.connect(self._on_video_done)
        self._video_worker.start()

    def _on_video_done(self, ok: bool, msg: str) -> None:
        generation = getattr(self, "_video_store_generation", None)
        if generation is not None:
            self._pair_retained_generations.discard(generation)
            self._video_store_generation = None
            self._cleanup_pair_store_dirs()
        self._export_progress.setValue(100 if ok else 0)
        self._export_progress.setFormat("Video export complete" if ok else "Video export failed")
        QTimer.singleShot(2500, self._export_progress.hide)
        if ok:
            QMessageBox.information(self, "Exported", f"Written to:\n{msg}")
        else:
            QMessageBox.warning(self, "Export Error", msg)

    def _export_hdf5(self) -> None:
        # Use shared memory directory
        start_dir = getattr(self._wizard.analysis, "last_hdf5_directory", os.path.expanduser("~"))
        default_path = os.path.join(start_dir, "dic_results.h5")

        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5", default_path, "HDF5 files (*.h5 *.hdf5)"
        )
        if not path:
            return

        # Save memory
        self._wizard.analysis.last_hdf5_directory = os.path.dirname(path)
        self._wizard.analysis.save_settings()

        if self._pair_mode and (
                self._pair_store is None or
                len(self._pair_bulk_ready) != len(self._pair_list)):
            self._pending_hdf5_path = path
            if self._pair_store is None:
                self._begin_pair_bulk()
            else:
                for index in range(len(self._pair_list)):
                    if not self._pair_store.has(index):
                        self._request_pair(index, priority=False)
            self._export_progress.show()
            self._export_progress.setFormat("Preparing temporal HDF5 data… %p%")
            return

        self._start_hdf5_worker(path)

    def _start_hdf5_worker(self, path: str) -> None:
        self._export_progress.show()
        self._export_progress.setValue(0)
        self._export_progress.setFormat("Exporting HDF5…  %p%")
        self._export_progress.setStyleSheet(
            f"QProgressBar {{ background: {_C_CARD}; border: 1px solid {_C_BORDER}; border-radius:3px; color: {_C_TEXT}; }}"
            f"QProgressBar::chunk {{ background: {_C_ACCENT}; border-radius: 3px; }}"
        )

        temporal_results = self._pair_store if self._pair_mode else None
        metadata = ({
            "mode": self._pair_sequence_mode,
            "strain_window": self._pair_strain_window,
        } if temporal_results is not None else {})
        self._export_worker = ExportWorker(
            self._wizard.analysis, path,
            temporal_results=temporal_results,
            temporal_pairs=(self._pair_list if temporal_results is not None else None),
            temporal_metadata=metadata, parent=self)
        self._hdf_store_generation = (
            self._pair_generation if temporal_results is not None else None)
        if self._hdf_store_generation is not None:
            self._pair_retained_generations.add(self._hdf_store_generation)
        self._export_worker.progress_export.connect(self._on_export_progress)
        self._export_worker.finished_export.connect(self._on_export_finished)
        self._export_worker.start()

    def _on_export_progress(self, val):
        self._export_progress.setValue(val)

    def _on_export_finished(self, success, result_str):
        generation = getattr(self, "_hdf_store_generation", None)
        if generation is not None:
            self._pair_retained_generations.discard(generation)
            self._hdf_store_generation = None
            self._cleanup_pair_store_dirs()
        if success:
            self._export_progress.setValue(100)
            self._export_progress.setFormat("Export complete")
            self._export_progress.setStyleSheet(
                f"QProgressBar {{ background: {_C_CARD}; border: 1px solid {_C_BORDER}; border-radius:3px; color: {_C_TEXT}; }}"
                f"QProgressBar::chunk {{ background: {_C_SUCCESS}; border-radius: 3px; }}"
            )
        else:
            self._export_progress.setFormat("Export failed")
            self._export_progress.setStyleSheet(
                f"QProgressBar {{ background: {_C_CARD}; border: 1px solid {_C_BORDER}; border-radius:3px; color: red; }}"
                f"QProgressBar::chunk {{ background: {_C_CARD}; }}"
            )
            QMessageBox.warning(self, "Export Error", result_str)
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_tab_style(self) -> None:
        active = (
            f"QToolButton {{ background:{_C_ACCENT}; color:#fff; border:none; "
            f"border-radius:3px; font-size:10px; font-weight:700; padding:3px 8px; }}"
        )
        inactive = (
            f"QToolButton {{ background:{_C_RAISED}; color:{_C_TEXT2}; "
            f"border:1px solid {_C_BORDER}; border-radius:3px; "
            f"font-size:10px; padding:3px 8px; }} "
            f"QToolButton:hover {{ background:{_C_BORDER}; color:{_C_TEXT}; }}"
        )
        for k, btn in self._field_btns.items():
            btn.setStyleSheet(active if btn.isChecked() else inactive)

    def _sep(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
        return f

    def _nav_btn(self, icon: str, slot) -> QToolButton:
        btn = QToolButton()
        btn.setText(icon)
        btn.setFixedSize(30, 30)
        btn.clicked.connect(slot)
        btn.setStyleSheet(
            f"QToolButton {{ background:{_C_CARD}; color:{_C_TEXT2}; "
            f"border:1px solid {_C_BORDER}; border-radius:3px; font-size:13px; }} "
            f"QToolButton:hover {{ background:{_C_BORDER}; color:{_C_TEXT}; }}"
        )
        return btn
