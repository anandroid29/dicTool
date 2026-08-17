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
import os
from typing import TYPE_CHECKING, Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QLinearGradient, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialog,
    QSlider, QComboBox, QCheckBox, QFrame, QSizePolicy,
    QFileDialog, QMessageBox, QSpinBox, QToolButton, QProgressDialog, QProgressBar,
    QListWidget, QListWidgetItem, QGroupBox, QGridLayout, QDoubleSpinBox,
    QRadioButton, QButtonGroup,
)

from src.core.units import LENGTH_UNIT_ORDER
from src.ui import render
from src.ui.render import RangeSpec

if TYPE_CHECKING:
    from src.ui.wizard import Wizard

from src.ui.image_canvas import ImageCanvas, marker_color

try:
    import cv2; _CV2 = True
except ImportError:
    _CV2 = False

_C_BG      = "#08111d"
_C_SURFACE = "#0e1c2e"
_C_CARD    = "#132035"
_C_RAISED  = "#1a2d47"
_C_BORDER  = "#1e3a5a"
_C_ACCENT  = "#3b82f6"
_C_TEXT    = "#e2e8f0"
_C_TEXT2   = "#94a3b8"
_C_TEXT3   = "#475569"
_C_SUCCESS = "#10b981"

FIELDS = {
    # 1. Displacements
    # Incremental first: "how far did it move between these two frames" is the
    # everyday question, and it is the one the cumulative fields cannot answer
    # once the sequence is long.
    "u_inc": ("Δu (frame-to-frame)", "px"),
    "v_inc": ("Δv (frame-to-frame)", "px"),
    "mag_inc": ("|Δ| (frame-to-frame)", "px"),
    "u": ("Displacement u (cumulative)", "px"),
    "v": ("Displacement v (cumulative)", "px"),

    # 2. Velocities
    "Vx": ("Velocity Vx", "px/s"),
    "Vy": ("Velocity Vy", "px/s"),
    "Veff": ("Effective Velocity", "px/s"),

    # 3. Strain Rates
    "Exx_rate": ("Strain Rate  Ėxx", "s⁻¹"),
    "Exy_rate": ("Strain Rate  Ėxy", "s⁻¹"),
    "Eyy_rate": ("Strain Rate  Ėyy", "s⁻¹"),
    "Eeff_rate": ("Effective Strain Rate", "s⁻¹"),

    # 4. Accumulated Strains
    "Exx": ("Strain  Exx", "ε"),
    "Exy": ("Strain  Exy", "ε"),
    "Eyy": ("Strain  Eyy", "ε"),
    "Eeff": ("Effective Strain", "ε"),
}

# Field families, in the order they appear in the category dropdown. Only the
# members of the selected family get a button in the toolbar.
FIELD_GROUPS = {
    "Displacement": ["u_inc", "v_inc", "mag_inc", "u", "v"],
    "Velocity":     ["Vx", "Vy", "Veff"],
    "Strain rate":  ["Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate"],
    "Strain":       ["Exx", "Exy", "Eyy", "Eeff"],
}

# Short button captions, now that the family is named by the dropdown.
_FIELD_SHORT = {
    "u": "u", "v": "v",
    "u_inc": "Δu", "v_inc": "Δv", "mag_inc": "|Δ|",
    "Vx": "Vx", "Vy": "Vy", "Veff": "eff",
    "Exx_rate": "Ėxx", "Exy_rate": "Ėxy", "Eyy_rate": "Ėyy", "Eeff_rate": "Ėeff",
    "Exx": "Exx", "Exy": "Exy", "Eyy": "Eyy", "Eeff": "Eeff",
}


def _field_short(key: str) -> str:
    return _FIELD_SHORT.get(key, key)


def _group_of(field: str) -> str:
    for g, keys in FIELD_GROUPS.items():
        if field in keys:
            return g
    return next(iter(FIELD_GROUPS))


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


class ExportWorker(QThread):
    finished_export = pyqtSignal(bool, str)
    progress_export = pyqtSignal(int)

    def __init__(self, analysis, path, parent=None):
        super().__init__(parent)
        self.analysis = analysis
        self.path = path

    def run(self):
        try:
            def prog_cb(frac):
                self.progress_export.emit(int(frac * 100))
            self.analysis.export_hdf5(self.path, progress_cb=prog_cb)
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

    def __init__(self, analysis, spec, path, markers, parent=None):
        super().__init__(parent)
        self._analysis = analysis
        self._spec = spec
        self._path = path
        self._markers = list(markers or [])
        self.cancel_flag = [False]

    def run(self):
        try:
            from src.ui.video_export import export_video
            out = export_video(
                self._analysis, self._spec, self._path, markers=self._markers,
                progress_cb=lambda f, _m: self.progress.emit(int(f * 100)),
                cancel_flag=self.cancel_flag)
            self.done.emit(True, out)
        except Exception as e:
            self.done.emit(False, str(e))


class _ColorBar(QWidget):
    """A thin horizontal gradient bar with vmin/vmax labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._colors = [(8,17,29)] * 2
        self._vmin = self._vmax = 0.0
        self._unit = ""

    def update_bar(self, vmin, vmax, unit, colors):
        self._vmin, self._vmax, self._unit = vmin, vmax, unit
        self._colors = colors
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        lm, rm, th = 6, 6, 12

        w = self.width() - lm - rm
        grad = QLinearGradient(lm, 0, lm + w, 0)
        n = len(self._colors)
        for i, (r, g, b) in enumerate(self._colors):
            grad.setColorAt(i / max(n - 1, 1), QColor(r, g, b))

        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(lm, 4, w, th, 3, 3)

        p.setPen(QColor(_C_TEXT2))
        font = QFont("Fira Code, Consolas, monospace", 9)
        p.setFont(font)
        vmin_s = f"{self._vmin:.4g}"
        vmax_s = f"{self._vmax:.4g} {self._unit}"
        p.drawText(lm, 4 + th + 13, vmin_s)
        fm = p.fontMetrics()
        p.drawText(lm + w - fm.horizontalAdvance(vmax_s), 4 + th + 13, vmax_s)
        p.end()


class ResultsPage(QWidget):
    """Step 6 — results viewer with correct frame-by-frame image updates."""

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._frame  = 0
        self._field  = "Eeff_rate"
        # Frame-pair average. When set, the viewer shows this instead of a
        # single frame and the scrubber is inert -- the average has no position
        # in the sequence to scrub to.
        self._pair_avg = None
        self._pair_list: list = []
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(200)
        self._play_timer.timeout.connect(self._advance)
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
        self._scale_auto_rb.setChecked(True)
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
        top = QWidget()
        top.setFixedHeight(52)
        top.setStyleSheet(f"background:{_C_SURFACE}; border-bottom:1px solid {_C_BORDER};")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(16, 0, 16, 0)
        top_lay.setSpacing(8)

        new_btn = QPushButton("← New Session")
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
            "Trace trajectories from markers you place on the image")
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
            f" padding:3px 10px; border-radius:4px; font-size:11px;}}"
            f"QPushButton:checked{{background:{_C_ACCENT}; color:#ffffff; border:1px solid {_C_ACCENT};"
            f" font-weight:700;}}")
        self._place_btn.toggled.connect(self._on_place_toggled)
        top_lay.addWidget(self._place_btn)

        self._marker_count_lbl = QLabel("0 markers")
        self._marker_count_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        top_lay.addWidget(self._marker_count_lbl)

        self._clear_markers_btn = QPushButton("Clear")
        self._clear_markers_btn.setFixedHeight(28)
        self._clear_markers_btn.setToolTip("Remove all markers")
        self._clear_markers_btn.setStyleSheet(
            f"background:{_C_CARD}; color:{_C_TEXT2}; border:1px solid {_C_BORDER};"
            f" padding:3px 10px; border-radius:4px; font-size:11px;")
        self._clear_markers_btn.clicked.connect(self._clear_markers)
        top_lay.addWidget(self._clear_markers_btn)

        self._trail_lbl = QLabel("Trail:")
        self._trail_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        top_lay.addWidget(self._trail_lbl)

        self._trail_combo = QComboBox()
        self._trail_combo.addItem("Full", 0)
        for n in (10, 25, 50, 100):
            self._trail_combo.addItem(f"{n} frames", n)
        self._trail_combo.setToolTip("How much trajectory history to draw")
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
        top_lay.addWidget(cmap_lbl)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(CMAPS)
        self._cmap_combo.setCurrentText(DEFAULT_CMAP)
        self._cmap_combo.setFixedWidth(100)
        self._cmap_combo.currentTextChanged.connect(self._refresh_overlay)
        top_lay.addWidget(self._cmap_combo)

        self._sym_chk = QCheckBox("Sym")
        self._sym_chk.setToolTip("Centre colormap around zero")
        self._sym_chk.stateChanged.connect(self._refresh_overlay)
        top_lay.addWidget(self._sym_chk)

        scale_lbl = QLabel("Scale:")
        scale_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        top_lay.addWidget(scale_lbl)
        for rb in (self._scale_auto_rb, self._scale_global_rb, self._scale_manual_rb):
            top_lay.addWidget(rb)

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
            top_lay.addWidget(sb)

        self._range_fit_btn = QPushButton("Fit")
        self._range_fit_btn.setFixedWidth(40)
        self._range_fit_btn.setToolTip(
            "Fill the limits from this frame's data, then keep them pinned.")
        self._range_fit_btn.setEnabled(False)
        self._range_fit_btn.clicked.connect(self._fit_range_to_frame)
        top_lay.addWidget(self._range_fit_btn)

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
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        body_lay.addWidget(self._canvas, 1)

        # Right sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(
            f"background:{_C_SURFACE}; border-left:1px solid {_C_BORDER};"
        )
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(16, 20, 16, 20)
        sb_lay.setSpacing(14)

        # Stats
        stats_hdr = QLabel("STATISTICS")
        stats_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;"
        )
        sb_lay.addWidget(stats_hdr)

        self._stat_labels: dict[str, QLabel] = {}
        for stat in ("Mean", "Std Dev", "Min", "Max", "Valid px"):
            row = QHBoxLayout()
            k_lbl = QLabel(stat + ":")
            k_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
            k_lbl.setFixedWidth(64)
            row.addWidget(k_lbl)
            v_lbl = QLabel("—")
            v_lbl.setStyleSheet(
                f"color:{_C_TEXT}; font-size:11px; "
                f"font-family:'Fira Code','Cascadia Code',monospace;"
            )
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(v_lbl, 1)
            sb_lay.addLayout(row)
            self._stat_labels[stat] = v_lbl

        sb_lay.addWidget(self._sep())

        # Colorbar
        cb_hdr = QLabel("COLORBAR")
        cb_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;"
        )
        sb_lay.addWidget(cb_hdr)
        self._colorbar = _ColorBar()
        sb_lay.addWidget(self._colorbar)

        sb_lay.addWidget(self._sep())

        # Export
        exp_hdr = QLabel("EXPORT")
        exp_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;"
        )
        sb_lay.addWidget(exp_hdr)

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
            f" border:1px dashed {_C_BORDER}; border-radius:4px; padding:7px;")
        mp_lay.addWidget(self._marker_hint)

        self._marker_list = QListWidget()
        self._marker_list.setFixedHeight(132)
        self._marker_list.setStyleSheet(
            f"QListWidget{{background:{_C_CARD}; border:1px solid {_C_BORDER};"
            f" border-radius:4px; font-size:10px; padding:2px;}}"
            f"QListWidget::item{{padding:3px 2px;}}"
            f"QListWidget::item:selected{{background:{_C_RAISED}; border-left:2px solid {_C_ACCENT};}}")
        self._marker_list.currentRowChanged.connect(self._canvas.select_marker)
        mp_lay.addWidget(self._marker_list)

        self._del_marker_btn = QPushButton("Remove selected")
        self._del_marker_btn.setFixedHeight(26)
        self._del_marker_btn.setStyleSheet(
            f"background:{_C_CARD}; color:{_C_TEXT2}; border:1px solid {_C_BORDER};"
            f" border-radius:4px; font-size:10px;")
        self._del_marker_btn.clicked.connect(
            lambda: self._canvas.remove_marker(self._canvas.selected_marker))
        mp_lay.addWidget(self._del_marker_btn)

        self._marker_panel.setVisible(False)
        sb_lay.addWidget(self._marker_panel)

        # ── Frame-pair average ──────────────────────────────────────
        pair_hdr = QLabel("FRAME PAIRS")
        pair_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;")
        sb_lay.addWidget(pair_hdr)

        self._pair_btn = QPushButton("Average frame pairs…")
        self._pair_btn.setFixedHeight(30)
        self._pair_btn.setToolTip(
            "Pick any number of frame pairs and average the displacement,\n"
            "velocity and strain rate measured across them.\n\n"
            "Cumulative strain is excluded — it carries the history before\n"
            "each pair begins, so it cannot be averaged across pairs.")
        self._pair_btn.clicked.connect(self._open_pair_dialog)
        sb_lay.addWidget(self._pair_btn)

        self._pair_banner = QLabel("")
        self._pair_banner.setWordWrap(True)
        self._pair_banner.setStyleSheet(
            f"color:{_C_SUCCESS}; font-size:10px; background:{_C_CARD};"
            f" border:1px solid {_C_SUCCESS}; border-radius:4px; padding:6px;")
        self._pair_banner.setVisible(False)
        sb_lay.addWidget(self._pair_banner)

        self._pair_exit_btn = QPushButton("← Back to single frames")
        self._pair_exit_btn.setFixedHeight(26)
        self._pair_exit_btn.setStyleSheet(
            f"background:{_C_CARD}; color:{_C_TEXT2}; border:1px solid {_C_BORDER};"
            f" border-radius:4px; font-size:10px;")
        self._pair_exit_btn.clicked.connect(self._clear_pair_average)
        self._pair_exit_btn.setVisible(False)
        sb_lay.addWidget(self._pair_exit_btn)

        sb_lay.addWidget(self._sep())

        for label, slot in [("CSV (this frame)", self._export_csv),
                             ("HDF5 (all frames)", self._export_hdf5),
                             ("Video / image sequence…", self._export_video)]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.clicked.connect(slot)
            sb_lay.addWidget(btn)

        self._export_progress = QProgressBar()
        self._export_progress.setFixedHeight(24)
        self._export_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._export_progress.setStyleSheet(
            f"QProgressBar {{ background: {_C_CARD}; border: 1px solid {_C_BORDER}; border-radius: 4px; color: {_C_TEXT}; }}"
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
        self._reset_view_btn = QPushButton("Reset Zoom")
        self._reset_view_btn.setFixedHeight(30)
        self._reset_view_btn.clicked.connect(self._canvas.fit_image)
        bot_lay.addWidget(self._reset_view_btn)
        # ──────────────────────────────────────────────────────────────

        prev_btn = self._nav_btn("◀", self._prev_frame)
        bot_lay.addWidget(prev_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        bot_lay.addWidget(self._slider, 1)

        next_btn = self._nav_btn("▶", self._next_frame)
        bot_lay.addWidget(next_btn)

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
        self._fps_spin.valueChanged.connect(
            lambda v: self._play_timer.setInterval(1000 // v)
        )
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
            "Physical size of one pixel. Set this and displacements and\n"
            "velocities are reported in real units instead of pixels.\n"
            "Strain and strain rate are ratios and never change.")
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
                                    "This frame has no valid correlation data yet.")
            return
        (rx, ry), resid = mapped
        if resid > 25.0:
            QMessageBox.information(
                self, "Outside analysed region",
                "That point isn't inside the analysed region for this frame.\n"
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
        self._place_btn.setChecked(False)
        self._streak_chk.setChecked(False)
        self._trail_combo.setCurrentIndex(0)
        self._canvas.set_marker_mode(False)
        self._canvas.clear_markers()
        self._marker_list.clear()

    def _rebuild_marker_list(self) -> None:
        self._marker_list.blockSignals(True)
        self._marker_list.clear()
        analysis = self._wizard.analysis
        pts = self._canvas.markers()
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
        pts = self._canvas.markers()
        self._canvas.set_marker_draw_positions(analysis.marker_positions(pts, idx))
        if not pts:
            self._canvas.set_streaklines(None)
            return
        trail = self._trail_combo.currentData() or 0
        trajs = analysis.get_trajectories_from_seeds(pts, idx, trail)
        self._canvas.set_streaklines(
            [t["points"] for t in trajs],
            colors=[marker_color(i) for i in range(len(trajs))],
            lost_flags=[t["lost_at"] is not None for t in trajs],
        )

    def on_before_show(self) -> None:
        """Cheap synchronous blanking so no stale frame is ever painted."""
        try:
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
        # Markers are indexed against the previous run's displacement fields, so
        # they are meaningless for a new sequence.
        self._canvas.clear_markers()
        self._marker_list.clear()
        # Same reasoning for a stored pair average: its frame indices and its
        # arrays belong to the run that produced them. Keeping it across a
        # re-analysis would silently show the previous run's numbers.
        self._pair_avg = None
        self._pair_list = []
        self._sync_pair_ui()
        self._sync_calibration_controls()
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._frame = 0
        self._show_frame(0)

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
            path = analysis.def_paths[idx]

            # Inline robust image loading
            img = None
            try:
                import cv2
                img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img_cv is not None:
                    img = img_cv.astype(np.float64) / 255.0
            except Exception:
                pass

            if img is None:
                try:
                    from PIL import Image
                    img = np.array(Image.open(path).convert("L"), dtype=np.float64) / 255.0
                except Exception as e:
                    print(f"Failed to load image {path}: {e}")

            if img is not None:
                # FORCE DEEP CONTIGUOUS COPY to prevent 0xC0000409 crash
                safe_img = np.ascontiguousarray(img * 0.45, dtype=np.float64)
                keep = self._canvas._image_arr is not None
                self._canvas.set_image(safe_img, keep_view=keep)
            else:
                self._canvas.clear_result_overlay()

        # 2. ── Render field overlay ──────────────────────────────────
        result = analysis.results[idx]
        arr, _ = self._display_array(result)
        if arr is not None and np.any(np.isfinite(arr)):
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
        idx = max(0, min(self._frame, len(results) - 1))
        arr, _ = self._display_array(results[idx])
        if arr is None:
            return
        finite = np.isfinite(arr)
        if not finite.any():
            return
        vals = arr[finite]
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
                         symmetric=self._sym_chk.isChecked())

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
            lo, hi = analysis.get_global_range(self._field)
            global_rng = (lo * factor, hi * factor)
        elif spec.mode == "global":
            # The sequence-wide range is measured on per-frame fields, whose
            # magnitudes differ from a pair's (cumulative u spans the whole run,
            # a pair's Δu spans one interval). Applying it here would flatten the
            # image to one colour, so the averaged field scales to itself.
            spec = RangeSpec(mode="auto", vmin=None, vmax=None,
                             symmetric=spec.symmetric)

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
                spacing=analysis.params.subset_spacing)
            self._canvas.set_result_overlay_rgba(rgba)
            self._update_colorbar(render.get_cmap(cmap_name, 256), vmin, vmax)
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

    def _unit_factor(self) -> tuple:
        cal = self._wizard.analysis.calibration
        base = FIELDS.get(self._field, ("", ""))[1]
        return cal.factor_and_unit(self._field, base)

    def _display_array(self, result):
        """The selected field in display units, plus its unit label."""
        arr = getattr(result, self._field, None)
        factor, unit = self._unit_factor()
        if arr is None or factor == 1.0:
            return arr, unit
        return arr * factor, unit

    def _update_colorbar(self, cmap, vmin, vmax):
        n_bar = 64
        bar_colors = []
        for i in range(n_bar):
            r, g, b, _ = cmap(i / (n_bar - 1))
            bar_colors.append((int(r * 255), int(g * 255), int(b * 255)))
        _, unit = self._unit_factor()
        self._colorbar.update_bar(vmin, vmax, unit, bar_colors)

    def _update_stats(self, result) -> None:
        arr, unit = self._display_array(result)
        if arr is None:
            for v in self._stat_labels.values():
                v.setText("—")
            return

        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            for v in self._stat_labels.values():
                v.setText("—")
            return

        suffix = f" {unit}" if unit else ""
        self._stat_labels["Mean"].setText(f"{valid.mean():.4g}{suffix}")
        self._stat_labels["Std Dev"].setText(f"{valid.std():.4g}{suffix}")
        self._stat_labels["Min"].setText(f"{valid.min():.4g}{suffix}")
        self._stat_labels["Max"].setText(f"{valid.max():.4g}{suffix}")
        self._stat_labels["Valid px"].setText(f"{valid.size:,}")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_slider(self, val: int) -> None:
        self._frame = val
        self._show_frame(val)

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
        if self._pair_avg is not None:
            self._show_pair_average()
        else:
            self._show_frame(self._frame)

    # ------------------------------------------------------------------
    # Frame-pair average
    # ------------------------------------------------------------------

    def _open_pair_dialog(self) -> None:
        analysis = self._wizard.analysis
        n = len(analysis.results)
        if n < 2:
            QMessageBox.information(
                self, "PyDIC",
                "Frame-pair averaging needs at least two analysed frames.")
            return

        from src.ui.pages.frame_pair_dialog import FramePairDialog
        dlg = FramePairDialog(n, analysis.fps, self._pair_list, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        pairs = dlg.pairs()
        try:
            avg = analysis.average_pairs(pairs)
        except Exception as exc:
            QMessageBox.warning(self, "Frame-Pair Average", str(exc))
            return

        self._pair_list = pairs
        self._pair_avg = avg

        # Cumulative strain is not defined for an average of pairs, so move off
        # it rather than showing an all-NaN field with no explanation.
        if self._field in ("Exx", "Exy", "Eyy", "Eeff"):
            self._select_field("mag_inc")

        self._sync_pair_ui()
        self._refresh_overlay()

    def _clear_pair_average(self) -> None:
        self._pair_avg = None
        self._sync_pair_ui()
        self._refresh_overlay()

    def _sync_pair_ui(self) -> None:
        """Show the averaging state and lock out controls it makes meaningless."""
        active = self._pair_avg is not None
        self._pair_banner.setVisible(active)
        self._pair_exit_btn.setVisible(active)
        self._pair_btn.setText(
            "Edit frame pairs…" if active else "Average frame pairs…")

        if active:
            n = len(self._pair_list)
            label = ", ".join(f"{a + 1}→{b + 1}" for a, b in self._pair_list[:4])
            if n > 4:
                label += f", +{n - 4} more"
            self._pair_banner.setText(
                f"<b>Averaging {n} frame pair{'s' if n != 1 else ''}</b><br>{label}"
                f"<br>Cumulative strain unavailable in this mode.")

        # Scrubbing and playback have no meaning for a single averaged field.
        if active and self._play_btn.isChecked():
            self._play_btn.setChecked(False)
            self._toggle_play(False)
        for w in (self._slider, self._play_btn):
            w.setEnabled(not active)

        # Cumulative strain buttons cannot be honoured while averaging.
        for k, btn in self._field_btns.items():
            if k in ("Exx", "Exy", "Eyy", "Eeff"):
                btn.setEnabled(not active)
                btn.setToolTip(
                    "Not available while averaging frame pairs — cumulative "
                    "strain includes the history before each pair begins."
                    if active else "")

    def _show_pair_average(self) -> None:
        """Render the averaged field over the reference image.

        The reference is the right backdrop here: an average spans several
        intervals, so no single deformed frame is the one it belongs to.
        """
        analysis = self._wizard.analysis
        res = self._pair_avg
        if res is None:
            return

        ref = analysis.reference_image
        if ref is not None:
            safe_img = np.ascontiguousarray(ref * 0.45, dtype=np.float64)
            keep = self._canvas._image_arr is not None
            self._canvas.set_image(safe_img, keep_view=keep)

        arr, _ = self._display_array(res)
        if arr is not None and np.any(np.isfinite(arr)):
            self._apply_overlay(arr)
        else:
            self._canvas.set_result_overlay_rgba(None)

        self._canvas.set_streaklines(None)
        self._canvas.set_markers([])

        self._update_stats(res)
        n = len(self._pair_list)
        self._frame_lbl.setText(f"Average of {n} pair{'s' if n != 1 else ''}")

    def _prev_frame(self) -> None:
        self._slider.setValue(max(0, self._slider.value() - 1))

    def _next_frame(self) -> None:
        self._slider.setValue(min(self._slider.maximum(), self._slider.value() + 1))

    def _advance(self) -> None:
        nxt = self._slider.value() + 1
        if nxt > self._slider.maximum():
            nxt = 0
        self._slider.setValue(nxt)

    def _toggle_play(self, checked: bool) -> None:
        if checked:
            self._play_btn.setText("⏹  Stop")
            self._play_timer.start()
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
            QMessageBox.warning(self, "Export Error", str(e))

    def _export_video(self) -> None:
        analysis = self._wizard.analysis
        if not analysis.results:
            QMessageBox.warning(self, "Nothing to export", "Run an analysis first.")
            return

        from src.ui.pages.video_export_dialog import VideoExportDialog
        from src.ui.video_export import CODECS, export_video

        dlg = VideoExportDialog(FIELDS, CMAPS, len(analysis.results),
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
        markers = self._canvas.markers()
        trail = self._trail_combo.currentData() or 0
        spec.trail = trail

        self._video_worker = _VideoWorker(analysis, spec, path, markers, self)
        self._video_worker.progress.connect(self._export_progress.setValue)
        self._video_worker.done.connect(self._on_video_done)
        self._video_worker.start()

    def _on_video_done(self, ok: bool, msg: str) -> None:
        self._export_progress.setValue(100 if ok else 0)
        self._export_progress.setFormat("Video export complete" if ok else "Video export failed ✗")
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

        self._export_progress.show()
        self._export_progress.setValue(0)
        self._export_progress.setFormat("Exporting HDF5... %p%")
        self._export_progress.setStyleSheet(
            f"QProgressBar {{ background: {_C_CARD}; border: 1px solid {_C_BORDER}; border-radius: 4px; color: {_C_TEXT}; }}"
            f"QProgressBar::chunk {{ background: {_C_ACCENT}; border-radius: 3px; }}"
        )

        self._export_worker = ExportWorker(self._wizard.analysis, path, self)
        self._export_worker.progress_export.connect(self._on_export_progress)
        self._export_worker.finished_export.connect(self._on_export_finished)
        self._export_worker.start()

    def _on_export_progress(self, val):
        self._export_progress.setValue(val)

    def _on_export_finished(self, success, result_str):
        if success:
            self._export_progress.setValue(100)
            self._export_progress.setFormat("Export Complete")
            self._export_progress.setStyleSheet(
                f"QProgressBar {{ background: {_C_CARD}; border: 1px solid {_C_BORDER}; border-radius: 4px; color: {_C_TEXT}; }}"
                f"QProgressBar::chunk {{ background: {_C_SUCCESS}; border-radius: 3px; }}"
            )
        else:
            self._export_progress.setFormat("Export Failed ✗")
            self._export_progress.setStyleSheet(
                f"QProgressBar {{ background: {_C_CARD}; border: 1px solid {_C_BORDER}; border-radius: 4px; color: red; }}"
                f"QProgressBar::chunk {{ background: {_C_CARD}; }}"
            )
            QMessageBox.warning(self, "Export Error", result_str)
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_tab_style(self) -> None:
        active = (
            f"QToolButton {{ background:{_C_ACCENT}; color:#fff; border:none; "
            f"border-radius:5px; font-size:10px; font-weight:700; padding:3px 8px; }}"
        )
        inactive = (
            f"QToolButton {{ background:{_C_RAISED}; color:{_C_TEXT2}; "
            f"border:1px solid {_C_BORDER}; border-radius:5px; "
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
            f"border:1px solid {_C_BORDER}; border-radius:5px; font-size:13px; }} "
            f"QToolButton:hover {{ background:{_C_BORDER}; color:{_C_TEXT}; }}"
        )
        return btn


# ---------------------------------------------------------------------------
# Image loading helper (frame-level, fast path)
# ---------------------------------------------------------------------------

def _load_gray(path: str) -> Optional[np.ndarray]:
    """Return float64 greyscale [0,1] or None on error."""
    try:
        if _CV2:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            mx = float(np.iinfo(img.dtype).max) if img.dtype.kind == "u" else 1.0
            return img.astype(np.float64) / mx
        else:
            from PIL import Image as PILImage
            return np.asarray(PILImage.open(path).convert("L"), np.float64) / 255.0
    except Exception:
        return None
