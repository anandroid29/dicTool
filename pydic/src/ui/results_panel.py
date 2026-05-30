"""
results_panel.py
================
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from src.core.analysis import DICAnalysis, PairResult

# ---------------------------------------------------------------------------
# Colour maps available in the selector
# ---------------------------------------------------------------------------

CMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "jet", "turbo", "hot", "cool", "rainbow",
    "RdBu_r", "seismic", "bwr", "coolwarm",
    "Spectral_r", "PiYG", "PRGn",
    "gray", "bone",
]

FIELD_LABELS = {
    "u":    ("Displacement u", "px"),
    "v":    ("Displacement v", "px"),
    "Exx":  ("Strain E_xx",    "ε"),
    "Exy":  ("Strain E_xy",    "ε"),
    "Eyy":  ("Strain E_yy",    "ε"),
    "Eeff": ("Effective Strain", "ε"),
}

# ---------------------------------------------------------------------------
# Tiny colourbar widget (rendered with QPainter, no matplotlib dependency)
# ---------------------------------------------------------------------------

class ColourBar(QWidget):
    """Horizontal gradient colourbar with min/max/unit labels."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(28)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._vmin: float = 0.0
        self._vmax: float = 1.0
        self._unit: str = ""
        self._colors: List[tuple] = [(0, 0, 128), (0, 255, 0), (255, 0, 0)]  # default

    def set_data(self, vmin: float, vmax: float, unit: str,
                 colors: List[tuple]) -> None:
        self._vmin = vmin
        self._vmax = vmax
        self._unit = unit
        self._colors = colors
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bar_h = 14
        bar_y = 4
        lm, rm = 6, 6

        w = self.width() - lm - rm
        grad = QLinearGradient(lm, 0, lm + w, 0)
        n = len(self._colors)
        for i, (r, g, b) in enumerate(self._colors):
            grad.setColorAt(i / max(n - 1, 1), QColor(r, g, b))

        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(lm, bar_y, w, bar_h, 3, 3)

        # Border
        p.setPen(QPen(QColor("#30363d"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(lm, bar_y, w, bar_h, 3, 3)

        # Labels
        p.setPen(QColor("#8b949e"))
        font = QFont("Fira Code, Consolas, monospace", 8)
        p.setFont(font)
        vmin_s = f"{self._vmin:.4g}"
        vmax_s = f"{self._vmax:.4g} {self._unit}"
        p.drawText(lm, bar_y + bar_h + 12, vmin_s)
        fm = p.fontMetrics()
        p.drawText(lm + w - fm.horizontalAdvance(vmax_s), bar_y + bar_h + 12, vmax_s)
        p.end()


# ---------------------------------------------------------------------------
# Statistics table
# ---------------------------------------------------------------------------

_STAT_COLS = ["Field", "Mean", "Std Dev", "Min", "Max", "Valid px"]

class StatTable(QTableWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, len(_STAT_COLS), parent)
        self.setHorizontalHeaderLabels(_STAT_COLS)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setMaximumHeight(200)
        self.setMinimumHeight(110)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QTableWidget {
                background: #0d1117;
                alternate-background-color: #161b22;
                color: #e6edf3;
                gridline-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                font-size: 11px;
                font-family: 'Fira Code', 'Consolas', monospace;
            }
            QHeaderView::section {
                background: #161b22;
                color: #8b949e;
                border: none;
                border-bottom: 1px solid #30363d;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            QTableWidget::item:selected {
                background: #1f3d6e;
            }
        """)

    def populate(self, result: "PairResult") -> None:
        self.setRowCount(0)
        if result is None:
            return
        fields = {
            "u":    getattr(result, "u",    None),
            "v":    getattr(result, "v",    None),
            "Exx":  getattr(result, "Exx",  None),
            "Exy":  getattr(result, "Exy",  None),
            "Eyy":  getattr(result, "Eyy",  None),
            "Eeff": getattr(result, "Eeff", None),
        }
        for field_key, arr in fields.items():
            if arr is None:
                continue
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                continue
            row = self.rowCount()
            self.insertRow(row)
            label, unit = FIELD_LABELS.get(field_key, (field_key, ""))
            data = [
                label,
                f"{valid.mean():.5g}",
                f"{valid.std():.5g}",
                f"{valid.min():.5g}",
                f"{valid.max():.5g}",
                f"{valid.size:,}",
            ]
            for col, txt in enumerate(data):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    if col > 0 else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                )
                self.setItem(row, col, item)
        self.resizeColumnsToContents()


# ---------------------------------------------------------------------------
# Main results panel
# ---------------------------------------------------------------------------

class ResultsPanel(QWidget):
    """
    Right-hand results panel.

    Signals
    -------
    field_changed(field_key: str) — emitted when user picks a different field
    frame_changed(index: int)     — emitted when scrubber moves to a new frame
    colormap_changed(cmap: str, vmin: float | None, vmax: float | None)
    """

    field_changed = pyqtSignal(str)
    frame_changed = pyqtSignal(int)
    colormap_changed = pyqtSignal(str, object, object)  # cmap, vmin, vmax

    def __init__(self, analysis: "DICAnalysis", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._analysis = analysis
        self._current_frame: int = 0
        self._current_field: str = "u"
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(200)
        self._play_timer.timeout.connect(self._advance_frame)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Title ──────────────────────────────────────────────────────
        title = QLabel("Results")
        title.setStyleSheet("color:#e6edf3; font-size:14px; font-weight:600;")
        root.addWidget(title)

        # ── Field tab bar ──────────────────────────────────────────────
        root.addWidget(self._build_field_tabs())

        # ── Colourmap controls ─────────────────────────────────────────
        root.addWidget(self._build_cmap_controls())

        # ── Colourbar ──────────────────────────────────────────────────
        self._colourbar = ColourBar()
        root.addWidget(self._colourbar)

        sep1 = self._separator()
        root.addWidget(sep1)

        # ── Statistics ─────────────────────────────────────────────────
        stat_label = QLabel("Statistics")
        stat_label.setStyleSheet("color:#8b949e; font-size:11px; font-weight:600;"
                                 "text-transform:uppercase; letter-spacing:0.5px;")
        root.addWidget(stat_label)
        self._stat_table = StatTable()
        root.addWidget(self._stat_table)

        sep2 = self._separator()
        root.addWidget(sep2)

        # ── Temporal scrubber ──────────────────────────────────────────
        root.addWidget(self._build_scrubber())

        sep3 = self._separator()
        root.addWidget(sep3)

        # ── Export buttons ─────────────────────────────────────────────
        root.addWidget(self._build_export_buttons())

        root.addStretch()

    def _build_field_tabs(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._field_btns: dict[str, QToolButton] = {}
        fields = list(FIELD_LABELS.keys())
        for key in fields:
            label, _ = FIELD_LABELS[key]
            short = key  # "u", "v", "Exx" …
            btn = QToolButton()
            btn.setText(short)
            btn.setToolTip(label)
            btn.setCheckable(True)
            btn.setChecked(key == self._current_field)
            btn.setFixedHeight(28)
            btn.setMinimumWidth(44)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, k=key: self._select_field(k))
            self._field_btns[key] = btn
            layout.addWidget(btn)

        layout.addStretch()
        self._apply_tab_style()
        return container

    def _apply_tab_style(self) -> None:
        active_css = (
            "QToolButton { background:#2f81f7; color:#ffffff; border:none; "
            "border-radius:5px; font-size:11px; font-weight:700; padding:0 10px; }"
        )
        inactive_css = (
            "QToolButton { background:#21262d; color:#8b949e; border:1px solid #30363d; "
            "border-radius:5px; font-size:11px; padding:0 10px; } "
            "QToolButton:hover { background:#2d333b; color:#e6edf3; }"
        )
        for key, btn in self._field_btns.items():
            btn.setStyleSheet(active_css if btn.isChecked() else inactive_css)

    def _build_cmap_controls(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        cmap_lbl = QLabel("Colormap")
        cmap_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        layout.addWidget(cmap_lbl)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(CMAPS)
        self._cmap_combo.setCurrentText("viridis")
        self._cmap_combo.setFixedWidth(110)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        layout.addWidget(self._cmap_combo)

        self._sym_check = QCheckBox("Symmetric")
        self._sym_check.setToolTip("Centre colormap around zero")
        self._sym_check.stateChanged.connect(self._on_cmap_changed)
        layout.addWidget(self._sym_check)

        self._rev_check = QCheckBox("Reverse")
        self._rev_check.stateChanged.connect(self._on_cmap_changed)
        layout.addWidget(self._rev_check)

        layout.addStretch()

        # N colours
        n_lbl = QLabel("N:")
        n_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        layout.addWidget(n_lbl)
        self._n_spin = QSpinBox()
        self._n_spin.setRange(8, 512)
        self._n_spin.setValue(256)
        self._n_spin.setFixedWidth(56)
        self._n_spin.valueChanged.connect(self._on_cmap_changed)
        layout.addWidget(self._n_spin)

        # Style all child widgets
        combo_style = (
            "QComboBox { background:#21262d; color:#e6edf3; border:1px solid #30363d; "
            "border-radius:5px; padding:2px 6px; font-size:11px; } "
            "QComboBox::drop-down { border:none; } "
            "QComboBox QAbstractItemView { background:#21262d; color:#e6edf3; "
            "selection-background-color:#2f81f7; border:1px solid #30363d; }"
        )
        self._cmap_combo.setStyleSheet(combo_style)
        check_style = (
            "QCheckBox { color:#8b949e; font-size:11px; } "
            "QCheckBox::indicator { width:13px; height:13px; border:1px solid #30363d; "
            "border-radius:3px; background:#21262d; } "
            "QCheckBox::indicator:checked { background:#2f81f7; border-color:#2f81f7; }"
        )
        self._sym_check.setStyleSheet(check_style)
        self._rev_check.setStyleSheet(check_style)
        spin_style = (
            "QSpinBox { background:#21262d; color:#e6edf3; border:1px solid #30363d; "
            "border-radius:5px; padding:2px 4px; font-size:11px; } "
            "QSpinBox::up-button, QSpinBox::down-button { border:none; }"
        )
        self._n_spin.setStyleSheet(spin_style)
        n_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        cmap_lbl.setStyleSheet("color:#8b949e; font-size:11px;")

        return container

    def _build_scrubber(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        hdr = QLabel("Temporal")
        hdr.setStyleSheet("color:#8b949e; font-size:11px; font-weight:600; "
                          "text-transform:uppercase; letter-spacing:0.5px;")
        layout.addWidget(hdr)

        # Slider row
        slider_row = QHBoxLayout()
        slider_row.setSpacing(6)

        self._prev_btn = self._icon_btn("◀", "Previous frame", self._prev_frame)
        slider_row.addWidget(self._prev_btn)

        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setMinimum(0)
        self._scrubber.setMaximum(0)
        self._scrubber.setValue(0)
        self._scrubber.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._scrubber.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #21262d;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2f81f7;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #1f4e8c;
                border-radius: 3px;
            }
        """)
        self._scrubber.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._scrubber, 1)

        self._next_btn = self._icon_btn("▶", "Next frame", self._next_frame)
        slider_row.addWidget(self._next_btn)

        layout.addLayout(slider_row)

        # Control row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self._frame_label = QLabel("No results")
        self._frame_label.setStyleSheet(
            "color:#8b949e; font-size:11px; font-family:'Fira Code','Consolas',monospace;"
        )
        ctrl_row.addWidget(self._frame_label)

        ctrl_row.addStretch()

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedWidth(72)
        self._play_btn.setCheckable(True)
        self._play_btn.clicked.connect(self._toggle_play)
        self._play_btn.setStyleSheet(
            "QPushButton { background:#21262d; color:#e6edf3; border:1px solid #30363d; "
            "border-radius:5px; font-size:11px; padding:4px 8px; } "
            "QPushButton:checked { background:#2f81f7; color:#fff; border-color:#2f81f7; } "
            "QPushButton:hover { background:#2d333b; }"
        )
        ctrl_row.addWidget(self._play_btn)

        fps_lbl = QLabel("FPS:")
        fps_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        ctrl_row.addWidget(fps_lbl)
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(5)
        self._fps_spin.setFixedWidth(48)
        self._fps_spin.valueChanged.connect(
            lambda v: self._play_timer.setInterval(1000 // v)
        )
        self._fps_spin.setStyleSheet(
            "QSpinBox { background:#21262d; color:#e6edf3; border:1px solid #30363d; "
            "border-radius:5px; padding:2px 4px; font-size:11px; } "
            "QSpinBox::up-button, QSpinBox::down-button { border:none; }"
        )
        ctrl_row.addWidget(self._fps_spin)

        layout.addLayout(ctrl_row)
        return container

    def _build_export_buttons(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        hdr = QLabel("Export")
        hdr.setStyleSheet("color:#8b949e; font-size:11px; font-weight:600; "
                          "text-transform:uppercase; letter-spacing:0.5px; margin-right:4px;")
        layout.addWidget(hdr)

        for label, tip, slot in [
            ("CSV",  "Export current frame as CSV",  self._export_csv),
            ("HDF5", "Export all frames as HDF5",    self._export_hdf5),
            ("PNG",  "Save current overlay as PNG",  self._export_png),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.setFixedHeight(28)
            btn.setFixedWidth(64)
            btn.clicked.connect(slot)
            btn.setStyleSheet(
                "QPushButton { background:#21262d; color:#c9d1d9; border:1px solid #30363d; "
                "border-radius:5px; font-size:11px; font-weight:600; } "
                "QPushButton:hover { background:#2d333b; color:#e6edf3; border-color:#8b949e; } "
                "QPushButton:pressed { background:#161b22; }"
            )
            layout.addWidget(btn)

        layout.addStretch()
        return container

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#21262d;")
        return line

    def _icon_btn(self, icon: str, tip: str, slot) -> QToolButton:
        btn = QToolButton()
        btn.setText(icon)
        btn.setToolTip(tip)
        btn.setFixedSize(26, 26)
        btn.clicked.connect(slot)
        btn.setStyleSheet(
            "QToolButton { background:#21262d; color:#c9d1d9; border:1px solid #30363d; "
            "border-radius:4px; font-size:11px; } "
            "QToolButton:hover { background:#2d333b; color:#e6edf3; } "
            "QToolButton:pressed { background:#161b22; }"
        )
        return btn

    # ------------------------------------------------------------------
    # Slots: field selection
    # ------------------------------------------------------------------

    def _select_field(self, key: str) -> None:
        self._current_field = key
        for k, btn in self._field_btns.items():
            btn.setChecked(k == key)
        self._apply_tab_style()
        self.field_changed.emit(key)

    # ------------------------------------------------------------------
    # Slots: colourmap
    # ------------------------------------------------------------------

    def _on_cmap_changed(self, *_) -> None:
        cmap = self._cmap_combo.currentText()
        if self._rev_check.isChecked():
            cmap = cmap + "_r" if not cmap.endswith("_r") else cmap[:-2]

        result = self._current_result()
        vmin_out = None
        vmax_out = None
        if result is not None:
            arr = self._get_field_array(result)
            if arr is not None:
                valid = arr[np.isfinite(arr)]
                if valid.size > 0:
                    vmin_out, vmax_out = float(valid.min()), float(valid.max())
                    if self._sym_check.isChecked():
                        lim = max(abs(vmin_out), abs(vmax_out))
                        vmin_out, vmax_out = -lim, lim

        self._update_colourbar(cmap, vmin_out, vmax_out)
        self.colormap_changed.emit(cmap, vmin_out, vmax_out)

    def _update_colourbar(self, cmap: str, vmin: Optional[float],
                          vmax: Optional[float]) -> None:
        """Sample the matplotlib colourmap to get RGB tuples for the gradient."""
        try:
            import matplotlib.cm as cm
            cmap_obj = cm.get_cmap(cmap, self._n_spin.value())
            n = 64
            colors = []
            for i in range(n):
                rgba = cmap_obj(i / (n - 1))
                colors.append(tuple(int(c * 255) for c in rgba[:3]))
            unit = FIELD_LABELS.get(self._current_field, ("", ""))[1]
            self._colourbar.set_data(
                vmin if vmin is not None else 0.0,
                vmax if vmax is not None else 1.0,
                unit,
                colors,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Slots: scrubber / playback
    # ------------------------------------------------------------------

    def _on_slider(self, value: int) -> None:
        self._current_frame = value
        n = len(self._analysis.results)
        self._frame_label.setText(
            f"Frame {value + 1} / {n}" if n > 0 else "No results"
        )
        result = self._current_result()
        if result is not None:
            self._stat_table.populate(result)
        self.frame_changed.emit(value)

    def _prev_frame(self) -> None:
        v = max(0, self._scrubber.value() - 1)
        self._scrubber.setValue(v)

    def _next_frame(self) -> None:
        v = min(self._scrubber.maximum(), self._scrubber.value() + 1)
        self._scrubber.setValue(v)

    def _advance_frame(self) -> None:
        next_v = self._scrubber.value() + 1
        if next_v > self._scrubber.maximum():
            next_v = 0  # loop
        self._scrubber.setValue(next_v)

    def _toggle_play(self, checked: bool) -> None:
        if checked:
            self._play_btn.setText("⏹  Stop")
            self._play_timer.start()
        else:
            self._play_btn.setText("▶  Play")
            self._play_timer.stop()

    # ------------------------------------------------------------------
    # Slots: export
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        if not self._analysis.results:
            self._warn("No results to export.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not directory:
            return
        try:
            self._analysis.export_csv(self._current_frame, directory)
            QMessageBox.information(self, "Exported",
                                    f"CSV files saved to:\n{directory}")
        except Exception as exc:
            self._warn(str(exc))

    def _export_hdf5(self) -> None:
        if not self._analysis.results:
            self._warn("No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5", "results.h5", "HDF5 files (*.h5 *.hdf5)")
        if not path:
            return
        try:
            self._analysis.export_hdf5(path)
            QMessageBox.information(self, "Exported", f"HDF5 saved to:\n{path}")
        except Exception as exc:
            self._warn(str(exc))

    def _export_png(self) -> None:
        """Ask the main window to grab the canvas pixmap and save it."""
        # Emitting field_changed with current key triggers canvas redraw;
        # the main window connects canvas.grab() on a dedicated export signal.
        # For simplicity we just trigger a re-render notification here.
        self.field_changed.emit(self._current_field)
        QMessageBox.information(self, "PNG Export",
                                "Use the canvas overlay directly — right-click the "
                                "canvas and choose 'Save image' from your OS.")

    # ------------------------------------------------------------------
    # Public API (called by main window)
    # ------------------------------------------------------------------

    def refresh_after_analysis(self) -> None:
        """Called after a DIC run completes to update all widgets."""
        n = len(self._analysis.results)
        self._scrubber.setMaximum(max(0, n - 1))
        self._scrubber.setValue(0)
        self._current_frame = 0
        self._frame_label.setText(f"Frame 1 / {n}" if n > 0 else "No results")
        result = self._current_result()
        if result is not None:
            self._stat_table.populate(result)
            self._on_cmap_changed()

    def current_field_key(self) -> str:
        return self._current_field

    def current_cmap(self) -> str:
        cmap = self._cmap_combo.currentText()
        if self._rev_check.isChecked():
            cmap = cmap + "_r" if not cmap.endswith("_r") else cmap[:-2]
        return cmap

    def current_cmap_n(self) -> int:
        return self._n_spin.value()

    def is_symmetric(self) -> bool:
        return self._sym_check.isChecked()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_result(self) -> Optional["PairResult"]:
        if not self._analysis.results:
            return None
        idx = min(self._current_frame, len(self._analysis.results) - 1)
        return self._analysis.results[idx]

    def _get_field_array(self, result: "PairResult") -> Optional[np.ndarray]:
        key = self._current_field
        return getattr(result, key, None)

    def _warn(self, msg: str) -> None:
        QMessageBox.warning(self, "PyDIC", msg)
