"""
Temporal plot of the tracked markers.

The field views answer "where is the strain rate high in this frame". This
answers the other half: "what happened at this point over the sequence". One
named curve per marker, so a point in the workpiece and a point that passes
through the shear zone can be read against each other.
"""
from __future__ import annotations

import csv
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout,
    QWidget,
)

from src.ui.theme import (
    C_ACCENT as _C_ACCENT, C_BORDER as _C_BORDER, C_CARD as _C_CARD,
    C_SURFACE as _C_SURFACE, C_TEXT as _C_TEXT, C_TEXT2 as _C_TEXT2,
    C_TEXT3 as _C_TEXT3,
)

# Only the quantities that are meaningful as a history at a point. Cumulative
# strain is excluded: it is a first-arrival field, not a per-frame measurement,
# so a curve of it would not mean what the axis label claims.
PLOTTABLE = [
    ("Eeff_rate", "Effective strain rate", "s⁻¹"),
    ("Exx_rate", "Strain rate Exx", "s⁻¹"),
    ("Eyy_rate", "Strain rate Eyy", "s⁻¹"),
    ("Exy_rate", "Strain rate Exy", "s⁻¹"),
    ("Veff", "Effective velocity", "px/s"),
    ("Vx", "Velocity Vx", "px/s"),
    ("Vy", "Velocity Vy", "px/s"),
    ("mag_inc", "Displacement per interval", "px"),
    ("u_inc", "Displacement u per interval", "px"),
    ("v_inc", "Displacement v per interval", "px"),
    ("corr", "Correlation coefficient", ""),
]


class MarkerPlotDialog(QDialog):
    """Curves of one field at each marker, against time or frame number."""

    def __init__(self, analysis, seeds, labels=None, parent=None,
                 field: str = "Eeff_rate"):
        super().__init__(parent)
        self.setWindowTitle("Marker history")
        self.resize(940, 620)
        self._analysis = analysis
        self._seeds = [(float(x), float(y)) for (x, y) in seeds]
        self._labels = list(labels or [f"M{i + 1}" for i in range(len(seeds))])
        self._series: dict = {}

        self.setStyleSheet(
            f"QDialog{{background:{_C_SURFACE};}}"
            f"QLabel{{color:{_C_TEXT2}; font-size:11px;}}"
            f"QComboBox,QListWidget{{background:{_C_CARD}; color:{_C_TEXT};"
            f" border:1px solid {_C_BORDER}; border-radius:3px; padding:3px;"
            f" font-size:11px;}}"
            f"QPushButton{{background:{_C_CARD}; color:{_C_TEXT};"
            f" border:1px solid {_C_BORDER}; border-radius:3px; padding:6px 12px;"
            f" font-size:11px;}}"
            f"QPushButton:hover{{border-color:{_C_ACCENT};}}"
            f"QCheckBox{{color:{_C_TEXT2}; font-size:11px;}}")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # ── controls ────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QLabel("Field"))
        self._field_combo = QComboBox()
        for name, label, unit in PLOTTABLE:
            self._field_combo.addItem(label, name)
        idx = self._field_combo.findData(field)
        self._field_combo.setCurrentIndex(max(0, idx))
        self._field_combo.setFixedWidth(210)
        self._field_combo.currentIndexChanged.connect(self._replot)
        top.addWidget(self._field_combo)

        top.addSpacing(12)
        top.addWidget(QLabel("x axis"))
        self._x_combo = QComboBox()
        self._x_combo.addItem("Time (s)", "time")
        self._x_combo.addItem("Frame", "frame")
        self._x_combo.setFixedWidth(110)
        self._x_combo.currentIndexChanged.connect(self._replot)
        top.addWidget(self._x_combo)

        top.addSpacing(12)
        self._smooth_chk = QCheckBox("Smooth")
        self._smooth_chk.setToolTip(
            "5-point moving average. Drawn over the raw curve, which stays\n"
            "visible faintly, so smoothing can never hide a feature.")
        self._smooth_chk.stateChanged.connect(self._replot)
        top.addWidget(self._smooth_chk)

        self._grid_chk = QCheckBox("Grid")
        self._grid_chk.setChecked(True)
        self._grid_chk.stateChanged.connect(self._replot)
        top.addWidget(self._grid_chk)

        top.addStretch()
        root.addLayout(top)

        # ── plot + marker list ──────────────────────────────────────
        mid = QHBoxLayout()
        mid.setSpacing(10)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(7.2, 4.4), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumWidth(560)
        self._ax = self._fig.add_subplot(111)
        mid.addWidget(self._canvas, 1)

        side = QWidget()
        side.setFixedWidth(190)
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(6)
        hdr = QLabel("MARKERS")
        hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700;"
            f" letter-spacing:0.8px;")
        side_lay.addWidget(hdr)
        self._list = QListWidget()
        self._list.itemChanged.connect(self._replot)
        side_lay.addWidget(self._list, 1)
        self._readout = QLabel("")
        self._readout.setWordWrap(True)
        self._readout.setStyleSheet(
            f"color:{_C_TEXT2}; font-size:10px; background:{_C_CARD};"
            f" border:1px solid {_C_BORDER}; border-radius:3px; padding:6px;")
        side_lay.addWidget(self._readout)
        mid.addWidget(side)
        root.addLayout(mid, 1)

        # ── footer ──────────────────────────────────────────────────
        foot = QHBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet(f"color:{_C_TEXT3}; font-size:10px;")
        foot.addWidget(self._status)
        foot.addStretch()
        save_img = QPushButton("Save image…")
        save_img.clicked.connect(self._save_image)
        foot.addWidget(save_img)
        save_csv = QPushButton("Save data…")
        save_csv.clicked.connect(self._save_csv)
        foot.addWidget(save_csv)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        foot.addWidget(close)
        root.addLayout(foot)

        self._canvas.mpl_connect("motion_notify_event", self._on_hover)

        self._gather()
        self._populate_list()
        self._replot()

    # ------------------------------------------------------------------
    def _gather(self) -> None:
        """Sample every plottable field at every marker, once."""
        results = self._analysis.results
        n = len(results)
        fps = float(getattr(self._analysis, "fps", 0.0) or 0.0)
        self._time = (np.arange(n) / fps if fps > 0 else
                      np.arange(n, dtype=float))
        self._have_time = fps > 0

        for m, (sx, sy) in enumerate(self._seeds):
            x, y = sx, sy
            tracked = True
            per_field = {name: np.full(n, np.nan) for name, _, _ in PLOTTABLE}
            for i in range(n):
                res = results[i]
                if not tracked:
                    break
                for name, _, _ in PLOTTABLE:
                    arr = getattr(res, name, None)
                    if arr is None:
                        continue
                    val = self._analysis._sample_sparse(arr, x, y)
                    if np.isfinite(val):
                        per_field[name][i] = val
                # Advance with the flow, exactly as the trajectories do, so the
                # curve follows the material rather than a fixed pixel.
                nxt = self._analysis._advect_step(res, x, y)
                if nxt is None:
                    tracked = False
                else:
                    x, y = nxt
            self._series[m] = per_field

    def _populate_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for m, label in enumerate(self._labels):
            item = QListWidgetItem(f"{label}   ({self._seeds[m][0]:.0f}, "
                                   f"{self._seeds[m][1]:.0f})")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, m)
            self._list.addItem(item)
        self._list.blockSignals(False)

    def _checked(self) -> list[int]:
        out = []
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
        """Moving average that ignores gaps rather than spreading them."""
        finite = np.isfinite(y)
        filled = np.where(finite, y, 0.0)
        kernel = np.ones(window)
        num = np.convolve(filled, kernel, mode="same")
        den = np.convolve(finite.astype(float), kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        return np.where(den > 0, out, np.nan)

    def _replot(self, *_) -> None:
        field = self._field_combo.currentData()
        label, unit = next(((l, u) for n, l, u in PLOTTABLE if n == field),
                           (field, ""))
        use_time = self._x_combo.currentData() == "time" and self._have_time
        xs = self._time if use_time else np.arange(len(self._time), dtype=float)

        self._ax.clear()
        colours = ["#4C9BE8", "#E8724C", "#5BC98B", "#C77DD8", "#E0B64C",
                   "#4CC5C9", "#E85D8A", "#9BA8E8"]

        drawn = 0
        for m in self._checked():
            y = self._series[m].get(field)
            if y is None or not np.isfinite(y).any():
                continue
            colour = colours[m % len(colours)]
            if self._smooth_chk.isChecked():
                self._ax.plot(xs, y, color=colour, lw=0.8, alpha=0.28)
                self._ax.plot(xs, self._smooth(y), color=colour, lw=1.7,
                              label=self._labels[m])
            else:
                self._ax.plot(xs, y, color=colour, lw=1.4,
                              label=self._labels[m])
            drawn += 1

        self._ax.set_xlabel("Time (s)" if use_time else "Frame")
        self._ax.set_ylabel(f"{label} [{unit}]" if unit else label)
        self._ax.grid(self._grid_chk.isChecked(), color=_C_BORDER, lw=0.5)
        for spine in self._ax.spines.values():
            spine.set_color(_C_BORDER)
        self._ax.tick_params(colors=_C_TEXT2, labelsize=9)
        self._ax.xaxis.label.set_color(_C_TEXT)
        self._ax.yaxis.label.set_color(_C_TEXT)
        self._ax.set_facecolor(_C_CARD)
        self._fig.patch.set_facecolor(_C_SURFACE)
        if drawn:
            legend = self._ax.legend(fontsize=9, framealpha=0.0)
            for text in legend.get_texts():
                text.set_color(_C_TEXT)
        else:
            self._ax.text(0.5, 0.5, "No data for this field",
                          transform=self._ax.transAxes, ha="center",
                          color=_C_TEXT3, fontsize=11)

        # Say plainly when the x axis is frame number because no capture rate
        # was set, rather than presenting frame index as if it were seconds.
        if not self._have_time:
            self._status.setText(
                "No capture rate set, so the x axis is frame number.")
        else:
            lost = [self._labels[m] for m in self._checked()
                    if not np.isfinite(self._series[m].get(field,
                                       np.array([np.nan]))).all()]
            self._status.setText(
                f"{drawn} of {len(self._seeds)} markers plotted"
                + (f" · gaps where tracking was lost: {', '.join(lost)}"
                   if lost else ""))

        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _on_hover(self, event) -> None:
        if event.inaxes is not self._ax or event.xdata is None:
            self._readout.setText("")
            return
        use_time = self._x_combo.currentData() == "time" and self._have_time
        xs = self._time if use_time else np.arange(len(self._time), dtype=float)
        if xs.size == 0:
            return
        i = int(np.clip(np.argmin(np.abs(xs - event.xdata)), 0, xs.size - 1))
        field = self._field_combo.currentData()
        unit = next((u for n, _, u in PLOTTABLE if n == field), "")
        head = (f"t = {xs[i]:.4g} s" if use_time else f"frame {int(xs[i])}")
        lines = [head]
        for m in self._checked():
            y = self._series[m].get(field)
            if y is None:
                continue
            v = y[i]
            lines.append(f"{self._labels[m]}: "
                         + ("—" if not np.isfinite(v) else f"{v:.4g} {unit}".strip()))
        self._readout.setText("\n".join(lines))

    # ------------------------------------------------------------------
    def _save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", "marker_history.png",
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)")
        if not path:
            return
        try:
            self._fig.savefig(path, dpi=200,
                              facecolor=self._fig.get_facecolor())
        except Exception as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self._status.setText(f"Saved {path}")

    def _save_csv(self) -> None:
        """The plotted curves, in the same shape the plot shows them."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plotted data", "marker_history.csv",
            "CSV files (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        field = self._field_combo.currentData()
        unit = next((u for n, _, u in PLOTTABLE if n == field), "")
        chosen = self._checked()
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                head = ["frame"] + (["time_s"] if self._have_time else [])
                head += [f"{self._labels[m]} [{unit}]" if unit
                         else self._labels[m] for m in chosen]
                w.writerow(head)
                for i in range(len(self._time)):
                    row = [i] + ([f"{self._time[i]:.6g}"] if self._have_time else [])
                    for m in chosen:
                        v = self._series[m].get(field, np.array([np.nan]))[i]
                        row.append("" if not np.isfinite(v) else f"{v:.6g}")
                    w.writerow(row)
        except Exception as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self._status.setText(f"Saved {path}")
