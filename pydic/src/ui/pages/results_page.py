"""
results_page.py — Step 5: Results viewer with correct frame synchronisation.

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
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QLinearGradient, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QCheckBox, QFrame, QSizePolicy,
    QFileDialog, QMessageBox, QSpinBox, QToolButton,
)

if TYPE_CHECKING:
    from src.ui.wizard import Wizard

from src.ui.image_canvas import ImageCanvas

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
    "u": ("Displacement u", "px"),
    "v": ("Displacement v", "px"),

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

CMAPS = ["inferno","magma","plasma","cividis","hot","afmhot",
         "gist_heat","copper","RdBu_r","seismic","bwr","coolwarm",
         "viridis","turbo","jet","gray"]

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
    """Step 5 — results viewer with correct frame-by-frame image updates."""

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._frame  = 0
        self._field  = "Eeff_rate"
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(200)
        self._play_timer.timeout.connect(self._advance)
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
        new_btn.clicked.connect(self._wizard.go_welcome)
        top_lay.addWidget(new_btn)

        top_lay.addSpacing(12)

        # Field selector tabs
        self._field_btns: dict[str, QToolButton] = {}
        for key, (label, _) in FIELDS.items():
            btn = QToolButton()
            btn.setText(key.replace("_rate", "̇ ").replace("_", " "))
            btn.setToolTip(label)
            btn.setCheckable(True)
            btn.setChecked(key == self._field)
            btn.clicked.connect(lambda c, k=key: self._select_field(k))
            top_lay.addWidget(btn)
            self._field_btns[key] = btn

        self._apply_tab_style()
        top_lay.addStretch()

        # Colormap
        cmap_lbl = QLabel("Colormap:")
        cmap_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        top_lay.addWidget(cmap_lbl)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(CMAPS)
        self._cmap_combo.setCurrentText("inferno")
        self._cmap_combo.setFixedWidth(100)
        self._cmap_combo.currentTextChanged.connect(self._refresh_overlay)
        top_lay.addWidget(self._cmap_combo)

        self._sym_chk = QCheckBox("Sym")
        self._sym_chk.setToolTip("Centre colormap around zero")
        self._sym_chk.stateChanged.connect(self._refresh_overlay)
        top_lay.addWidget(self._sym_chk)

        root.addWidget(top)

        # ── Body: canvas + right sidebar ──────────────────────────────
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        # Canvas
        self._canvas = ImageCanvas()
        self._canvas.seed_enabled = False  # Disable seed placement here
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

        for label, slot in [("CSV (this frame)", self._export_csv),
                             ("HDF5 (all frames)", self._export_hdf5)]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.clicked.connect(slot)
            sb_lay.addWidget(btn)

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

        root.addWidget(bottom)

    # ------------------------------------------------------------------
    # Public API — called by wizard
    # ------------------------------------------------------------------

    def on_enter(self) -> None:
        """Refresh after analysis completes."""
        n = len(self._wizard.analysis.results)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._frame = 0
        self._show_frame(0)

    # ------------------------------------------------------------------
    # Frame display — THE CRITICAL FIX
    # ------------------------------------------------------------------

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
            img = _load_gray(path)
            if img is not None:
                # Darken the grayscale background so the colormap overlay pops.
                # 0.28 keeps enough structural detail for context without
                # competing with the vivid field colours.
                self._canvas.set_image(img * 0.45)
        # (If image can't be loaded, canvas keeps whatever was last shown)

        # 2. ── Render field overlay ──────────────────────────────────
        result = analysis.results[idx]
        arr = getattr(result, self._field, None)
        if arr is not None and np.any(np.isfinite(arr)):
            self._apply_overlay(arr)
        else:
            self._canvas.set_result_overlay_rgba(None)

        # 3. ── Update sidebar ────────────────────────────────────────
        self._update_stats(result)
        self._frame_lbl.setText(f"Frame {idx + 1} / {n}")

    def _apply_overlay(self, arr: np.ndarray) -> None:
        """
        Convert field array to RGBA colormap overlay and push to canvas.

        The DIC grid is sparse (one solved subset every N pixels), so the raw
        overlay has tiny coloured dots on a transparent background. We dilate
        each dot to fill the Voronoi cell around it — giving a solid, vivid
        coverage across the whole ROI — then apply a soft Gaussian blur to
        smooth the block boundaries.
        """
        try:
            import matplotlib.cm as cm
            import matplotlib.colors as mc
            from scipy.ndimage import grey_dilation, binary_dilation, gaussian_filter

            valid_mask = np.isfinite(arr)
            if not valid_mask.any():
                return

            vmin, vmax = float(arr[valid_mask].min()), float(arr[valid_mask].max())
            if self._sym_chk.isChecked():
                lim = max(abs(vmin), abs(vmax))
                vmin, vmax = -lim, lim
            if vmin == vmax:
                vmax = vmin + 1e-12

            cmap_name = self._cmap_combo.currentText()
            cmap_obj  = cm.get_cmap(cmap_name, 256)
            norm      = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)

            rgba = cmap_obj(norm(arr), bytes=True).astype(np.float32)   # H×W×4
            rgba[~valid_mask] = 0

            # ── Mild saturation boost — vivid but not crushed ─────────
            # Shift each channel toward its distance from mid-grey (128),
            # making colours punchier without turning everything black.
            SATURATION = 1.35          # >1 = more vivid, 1.0 = unchanged
            for ch in range(3):
                ch_f = rgba[..., ch] / 255.0
                ch_f = np.clip(0.5 + (ch_f - 0.5) * SATURATION, 0.0, 1.0)
                rgba[..., ch] = (ch_f * 255.0).astype(np.float32)

            # ── Fill sparse grid gaps with morphological dilation ──────
            kernel_size = 13           # slightly larger for smoother fill
            struct = np.ones((kernel_size, kernel_size), dtype=bool)

            for ch in range(3):
                rgba[..., ch] = grey_dilation(rgba[..., ch].astype(np.uint8),
                                              structure=struct).astype(np.float32)

            covered = binary_dilation(valid_mask, structure=struct)
            rgba[~covered, 3] = 0
            rgba[covered,  3] = 195    # ~76% — vivid colour but greyscale stays visible

            # ── Soft blur to smooth block edges ────────────────────────
            sigma = 1.4
            for ch in range(4):
                rgba[..., ch] = gaussian_filter(rgba[..., ch], sigma=sigma)

            rgba[~covered, 3] = 0      # re-clip after blur

            self._canvas.set_result_overlay_rgba(rgba.astype(np.uint8))

            # Update colorbar — apply same darkening so legend matches canvas
            n_bar = 64
            bar_colors = []
            SATURATION = 1.35
            for i in range(n_bar):
                r, g, b, _ = cmap_obj(i / (n_bar - 1))
                r = max(0.0, min(1.0, 0.5 + (r - 0.5) * SATURATION))
                g = max(0.0, min(1.0, 0.5 + (g - 0.5) * SATURATION))
                b = max(0.0, min(1.0, 0.5 + (b - 0.5) * SATURATION))
                bar_colors.append((int(r*255), int(g*255), int(b*255)))
            unit = FIELDS.get(self._field, ("", ""))[1]
            self._colorbar.update_bar(vmin, vmax, unit, bar_colors)

        except Exception as exc:
            print(f"Overlay error: {exc}")

    def _update_stats(self, result) -> None:
        arr = getattr(result, self._field, None)
        if arr is None:
            for v in self._stat_labels.values():
                v.setText("—")
            return

        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            for v in self._stat_labels.values():
                v.setText("—")
            return

        self._stat_labels["Mean"].setText(f"{valid.mean():.4g}")
        self._stat_labels["Std Dev"].setText(f"{valid.std():.4g}")
        self._stat_labels["Min"].setText(f"{valid.min():.4g}")
        self._stat_labels["Max"].setText(f"{valid.max():.4g}")
        self._stat_labels["Valid px"].setText(f"{valid.size:,}")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_slider(self, val: int) -> None:
        self._frame = val
        self._show_frame(val)

    def _select_field(self, key: str) -> None:
        self._field = key
        for k, btn in self._field_btns.items():
            btn.setChecked(k == key)
        self._apply_tab_style()
        self._refresh_overlay()

    def _refresh_overlay(self, *_) -> None:
        self._show_frame(self._frame)

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
            self._wizard.analysis.export_csv(self._frame, directory)
            QMessageBox.information(self, "Exported", f"CSV files saved to:\n{directory}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _export_hdf5(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5", "results.h5", "HDF5 files (*.h5 *.hdf5)"
        )
        if not path:
            return
        try:
            self._wizard.analysis.export_hdf5(path)
            QMessageBox.information(self, "Exported", f"HDF5 saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

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
