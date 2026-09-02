"""
video_export_dialog.py — configure a video / image-sequence export.

Layout is rows x cols; each cell gets its own content, field, colormap,
background and colour range. A 1x1 grid with the current view preselected is the
common case and is what the dialog opens on.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QDialogButtonBox,
    QScrollArea, QWidget, QFrame, QColorDialog, QGroupBox,
)

from pydic.ui.render import PanelSpec, RangeSpec, BACKGROUND_CHOICES
from pydic.ui.video_export import ExportSpec, CODECS

# Palette comes from the single source of truth in theme.py. These were
# duplicated literals, which is why re-theming previously left pages behind.
from pydic.ui.theme import C_BORDER, C_SURFACE, C_TEXT, C_TEXT2, C_TEXT3

_C_BORDER = C_BORDER
_C_SURFACE = C_SURFACE
_C_TEXT = C_TEXT
_C_TEXT2 = C_TEXT2
_C_TEXT3 = C_TEXT3


if TYPE_CHECKING:
    pass

_C_DISABLED = "#3f4a5c"

CONTENTS = [("Result field", "field"),
            ("Raw frame", "image"),
            ("Empty", "empty")]


class _PanelEditor(QWidget):
    """Controls for one cell of the grid."""

    def __init__(self, index: int, fields: dict, cmaps: List[str],
                 default_field: str, default_cmap: str, parent=None):
        super().__init__(parent)
        self._solid = (0, 0, 0)
        lay = QGridLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(6)

        lay.setColumnMinimumWidth(0, 88)
        lay.setColumnMinimumWidth(2, 88)
        lay.setColumnStretch(1, 1)
        lay.setColumnStretch(3, 1)

        title = QLabel(f"Panel {index + 1}")
        title.setStyleSheet(f"color:{_C_TEXT}; font-weight:600; font-size:12px;")
        lay.addWidget(title, 0, 0, 1, 3)
        self.label = QCheckBox("Label")
        self.label.setChecked(True)
        self.label.setToolTip("Caption this panel with its field name.")
        lay.addWidget(self.label, 0, 3)

        # --- what this panel draws -----------------------------------------
        # Labels are kept as attributes so they can be greyed alongside their
        # control. A live label over a disabled widget still reads as available.
        self.lbl_shows = QLabel("Shows")
        lay.addWidget(self.lbl_shows, 1, 0)
        self.content = QComboBox()
        for label, key in CONTENTS:
            self.content.addItem(label, key)
        self.content.currentIndexChanged.connect(self._sync_enabled)
        lay.addWidget(self.content, 1, 1)

        self.streaks = QCheckBox("Overlay streaklines")
        self.streaks.setToolTip(
            "Draw the marker trajectories over this result or raw-frame panel.")
        lay.addWidget(self.streaks, 1, 2, 1, 2)

        # Background and its colour picker sit together: the picker only applies
        # to 'Solid colour', and separating them made it look permanently broken.
        self.lbl_bg = QLabel("Background")
        lay.addWidget(self.lbl_bg, 2, 0)
        self.background = QComboBox()
        self.background.addItems(BACKGROUND_CHOICES)
        self.background.currentTextChanged.connect(self._sync_enabled)
        lay.addWidget(self.background, 2, 1)

        self.colour_btn = QPushButton("Pick colour…")
        self.colour_btn.setToolTip("Applies only when Background is Solid colour.")
        self.colour_btn.clicked.connect(self._pick_colour)
        lay.addWidget(self.colour_btn, 2, 2, 1, 2)

        # --- colour mapping -------------------------------------------------
        self.colour_hdr = QLabel("COLOUR")
        self.colour_hdr.setStyleSheet(
            f"color:{_C_TEXT3}; font-size:9px; font-weight:700; letter-spacing:0.8px;")
        lay.addWidget(self.colour_hdr, 3, 0, 1, 4)

        self.lbl_field = QLabel("Field")
        lay.addWidget(self.lbl_field, 4, 0)
        self.field = QComboBox()
        for key, (label, _u) in fields.items():
            self.field.addItem(label, key)
        i = self.field.findData(default_field)
        if i >= 0:
            self.field.setCurrentIndex(i)
        lay.addWidget(self.field, 4, 1)

        self.lbl_cmap = QLabel("Colormap")
        lay.addWidget(self.lbl_cmap, 4, 2)
        self.cmap = QComboBox()
        self.cmap.addItems(cmaps)
        self.cmap.setCurrentText(default_cmap)
        lay.addWidget(self.cmap, 4, 3)

        self.lbl_range = QLabel("Range")
        lay.addWidget(self.lbl_range, 5, 0)
        self.range_mode = QComboBox()
        self.range_mode.addItem("Auto (per frame)", "auto")
        self.range_mode.addItem("Global (sequence)", "global")
        self.range_mode.addItem("Manual", "manual")
        self.range_mode.currentIndexChanged.connect(self._sync_enabled)
        lay.addWidget(self.range_mode, 5, 1)

        self.rng_row = rng_row = QWidget()
        rl = QHBoxLayout(rng_row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        self.vmin = QDoubleSpinBox(); self.vmin.setDecimals(4); self.vmin.setRange(-1e12, 1e12)
        self.vmax = QDoubleSpinBox(); self.vmax.setDecimals(4); self.vmax.setRange(-1e12, 1e12)
        self.vmin.setToolTip("Lower colour limit"); self.vmax.setToolTip("Upper colour limit")
        for sb in (self.vmin, self.vmax):
            sb.setMinimumWidth(84)
            rl.addWidget(sb)
        lay.addWidget(rng_row, 5, 2, 1, 2)

        self.symmetric = QCheckBox("Symmetric about 0")
        lay.addWidget(self.symmetric, 6, 1)
        self.colorbar = QCheckBox("Colourbar")
        self.colorbar.setChecked(True)
        lay.addWidget(self.colorbar, 6, 2, 1, 2)

        self.setStyleSheet(
            f"QLabel {{ color:{_C_TEXT2}; font-size:11px; }} "
            f"QWidget {{ background:{_C_SURFACE}; }}")
        self._update_colour_swatch()
        self._sync_enabled()

    def _update_colour_swatch(self) -> None:
        """Show the chosen colour on the button so it reads as a colour control."""
        r, g, b = self._solid
        fg = "#000" if (r * 299 + g * 587 + b * 114) / 1000 > 128 else "#fff"
        self.colour_btn.setStyleSheet(
            f"QPushButton {{ background:rgb({r},{g},{b}); color:{fg}; "
            f"border:1px solid {_C_BORDER}; border-radius:3px; padding:4px 8px; }} "
            f"QPushButton:disabled {{ background:{_C_SURFACE}; color:{_C_DISABLED}; }}")

    def _pick_colour(self):
        from PyQt6.QtGui import QColor
        c = QColorDialog.getColor(QColor(*self._solid), self, "Panel background")
        if c.isValid():
            self._solid = (c.red(), c.green(), c.blue())
            self._update_colour_swatch()

    def _sync_enabled(self, *_):
        """Show only the controls that do something for the chosen content.

        Irrelevant controls are HIDDEN, not greyed. A raw-frame panel has no
        field, so field / colormap / colour range / symmetric /
        colourbar mean nothing there; leaving them on screen greyed still makes
        the panel look like it has settings you failed to reach. The grid rows
        collapse when every widget in them is hidden, so each panel shrinks to
        exactly the controls that apply.
        """
        kind = self.content.currentData()
        is_field = kind == "field"
        draws = kind != "empty"
        manual = is_field and self.range_mode.currentData() == "manual"

        # Colour block: only meaningful for a result field.
        for w in (self.colour_hdr, self.lbl_field, self.field,
                  self.lbl_cmap, self.cmap, self.lbl_range, self.range_mode,
                  self.rng_row, self.symmetric, self.colorbar):
            w.setVisible(is_field)
        # vmin/vmax exist only for an explicit manual range.
        self.rng_row.setVisible(manual)

        # Anything that draws pixels needs a background and can be captioned.
        self.lbl_bg.setVisible(draws)
        self.background.setVisible(draws)
        self.label.setVisible(draws)

        # The picker applies only to a solid background -- hide it otherwise
        # rather than leaving a permanently dead button on screen.
        self.colour_btn.setVisible(draws and self.background.currentText() == "Solid colour")

        # Result and raw-frame panels can both carry trajectories; an empty
        # panel draws nothing at all.
        self.streaks.setVisible(kind in ("field", "image"))
        if not draws:
            self.streaks.setChecked(False)

    def spec(self) -> PanelSpec:
        return PanelSpec(
            content=self.content.currentData(),
            field=self.field.currentData(),
            cmap=self.cmap.currentText(),
            background=self.background.currentText(),
            solid_colour=self._solid,
            show_colorbar=self.colorbar.isChecked(),
            show_label=self.label.isChecked(),
            overlay_streaklines=self.streaks.isChecked(),
            range_spec=RangeSpec(
                mode=self.range_mode.currentData(),
                vmin=self.vmin.value(),
                vmax=self.vmax.value(),
                symmetric=self.symmetric.isChecked()),
        )


class VideoExportDialog(QDialog):
    def __init__(self, fields: dict, cmaps: List[str], n_frames: int,
                 current_field: str, current_cmap: str, fps: float = 25.0,
                 parent=None, source_size: tuple[int, int] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export video")
        self.setMinimumWidth(760)
        self._fields = fields
        self._cmaps = cmaps
        self._current_field = current_field
        self._current_cmap = current_cmap
        self._editors: List[_PanelEditor] = []

        root = QVBoxLayout(self)

        # -- layout / output ------------------------------------------------
        gb = QGroupBox("Layout and output")
        gl = QGridLayout(gb)

        gl.addWidget(QLabel("Rows"), 0, 0)
        self.rows = QSpinBox(); self.rows.setRange(1, 4); self.rows.setValue(1)
        self.rows.valueChanged.connect(self._rebuild_panels)
        gl.addWidget(self.rows, 0, 1)

        gl.addWidget(QLabel("Columns"), 0, 2)
        self.cols = QSpinBox(); self.cols.setRange(1, 4); self.cols.setValue(1)
        self.cols.valueChanged.connect(self._rebuild_panels)
        gl.addWidget(self.cols, 0, 3)

        gl.addWidget(QLabel("Cell size"), 0, 4)
        source_w, source_h = source_size or (640, 480)
        self.cell_w = QSpinBox(); self.cell_w.setRange(1, 16384); self.cell_w.setValue(int(source_w))
        self.cell_h = QSpinBox(); self.cell_h.setRange(1, 16384); self.cell_h.setValue(int(source_h))
        self.cell_w.setToolTip("Output width of each cell; defaults to the source frame width.")
        self.cell_h.setToolTip("Output height of each cell; defaults to the source frame height.")
        gl.addWidget(self.cell_w, 0, 5)
        gl.addWidget(self.cell_h, 0, 6)

        gl.addWidget(QLabel("Format"), 1, 0)
        self.codec = QComboBox(); self.codec.addItems(list(CODECS.keys()))
        self.codec.setToolTip(
            "mp4v plays almost everywhere.\n"
            "MJPG needs no external codec if mp4 fails.\n"
            "FFV1 is lossless but large.\n"
            "PNG sequence is the only option that keeps a transparent\n"
            "background; none of the video codecs carry alpha.")
        gl.addWidget(self.codec, 1, 1)

        gl.addWidget(QLabel("FPS"), 1, 2)
        self.fps = QDoubleSpinBox(); self.fps.setRange(0.1, 480.0); self.fps.setValue(float(fps))
        gl.addWidget(self.fps, 1, 3)

        gl.addWidget(QLabel("Frames"), 1, 4)
        self.first = QSpinBox(); self.first.setRange(1, max(1, n_frames)); self.first.setValue(1)
        self.last = QSpinBox(); self.last.setRange(1, max(1, n_frames)); self.last.setValue(max(1, n_frames))
        gl.addWidget(self.first, 1, 5)
        gl.addWidget(self.last, 1, 6)

        root.addWidget(gb)

        # -- panels ---------------------------------------------------------
        self._panel_host = QWidget()
        self._panel_lay = QVBoxLayout(self._panel_host)
        self._panel_lay.setContentsMargins(0, 0, 0, 0)
        self._panel_lay.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._panel_host)
        scroll.setMinimumHeight(300)
        root.addWidget(scroll, 1)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        root.addWidget(self._hint)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Export…")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        self.codec.currentTextChanged.connect(self._update_hint)
        self._rebuild_panels()

    def _update_hint(self, *_):
        seq = CODECS.get(self.codec.currentText(), (None, None))[1] is None
        transparent = any(e.background.currentText() == "Transparent"
                          for e in self._editors)
        msgs = []
        if transparent and not seq:
            msgs.append("⚠  A transparent background cannot be stored in a video "
                        "file; it will come out black. Choose the PNG image "
                        "sequence' to keep real transparency.")
        if seq:
            msgs.append("Frames are written as numbered PNGs into a folder next "
                        "to the filename you choose.")
        self._hint.setText("  ".join(msgs))

    def _rebuild_panels(self, *_):
        want = self.rows.value() * self.cols.value()
        while len(self._editors) > want:
            e = self._editors.pop()
            self._panel_lay.removeWidget(e)
            e.deleteLater()
        while len(self._editors) < want:
            i = len(self._editors)
            # First cell mirrors the current view; extra cells default to the
            # raw frame so a new grid is immediately meaningful.
            e = _PanelEditor(i, self._fields, self._cmaps,
                             self._current_field, self._current_cmap)
            if i > 0:
                e.content.setCurrentIndex(1)       # Raw frame
            e.background.currentTextChanged.connect(self._update_hint)
            self._editors.append(e)
            self._panel_lay.addWidget(e)
            sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
            self._panel_lay.addWidget(sep)
        self._update_hint()

    def spec(self) -> ExportSpec:
        return ExportSpec(
            rows=self.rows.value(),
            cols=self.cols.value(),
            panels=[e.spec() for e in self._editors],
            cell_w=self.cell_w.value(),
            cell_h=self.cell_h.value(),
            fps=self.fps.value(),
            codec=self.codec.currentText(),
            first=self.first.value() - 1,
            last=self.last.value() - 1,
        )
