"""
roi_page.py — Step 2: Define region of interest on full-screen canvas.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QToolButton, QButtonGroup,
    QSizePolicy, QFileDialog, QMessageBox, QComboBox,
    QSpinBox,
)
from strainx.ui.components import FooterButton

if TYPE_CHECKING:
    from strainx.ui.wizard import Wizard

from strainx.ui.image_canvas import ImageCanvas, ROITool

# Palette comes from the single source of truth in theme.py. These were
# duplicated literals, which is why re-theming previously left pages behind.
from strainx.ui.theme import C_ACCENT, C_BG, C_BORDER, C_CARD, C_SUCCESS, C_SURFACE, C_TEXT, C_TEXT2, C_TEXT3

_C_ACCENT = C_ACCENT
_C_BG = C_BG
_C_BORDER = C_BORDER
_C_CARD = C_CARD
_C_SUCCESS = C_SUCCESS
_C_SURFACE = C_SURFACE
_C_TEXT = C_TEXT
_C_TEXT2 = C_TEXT2
_C_TEXT3 = C_TEXT3




def _tool_btn(icon: str, tip: str) -> QToolButton:
    btn = QToolButton()
    btn.setText(icon)
    btn.setToolTip(tip)
    btn.setCheckable(True)
    btn.setFixedSize(44, 44)
    btn.setStyleSheet(
        f"QToolButton {{ background:{_C_CARD}; color:{_C_TEXT2}; "
        f"border:1px solid {_C_BORDER}; border-radius:3px; font-size:18px; }} "
        f"QToolButton:hover {{ background:{_C_BORDER}; color:{_C_TEXT}; }} "
        f"QToolButton:checked {{ background:{_C_ACCENT}; color:#fff; "
        f"border-color:{_C_ACCENT}; }} "
    )
    return btn


class ROIPage(QWidget):
    """Step 2 — draw ROI on the full reference image."""

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._editing_origin = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────
        top = QWidget()
        top.setFixedHeight(52)
        top.setStyleSheet(f"background:{_C_SURFACE}; border-bottom:1px solid {_C_BORDER};")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(20, 0, 20, 0)
        top_lay.setSpacing(16)

        back = QPushButton("← Back")
        back.setFixedWidth(90)
        back.clicked.connect(self._wizard.go_welcome)
        top_lay.addWidget(back)

        title = QLabel("Step 2  ·  Region of interest")
        title.setStyleSheet(f"color:{_C_TEXT}; font-size:13px; font-weight:600;")
        top_lay.addWidget(title)

        top_lay.addStretch()

        start_lbl = QLabel("Strain zero frame")
        start_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        start_lbl.setToolTip(
            "Accumulated strain is zero on this frame. It stays on the\n"
            "canvas while you draw the ROI and the origin line that\n"
            "seeds new material paths.")
        top_lay.addWidget(start_lbl)

        self._strain_start_spin = QSpinBox()
        self._strain_start_spin.setRange(0, 0)
        self._strain_start_spin.setFixedWidth(86)
        self._strain_start_spin.setPrefix("# ")
        self._strain_start_spin.setToolTip(
            "0 is the imported reference frame; 1 is the first deformed frame.")
        self._strain_start_spin.valueChanged.connect(
            self._on_strain_start_changed)
        top_lay.addWidget(self._strain_start_spin)

        self._roi_lbl = QLabel("No ROI drawn")
        self._roi_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        top_lay.addWidget(self._roi_lbl)

        root.addWidget(top)

        # ── Main area (tools + canvas) ────────────────────────────────
        main = QWidget()
        main_lay = QHBoxLayout(main)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # Left toolbar
        toolbar = QWidget()
        toolbar.setFixedWidth(64)
        toolbar.setStyleSheet(f"background:{_C_SURFACE}; border-right:1px solid {_C_BORDER};")
        tb_lay = QVBoxLayout(toolbar)
        tb_lay.setContentsMargins(10, 16, 10, 16)
        tb_lay.setSpacing(8)
        tb_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        channel_lbl = QLabel("Edit")
        channel_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:9px;")
        channel_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tb_lay.addWidget(channel_lbl)

        self._channel_group = QButtonGroup(self)
        self._channel_group.setExclusive(True)
        self._roi_channel_btn = _tool_btn("ROI", "Edit the pairwise analysis ROI")
        self._roi_channel_btn.setStyleSheet(
            self._roi_channel_btn.styleSheet().replace("font-size:18px", "font-size:10px"))
        self._origin_channel_btn = _tool_btn(
            "ε₀", "Draw the line/curve where zero-strain material enters")
        self._channel_group.addButton(self._roi_channel_btn)
        self._channel_group.addButton(self._origin_channel_btn)
        self._roi_channel_btn.setChecked(True)
        self._roi_channel_btn.clicked.connect(lambda: self._set_edit_channel(False))
        self._origin_channel_btn.clicked.connect(lambda: self._set_edit_channel(True))
        tb_lay.addWidget(self._roi_channel_btn)
        tb_lay.addWidget(self._origin_channel_btn)
        tb_lay.addSpacing(12)

        tools = [
            ("▱", "Rectangle ROI", ROITool.RECTANGLE),
            ("○", "Ellipse / Circle ROI", ROITool.CIRCLE),
            ("⬡", "Polygon ROI (click to place, dbl-click to close)", ROITool.POLYGON),
            ("⌁", "Strain-origin line / curve (click points or drag, then Done)",
             ROITool.POLYLINE),
            ("✕", "Erase from ROI", ROITool.ERASE),
        ]
        self._tool_btns = {}
        for icon, tip, tool in tools:
            btn = _tool_btn(icon, tip)
            self._tool_group.addButton(btn)
            btn.clicked.connect(lambda checked, t=tool: self._on_tool_selected(t))
            tb_lay.addWidget(btn)
            self._tool_btns[tool] = btn

        # The main analysis ROI opens in the most common drawing mode. The
        # strain-origin channel switches to its dedicated open-curve tool.
        self._tool_btns[ROITool.POLYLINE].setVisible(False)
        self._tool_btns[ROITool.RECTANGLE].setChecked(True)

        tb_lay.addSpacing(12)

        # Seed
        seed_lbl = QLabel("Seed")
        seed_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:9px; text-align:center;")
        seed_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tb_lay.addWidget(seed_lbl)

        self._seed_btn = _tool_btn("⦿", "Place seed point for RG-DIC propagation")
        self._seed_btn.clicked.connect(
            lambda: self._canvas.set_roi_tool(ROITool.NONE)  # canvas handles seed on click
        )
        tb_lay.addWidget(self._seed_btn)

        tb_lay.addSpacing(12)

        # Undo sits with the tools, not at the far bottom of the column.
        # Parked below a stretch it was several hundred pixels from the actions
        # it reverses and read as part of the footer, so it went unnoticed --
        # an undo nobody can find is the same as no undo. Same 44x44 footprint
        # as the tools, and a label, because a bare glyph is not self-evident.
        edit_lbl = QLabel("History")
        edit_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:9px;")
        edit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tb_lay.addWidget(edit_lbl)

        self._undo_btn = QPushButton("↶")
        self._undo_btn.setToolTip(
            "Undo the last region, erase stroke or seed  (Ctrl+Z)")
        self._undo_btn.setFixedSize(44, 44)
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self._undo)
        # The accent border when there is something to undo is the cue that
        # the action is available at all.
        self._undo_btn.setStyleSheet(
            f"QPushButton {{ background:{_C_CARD}; color:{_C_TEXT}; "
            f"border:1px solid {_C_ACCENT}; border-radius:3px; font-size:20px; }} "
            f"QPushButton:hover {{ background:{_C_ACCENT}; color:#fff; }} "
            f"QPushButton:disabled {{ background:{_C_CARD}; color:{_C_TEXT3}; "
            f"border:1px solid {_C_BORDER}; }}"
        )
        tb_lay.addWidget(self._undo_btn)

        tb_lay.addStretch()

        # Clear button
        clr_btn = QPushButton("⟳")
        clr_btn.setToolTip("Clear the mask being edited.")
        clr_btn.setFixedSize(44, 36)
        clr_btn.clicked.connect(self._clear_active_mask)
        clr_btn.setStyleSheet(
            f"QPushButton {{ background:{_C_CARD}; color:{_C_TEXT2}; "
            f"border:1px solid {_C_BORDER}; border-radius:3px; font-size:16px; }} "
            f"QPushButton:hover {{ background:{_C_BORDER}; color:{_C_TEXT}; }}"
        )
        tb_lay.addWidget(clr_btn)

        main_lay.addWidget(toolbar)

        # Canvas
        self._canvas = ImageCanvas()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        self._canvas.roi_changed.connect(self._on_roi_changed)
        self._canvas.seed_placed.connect(self._on_seed_placed)
        self._canvas.undo_availability_changed.connect(self._undo_btn.setEnabled)

        # Ctrl+Z as well as the button. WindowShortcut scope keeps it inert on
        # the other wizard steps, which have nothing to undo.
        self._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._undo_shortcut.activated.connect(self._undo)
        self._canvas.set_roi_tool(ROITool.RECTANGLE)
        main_lay.addWidget(self._canvas, 1)

        root.addWidget(main, 1)

        # ── Footer ────────────────────────────────────────────────────
        footer = QWidget()
        footer.setFixedHeight(58)
        footer.setStyleSheet(f"background:{_C_SURFACE}; border-top:1px solid {_C_BORDER};")
        foot_lay = QHBoxLayout(footer)
        foot_lay.setContentsMargins(20, 0, 20, 0)

        full_btn = QPushButton("Use full image")
        full_btn.setFixedWidth(130)
        full_btn.clicked.connect(self._use_full)
        foot_lay.addWidget(full_btn)

        # ────────Reset Zoom Button ────────────────────────────────────
        self._reset_view_btn = QPushButton("Reset zoom")
        self._reset_view_btn.setFixedWidth(110)
        self._reset_view_btn.clicked.connect(self._canvas.fit_image)
        foot_lay.addWidget(self._reset_view_btn)
        # ──────────────────────────────────────────────────────────────

        load_btn = QPushButton("Load ROI from file")
        load_btn.setFixedWidth(180)
        load_btn.setToolTip(
            "Load an ROI mask from:\n"
            "  • PNG / TIF / JPG image  (white = ROI)\n"
            "  • NumPy .npy array\n"
            "  • Ncorr .mat / .h5 file"
        )
        load_btn.clicked.connect(self._load_roi_from_file)
        foot_lay.addWidget(load_btn)

        foot_lay.addStretch()

        # Dynamic ROI is chosen here, next to the static ROI it refines, so the
        # answer is known before we decide whether the tuning step is even
        # needed. "None" skips that step entirely.
        dyn_lbl = QLabel("Dynamic ROI")
        dyn_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        foot_lay.addWidget(dyn_lbl)

        self._cb_dynamic = QComboBox()
        self._cb_dynamic.addItems(["None", "Contrast", "Edge Detection", "Hybrid"])
        self._cb_dynamic.setFixedWidth(134)
        self._cb_dynamic.setToolTip(
            "Per-frame texture masking, applied inside the ROI you drew.\n"
            "Useful when material leaves the field, as in cutting tests.\n\n"
            "Any setting other than None adds a tuning step for the\n"
            "threshold and for forcing regions in or out by hand.")
        self._cb_dynamic.currentTextChanged.connect(self._on_dynamic_changed)
        foot_lay.addWidget(self._cb_dynamic)

        foot_lay.addSpacing(18)

        self._seed_status = QLabel("No seed. Defaults to the ROI centroid.")
        self._seed_status.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        foot_lay.addWidget(self._seed_status)

        foot_lay.addSpacing(24)

        self._next_btn = FooterButton("Parameters  →")
        self._next_btn.setProperty("class", "accent")
        self._next_btn.setFixedHeight(36)
        self._next_btn.setMinimumWidth(150)
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._wizard.go_after_roi)
        foot_lay.addWidget(self._next_btn)

        # Finishing an open strain-origin curve is a page-level action, not a
        # drawing tool. Keep it full-sized and at the extreme bottom-right.
        self._finish_line_btn = FooterButton("Done")
        self._finish_line_btn.setToolTip(
            "Finish and save the strain-origin line")
        self._finish_line_btn.setFixedHeight(38)
        self._finish_line_btn.setMinimumWidth(120)
        self._finish_line_btn.setVisible(False)
        self._finish_line_btn.clicked.connect(self._finish_strain_line)
        foot_lay.addWidget(self._finish_line_btn)
        self._canvas.shape_drawing_changed.connect(
            self._on_shape_drawing_changed)

        root.addWidget(footer)

    # ------------------------------------------------------------------
    def _on_dynamic_changed(self, text: str) -> None:
        analysis = self._wizard.analysis
        old = getattr(analysis.params, "dynamic_roi", "None")
        analysis.params.dynamic_roi = text
        if old != text:
            analysis.results.clear()
            # The choice lives on this page, so cache it here rather than
            # relying on a later Parameters/Run action the user may never reach.
            analysis.save_settings()
        self._next_btn.setText("Dynamic ROI  →" if text not in ("None", "")
                               else "Parameters  →")

    def _sync_dynamic_combo(self) -> None:
        cur = getattr(self._wizard.analysis.params, "dynamic_roi", "None") or "None"
        self._cb_dynamic.blockSignals(True)
        self._cb_dynamic.setCurrentText(cur)
        self._cb_dynamic.blockSignals(False)
        self._next_btn.setText("Dynamic ROI  →" if cur not in ("None", "")
                               else "Parameters  →")

    def _on_strain_start_changed(self, frame: int) -> None:
        analysis = self._wizard.analysis
        frame = min(max(0, int(frame)), len(analysis.def_paths))
        if getattr(analysis, "strain_start_frame", 0) != frame:
            analysis.strain_start_frame = frame
            analysis.results.clear()
        image = analysis.frame_image(frame)
        if image is not None:
            self._canvas.set_image(image, keep_view=True)

    def _set_edit_channel(self, origin: bool) -> None:
        # Do not discard a line merely because the operator switches channels.
        # finish_active_shape() is a no-op until at least two points exist.
        if self._editing_origin and not origin:
            self._canvas.finish_active_shape()
        self._editing_origin = bool(origin)
        self._roi_channel_btn.setChecked(not self._editing_origin)
        self._origin_channel_btn.setChecked(self._editing_origin)
        self._canvas.set_roi_role(self._editing_origin)
        self._canvas.seed_enabled = not self._editing_origin
        self._seed_btn.setEnabled(not self._editing_origin)
        for tool in (ROITool.RECTANGLE, ROITool.CIRCLE, ROITool.POLYGON):
            self._tool_btns[tool].setVisible(not self._editing_origin)
        self._tool_btns[ROITool.POLYLINE].setVisible(self._editing_origin)

        main_roi = self._wizard.analysis.roi_mask
        self._canvas.set_context_mask(main_roi if self._editing_origin else None)
        self._canvas.set_draw_constraint(main_roi if self._editing_origin else None)

        default_tool = (ROITool.POLYLINE if self._editing_origin
                        else ROITool.RECTANGLE)
        self._tool_btns[default_tool].setChecked(True)
        self._canvas.set_roi_tool(default_tool)
        mask = (self._wizard.analysis.strain_origin_mask
                if self._editing_origin else self._wizard.analysis.roi_mask)
        if mask is None:
            self._canvas.clear_roi()
        else:
            self._canvas.set_roi_mask(mask)
        if not self._editing_origin and getattr(self._wizard, "seed_xy", None) is not None:
            self._canvas.set_seed_xy(self._wizard.seed_xy)
        self._sync_strain_drawing_actions()
        self._refresh_mask_status()

    def _on_tool_selected(self, tool: ROITool) -> None:
        if self._canvas._tool == ROITool.POLYLINE and tool != ROITool.POLYLINE:
            self._canvas.finish_active_shape()
        self._canvas.set_roi_tool(tool)

    def _finish_strain_line(self) -> None:
        if not self._canvas.finish_active_shape():
            QMessageBox.information(
                self, "Unfinished strain-origin line",
                "Draw at least two points, then press Done.")
            return
        # Stay ready to add another disconnected inlet segment if needed.
        self._tool_btns[ROITool.POLYLINE].setChecked(True)
        self._canvas.set_roi_tool(ROITool.POLYLINE)

    def _on_shape_drawing_changed(self, drawing: bool) -> None:
        self._sync_strain_drawing_actions(bool(drawing))

    def _sync_strain_drawing_actions(self, drawing=None) -> None:
        """Swap the navigation button for Done only during an active curve."""
        if drawing is None:
            drawing = bool(
                self._editing_origin and
                self._canvas._tool == ROITool.POLYLINE and
                self._canvas._poly_pts)
        drawing = bool(self._editing_origin and drawing)
        self._finish_line_btn.setVisible(drawing)
        self._next_btn.setVisible(not drawing)

    def _refresh_mask_status(self) -> None:
        analysis = self._wizard.analysis
        roi_n = int(analysis.roi_mask.sum()) if analysis.roi_mask is not None else 0
        origin_n = (int(analysis.strain_origin_mask.sum())
                    if analysis.strain_origin_mask is not None else 0)
        if roi_n and origin_n:
            self._roi_lbl.setText(
                f"ROI: {roi_n:,} px  ·  strain origin: {origin_n:,} px")
            self._roi_lbl.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")
        elif roi_n:
            self._roi_lbl.setText(
                f"ROI: {roi_n:,} px  ·  draw amber strain-origin line")
            self._roi_lbl.setStyleSheet("color:#c2954f; font-size:11px;")
        else:
            self._roi_lbl.setText("Draw the analysis ROI first")
            self._roi_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        self._next_btn.setEnabled(roi_n > 0 and origin_n > 0)

    def on_enter(self) -> None:
        """Called by wizard when this page becomes visible."""
        self._sync_dynamic_combo()
        analysis = self._wizard.analysis
        self._strain_start_spin.blockSignals(True)
        self._strain_start_spin.setRange(0, len(analysis.def_paths))
        start = min(max(0, int(getattr(analysis, "strain_start_frame", 0))),
                    len(analysis.def_paths))
        analysis.strain_start_frame = start
        self._strain_start_spin.setValue(start)
        self._strain_start_spin.blockSignals(False)
        img = analysis.frame_image(start)
        if img is not None:
            # Only set the image if it's new so we don't reset the user's pan/zoom
            if self._canvas._image_arr is not img:
                self._canvas.set_image(img)
                self._canvas.zoom_fit()
                # Masks from a different frame are not states this image could
                # be undone to.
                self._canvas.clear_undo_history()

        # Restore ROI Mask & update labels
        mask = (analysis.strain_origin_mask if self._editing_origin
                else analysis.roi_mask)
        self._canvas.set_roi_role(self._editing_origin)
        self._canvas.seed_enabled = not self._editing_origin
        self._seed_btn.setEnabled(not self._editing_origin)
        for tool in (ROITool.RECTANGLE, ROITool.CIRCLE, ROITool.POLYGON):
            self._tool_btns[tool].setVisible(not self._editing_origin)
        self._tool_btns[ROITool.POLYLINE].setVisible(self._editing_origin)
        self._canvas.set_context_mask(
            analysis.roi_mask if self._editing_origin else None)
        self._canvas.set_draw_constraint(
            analysis.roi_mask if self._editing_origin else None)
        default_tool = (ROITool.POLYLINE if self._editing_origin
                        else ROITool.RECTANGLE)
        self._tool_btns[default_tool].setChecked(True)
        self._canvas.set_roi_tool(default_tool)
        if mask is not None:
            self._canvas.set_roi_mask(mask)
        else:
            # Do not leave the previous video's outline painted over a new
            # reference when the model correctly has no ROI yet.
            self._canvas.clear_roi()
        self._sync_strain_drawing_actions()
        self._refresh_mask_status()

        # Restore Seed & update labels
        if getattr(self._wizard, "seed_xy", None) is not None:
            sx, sy = self._wizard.seed_xy
            self._canvas.set_seed_xy((sx, sy))
            self._seed_status.setText(f"Seed: ({sx}, {sy})")
            self._seed_status.setStyleSheet("color:#6a9c74; font-size:11px;")
        else:
            self._seed_status.setText("No seed. Defaults to the ROI centroid.")
            self._seed_status.setStyleSheet("color:#94a3b8; font-size:11px;")

        # Hide the subset radius square/circle on the ROI page
        if hasattr(self._canvas, "set_subset_radius"):
            self._canvas.set_subset_radius(None)

    def _clear_active_mask(self) -> None:
        """Clear only the currently selected analysis/origin channel."""
        self._canvas.clear_roi()
        if self._editing_origin:
            self._wizard.analysis.clear_strain_origin()
        else:
            self._wizard.analysis.clear_roi()
            self._wizard.seed_xy = None
            self._seed_status.setText("No seed. Defaults to the ROI centroid.")
            self._seed_status.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        self._refresh_mask_status()

    def _on_roi_changed(self, mask: np.ndarray) -> None:
        # Defence in depth for the crash above: an exception raised in a Qt
        # slot is not caught anywhere and takes the process with it, so this
        # entry point must not assume it was handed an array.
        if mask is None:
            img = self._canvas._image_arr
            if img is None:
                return
            mask = np.zeros(img.shape, dtype=bool)

        if self._editing_origin:
            self._wizard.analysis.set_strain_origin_mask(mask)
            clipped = self._wizard.analysis.strain_origin_mask
            if clipped is not None and not np.array_equal(clipped, mask):
                self._canvas.set_roi_mask(clipped)
            self._refresh_mask_status()
            return

        self._wizard.analysis.set_roi_mask(mask)
        self._refresh_mask_status()

        # Verify existing seed is still inside the newly edited ROI
        if getattr(self._wizard, 'seed_xy', None) is not None:
            sx, sy = self._wizard.seed_xy
            h, w = mask.shape
            # A seed outside the array is as invalid as one outside the mask,
            # and indexing it would raise rather than simply drop the seed.
            if not (0 <= sy < h and 0 <= sx < w) or not mask[sy, sx]:
                self._wizard.seed_xy = None
                self._canvas.set_seed_xy(None)
                self._seed_status.setText("No seed. Defaults to the ROI centroid.")
                self._seed_status.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")

    def _on_seed_placed(self, x: int, y: int) -> None:
        # Negative coordinates mean the seed was removed rather than placed --
        # undo emits that when the region containing the seed is restored away.
        if x < 0 or y < 0:
            self._wizard.seed_xy = None
            self._seed_status.setText("No seed. Defaults to the ROI centroid.")
            self._seed_status.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
            return
        self._wizard.seed_xy = (x, y)
        self._seed_status.setText(f"Seed: ({x}, {y})")
        self._seed_status.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")

    def _undo(self) -> None:
        """Step back one drawing action on the canvas."""
        if self._canvas.undo():
            self._refresh_mask_status()

    def _use_full(self) -> None:
        self._set_edit_channel(False)
        ana = self._wizard.analysis
        ana.set_full_roi()
        # Show full-image mask on canvas
        if ana.roi_mask is not None:
            self._canvas._roi_mask = ana.roi_mask.copy()
            self._canvas.update()
            self._roi_lbl.setText(f"ROI: {int(ana.roi_mask.sum()):,} px (full image)")
            self._roi_lbl.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")
            self._refresh_mask_status()

    def _load_roi_from_file(self) -> None:
        """Load a pre-defined ROI mask from a file (image, npy, or Ncorr MAT)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROI Mask",
            "",
            "ROI files (*.png *.tif *.tiff *.jpg *.bmp *.npy *.mat *.h5 *.hdf5);;"
            "Image files (*.png *.tif *.tiff *.jpg *.bmp);;"
            "NumPy (*.npy);;"
            "Ncorr MAT / HDF5 (*.mat *.h5 *.hdf5);;"
            "All Files (*)"
        )
        if not path:
            return
        try:
            ana = self._wizard.analysis
            if self._editing_origin:
                ana.set_strain_origin_from_file(path)
                mask = ana.strain_origin_mask
            else:
                ana.set_roi_from_file(path)
                mask = ana.roi_mask
            if mask is not None:
                self._canvas.set_roi_mask(mask)
                self._refresh_mask_status()
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not load ROI",
                f"Could not load ROI from:\n{path}\n\n{exc}"
            )
