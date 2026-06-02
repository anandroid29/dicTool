"""
welcome_page.py — Step 1: Import video or images
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, List, Optional
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QEvent
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QFrame, QSizePolicy,
    QGroupBox, QSpinBox, QRadioButton
)

if TYPE_CHECKING:
    from src.ui.wizard import Wizard
from src.ui.components import FooterButton

# color shortcuts
_C_SURFACE = "#0e1c2e"
_C_CARD    = "#132035"
_C_BORDER  = "#1e3a5a"
_C_ACCENT  = "#3b82f6"
_C_TEXT    = "#e2e8f0"
_C_TEXT2   = "#94a3b8"
_C_TEXT3   = "#475569"
_C_SUCCESS = "#10b981"


class _FocusFilter(QObject):
    """Event filter to auto-check a radio button when a widget is interacted with."""
    def __init__(self, target_radio: QRadioButton):
        super().__init__()
        self._target = target_radio

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn, QEvent.Type.Wheel):
            if not self._target.isChecked():
                self._target.setChecked(True)
        return super().eventFilter(obj, event)


class _ImportCard(QFrame):
    """Clickable card for either 'Import Video' or 'Load Images'."""

    clicked = pyqtSignal()

    def __init__(self, icon: str, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setFixedSize(220, 160)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._normal_style = (
            f"QFrame {{ background:{_C_CARD}; border:1px solid {_C_BORDER}; "
            f"border-radius:12px; }} "
        )
        self._hover_style = (
            f"QFrame {{ background:#1a2d47; border:2px solid {_C_ACCENT}; "
            f"border-radius:12px; }} "
        )
        self.setStyleSheet(self._normal_style)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 22, 20, 22)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ic = QLabel(icon)
        ic.setStyleSheet("font-size:34px; background:transparent; border:none;")
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ic)

        t = QLabel(title)
        t.setStyleSheet(f"color:{_C_TEXT}; font-size:14px; font-weight:700; background:transparent; border:none;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)

        s = QLabel(subtitle)
        s.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px; background:transparent; border:none;")
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.setWordWrap(True)
        lay.addWidget(s)

    def enterEvent(self, e):
        self.setStyleSheet(self._hover_style)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.setStyleSheet(self._normal_style)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


class WelcomePage(QWidget):
    """Step 1 — import video or images."""

    ready = pyqtSignal()       # emitted when ref + at least 1 deformed loaded

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Hero section ──────────────────────────────────────────────
        hero = QWidget()
        hero.setStyleSheet(f"background:#0e1c2e;")
        hero_lay = QVBoxLayout(hero)
        hero_lay.setContentsMargins(60, 40, 60, 30)
        hero_lay.setSpacing(6)

        logo = QLabel("PyDIC")
        logo.setStyleSheet(
            f"color:{_C_ACCENT}; font-size:38px; font-weight:800; letter-spacing:2px;"
        )
        hero_lay.addWidget(logo)

        tagline = QLabel("Digital Image Correlation  ·  Professional Analysis Suite")
        tagline.setStyleSheet(f"color:{_C_TEXT2}; font-size:14px;")
        hero_lay.addWidget(tagline)

        root.addWidget(hero)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
        root.addWidget(sep)

        # ── Body ──────────────────────────────────────────────────────
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(60, 30, 60, 30)
        body_lay.setSpacing(24)

        step_lbl = QLabel("Step 1  —  Import your footage")
        step_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:13px; font-weight:600;")
        body_lay.addWidget(step_lbl)

        # Cards row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(24)
        cards_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._card_video = _ImportCard(
            "🎬", "Import Video",
            "MP4 · AVI · MOV · MKV\nAuto-extract frames"
        )
        self._card_video.clicked.connect(self._import_video)
        cards_row.addWidget(self._card_video)

        self._card_images = _ImportCard(
            "🖼", "Load Images",
            "PNG · TIF · JPEG · BMP\nManual selection"
        )
        self._card_images.clicked.connect(self._load_images)
        cards_row.addWidget(self._card_images)

        self._card_hdf5 = _ImportCard(
            "🗄️", "Load Session",
            "HDF5 (.h5)\nRestore previous analysis"
        )
        self._card_hdf5.clicked.connect(self._load_hdf5)
        cards_row.addWidget(self._card_hdf5)

        cards_row.addStretch()
        body_lay.addLayout(cards_row)

        # ── Frame Sampling Settings ───────────────────────────────────
        self.frame_options_grp = QGroupBox("FRAME SAMPLING SETTINGS")
        self.frame_options_grp.setMinimumHeight(140)
        self.frame_options_grp.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.frame_options_grp.setStyleSheet(f"""
            QGroupBox {{ border: 1px solid {_C_BORDER}; border-radius: 8px; padding-top: 18px; font-weight: bold; color: {_C_TEXT3}; font-size: 11px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {_C_TEXT2}; }}
            QRadioButton {{ color: {_C_TEXT2}; font-size: 11px; }}
            QRadioButton::indicator {{ width: 12px; height: 12px; border-radius: 6px; border: 1px solid {_C_BORDER}; background: {_C_SURFACE}; }}
            QRadioButton::indicator:checked {{ background: {_C_ACCENT}; border: 1px solid {_C_ACCENT}; }}
        """)

        opts_layout = QVBoxLayout(self.frame_options_grp)
        opts_layout.setContentsMargins(20, 16, 20, 16)
        opts_layout.setSpacing(12)

        # Step Spinbox
        step_lay = QHBoxLayout()
        step_lay.addWidget(QLabel("Load every:"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setValue(1)
        self.step_spin.setStyleSheet(f"background:{_C_SURFACE}; color:{_C_TEXT}; border:1px solid {_C_BORDER}; padding:4px 8px; border-radius:4px;")
        step_lay.addWidget(self.step_spin)

        lbl1 = QLabel("frame(s)")
        lbl1.setStyleSheet(f"color:{_C_TEXT2};")
        step_lay.addWidget(lbl1)
        step_lay.addStretch()
        opts_layout.addLayout(step_lay)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
        opts_layout.addWidget(div)

        # Max Frames Radio Buttons
        self.radio_all_frames = QRadioButton("Load all available deformed frames")
        self.radio_all_frames.setChecked(True)
        opts_layout.addWidget(self.radio_all_frames)

        limit_lay = QHBoxLayout()
        self.radio_limit_frames = QRadioButton("Limit maximum deformed frames to:")
        limit_lay.addWidget(self.radio_limit_frames)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(1, 999999)
        self.max_frames_spin.setValue(30)
        self.max_frames_spin.setStyleSheet(f"background:{_C_SURFACE}; color:{_C_TEXT}; border:1px solid {_C_BORDER}; padding:4px 8px; border-radius:4px;")

        # CRITICAL FIX: Intercept mouse clicks, scrolling, and keyboard focus to auto-select the radio button
        self._spin_focus_filter = _FocusFilter(self.radio_limit_frames)
        self.max_frames_spin.installEventFilter(self._spin_focus_filter)

        limit_lay.addWidget(self.max_frames_spin)
        limit_lay.addStretch()
        opts_layout.addLayout(limit_lay)

        body_lay.addWidget(self.frame_options_grp)
        # ──────────────────────────────────────────────────────────────

        # Status area
        self._status_box = QFrame()
        self._status_box.setMinimumHeight(80)
        self._status_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._status_box.setStyleSheet(
            f"background:{_C_CARD}; border:1px solid {_C_BORDER}; border-radius:10px;"
        )
        self._status_box.setVisible(False)
        status_lay = QVBoxLayout(self._status_box)
        status_lay.setContentsMargins(20, 16, 20, 16)
        status_lay.setSpacing(6)

        self._status_ref = QLabel("")
        self._status_ref.setStyleSheet(f"color:{_C_TEXT}; font-size:12px; border:none;")
        status_lay.addWidget(self._status_ref)

        self._status_def = QLabel("")
        self._status_def.setStyleSheet(f"color:{_C_TEXT}; font-size:12px; border:none;")
        status_lay.addWidget(self._status_def)

        self._status_fps = QLabel("")
        self._status_fps.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px; border:none;")
        status_lay.addWidget(self._status_fps)

        body_lay.addWidget(self._status_box)
        body_lay.addStretch()

        root.addWidget(body, 1)

        # ── Footer nav ────────────────────────────────────────────────
        footer = QWidget()
        footer.setStyleSheet(f"background:#0e1c2e; border-top:1px solid {_C_BORDER};")
        footer_lay = QHBoxLayout(footer)
        footer_lay.setContentsMargins(60, 14, 60, 14)
        footer_lay.addStretch()

        self._next_btn = FooterButton("Define ROI  →")
        self._next_btn.setProperty("class", "accent")
        self._next_btn.setFixedHeight(38)
        self._next_btn.setMinimumWidth(160)
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self._wizard.go_roi)
        footer_lay.addWidget(self._next_btn)

        root.addWidget(footer)

    # ------------------------------------------------------------------
    def _import_video(self) -> None:
        from src.ui.video_importer import VideoImporterDialog
        dlg = VideoImporterDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted or not dlg.extracted_paths:
            return

        paths   = dlg.extracted_paths
        ref_idx = dlg.reference_index
        ref_path = paths[ref_idx]
        def_paths = [p for i, p in enumerate(paths) if i != ref_idx]

        # Apply Slicing Logic
        step = self.step_spin.value()
        def_paths = def_paths[::step]

        if self.radio_limit_frames.isChecked():
            max_f = self.max_frames_spin.value()
            def_paths = def_paths[:max_f]

        analysis = self._wizard.analysis

        original_fps = dlg._fps if hasattr(dlg, '_fps') else 1.0
        analysis.fps = original_fps / step

        analysis.set_reference(ref_path)
        analysis.clear_deformed()
        for p in def_paths:
            analysis.add_deformed(p)

        self._update_status(ref_path, def_paths, analysis.fps)

    def _load_images(self) -> None:
        analysis = self._wizard.analysis

        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        valid_exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
        img_files = []
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in valid_exts:
                img_files.append(os.path.join(folder, f))

        img_files.sort()

        if len(img_files) < 2:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Not Enough Images", "The folder must contain at least 2 images.")
            return

        step = self.step_spin.value()
        ref = img_files[0]
        defs = img_files[1:]

        defs = defs[::step]

        if self.radio_limit_frames.isChecked():
            max_f = self.max_frames_spin.value()
            defs = defs[:max_f]

        if len(defs) == 0:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Empty Selection", "Your sampling settings filtered out all deformed frames.")
            return

        original_fps = 1.0
        import json
        meta_path = os.path.join(folder, "dic_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    if "fps" in meta:
                        original_fps = float(meta["fps"])
            except Exception:
                pass

        analysis.fps = original_fps / step

        analysis.set_reference(ref)
        analysis.clear_deformed()
        for p in defs:
            analysis.add_deformed(p)

        self._update_status(ref, defs, analysis.fps)

    def _load_hdf5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load HDF5 Session", "", "HDF5 Files (*.h5 *.hdf5)")
        if not path:
            return
        try:
            self._wizard.analysis.load_hdf5(path)
            self._wizard.go_results()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load Error", f"Failed to load session:\n{e}")

    def _update_status(self, ref: str, defs: list, fps: float) -> None:
        self._status_ref.setText(
            f"✓  Reference: {os.path.basename(ref)}"
        )
        self._status_ref.setStyleSheet(f"color:{_C_SUCCESS}; font-size:12px; border:none;")
        self._status_def.setText(
            f"✓  {len(defs)} deformed frame{'s' if len(defs)!=1 else ''} loaded"
        )
        self._status_def.setStyleSheet(f"color:{_C_SUCCESS}; font-size:12px; border:none;")
        if fps > 1.0:
            self._status_fps.setText(f"   Effective sample rate: {fps:.2f} fps  ·  "
                                     f"Δt = {1000/fps:.1f} ms per frame")
        else:
            self._status_fps.setText("   No fps metadata — strain rate will use Δt = 1 s")
        self._status_box.setVisible(True)
        self._next_btn.setEnabled(len(defs) > 0)