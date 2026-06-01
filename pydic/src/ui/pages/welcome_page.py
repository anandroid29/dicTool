"""
welcome_page.py — Step 1: Import video or images
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, List, Optional
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QFrame, QSizePolicy,
)

if TYPE_CHECKING:
    from src.ui.wizard import Wizard


# color shortcuts
_C_CARD    = "#132035"
_C_BORDER  = "#1e3a5a"
_C_ACCENT  = "#3b82f6"
_C_TEXT    = "#e2e8f0"
_C_TEXT2   = "#94a3b8"
_C_SUCCESS = "#10b981"


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
        hero_lay.setContentsMargins(60, 60, 60, 50)
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
        body_lay.setContentsMargins(60, 48, 60, 48)
        body_lay.setSpacing(32)

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

        cards_row.addStretch()
        body_lay.addLayout(cards_row)

        # Status area
        self._status_box = QFrame()
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

        self._next_btn = QPushButton("Define ROI  →")
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

        analysis = self._wizard.analysis
        analysis.fps = dlg._fps if hasattr(dlg, '_fps') else 1.0
        analysis.set_reference(ref_path)
        analysis.clear_deformed()
        for p in def_paths:
            analysis.add_deformed(p)

        self._update_status(ref_path, def_paths, analysis.fps)

    def _load_images(self) -> None:
        analysis = self._wizard.analysis

        # 1. Ask for a folder
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        # 2. Gather and sort image files
        valid_exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
        img_files = []
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in valid_exts:
                img_files.append(os.path.join(folder, f))

        img_files.sort()  # Lexicographical sort

        if len(img_files) < 2:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Not Enough Images", "The folder must contain at least 2 images.")
            return

        # 3. Assign reference and deformed
        ref = img_files[0]
        defs = img_files[1:]

        analysis.set_reference(ref)
        analysis.clear_deformed()
        for p in defs:
            analysis.add_deformed(p)

        # 4. Look for invisible metadata file to restore FPS
        import json
        meta_path = os.path.join(folder, "dic_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    if "fps" in meta:
                        analysis.fps = float(meta["fps"])
            except Exception:
                pass

        self._update_status(ref, defs, analysis.fps)
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
            self._status_fps.setText(f"   Video: {fps:.2f} fps  ·  "
                                     f"Δt = {1000/fps:.1f} ms per frame")
        else:
            self._status_fps.setText("   No fps metadata — strain rate will use Δt = 1 s")
        self._status_box.setVisible(True)
        self._next_btn.setEnabled(len(defs) > 0)
