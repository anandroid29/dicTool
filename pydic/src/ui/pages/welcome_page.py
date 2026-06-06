"""
welcome_page.py — Step 1: Import video or images
"""
from __future__ import annotations
import os
import json
from typing import TYPE_CHECKING, List, Optional
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QFrame, QSizePolicy,
    QGroupBox, QSpinBox, QRadioButton, QDialog,
    QDialogButtonBox, QComboBox, QMessageBox, QLineEdit,
    QDoubleSpinBox
)

if TYPE_CHECKING:
    from src.ui.wizard import Wizard
from src.ui.components import FooterButton

# color shortcuts
_C_BG      = "#08111d"
_C_SURFACE = "#0e1c2e"
_C_CARD    = "#132035"
_C_BORDER  = "#1e3a5a"
_C_ACCENT  = "#3b82f6"
_C_TEXT    = "#e2e8f0"
_C_TEXT2   = "#94a3b8"
_C_TEXT3   = "#475569"
_C_SUCCESS = "#10b981"


class _FocusFilter(QObject):
    def __init__(self, target_radio: QRadioButton, parent=None):
        super().__init__(parent)
        self._target = target_radio

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        try:
            if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn, QEvent.Type.Wheel):
                if not self._target.isChecked():
                    self._target.setChecked(True)
        except Exception:
            pass
        return False


class _ImportCard(QFrame):
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


class ImageLoadSettingsDialog(QDialog):
    def __init__(self, image_files: List[str], folder: str, fps_from_meta: Optional[float] = None, parent=None):
        super().__init__(parent)
        self._folder = folder
        self._fps_from_meta = fps_from_meta
        self.setWindowTitle("Image Loading Settings")
        self.setMinimumWidth(500)
        self.setStyleSheet(f"""
            QDialog {{ background:{_C_BG}; }}
            QLabel {{ color:{_C_TEXT}; font-size:12px; }}
            QRadioButton {{ color:{_C_TEXT}; font-size:12px; spacing: 8px; }}
            QRadioButton::indicator {{ width:16px; height:16px; border-radius:9px; border:2px solid {_C_BORDER}; background:{_C_SURFACE}; }}
            QRadioButton::indicator:checked {{ background:{_C_ACCENT}; border:3px solid #ffffff; }}
            QSpinBox {{ background:{_C_SURFACE}; color:{_C_TEXT}; border:1px solid {_C_BORDER}; padding:4px 8px; border-radius:4px; }}
            QLineEdit {{ background:{_C_SURFACE}; color:{_C_TEXT}; border:1px solid {_C_BORDER}; padding:6px 10px; border-radius:4px; }}
            QPushButton {{ background:{_C_SURFACE}; color:{_C_TEXT}; border:1px solid {_C_BORDER}; padding:6px 12px; border-radius:4px; }}
            QPushButton:hover {{ background:{_C_BORDER}; border:1px solid {_C_ACCENT}; }}
        """)

        lay = QVBoxLayout(self)
        lay.setSpacing(16)

        # 1. ROI Selection
        roi_lay = QVBoxLayout()
        roi_lay.addWidget(QLabel("Select an image to use as the ROI Mask (Optional):"))

        roi_row = QHBoxLayout()
        self.roi_edit = QLineEdit()
        self.roi_edit.setPlaceholderText("No ROI selected (Draw manually later)")
        self.roi_edit.setReadOnly(True)
        roi_row.addWidget(self.roi_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_roi)
        roi_row.addWidget(browse_btn)

        clear_btn = QPushButton("✕")
        clear_btn.setFixedWidth(32)
        clear_btn.clicked.connect(lambda: self.roi_edit.clear())
        roi_row.addWidget(clear_btn)

        roi_lay.addLayout(roi_row)
        lay.addLayout(roi_lay)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
        lay.addWidget(div)

        # 2. Camera FPS Rate
        fps_lay = QHBoxLayout()
        if self._fps_from_meta is not None:
            lbl = QLabel(f"Camera frame rate: {self._fps_from_meta:.2f} Hz (detected from metadata)")
            lbl.setStyleSheet(f"color:{_C_SUCCESS}; font-size:12px; font-weight:bold;")
            fps_lay.addWidget(lbl)
            fps_lay.addStretch()
            self.fps_spin = None
        else:
            fps_lay.addWidget(QLabel("Camera frame rate:"))
            self.fps_spin = QDoubleSpinBox()
            self.fps_spin.setRange(0.01, 1000000.0)
            self.fps_spin.setValue(1.0)
            self.fps_spin.setDecimals(2)
            self.fps_spin.setStyleSheet(f"background:{_C_SURFACE}; color:{_C_TEXT}; border:1px solid {_C_BORDER}; padding:4px 8px; border-radius:4px;")
            fps_lay.addWidget(self.fps_spin)

            lbl_hz = QLabel("Hz")
            lbl_hz.setStyleSheet(f"color:{_C_TEXT2};")
            fps_lay.addWidget(lbl_hz)
            fps_lay.addStretch()
        lay.addLayout(fps_lay)

        # Divider
        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
        lay.addWidget(div2)

        # 3. Step Spinbox
        step_lay = QHBoxLayout()
        step_lay.addWidget(QLabel("Load every:"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setValue(1)
        step_lay.addWidget(self.step_spin)
        lbl1 = QLabel("frame(s)")
        lbl1.setStyleSheet(f"color:{_C_TEXT2};")
        step_lay.addWidget(lbl1)
        step_lay.addStretch()
        lay.addLayout(step_lay)

        # 4. Max Frames Radio Buttons
        self.radio_all_frames = QRadioButton("Load all available deformed frames")
        self.radio_all_frames.setChecked(True)
        lay.addWidget(self.radio_all_frames)

        limit_lay = QHBoxLayout()
        self.radio_limit_frames = QRadioButton("Limit maximum deformed frames to:")
        limit_lay.addWidget(self.radio_limit_frames)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(1, 999999)
        self.max_frames_spin.setValue(30)
        self._spin_focus_filter = _FocusFilter(self.radio_limit_frames, parent=self)
        self.max_frames_spin.installEventFilter(self._spin_focus_filter)

        limit_lay.addWidget(self.max_frames_spin)
        limit_lay.addStretch()
        lay.addLayout(limit_lay)

        # Buttons
        btns = QDialogButtonBox()
        btns.setStandardButtons(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _browse_roi(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ROI Mask Image", self._folder,
            "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if path:
            self.roi_edit.setText(path)

    def get_settings(self):
        step = self.step_spin.value()
        limit = self.max_frames_spin.value() if self.radio_limit_frames.isChecked() else None
        roi_path = self.roi_edit.text().strip()
        user_fps = self.fps_spin.value() if self.fps_spin else None
        return step, limit, roi_path if roi_path else None, user_fps


class WelcomePage(QWidget):
    ready = pyqtSignal()

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

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

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(60, 30, 60, 30)
        body_lay.setSpacing(24)

        step_lbl = QLabel("Step 1  —  Import your footage")
        step_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:13px; font-weight:600;")
        body_lay.addWidget(step_lbl)

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

    def _import_video(self) -> None:
        from src.ui.video_importer import VideoImporterDialog
        dlg = VideoImporterDialog(self)

        if dlg.exec() == 0 or not dlg.extracted_paths:
            return

        paths   = dlg.extracted_paths
        ref_idx = dlg.reference_index
        ref_path = paths[ref_idx]

        def_paths = [p for i, p in enumerate(paths) if i != ref_idx]
        analysis = self._wizard.analysis

        original_fps = 1.0
        out_dir = os.path.dirname(paths[0])
        meta_path = os.path.join(out_dir, "dic_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    if "fps" in meta:
                        original_fps = float(meta["fps"])
            except Exception:
                pass

        analysis.fps = original_fps

        analysis.set_reference(ref_path)
        analysis.clear_deformed()
        for p in def_paths:
            analysis.add_deformed(p)

        self._update_status(ref_path, def_paths, analysis.fps)

    def _load_images(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        valid_exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
        img_files = []
        try:
            for f in os.listdir(folder):
                if os.path.splitext(f)[1].lower() in valid_exts:
                    img_files.append(os.path.join(folder, f))
        except Exception as e:
            QMessageBox.critical(self, "Folder Error", f"Could not read folder:\n{e}")
            return

        img_files.sort()

        if len(img_files) < 2:
            QMessageBox.warning(self, "Not Enough Images", "The folder must contain at least 2 images.")
            return

        # Attempt to read existing metadata before showing the dialog
        original_fps = None
        meta_path = os.path.join(folder, "dic_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    if "fps" in meta:
                        original_fps = float(meta["fps"])
            except Exception:
                pass

        try:
            dlg = ImageLoadSettingsDialog(img_files, folder, original_fps, self)
            if dlg.exec() == 0:
                return
            step, limit, roi_path, user_fps = dlg.get_settings()
        except Exception as e:
            QMessageBox.critical(self, "Dialog Error", f"Failed to open settings dialog:\n{e}")
            return

        if roi_path and roi_path in img_files:
            img_files.remove(roi_path)

        if len(img_files) < 2:
            QMessageBox.warning(self, "Invalid Selection", "Not enough images remaining after excluding the ROI mask.")
            return

        ref = img_files[0]
        defs = img_files[1:]

        defs = defs[::step]

        if limit is not None:
            defs = defs[:limit]

        if len(defs) == 0:
            QMessageBox.warning(self, "Empty Selection", "Your sampling settings filtered out all deformed frames.")
            return

        # Determine the final base FPS
        base_fps = original_fps if original_fps is not None else (user_fps if user_fps is not None else 1.0)

        analysis = self._wizard.analysis
        analysis.fps = base_fps / step

        analysis.set_reference(ref)
        analysis.clear_deformed()
        for p in defs:
            analysis.add_deformed(p)

        if roi_path:
            try:
                analysis.set_roi_from_file(roi_path)
            except Exception as e:
                QMessageBox.warning(self, "ROI Load Error", f"Could not load selected ROI mask:\n{e}")

        self._update_status(ref, defs, analysis.fps)

    def _load_hdf5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load HDF5 Session", "", "HDF5 Files (*.h5 *.hdf5)")
        if not path:
            return
        try:
            self._wizard.analysis.load_hdf5(path)
            self._wizard.go_results()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load session:\n{e}")

    def _update_status(self, ref: str, defs: list, fps: float) -> None:
        self._status_ref.setText(f"✓  Reference: {os.path.basename(ref)}")
        self._status_ref.setStyleSheet(f"color:{_C_SUCCESS}; font-size:12px; border:none;")
        self._status_def.setText(f"✓  {len(defs)} deformed frame{'s' if len(defs)!=1 else ''} loaded")
        self._status_def.setStyleSheet(f"color:{_C_SUCCESS}; font-size:12px; border:none;")
        if fps > 1.0:
            self._status_fps.setText(f"   Effective sample rate: {fps:.2f} fps  ·  Δt = {1000/fps:.1f} ms per frame")
        else:
            self._status_fps.setText("   No fps metadata — strain rate will use Δt = 1 s")
        self._status_box.setVisible(True)
        self._next_btn.setEnabled(len(defs) > 0)