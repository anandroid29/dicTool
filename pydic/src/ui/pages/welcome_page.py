"""
welcome_page.py — Step 1: Import video or images
"""
from __future__ import annotations
import os
import json
from typing import TYPE_CHECKING, List, Optional
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QObject, QEvent, QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QFrame, QSizePolicy,
    QGroupBox, QSpinBox, QRadioButton, QDialog,
    QDialogButtonBox, QComboBox, QMessageBox, QLineEdit,
    QDoubleSpinBox, QProgressDialog
)

from src.core.units import Calibration, LENGTH_UNIT_ORDER

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


class _HDF5LoadWorker(QObject):
    """Load a complete session without blocking Qt's GUI thread."""

    progress = pyqtSignal(int, str)
    loaded = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    @pyqtSlot()
    def run(self) -> None:
        try:
            from src.core.analysis import DICAnalysis
            analysis = DICAnalysis()
            analysis.load_hdf5(
                self.path,
                progress_cb=lambda frac, message: self.progress.emit(
                    int(round(frac * 100.0)), message))
            self.loaded.emit(analysis, self.path)
        except Exception as exc:
            self.failed.emit(str(exc))


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
    def __init__(self, image_files: List[str], folder: str,
                 fps_from_meta: Optional[float] = None,
                 initial_calibration: Optional[Calibration] = None,
                 parent=None):
        super().__init__(parent)
        self._folder = folder
        self._fps_from_meta = fps_from_meta
        self._initial_calibration = initial_calibration or Calibration()
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
        # Always editable. A detected value only pre-fills the box: metadata can
        # carry a container playback rate rather than the true capture rate, and
        # a locked-in wrong rate silently rescales every velocity and strain rate.
        fps_lay = QHBoxLayout()
        fps_lay.addWidget(QLabel("Camera frame rate:"))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.01, 10000000.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setValue(
            self._fps_from_meta if self._fps_from_meta is not None else 1.0)
        self.fps_spin.setStyleSheet(f"background:{_C_SURFACE}; color:{_C_TEXT}; border:1px solid {_C_BORDER}; padding:4px 8px; border-radius:4px;")
        self.fps_spin.setToolTip(
            "Sample rate of this already-extracted image sequence, in Hz.\n\n"
            "This is the number of analysed images per second, after any frame\n"
            "skipping performed during video extraction. PyDIC uses Δt = 1 / rate\n"
            "for velocity and strain-rate calculations.\n\n"
            "A value detected from dic_metadata.json is pre-filled but remains\n"
            "editable if the metadata is incorrect."
        )
        fps_lay.addWidget(self.fps_spin)

        lbl_hz = QLabel("Hz")
        lbl_hz.setStyleSheet(f"color:{_C_TEXT2};")
        fps_lay.addWidget(lbl_hz)

        if self._fps_from_meta is not None:
            det = QLabel(f"detected {self._fps_from_meta:.2f} Hz — override if wrong")
            det.setStyleSheet(f"color:{_C_SUCCESS}; font-size:11px;")
            fps_lay.addWidget(det)
        fps_lay.addStretch()
        lay.addLayout(fps_lay)

        # 2b. Spatial scale — the space counterpart to the frame rate above.
        # Without it displacements can only ever be reported in pixels, which is
        # rarely the unit anyone actually wants to publish.
        scale_lay = QHBoxLayout()
        scale_lay.addWidget(QLabel("Pixel size:"))
        self.px_size_spin = QDoubleSpinBox()
        self.px_size_spin.setRange(0.0, 1e9)
        self.px_size_spin.setDecimals(6)
        initial_unit = self._initial_calibration.display_unit
        self.px_size_spin.setValue(
            self._initial_calibration.pixel_size_in(initial_unit) or 0.0)
        self.px_size_spin.setSpecialValueText("— unknown —")
        self.px_size_spin.setToolTip(
            "Physical size of one pixel, so results read in real units.\n"
            "Leave at 0 to work in pixels; it can also be set later on the\n"
            "results page without re-running the analysis.")
        self.px_size_spin.setStyleSheet(
            f"background:{_C_SURFACE}; color:{_C_TEXT}; border:1px solid {_C_BORDER}; "
            f"padding:4px 8px; border-radius:4px;")
        scale_lay.addWidget(self.px_size_spin)

        self.px_unit_combo = QComboBox()
        self.px_unit_combo.addItems(LENGTH_UNIT_ORDER)
        self.px_unit_combo.setCurrentText(initial_unit)
        self.px_unit_combo.setStyleSheet(
            f"background:{_C_SURFACE}; color:{_C_TEXT}; border:1px solid {_C_BORDER}; "
            f"padding:4px 8px; border-radius:4px;")
        scale_lay.addWidget(self.px_unit_combo)

        per_lbl = QLabel("per pixel")
        per_lbl.setStyleSheet(f"color:{_C_TEXT2};")
        scale_lay.addWidget(per_lbl)
        scale_lay.addStretch()
        lay.addLayout(scale_lay)

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
        # Pre-filled from metadata when present, so this already carries the
        # detected rate unless the operator changed it.
        user_fps = self.fps_spin.value()
        return step, limit, roi_path if roi_path else None, user_fps

    def get_calibration(self):
        """Calibration chosen in this dialog (uncalibrated when pixel size is 0)."""
        val = float(self.px_size_spin.value())
        unit = self.px_unit_combo.currentText()
        return Calibration.from_pixel_size(val, unit) if val > 0 else Calibration(None, unit)


class WelcomePage(QWidget):
    ready = pyqtSignal()

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._hdf5_thread: Optional[QThread] = None
        self._hdf5_worker: Optional[_HDF5LoadWorker] = None
        self._hdf5_progress: Optional[QProgressDialog] = None
        self._build_ui()

    def _get_safe_start_dir(self, attr_name: str) -> str:
        """Guarantees the file dialog never defaults to the project 'src' root."""
        d = getattr(self._wizard.analysis, attr_name, "")
        if not d or not os.path.isdir(d):
            d = os.path.expanduser("~")
        return d

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

        start_dir = self._get_safe_start_dir("last_video_directory")
        dlg = VideoImporterDialog(
            self, start_dir=start_dir,
            initial_calibration=self._wizard.analysis.calibration)

        if dlg.exec() == 0 or not dlg.extracted_paths:
            return

        # CRITICAL FIX: Save the source video folder, not the extracted frames folder!
        if hasattr(dlg, 'video_path') and dlg.video_path:
            self._wizard.analysis.last_video_directory = os.path.dirname(dlg.video_path)
            self._wizard.analysis.save_settings()

        paths = dlg.extracted_paths
        ref_idx = dlg.reference_index
        ref_path = paths[ref_idx]

        # Immediate-frame DIC requires a chronological chain beginning at the
        # selected reference. Frames before it cannot be inserted ahead of it,
        # and joining the frame before and after it after removing only the
        # reference creates a false, non-adjacent interval.
        def_paths = paths[ref_idx + 1:]
        if not def_paths:
            QMessageBox.warning(
                self, "Reference has no following frames",
                "Choose a reference frame earlier in the extracted sequence. "
                "Updated-Lagrangian analysis proceeds forward from it.")
            return
        analysis = self._wizard.analysis
        analysis.calibration = dlg.get_calibration()
        analysis.save_settings()

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
        self._wizard.seed_xy = None
        for p in def_paths:
            analysis.add_deformed(p)

        self._update_status(ref_path, def_paths, analysis.fps)

    def _load_images(self) -> None:
        start_dir = self._get_safe_start_dir("last_image_directory")
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", start_dir)
        if not folder:
            return

        self._wizard.analysis.last_image_directory = folder
        self._wizard.analysis.save_settings()

        valid_exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
        img_files = []
        roi_auto_path = None  # Track an auto-detected ROI

        try:
            for f in os.listdir(folder):
                ext = os.path.splitext(f)[1].lower()
                full_path = os.path.join(folder, f)

                # Logic: If it contains 'roi' in name and is an image, auto-select it
                if "roi" in f.lower() and ext in valid_exts:
                    roi_auto_path = full_path
                elif ext in valid_exts:
                    img_files.append(full_path)
        except Exception as e:
            QMessageBox.critical(self, "Folder Error", f"Could not read folder:\n{e}")
            return

        img_files.sort()

        if len(img_files) < 2:
            QMessageBox.warning(self, "Not Enough Images", "The folder must contain at least 2 images.")
            return

        # Attempt to read metadata
        original_fps = None
        metadata_calibration = None
        meta_path = os.path.join(folder, "dic_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    if "fps" in meta:
                        original_fps = float(meta["fps"])
                    if "calibration" in meta:
                        metadata_calibration = Calibration.from_dict(meta["calibration"])
            except Exception:
                pass

        try:
            # We pass the auto-detected path to the dialog so it's pre-filled
            dlg = ImageLoadSettingsDialog(
                img_files, folder, original_fps,
                initial_calibration=(metadata_calibration or
                                     self._wizard.analysis.calibration),
                parent=self)
            if roi_auto_path:
                dlg.roi_edit.setText(roi_auto_path)

            if dlg.exec() == 0:
                return
            step, limit, roi_path, user_fps = dlg.get_settings()
            self._wizard.analysis.calibration = dlg.get_calibration()
            # Persist immediately. Setting it here and relying on some later
            # save meant a pixel size entered at import was lost if the session
            # ended before anything else happened to write the settings.
            self._wizard.analysis.save_settings()
        except Exception as e:
            QMessageBox.critical(self, "Dialog Error", f"Failed to open settings dialog:\n{e}")
            return

        if roi_path:
            roi_norm = os.path.normpath(roi_path)
            img_files = [f for f in img_files if os.path.normpath(f) != roi_norm]

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

        # The dialog's spin box is seeded from metadata, so it is the detected
        # rate unless the operator overrode it -- preferring original_fps here
        # would throw that override away.
        base_fps = user_fps if user_fps is not None else (
            original_fps if original_fps is not None else 1.0)

        analysis = self._wizard.analysis
        analysis.fps = base_fps / step

        analysis.set_reference(ref)
        analysis.clear_deformed()
        self._wizard.seed_xy = None
        for p in defs:
            analysis.add_deformed(p)

        if roi_path:
            try:
                analysis.set_roi_from_file(roi_path)
            except Exception as e:
                QMessageBox.warning(self, "ROI Load Error", f"Could not load selected ROI mask:\n{e}")

        self._update_status(ref, defs, analysis.fps)

    def _load_hdf5(self) -> None:
        if self._hdf5_thread is not None and self._hdf5_thread.isRunning():
            return
        start_dir = self._get_safe_start_dir("last_hdf5_directory")
        path, _ = QFileDialog.getOpenFileName(self, "Load HDF5 Session", start_dir, "HDF5 Files (*.h5 *.hdf5)")
        if not path:
            return

        progress = QProgressDialog(
            f"Opening {os.path.basename(path)}…", "", 0, 100, self)
        progress.setWindowTitle("Loading HDF5 session")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setCancelButton(None)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.setMinimumWidth(430)
        progress.show()

        thread = QThread(self)
        worker = _HDF5LoadWorker(path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_hdf5_progress)
        worker.loaded.connect(self._on_hdf5_loaded)
        worker.failed.connect(self._on_hdf5_failed)
        worker.loaded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.loaded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_hdf5_loader)

        self._hdf5_progress = progress
        self._hdf5_thread = thread
        self._hdf5_worker = worker
        thread.start()

    def _on_hdf5_progress(self, value: int, message: str) -> None:
        if self._hdf5_progress is not None:
            self._hdf5_progress.setLabelText(message)
            self._hdf5_progress.setValue(max(0, min(100, int(value))))

    def _close_hdf5_progress(self) -> None:
        if self._hdf5_progress is not None:
            self._hdf5_progress.setValue(100)
            self._hdf5_progress.close()
            self._hdf5_progress.deleteLater()
            self._hdf5_progress = None

    def _on_hdf5_loaded(self, analysis, path: str) -> None:
        self._close_hdf5_progress()
        analysis.last_hdf5_directory = os.path.dirname(path)
        try:
            analysis.save_settings()
        except Exception:
            pass

        # Swap only after the new session is complete. If loading fails, the
        # currently open analysis remains untouched and usable.
        self._wizard.analysis = analysis
        self._wizard.seed_xy = None
        self._wizard.use_gpu = bool(getattr(analysis, "prefer_gpu", True))
        self._wizard.go_results()

    def _on_hdf5_failed(self, message: str) -> None:
        self._close_hdf5_progress()
        QMessageBox.critical(
            self, "Load Error", f"Failed to load session:\n{message}")

    def _clear_hdf5_loader(self) -> None:
        self._hdf5_worker = None
        self._hdf5_thread = None

    def _update_status(self, ref: str, defs: list, fps: float) -> None:
        self._status_ref.setText(f"Reference: {os.path.basename(ref)}")
        self._status_ref.setStyleSheet(f"color:{_C_SUCCESS}; font-size:12px; border:none;")
        self._status_def.setText(f"{len(defs)} deformed frame{'s' if len(defs)!=1 else ''} loaded")
        self._status_def.setStyleSheet(f"color:{_C_SUCCESS}; font-size:12px; border:none;")
        if fps > 1.0:
            time_text = (f"Effective sample rate: {fps:.2f} fps  ·  "
                         f"Δt = {1000/fps:.1f} ms per frame")
        else:
            time_text = "No fps metadata — strain rate will use Δt = 1 s"
        scale_text = self._wizard.analysis.calibration.describe()
        self._status_fps.setText(f"   {time_text}  ·  {scale_text}")
        self._status_box.setVisible(True)
        self._next_btn.setEnabled(len(defs) > 0)
