"""
welcome_page.py — Step 1: Import video or images
"""
from __future__ import annotations
import os
import json
from typing import TYPE_CHECKING, Optional
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QObject, QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QFrame, QSizePolicy,
    QMessageBox, QProgressDialog,
)

from src.core.units import Calibration

if TYPE_CHECKING:
    from src.ui.wizard import Wizard
from src.ui.components import FooterButton
from src.ui.image_importer import ImageSequenceImporterDialog

# Palette comes from the single source of truth in theme.py. These were
# duplicated literals, which is why re-theming previously left pages behind.
from src.ui.theme import C_ACCENT, C_BG, C_BORDER, C_CARD, C_SUCCESS, C_SURFACE, C_TEXT, C_TEXT2, C_TEXT3

_C_ACCENT = C_ACCENT
_C_BG = C_BG
_C_BORDER = C_BORDER
_C_CARD = C_CARD
_C_SUCCESS = C_SUCCESS
_C_SURFACE = C_SURFACE
_C_TEXT = C_TEXT
_C_TEXT2 = C_TEXT2
_C_TEXT3 = C_TEXT3


# color shortcuts


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


class _ImportCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, source_type: str, title: str, subtitle: str,
                 action: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setMinimumSize(230, 176)
        self.setMaximumWidth(310)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._normal_style = (
            f"QFrame {{ background:{_C_CARD}; border:1px solid {_C_BORDER}; "
            f"border-radius:3px; }}"
        )
        self._hover_style = (
            f"QFrame {{ background:#31353a; border:1px solid {_C_ACCENT}; "
            f"border-radius:3px; }}"
        )
        self.setStyleSheet(self._normal_style)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(9)

        kind = QLabel(source_type.upper())
        kind.setStyleSheet(
            f"color:#74a6c9; background:#282b2e; border:1px solid #3c4247; "
            "border-radius:3px; padding:3px 8px; font-size:9px; "
            "font-weight:700; letter-spacing:1px;")
        kind.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        lay.addWidget(kind, 0, Qt.AlignmentFlag.AlignLeft)

        t = QLabel(title)
        t.setStyleSheet(
            f"color:{_C_TEXT}; font-size:17px; font-weight:700; "
            "background:transparent; border:none;")
        lay.addWidget(t)

        s = QLabel(subtitle)
        s.setStyleSheet(
            f"color:{_C_TEXT2}; font-size:11px; line-height:1.35; "
            "background:transparent; border:none;")
        s.setWordWrap(True)
        lay.addWidget(s)

        lay.addStretch()
        action_label = QLabel(action.upper())
        action_label.setStyleSheet(
            f"color:{_C_ACCENT}; font-size:10px; font-weight:700; "
            "letter-spacing:0.8px; background:transparent; border:none;")
        lay.addWidget(action_label)

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


# Compatibility name retained for integrations that imported the old dialog.
ImageLoadSettingsDialog = ImageSequenceImporterDialog


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
        self.setStyleSheet(f"background:{_C_BG};")

        hero = QWidget()
        hero.setObjectName("welcomeHero")
        hero.setStyleSheet(
            f"QWidget#welcomeHero {{ background:{_C_SURFACE}; }} "
            "QWidget#welcomeHero QLabel { background:transparent; }")
        hero_lay = QVBoxLayout(hero)
        hero_lay.setContentsMargins(64, 30, 64, 28)
        hero_lay.setSpacing(8)

        product = QLabel("PYDIC  /  MEASUREMENT WORKSPACE")
        product.setStyleSheet(
            f"color:#74a6c9; font-size:10px; font-weight:700; "
            "letter-spacing:1.8px;")
        hero_lay.addWidget(product)

        logo = QLabel("Measure motion. Quantify deformation.")
        logo.setStyleSheet(
            f"color:{_C_TEXT}; font-size:30px; font-weight:750;"
        )
        hero_lay.addWidget(logo)

        tagline = QLabel(
            "Digital image correlation for displacement, velocity, strain "
            "rate and accumulated strain.")
        tagline.setStyleSheet(f"color:{_C_TEXT2}; font-size:13px;")
        tagline.setWordWrap(True)
        hero_lay.addWidget(tagline)

        capabilities = QHBoxLayout()
        capabilities.setSpacing(8)
        for text in ("DISPLACEMENT", "VELOCITY", "STRAIN RATE", "STRAIN"):
            chip = QLabel(text)
            chip.setStyleSheet(
                "color:#a2a8ad; background:#282b2e; border:1px solid #3c4247; "
                "border-radius:3px; padding:4px 9px; font-size:9px; "
                "font-weight:650; letter-spacing:0.6px;")
            chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            capabilities.addWidget(chip)
        capabilities.addStretch()
        hero_lay.addLayout(capabilities)
        root.addWidget(hero)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
        root.addWidget(sep)

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(64, 28, 64, 28)
        body_lay.setSpacing(18)

        section_title = QLabel("Choose a data source")
        section_title.setStyleSheet(
            f"color:{_C_TEXT}; font-size:18px; font-weight:700;")
        body_lay.addWidget(section_title)
        section_copy = QLabel(
            "Start from a recording, an image sequence, or a saved session.")
        section_copy.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        body_lay.addWidget(section_copy)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)

        self._card_video = _ImportCard(
            "Recording", "Import video",
            "Select an MP4, AVI, MOV, or MKV recording and extract the frames "
            "needed for analysis.", "Select video"
        )
        self._card_video.clicked.connect(self._import_video)
        cards_row.addWidget(self._card_video)

        self._card_images = _ImportCard(
            "Sequence", "Load image folder",
            "Use an existing PNG, TIFF, JPEG, or BMP sequence with explicit "
            "frame-rate and scale settings.", "Select folder"
        )
        self._card_images.clicked.connect(self._load_images)
        cards_row.addWidget(self._card_images)

        self._card_hdf5 = _ImportCard(
            "Session", "Resume analysis",
            "Open a saved HDF5 result set and return directly to inspection "
            "and export.", "Open HDF5"
        )
        self._card_hdf5.clicked.connect(self._load_hdf5)
        cards_row.addWidget(self._card_hdf5)

        body_lay.addLayout(cards_row)

        self._status_box = QFrame()
        self._status_box.setMinimumHeight(96)
        self._status_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._status_box.setStyleSheet(
            f"background:#0f2035; border:1px solid #245079; border-radius:3px;"
        )
        self._status_box.setVisible(False)
        status_lay = QVBoxLayout(self._status_box)
        status_lay.setContentsMargins(20, 14, 20, 14)
        status_lay.setSpacing(5)

        status_title = QLabel("DATASET READY")
        status_title.setStyleSheet(
            "color:#74a6c9; font-size:9px; font-weight:700; "
            "letter-spacing:1px; border:none;")
        status_lay.addWidget(status_title)

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
        footer.setStyleSheet(
            f"background:{_C_SURFACE}; border-top:1px solid {_C_BORDER};")
        footer_lay = QHBoxLayout(footer)
        footer_lay.setContentsMargins(60, 14, 60, 14)
        footer_lay.addStretch()

        self._next_btn = FooterButton("Continue to ROI")
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
                "Choose a reference frame earlier in the sequence. "
                "Analysis proceeds forward from it.")
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
            QMessageBox.critical(self, "Could not read folder", f"Could not read folder:\n{e}")
            return

        img_files.sort()

        if len(img_files) < 2:
            QMessageBox.warning(self, "Not enough images", "The folder must contain at least 2 images.")
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
            _step, _limit, roi_path, _source_fps = dlg.get_settings()
            selected_paths = dlg.selected_paths()
            effective_fps = dlg.effective_fps()
            self._wizard.analysis.calibration = dlg.get_calibration()
            # Persist immediately. Setting it here and relying on some later
            # save meant a pixel size entered at import was lost if the session
            # ended before anything else happened to write the settings.
            self._wizard.analysis.save_settings()
        except Exception as e:
            QMessageBox.critical(self, "Could not open settings", f"Failed to open settings dialog:\n{e}")
            return

        if len(selected_paths) < 2:
            QMessageBox.warning(
                self, "Invalid selection",
                "The selection must contain one reference image and at least "
                "one deformed image.")
            return

        # The importer returns exactly what its scrubber and playback preview
        # showed: the first path is the chosen reference, followed by sampled
        # deformed frames. No second sampling pipeline is applied here.
        ref = selected_paths[0]
        defs = selected_paths[1:]

        analysis = self._wizard.analysis
        analysis.fps = effective_fps

        analysis.set_reference(ref)
        analysis.clear_deformed()
        self._wizard.seed_xy = None
        for p in defs:
            analysis.add_deformed(p)

        if roi_path:
            try:
                analysis.set_roi_from_file(roi_path)
            except Exception as e:
                QMessageBox.warning(self, "Could not load ROI", f"Could not load selected ROI mask:\n{e}")

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
            self, "Could not load session", f"Failed to load session:\n{message}")

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
            time_text = "No frame-rate metadata. Strain rate will use \u0394t = 1 s"
        scale_text = self._wizard.analysis.calibration.describe()
        self._status_fps.setText(f"   {time_text}  ·  {scale_text}")
        self._status_box.setVisible(True)
        self._next_btn.setEnabled(len(defs) > 0)
