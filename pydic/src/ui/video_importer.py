"""
video_importer.py
=================
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import (
    Qt, QObject, QThread, pyqtSignal, pyqtSlot, QTimer,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QLineEdit,
    QProgressBar, QFileDialog, QGroupBox, QSizePolicy,
    QMessageBox, QFrame, QCheckBox, QComboBox,
)

from src.core.units import Calibration, LENGTH_UNIT_ORDER

# Defer CV2 import to prevent UI freeze on startup
if TYPE_CHECKING:
    import cv2


# ---------------------------------------------------------------------------
# Extraction worker (runs on a QThread)
# ---------------------------------------------------------------------------

class ExtractionWorker(QObject):
    progress  = pyqtSignal(int, int)   # frames_done, frames_total
    finished  = pyqtSignal(list)       # sorted list of output paths
    error     = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        out_dir: str,
        start: int,
        end: int,
        step: int,
        grayscale: bool,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._out_dir    = out_dir
        self._start      = start
        self._end        = end
        self._step       = step
        self._grayscale  = grayscale
        self._cancel     = False

    def cancel(self) -> None:
        self._cancel = True

    @pyqtSlot()
    def run(self) -> None:
        import cv2 # IMPORT CV2 HERE IN THE BACKGROUND THREAD

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self.error.emit(f"Cannot open video:\n{self._video_path}")
            return

        os.makedirs(self._out_dir, exist_ok=True)
        frame_indices = list(range(self._start, self._end + 1, self._step))
        total = len(frame_indices)
        paths: List[str] = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, self._start)
        current_frame = self._start
        done = 0

        for idx in frame_indices:
            if self._cancel:
                break
            # Seek if needed (step > 1)
            if current_frame != idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                current_frame = idx

            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1

            if self._grayscale:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            fname = os.path.join(self._out_dir, f"frame_{idx:06d}.png")
            cv2.imwrite(fname, frame if self._grayscale else
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            paths.append(fname)
            done += 1
            self.progress.emit(done, total)

        cap.release()
        if not self._cancel:
            self.finished.emit(sorted(paths))
        else:
            self.finished.emit([])


# ---------------------------------------------------------------------------
# Thin preview label
# ---------------------------------------------------------------------------

class _PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(240, 135)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background:#1a1c1e; border:1px solid #3c4247; border-radius:3px;"
            "color:#6b7378; font-size:11px;"
        )
        self.setText("No preview")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setFixedHeight(160)

    def show_frame(self, bgr: np.ndarray) -> None:
        import cv2 # IMPORT CV2 LOCALLY

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr.ndim == 3 else \
              cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        px = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(px)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

_BTN = (
    "QPushButton { background:#212426; color:#e6e8ea; border:1px solid #3c4247; "
    "border-radius:3px; font-size:11px; padding:4px 12px; } "
    "QPushButton:hover { background:#31353a; color:#e6e8ea; border-color:#a2a8ad; } "
    "QPushButton:pressed { background:#1a1c1e; }"
)
_BTN_ACCENT = (
    "QPushButton { background:#5a8cb0; color:#fff; border:none; "
    "border-radius:3px; font-size:12px; font-weight:700; padding:6px 16px; } "
    "QPushButton:hover { background:#74a6c9; } "
    "QPushButton:pressed { background:#41697f; } "
    "QPushButton:disabled { background:#212426; color:#6b7378; }"
)
_SPIN = (
    "QSpinBox, QDoubleSpinBox { background:#212426; color:#e6e8ea; "
    "border:1px solid #3c4247; "
    "border-radius:3px; padding:3px 6px; font-size:11px; } "
    "QSpinBox::up-button, QSpinBox::down-button, "
    "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button "
    "{ border:none; width:16px; }"
)
_LABEL_DIM = "color:#a2a8ad; font-size:11px;"
_LABEL_PRI = "color:#e6e8ea; font-size:11px;"


class VideoImporterDialog(QDialog):
    """
    Open a video, pick a frame range, and extract frames as PNG images.
    """

    def __init__(
        self,
        parent=None,
        initial_video: Optional[str] = None,
        start_dir: str = "", # ADDED START DIRECTORY TRACKING
        initial_calibration: Optional[Calibration] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import video frames")
        self.setMinimumWidth(520)
        self.setModal(True)
        self.setStyleSheet(
            "QDialog { background:#1a1c1e; } "
            "QGroupBox { color:#a2a8ad; font-size:10px; font-weight:700; "
            "  text-transform:uppercase; letter-spacing:0.5px; "
            "  border:1px solid #212426; border-radius:3px; "
            "  margin-top:8px; padding-top:12px; } "
            "QGroupBox::title { subcontrol-origin:margin; left:8px; } "
            "QLabel { color:#e6e8ea; } "
            "QLineEdit { background:#212426; color:#e6e8ea; border:1px solid #3c4247; "
            "  border-radius:3px; padding:4px 8px; font-size:11px; } "
            "QCheckBox { color:#a2a8ad; font-size:11px; } "
            "QCheckBox::indicator { width:13px; height:13px; border:1px solid #3c4247; "
            "  border-radius:3px; background:#212426; } "
            "QCheckBox::indicator:checked { background:#5a8cb0; border-color:#5a8cb0; } "
            "QSlider::groove:horizontal { background:#212426; height:5px; border-radius:3px; } "
            "QSlider::handle:horizontal { background:#5a8cb0; width:13px; height:13px; "
            "  margin:-4px 0; border-radius:3px; } "
            "QSlider::sub-page:horizontal { background:#41697f; border-radius:3px; } "
            "QProgressBar { background:#212426; border:1px solid #3c4247; border-radius:3px; "
            "  height:8px; text-align:center; } "
            "QProgressBar::chunk { background:#5a8cb0; border-radius:3px; }"
        )

        # State
        self.start_dir = start_dir if start_dir and os.path.isdir(start_dir) else os.path.expanduser("~")
        self._video_path: Optional[str] = None
        self._cap:        Optional['cv2.VideoCapture'] = None
        self._total_frames: int = 0
        # Playback rate stored in the container. For a high-speed camera this
        # is NOT the rate the frames were captured at -- a 2000 Hz recording is
        # routinely written out as a 50 fps file for normal-speed viewing.
        self._fps:          float = 25.0
        self.extracted_paths: List[str] = []
        self.reference_index: int = 0
        self._initial_calibration = initial_calibration or Calibration()
        self._auto_output_path = ""

        self._worker: Optional[ExtractionWorker] = None
        self._thread: Optional[QThread] = None

        self._build_ui()

        if initial_video:
            self._load_video(initial_video)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Title ──────────────────────────────────────────────────────
        title = QLabel("Import video frames")
        title.setStyleSheet("color:#e6e8ea; font-size:15px; font-weight:700;")
        root.addWidget(title)

        sub = QLabel("Extract frames from a video file for use as DIC images.")
        sub.setStyleSheet(_LABEL_DIM)
        sub.setWordWrap(True)
        root.addWidget(sub)

        root.addWidget(self._separator())

        # ── Video file ─────────────────────────────────────────────────
        vgrp = QGroupBox("Video file")
        vlay = QVBoxLayout(vgrp)
        vlay.setSpacing(6)

        file_row = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("No video selected…")
        self._file_edit.setReadOnly(True)
        file_row.addWidget(self._file_edit, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.setStyleSheet(_BTN)
        btn_browse.clicked.connect(self._browse_video)
        file_row.addWidget(btn_browse)
        vlay.addLayout(file_row)

        self._info_lbl = QLabel("")
        self._info_lbl.setStyleSheet(_LABEL_DIM)
        vlay.addWidget(self._info_lbl)

        # ── Source capture rate ────────────────────────────────────────
        # This is the acquisition rate before the extraction step. The actual
        # sample rate used by DIC is shown immediately below and is computed as
        # source rate / extraction step.
        cap_row = QHBoxLayout()
        cap_lbl = QLabel("Source capture rate:")
        cap_lbl.setStyleSheet(_LABEL_PRI)
        cap_row.addWidget(cap_lbl)
        self._capture_fps_spin = QDoubleSpinBox()
        self._capture_fps_spin.setRange(0.01, 10_000_000.0)
        self._capture_fps_spin.setDecimals(2)
        self._capture_fps_spin.setValue(25.0)
        self._capture_fps_spin.setStyleSheet(_SPIN)
        capture_tip = (
            "Frame rate of the source recording before frame extraction.\n\n"
            "For an ordinary video, leave the detected value unchanged. Override it\n"
            "only when the file metadata is wrong, or when high-speed footage was\n"
            "saved with a different playback rate.\n\n"
            "If Step is N, PyDIC analyses every Nth frame and uses:\n"
            "effective sample rate = source capture rate / N\n"
            "Δt = N / source capture rate.\n\n"
            "The effective rate and Δt are shown below. They control velocity and\n"
            "strain-rate calculations; this setting does not alter the video itself."
        )
        cap_lbl.setToolTip(capture_tip)
        self._capture_fps_spin.setToolTip(capture_tip)
        self._capture_fps_spin.valueChanged.connect(self._on_capture_rate_changed)
        cap_row.addWidget(self._capture_fps_spin)
        cap_unit = QLabel("Hz")
        cap_unit.setStyleSheet(_LABEL_DIM)
        cap_unit.setToolTip(capture_tip)
        cap_row.addWidget(cap_unit)
        cap_row.addStretch()
        vlay.addLayout(cap_row)

        self._capture_note = QLabel("")
        self._capture_note.setStyleSheet(_LABEL_DIM)
        self._capture_note.setWordWrap(True)
        vlay.addWidget(self._capture_note)

        # Spatial calibration belongs at extraction time beside temporal
        # calibration. It is written into dic_metadata.json and handed directly
        # back to the analysis, so video and image-sequence imports behave the
        # same way and reopening the extracted folder restores the mapping.
        scale_row = QHBoxLayout()
        scale_lbl = QLabel("Pixel-to-length mapping:")
        scale_lbl.setStyleSheet(_LABEL_PRI)
        scale_row.addWidget(scale_lbl)
        one_px = QLabel("1 px =")
        one_px.setStyleSheet(_LABEL_DIM)
        scale_row.addWidget(one_px)
        self._px_size_spin = QDoubleSpinBox()
        self._px_size_spin.setRange(0.0, 1e9)
        self._px_size_spin.setDecimals(6)
        self._px_size_spin.setSpecialValueText("— unknown —")
        self._px_size_spin.setStyleSheet(_SPIN)
        self._px_size_spin.setToolTip(
            "Physical length of one source-video pixel.\n"
            "Leave at 0 to report displacement in pixels.")
        scale_row.addWidget(self._px_size_spin)
        self._px_unit_combo = QComboBox()
        self._px_unit_combo.addItems(LENGTH_UNIT_ORDER)
        self._px_unit_combo.setStyleSheet(
            "QComboBox { background:#212426; color:#e6e8ea; "
            "border:1px solid #3c4247; border-radius:3px; padding:3px 6px; }")
        scale_row.addWidget(self._px_unit_combo)
        scale_row.addStretch()
        vlay.addLayout(scale_row)

        initial_unit = self._initial_calibration.display_unit
        self._px_unit_combo.setCurrentText(initial_unit)
        self._px_size_spin.setValue(
            self._initial_calibration.pixel_size_in(initial_unit) or 0.0)

        root.addWidget(vgrp)

        # ── Preview ────────────────────────────────────────────────────
        self._preview = _PreviewLabel()
        root.addWidget(self._preview)

        # Preview scrubber
        scrub_row = QHBoxLayout()
        scrub_lbl = QLabel("Preview frame:")
        scrub_lbl.setStyleSheet(_LABEL_DIM)
        scrub_row.addWidget(scrub_lbl)
        self._preview_slider = QSlider(Qt.Orientation.Horizontal)
        self._preview_slider.setEnabled(False)
        self._preview_slider.valueChanged.connect(self._on_preview_slider)
        scrub_row.addWidget(self._preview_slider, 1)
        self._preview_frame_lbl = QLabel("—")
        self._preview_frame_lbl.setStyleSheet(
            _LABEL_DIM + " font-family:'Fira Code','Consolas',monospace; min-width:60px;")
        scrub_row.addWidget(self._preview_frame_lbl)
        root.addLayout(scrub_row)

        # ── Frame range ────────────────────────────────────────────────
        rgrp = QGroupBox("Frame range")
        rgrid = QGridLayout(rgrp)
        rgrid.setSpacing(8)

        def spin(lo, hi, val, tip="") -> QSpinBox:
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setStyleSheet(_SPIN)
            if tip:
                s.setToolTip(tip)
            return s

        rgrid.addWidget(QLabel("Start frame:"), 0, 0)
        self._start_spin = spin(0, 0, 0, "First frame to extract (0-based)")
        rgrid.addWidget(self._start_spin, 0, 1)

        rgrid.addWidget(QLabel("End frame:"), 0, 2)
        self._end_spin = spin(0, 0, 0, "Last frame to extract (inclusive)")
        rgrid.addWidget(self._end_spin, 0, 3)

        rgrid.addWidget(QLabel("Step:"), 1, 0)
        self._step_spin = spin(1, 100, 1,
                               "Extract every Nth frame (1 = every frame)")
        self._step_spin.valueChanged.connect(self._update_count_label)
        rgrid.addWidget(self._step_spin, 1, 1)

        self._count_lbl = QLabel("— frames")
        self._count_lbl.setStyleSheet(_LABEL_DIM)
        rgrid.addWidget(self._count_lbl, 1, 2, 1, 2)

        self._start_spin.valueChanged.connect(self._update_count_label)
        self._end_spin.valueChanged.connect(self._update_count_label)

        for r, c, lbl in [(0,0,"Start frame:"),(0,2,"End frame:"),
                          (1,0,"Step:"),(1,2,"")]:
            w = rgrid.itemAtPosition(r, c)
            if w and isinstance(w.widget(), QLabel):
                w.widget().setStyleSheet(_LABEL_DIM)

        root.addWidget(rgrp)

        # ── Reference frame ────────────────────────────────────────────
        ref_grp = QGroupBox("Reference frame")
        ref_lay = QHBoxLayout(ref_grp)
        ref_lay.setSpacing(8)
        ref_lbl = QLabel("Use extracted frame #")
        ref_lbl.setStyleSheet(_LABEL_DIM)
        ref_lay.addWidget(ref_lbl)
        self._ref_spin = spin(0, 0, 0,
            "Which extracted frame to use as the DIC reference (0 = first)")
        ref_lay.addWidget(self._ref_spin)
        ref_note = QLabel("as reference image")
        ref_note.setStyleSheet(_LABEL_DIM)
        ref_lay.addWidget(ref_note)
        ref_lay.addStretch()
        root.addWidget(ref_grp)

        # ── Output ─────────────────────────────────────────────────────
        ogrp = QGroupBox("Output")
        olay = QVBoxLayout(ogrp)
        olay.setSpacing(6)

        out_row = QHBoxLayout()
        out_lbl = QLabel("Save frames to:")
        out_lbl.setStyleSheet(_LABEL_DIM)
        out_row.addWidget(out_lbl)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Auto: temporary folder beside the video")
        out_row.addWidget(self._out_edit, 1)
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(28)
        btn_out.setStyleSheet(_BTN)
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(btn_out)
        olay.addLayout(out_row)

        self._gray_check = QCheckBox("Convert to greyscale (recommended for DIC)")
        self._gray_check.setChecked(True)
        olay.addWidget(self._gray_check)

        root.addWidget(ogrp)

        # ── Progress ───────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._extract_status = QLabel("")
        self._extract_status.setStyleSheet(_LABEL_DIM)
        self._extract_status.setVisible(False)
        root.addWidget(self._extract_status)

        root.addWidget(self._separator())

        # ── Buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(_BTN)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        self._extract_btn = QPushButton("Extract frames")
        self._extract_btn.setStyleSheet(_BTN_ACCENT)
        self._extract_btn.setEnabled(False)
        self._extract_btn.clicked.connect(self._start_extraction)
        btn_row.addWidget(self._extract_btn)

        root.addLayout(btn_row)

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#212426;")
        return line

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", self.start_dir,
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v);;"
            "All files (*)"
        )
        if path:
            self.start_dir = os.path.dirname(path)
            self._load_video(path)

    def _load_video(self, path: str) -> None:
        import cv2 # IMPORT CV2 LOCALLY

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "PyDIC", f"Cannot open:\n{path}")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self._cap:
            self._cap.release()
        self._cap           = cap
        self._video_path    = path
        self._total_frames  = total
        self._fps           = fps

        self._file_edit.setText(os.path.basename(path))
        duration = total / fps if fps else 0
        self._info_lbl.setText(
            f"{w}×{h} px  •  {total} frames  •  {fps:.2f} fps (playback)  •  "
            f"{duration:.1f} s"
        )

        # Seed the capture rate with the container's value: correct for ordinary
        # footage, and the right starting point to edit for high-speed footage.
        self._capture_fps_spin.blockSignals(True)
        self._capture_fps_spin.setValue(fps if fps > 0 else 25.0)
        self._capture_fps_spin.blockSignals(False)

        for sp in (self._start_spin, self._end_spin,
                   self._ref_spin, self._preview_slider):
            sp.setMaximum(max(0, total - 1))
        self._start_spin.setValue(0)
        self._end_spin.setValue(max(0, total - 1))
        self._preview_slider.setEnabled(True)
        self._preview_slider.setValue(0)

        base = os.path.splitext(path)[0] + "_frames"
        # If the user chooses a second video in the same dialog, follow it with
        # a new automatic output folder. Preserve only a genuinely hand-edited
        # destination; the old automatic path belonged to the previous video.
        if not self._out_edit.text() or self._out_edit.text() == self._auto_output_path:
            self._out_edit.setText(base)
        self._auto_output_path = base

        self._update_count_label()
        self._extract_btn.setEnabled(True)
        self._show_frame(0)

    def _show_frame(self, idx: int) -> None:
        import cv2 # IMPORT CV2 LOCALLY

        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if ret:
            self._preview.show_frame(frame)
        # Real capture time, not playback position.
        rate = self._capture_fps_spin.value()
        t = idx / rate if rate else 0
        self._preview_frame_lbl.setText(
            f"{idx}  ({t:.4g}s)"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_preview_slider(self, val: int) -> None:
        self._show_frame(val)

    def _update_count_label(self) -> None:
        s = self._start_spin.value()
        e = self._end_spin.value()
        step = self._step_spin.value()

        if e < s:
            self._count_lbl.setText("Invalid range: end is before start")
            self._count_lbl.setStyleSheet("color:#bf6259; font-size:11px;")  # Red warning
            self._ref_spin.setMaximum(0)
            if hasattr(self, '_extract_btn'):
                self._extract_btn.setEnabled(False)
            return

        count = max(0, (e - s) // step + 1)
        self._count_lbl.setText(f"{count} frames will be extracted")
        self._count_lbl.setStyleSheet(_LABEL_DIM)  # Reset to standard color
        self._ref_spin.setMaximum(max(0, count - 1))

        if self._video_path and hasattr(self, '_extract_btn'):
            self._extract_btn.setEnabled(True)

        self._update_capture_note()

    def effective_fps(self) -> float:
        """Sample rate of the extracted sequence, in Hz.

        Capture rate divided by the extraction step: keeping every 4th frame of
        a 2000 Hz recording samples the motion at 500 Hz, and that is the rate
        Δt must come from.
        """
        step = self._step_spin.value() if hasattr(self, "_step_spin") else 1
        fps = self._capture_fps_spin.value()
        return fps / step if step > 0 else fps

    def _update_capture_note(self) -> None:
        note = getattr(self, "_capture_note", None)
        if note is None:
            return
        eff = self.effective_fps()
        if eff <= 0:
            note.setText("")
            return
        txt = f"Δt = {1000.0 / eff:.4g} ms between extracted frames  ·  sampled at {eff:,.2f} Hz"
        container = self._fps
        # Only worth flagging once a video is loaded and the two disagree.
        if self._video_path and container > 0:
            ratio = self._capture_fps_spin.value() / container
            if ratio > 1.01:
                txt += f"   (file plays back at {container:.2f} fps, {ratio:.4g}\u00d7 slow motion)"
            elif ratio < 0.99:
                txt += f"   (file reports {container:.2f} fps)"
        note.setText(txt)

    def _on_capture_rate_changed(self, *_args) -> None:
        """Keep both the derived-rate note and preview timestamp in sync."""
        self._update_capture_note()
        if self._cap is not None and self._preview_slider.isEnabled():
            self._show_frame(self._preview_slider.value())

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.start_dir)
        if path:
            if self._video_path:
                vid_name = os.path.splitext(os.path.basename(self._video_path))[0]
                subfolder = f"{vid_name}_frames"
                self._out_edit.setText(os.path.join(path, subfolder))
            else:
                self._out_edit.setText(path)
    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _start_extraction(self) -> None:
        if not self._video_path:
            return

        start = self._start_spin.value()
        end   = self._end_spin.value()
        step  = self._step_spin.value()
        gray  = self._gray_check.isChecked()
        out   = self._out_edit.text().strip()

        if not out:
            out = os.path.join(
                tempfile.gettempdir(),
                "pydic_frames_" + os.path.splitext(
                    os.path.basename(self._video_path))[0]
            )

        if os.path.isdir(out) and os.listdir(out):
            ans = QMessageBox.question(
                self, "Output folder is not empty",
                f"The folder already contains files:\n{out}\n\n"
                "Existing PNG files will be overwritten. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        # Lock UI
        self._extract_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._extract_status.setVisible(True)
        self._extract_status.setText("Extracting frames…")

        self._thread = QThread()
        self._worker = ExtractionWorker(
            self._video_path, out, start, end, step, gray
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @pyqtSlot(int, int)
    def _on_progress(self, done: int, total: int) -> None:
        pct = int(done / total * 100) if total else 0
        self._progress.setValue(pct)
        self._extract_status.setText(f"Extracting… {done}/{total} frames")

    @pyqtSlot(list)
    def _on_done(self, paths: List[str]) -> None:
        self._progress.setValue(100)
        if not paths:
            self._extract_status.setText("Cancelled.")
            self._extract_btn.setEnabled(True)
            return

        self.extracted_paths = paths
        self.reference_index = min(
            self._ref_spin.value(), len(paths) - 1
        )
        self._extract_status.setText(
            f"Done. {len(paths)} frames saved."
        )

        try:
            import json
            out_dir = os.path.dirname(paths[0])
            meta_path = os.path.join(out_dir, "dic_metadata.json")

            # Derived from the operator-supplied capture rate, not the
            # container's playback rate, so downstream Δt is real time.
            with open(meta_path, "w") as f:
                json.dump({
                    "fps": self.effective_fps(),
                    "capture_fps": self._capture_fps_spin.value(),
                    "container_fps": self._fps,
                    "extraction_step": self._step_spin.value(),
                    "calibration": self.get_calibration().to_dict(),
                }, f)
        except Exception as e:
            print(f"Failed to save metadata: {e}")

        self.accept()

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._extract_status.setVisible(False)
        self._extract_btn.setEnabled(True)
        QMessageBox.critical(self, "Extraction failed", msg)

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
        if self._cap:
            self._cap.release()
            self._cap = None
        self.reject()

    @property
    def video_path(self) -> str:
        return self._video_path or ""

    def get_calibration(self) -> Calibration:
        value = float(self._px_size_spin.value())
        unit = self._px_unit_combo.currentText()
        return (Calibration.from_pixel_size(value, unit)
                if value > 0 else Calibration(None, unit))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker:
            self._worker.cancel()
        if self._cap:
            self._cap.release()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        event.accept()
