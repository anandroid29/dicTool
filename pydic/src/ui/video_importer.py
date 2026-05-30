"""
video_importer.py
=================
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import (
    Qt, QObject, QThread, pyqtSignal, pyqtSlot, QTimer,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QLineEdit,
    QProgressBar, QFileDialog, QGroupBox, QSizePolicy,
    QMessageBox, QFrame, QCheckBox,
)


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
            "background:#0d1117; border:1px solid #30363d; border-radius:6px;"
            "color:#484f58; font-size:11px;"
        )
        self.setText("No preview")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setFixedHeight(160)

    def show_frame(self, bgr: np.ndarray) -> None:
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
    "QPushButton { background:#21262d; color:#c9d1d9; border:1px solid #30363d; "
    "border-radius:5px; font-size:11px; padding:4px 12px; } "
    "QPushButton:hover { background:#2d333b; color:#e6edf3; border-color:#8b949e; } "
    "QPushButton:pressed { background:#161b22; }"
)
_BTN_ACCENT = (
    "QPushButton { background:#2f81f7; color:#fff; border:none; "
    "border-radius:5px; font-size:12px; font-weight:700; padding:6px 16px; } "
    "QPushButton:hover { background:#388bfd; } "
    "QPushButton:pressed { background:#1f6feb; } "
    "QPushButton:disabled { background:#21262d; color:#484f58; }"
)
_SPIN = (
    "QSpinBox { background:#21262d; color:#e6edf3; border:1px solid #30363d; "
    "border-radius:5px; padding:3px 6px; font-size:11px; } "
    "QSpinBox::up-button, QSpinBox::down-button { border:none; width:16px; }"
)
_LABEL_DIM = "color:#8b949e; font-size:11px;"
_LABEL_PRI = "color:#e6edf3; font-size:11px;"


class VideoImporterDialog(QDialog):
    """
    Open a video, pick a frame range, and extract frames as PNG images.

    Usage
    -----
    dlg = VideoImporterDialog(parent, video_path)   # pre-fill path (optional)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        paths = dlg.extracted_paths   # list[str], sorted
        ref   = dlg.reference_index   # int — which frame to use as reference
    """

    def __init__(
        self,
        parent=None,
        initial_video: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Video Frames")
        self.setMinimumWidth(520)
        self.setModal(True)
        self.setStyleSheet(
            "QDialog { background:#0d1117; } "
            "QGroupBox { color:#8b949e; font-size:10px; font-weight:700; "
            "  text-transform:uppercase; letter-spacing:0.5px; "
            "  border:1px solid #21262d; border-radius:6px; "
            "  margin-top:8px; padding-top:12px; } "
            "QGroupBox::title { subcontrol-origin:margin; left:8px; } "
            "QLabel { color:#e6edf3; } "
            "QLineEdit { background:#21262d; color:#e6edf3; border:1px solid #30363d; "
            "  border-radius:5px; padding:4px 8px; font-size:11px; } "
            "QCheckBox { color:#8b949e; font-size:11px; } "
            "QCheckBox::indicator { width:13px; height:13px; border:1px solid #30363d; "
            "  border-radius:3px; background:#21262d; } "
            "QCheckBox::indicator:checked { background:#2f81f7; border-color:#2f81f7; } "
            "QSlider::groove:horizontal { background:#21262d; height:5px; border-radius:3px; } "
            "QSlider::handle:horizontal { background:#2f81f7; width:13px; height:13px; "
            "  margin:-4px 0; border-radius:6px; } "
            "QSlider::sub-page:horizontal { background:#1f4e8c; border-radius:3px; } "
            "QProgressBar { background:#21262d; border:1px solid #30363d; border-radius:4px; "
            "  height:8px; text-align:center; } "
            "QProgressBar::chunk { background:#2f81f7; border-radius:4px; }"
        )

        # State
        self._video_path: Optional[str] = None
        self._cap:        Optional[cv2.VideoCapture] = None
        self._total_frames: int = 0
        self._fps:          float = 25.0
        self.extracted_paths: List[str] = []
        self.reference_index: int = 0   # index within extracted_paths

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
        title = QLabel("Import Video Frames")
        title.setStyleSheet("color:#e6edf3; font-size:15px; font-weight:700;")
        root.addWidget(title)

        sub = QLabel("Extract individual frames from a video file for use as DIC images.")
        sub.setStyleSheet(_LABEL_DIM)
        sub.setWordWrap(True)
        root.addWidget(sub)

        root.addWidget(self._separator())

        # ── Video file ─────────────────────────────────────────────────
        vgrp = QGroupBox("Video File")
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
        rgrp = QGroupBox("Frame Range")
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

        # Connect start/end to live update
        self._start_spin.valueChanged.connect(self._update_count_label)
        self._end_spin.valueChanged.connect(self._update_count_label)

        # Labels for spin boxes
        for r, c, lbl in [(0,0,"Start frame:"),(0,2,"End frame:"),
                          (1,0,"Step:"),(1,2,"")]:
            w = rgrid.itemAtPosition(r, c)
            if w and isinstance(w.widget(), QLabel):
                w.widget().setStyleSheet(_LABEL_DIM)

        root.addWidget(rgrp)

        # ── Reference frame ────────────────────────────────────────────
        ref_grp = QGroupBox("Reference Frame")
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
        self._out_edit.setPlaceholderText("(auto — temp folder next to video)")
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

        self._extract_btn = QPushButton("Extract Frames")
        self._extract_btn.setStyleSheet(_BTN_ACCENT)
        self._extract_btn.setEnabled(False)
        self._extract_btn.clicked.connect(self._start_extraction)
        btn_row.addWidget(self._extract_btn)

        root.addLayout(btn_row)

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#21262d;")
        return line

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v);;"
            "All files (*)"
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str) -> None:
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
            f"{w}×{h} px  •  {total} frames  •  {fps:.2f} fps  •  "
            f"{duration:.1f} s"
        )

        # Update range spinboxes
        for sp in (self._start_spin, self._end_spin,
                   self._ref_spin, self._preview_slider):
            sp.setMaximum(max(0, total - 1))
        self._start_spin.setValue(0)
        self._end_spin.setValue(max(0, total - 1))
        self._preview_slider.setEnabled(True)
        self._preview_slider.setValue(0)

        # Default output dir
        if not self._out_edit.text():
            base = os.path.splitext(path)[0] + "_frames"
            self._out_edit.setText(base)

        self._update_count_label()
        self._extract_btn.setEnabled(True)
        self._show_frame(0)

    def _show_frame(self, idx: int) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if ret:
            self._preview.show_frame(frame)
        t = idx / self._fps if self._fps else 0
        self._preview_frame_lbl.setText(
            f"{idx}  ({t:.2f}s)"
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
            self._end_spin.setValue(s)
            e = s
        count = max(0, (e - s) // step + 1)
        self._count_lbl.setText(f"{count} frames will be extracted")
        self._ref_spin.setMaximum(max(0, count - 1))

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output Directory")
        if path:
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
                self, "Output folder not empty",
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
            f"Done — {len(paths)} frames saved."
        )
        self.accept()   # close dialog with Accepted

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._extract_status.setVisible(False)
        self._extract_btn.setEnabled(True)
        QMessageBox.critical(self, "Extraction Error", msg)

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
        if self._cap:
            self._cap.release()
            self._cap = None
        self.reject()

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
