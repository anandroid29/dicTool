"""Interactive importer for an existing image sequence."""

from __future__ import annotations

import os
import time
import re
from collections import OrderedDict
from typing import List, Optional

from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QImageReader, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QSpinBox, QVBoxLayout,
)

from pydic.core.units import Calibration, LENGTH_UNIT_ORDER


_BG = "#1a1c1e"
_SURFACE = "#212426"
_CARD = "#282b2e"
_BORDER = "#3c4247"
_ACCENT = "#5a8cb0"
_TEXT = "#e2e8f0"
_TEXT2 = "#94a3b8"
_SUCCESS = "#6a9c74"
_ERROR = "#bf6259"


def _natural_sort_key(path: str):
    """Sort frame2 before frame10 while keeping ordinary names stable."""
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", os.path.basename(path))]


class _SequencePreview(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            f"background:#050b13; color:{_TEXT2}; border:1px solid {_BORDER}; "
            "border-radius:3px;")
        self.setText("No selected frame")

    def show_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._source = pixmap
        if pixmap is None or pixmap.isNull():
            self.show_message("This image could not be previewed")
            return
        self._rescale()

    def show_message(self, message: str) -> None:
        self._source = None
        self.clear()
        self.setText(message)

    def _rescale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        self.setPixmap(self._source.scaled(
            max(1, self.width() - 12), max(1, self.height() - 12),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()


class ImageSequenceImporterDialog(QDialog):
    """Choose, sample, and preview the exact image sequence DIC will receive."""

    def __init__(self, image_files: List[str], folder: str,
                 fps_from_meta: Optional[float] = None,
                 initial_calibration: Optional[Calibration] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._folder = folder
        self._image_files = sorted(image_files, key=_natural_sort_key)
        self._fps_from_meta = fps_from_meta
        self._initial_calibration = initial_calibration or Calibration()
        self._selected_indices: List[int] = []
        self._thumb_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._updating_selection = False

        self._play_timer = QTimer(self)
        # Single-shot and self-rescheduling: each preview frame decodes an
        # image, which on a slow disk can outlast the interval. A repeating
        # timer would queue those overruns until the dialog stopped responding.
        self._play_timer.setSingleShot(True)
        self._play_timer.timeout.connect(self._advance_preview)

        self.setWindowTitle("Load image sequence")
        self.setMinimumSize(760, 720)
        self.resize(900, 790)
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{ background:{_BG}; }}
            QLabel {{ color:{_TEXT}; font-size:11px; }}
            QGroupBox {{ color:{_TEXT2}; font-size:10px; font-weight:700;
                border:1px solid {_BORDER}; border-radius:3px;
                margin-top:8px; padding-top:12px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:9px; }}
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
                background:{_SURFACE}; color:{_TEXT}; border:1px solid {_BORDER};
                border-radius:3px; padding:4px 7px; }}
            QPushButton {{ background:{_SURFACE}; color:{_TEXT};
                border:1px solid {_BORDER}; border-radius:3px; padding:5px 12px; }}
            QPushButton:hover {{ background:{_CARD}; border-color:{_ACCENT}; }}
            QPushButton:checked {{ background:{_ACCENT}; color:white; }}
            QPushButton:disabled {{ color:#475569; border-color:#31353a; }}
            QCheckBox {{ color:{_TEXT2}; font-size:11px; }}
            QSlider::groove:horizontal {{ background:{_CARD}; height:5px;
                border-radius:3px; }}
            QSlider::handle:horizontal {{ background:{_ACCENT}; width:13px;
                margin:-4px 0; border-radius:3px; }}
        """)
        self._build_ui()
        self._initialise_sequence()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        title = QLabel("Load image sequence")
        title.setStyleSheet(f"color:{_TEXT}; font-size:17px; font-weight:700;")
        root.addWidget(title)
        self._source_info = QLabel("")
        self._source_info.setStyleSheet(f"color:{_TEXT2};")
        root.addWidget(self._source_info)

        self._preview = _SequencePreview()
        root.addWidget(self._preview, 1)

        preview_controls = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedWidth(72)
        self._play_btn.toggled.connect(self._toggle_playback)
        preview_controls.addWidget(self._play_btn)

        previous_btn = QPushButton("Previous")
        previous_btn.clicked.connect(lambda: self._step_preview(-1))
        preview_controls.addWidget(previous_btn)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(lambda: self._step_preview(1))
        preview_controls.addWidget(next_btn)

        self._preview_slider = QSlider(Qt.Orientation.Horizontal)
        self._preview_slider.valueChanged.connect(self._show_preview)
        preview_controls.addWidget(self._preview_slider, 1)

        preview_controls.addWidget(QLabel("Preview speed:"))
        self._preview_fps_spin = QDoubleSpinBox()
        self._preview_fps_spin.setRange(0.5, 60.0)
        self._preview_fps_spin.setDecimals(1)
        self._preview_fps_spin.setValue(10.0)
        self._preview_fps_spin.setSuffix(" fps")
        self._preview_fps_spin.valueChanged.connect(
            self._update_playback_interval)
        preview_controls.addWidget(self._preview_fps_spin)
        self._loop_check = QCheckBox("Loop")
        self._loop_check.setChecked(True)
        preview_controls.addWidget(self._loop_check)
        root.addLayout(preview_controls)

        self._preview_info = QLabel("")
        self._preview_info.setStyleSheet(
            f"color:{_TEXT2}; font-family:'Consolas',monospace;")
        root.addWidget(self._preview_info)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(10)

        range_group = QGroupBox("Frame selection")
        range_grid = QGridLayout(range_group)
        range_grid.setSpacing(7)

        range_grid.addWidget(QLabel("Start position:"), 0, 0)
        self.start_spin = QSpinBox()
        self.start_spin.setToolTip(
            "Skip images before this position; counting starts at 0.\n"
            "The start image is normally the reference.")
        range_grid.addWidget(self.start_spin, 0, 1)

        range_grid.addWidget(QLabel("End position:"), 0, 2)
        self.end_spin = QSpinBox()
        self.end_spin.setToolTip("Last source image to use, inclusive.")
        range_grid.addWidget(self.end_spin, 0, 3)

        range_grid.addWidget(QLabel("Load every:"), 1, 0)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, max(1, len(self._image_files)))
        self.step_spin.setValue(1)
        self.step_spin.setSuffix(" frame(s)")
        range_grid.addWidget(self.step_spin, 1, 1)

        range_grid.addWidget(QLabel("Reference position:"), 1, 2)
        self.reference_spin = QSpinBox()
        self.reference_spin.setToolTip(
            "Position within the sampled sequence. Earlier frames are shown\n"
            "for choosing the reference but are not analysed.")
        range_grid.addWidget(self.reference_spin, 1, 3)

        self.limit_check = QCheckBox("Limit deformed frames to")
        range_grid.addWidget(self.limit_check, 2, 0, 1, 2)
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(1, max(1, len(self._image_files) - 1))
        self.max_frames_spin.setValue(min(100, max(1, len(self._image_files) - 1)))
        self.max_frames_spin.setEnabled(False)
        range_grid.addWidget(self.max_frames_spin, 2, 2)

        self._selection_summary = QLabel("")
        self._selection_summary.setWordWrap(True)
        self._selection_summary.setStyleSheet(f"color:{_SUCCESS};")
        range_grid.addWidget(self._selection_summary, 3, 0, 1, 4)
        settings_row.addWidget(range_group, 3)

        timing_group = QGroupBox("Timing and scale")
        timing_grid = QGridLayout(timing_group)
        timing_grid.setSpacing(7)
        timing_grid.addWidget(QLabel("Source sequence rate:"), 0, 0)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.01, 10_000_000.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setValue(
            self._fps_from_meta if self._fps_from_meta is not None else 1.0)
        self.fps_spin.setSuffix(" Hz")
        self.fps_spin.setToolTip(
            "Rate of the source images, before Load every sampling.")
        timing_grid.addWidget(self.fps_spin, 0, 1, 1, 2)

        timing_grid.addWidget(QLabel("Pixel size:"), 1, 0)
        self.px_size_spin = QDoubleSpinBox()
        self.px_size_spin.setRange(0.0, 1e9)
        self.px_size_spin.setDecimals(6)
        self.px_size_spin.setSpecialValueText("unknown")
        initial_unit = self._initial_calibration.display_unit
        self.px_size_spin.setValue(
            self._initial_calibration.pixel_size_in(initial_unit) or 0.0)
        timing_grid.addWidget(self.px_size_spin, 1, 1)
        self.px_unit_combo = QComboBox()
        self.px_unit_combo.addItems(LENGTH_UNIT_ORDER)
        self.px_unit_combo.setCurrentText(initial_unit)
        timing_grid.addWidget(self.px_unit_combo, 1, 2)

        self._timing_summary = QLabel("")
        self._timing_summary.setWordWrap(True)
        self._timing_summary.setStyleSheet(f"color:{_TEXT2};")
        timing_grid.addWidget(self._timing_summary, 2, 0, 1, 3)
        settings_row.addWidget(timing_group, 2)
        root.addLayout(settings_row)

        roi_group = QGroupBox("Optional ROI mask")
        roi_row = QHBoxLayout(roi_group)
        self.roi_edit = QLineEdit()
        self.roi_edit.setReadOnly(True)
        self.roi_edit.setPlaceholderText("No mask selected. Draw the ROI later.")
        roi_row.addWidget(self.roi_edit, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_roi)
        roi_row.addWidget(browse_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.roi_edit.clear)
        roi_row.addWidget(clear_btn)
        root.addWidget(roi_group)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Load selected sequence")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self.start_spin.valueChanged.connect(self._on_start_changed)
        self.end_spin.valueChanged.connect(self._selection_changed)
        self.step_spin.valueChanged.connect(self._selection_changed)
        self.reference_spin.valueChanged.connect(self._selection_changed)
        self.limit_check.toggled.connect(self.max_frames_spin.setEnabled)
        self.limit_check.toggled.connect(self._selection_changed)
        self.max_frames_spin.valueChanged.connect(self._selection_changed)
        self.fps_spin.valueChanged.connect(self._selection_changed)
        self.roi_edit.textChanged.connect(self._selection_changed)
        self._update_playback_interval()

    def _initialise_sequence(self) -> None:
        count = len(self._image_files)
        dimensions = ""
        if self._image_files:
            size = QImageReader(self._image_files[0]).size()
            if size.isValid():
                dimensions = f"  ·  {size.width()}×{size.height()} px"
        self._source_info.setText(
            f"{os.path.basename(os.path.normpath(self._folder))}  ·  "
            f"{count:,} image{'s' if count != 1 else ''}{dimensions}  ·  "
            "natural filename order")
        self.start_spin.setRange(0, max(0, count - 2))
        self.end_spin.setRange(1 if count > 1 else 0, max(0, count - 1))
        self.end_spin.setValue(max(0, count - 1))
        self._selection_changed()

    def _on_start_changed(self, value: int) -> None:
        self.end_spin.setMinimum(min(max(0, len(self._image_files) - 1), value + 1))
        self._selection_changed()

    def _sampled_source_indices(self) -> List[int]:
        if len(self._image_files) < 2:
            return []
        start = self.start_spin.value()
        end = self.end_spin.value()
        if end <= start:
            return []
        return list(range(start, end + 1, self.step_spin.value()))

    def selected_source_indices(self) -> List[int]:
        sampled = self._sampled_source_indices()
        if len(sampled) < 2:
            return []
        selected = sampled[min(self.reference_spin.value(), len(sampled) - 1):]
        if self.limit_check.isChecked():
            selected = selected[:self.max_frames_spin.value() + 1]
        roi_path = os.path.normcase(os.path.normpath(self.roi_edit.text().strip()))
        if roi_path:
            selected = [idx for idx in selected if os.path.normcase(
                os.path.normpath(self._image_files[idx])) != roi_path]
        return selected

    def selected_paths(self) -> List[str]:
        return [self._image_files[idx] for idx in self.selected_source_indices()]

    def effective_fps(self) -> float:
        return self.fps_spin.value() / max(1, self.step_spin.value())

    def _selection_changed(self, *_args) -> None:
        if self._updating_selection:
            return
        self._updating_selection = True
        try:
            sampled = self._sampled_source_indices()
            self.reference_spin.setMaximum(max(0, len(sampled) - 2))
            indices = self.selected_source_indices()
            self._selected_indices = indices

            previous = self._preview_slider.value()
            self._preview_slider.setRange(0, max(0, len(indices) - 1))
            self._preview_slider.setEnabled(bool(indices))
            self._preview_slider.setValue(min(previous, max(0, len(indices) - 1)))

            valid = len(indices) >= 2
            ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
            ok.setEnabled(valid)
            self._play_btn.setEnabled(valid)
            if valid:
                deformed = len(indices) - 1
                self._selection_summary.setStyleSheet(f"color:{_SUCCESS};")
                self._selection_summary.setText(
                    f"Will load {len(indices):,} images: 1 reference + "
                    f"{deformed:,} deformed frames")
            else:
                self._selection_summary.setStyleSheet(f"color:{_ERROR};")
                self._selection_summary.setText(
                    "Select at least two images after sampling and reference selection.")

            eff = self.effective_fps()
            duration = (len(indices) - 1) / eff if valid and eff > 0 else 0.0
            self._timing_summary.setText(
                f"Effective rate: {eff:,.3g} Hz  ·  Δt: {1.0 / eff:.5g} s  ·  "
                f"selected duration: {duration:.5g} s")
        finally:
            self._updating_selection = False
        self._show_preview(self._preview_slider.value())

    def _thumbnail(self, path: str) -> Optional[QPixmap]:
        cached = self._thumb_cache.get(path)
        if cached is not None:
            self._thumb_cache.move_to_end(path)
            return cached

        reader = QImageReader(path)
        reader.setAutoTransform(True)
        source_size = reader.size()
        target = QSize(900, 500)
        if source_size.isValid() and (source_size.width() > target.width() or
                                      source_size.height() > target.height()):
            reader.setScaledSize(source_size.scaled(
                target, Qt.AspectRatioMode.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            return None
        pixmap = QPixmap.fromImage(image)
        self._thumb_cache[path] = pixmap
        while len(self._thumb_cache) > 4:
            self._thumb_cache.popitem(last=False)
        return pixmap

    def _show_preview(self, position: int) -> None:
        if not self._selected_indices:
            self._preview.show_message("No selected frame")
            self._preview_info.setText("No frame selected")
            return
        position = max(0, min(position, len(self._selected_indices) - 1))
        source_index = self._selected_indices[position]
        path = self._image_files[source_index]
        self._preview.show_pixmap(self._thumbnail(path))
        role = "REFERENCE" if position == 0 else f"DEFORMED {position}"
        self._preview_info.setText(
            f"{role}  ·  selected {position + 1}/{len(self._selected_indices)}  ·  "
            f"source position {source_index}  ·  {os.path.basename(path)}")

    def _toggle_playback(self, playing: bool) -> None:
        if playing and len(self._selected_indices) >= 2:
            self._play_btn.setText("Pause")
            self._play_timer.start(self._playback_interval_ms())
        else:
            self._play_btn.setText("Play")
            self._play_timer.stop()

    def _playback_interval_ms(self) -> int:
        return max(16, int(round(1000.0 / self._preview_fps_spin.value())))

    def _update_playback_interval(self, *_args) -> None:
        # Read fresh on each tick, so a rate change applies at the next frame.
        self._play_timer.setInterval(self._playback_interval_ms())

    def _advance_preview(self) -> None:
        if not self._play_btn.isChecked() or not self._selected_indices:
            self._play_btn.setChecked(False)
            return
        started = time.perf_counter()
        position = self._preview_slider.value() + 1
        if position >= len(self._selected_indices):
            if self._loop_check.isChecked():
                position = 0
            else:
                self._play_btn.setChecked(False)
                return
        self._preview_slider.setValue(position)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._play_timer.start(
            int(max(1.0, self._playback_interval_ms() - elapsed_ms)))

    def _step_preview(self, delta: int) -> None:
        self._play_btn.setChecked(False)
        if self._selected_indices:
            self._preview_slider.setValue(max(
                0, min(len(self._selected_indices) - 1,
                       self._preview_slider.value() + delta)))

    def _browse_roi(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ROI Mask Image", self._folder,
            "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp);;All Files (*)")
        if path:
            self.roi_edit.setText(path)

    def get_settings(self):
        """Backward-compatible access to the original four settings."""
        limit = self.max_frames_spin.value() if self.limit_check.isChecked() else None
        roi_path = self.roi_edit.text().strip() or None
        return self.step_spin.value(), limit, roi_path, self.fps_spin.value()

    def get_calibration(self) -> Calibration:
        value = float(self.px_size_spin.value())
        unit = self.px_unit_combo.currentText()
        return (Calibration.from_pixel_size(value, unit)
                if value > 0 else Calibration(None, unit))

    def done(self, result: int) -> None:
        self._play_timer.stop()
        super().done(result)
