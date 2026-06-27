"""
analysis_page.py — Step 4: Live progress during DIC analysis.
"""
from __future__ import annotations
import time
from typing import TYPE_CHECKING, Optional
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot, QObject
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QFrame, QSizePolicy,
)

if TYPE_CHECKING:
    from src.ui.wizard import Wizard


_C_SURFACE = "#0e1c2e"
_C_CARD    = "#132035"
_C_BORDER  = "#1e3a5a"
_C_ACCENT  = "#3b82f6"
_C_TEXT    = "#e2e8f0"
_C_TEXT2   = "#94a3b8"
_C_SUCCESS = "#10b981"
_C_DANGER  = "#ef4444"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _Worker(QObject):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, analysis, seed_xy, use_gpu: bool = False):
        super().__init__()
        self._analysis = analysis
        self._seed_xy  = seed_xy
        self._use_gpu = use_gpu  # <-- Save it

    @pyqtSlot()
    def run(self):
        try:
            self._analysis.run(
                progress_cb=lambda f, m: self.progress.emit(f, m),
                seed_xy=self._seed_xy,
                use_gpu=self._use_gpu,
            )
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Analysis page
# ---------------------------------------------------------------------------

class AnalysisPage(QWidget):
    """Step 4 — runs DIC in background, shows live progress."""

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._worker: Optional[_Worker] = None
        self._thread: Optional[QThread] = None
        self._t_start = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick_elapsed)
        self._last_thumb_idx = -1   # tracks len(results) to redraw on new result
        self._last_shown_frame = -1  # tracks frame number from progress msg
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

        self._title = QLabel("Step 4  ·  Running Analysis")
        self._title.setStyleSheet(f"color:{_C_TEXT}; font-size:13px; font-weight:600;")
        top_lay.addWidget(self._title)
        top_lay.addStretch()
        root.addWidget(top)

        # ── Body ──────────────────────────────────────────────────────
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(60, 40, 60, 40)
        body_lay.setSpacing(28)
        body_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Frame preview
        self._preview_lbl = QLabel()
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setMinimumHeight(200)
        self._preview_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self._preview_lbl.setStyleSheet(
            f"background:{_C_CARD}; border:1px solid {_C_BORDER}; "
            f"border-radius:8px; color:{_C_TEXT2}; font-size:13px;"
        )
        self._preview_lbl.setText("Preparing…")
        body_lay.addWidget(self._preview_lbl, 1)

        # (Frame label removed as per request)
        self._frame_lbl = None

        # Progress bar
        self._pbar = QProgressBar()
        self._pbar.setRange(0, 1000)
        self._pbar.setValue(0)
        self._pbar.setFixedHeight(10)
        body_lay.addWidget(self._pbar)

        # Status text
        self._status_lbl = QLabel("Initialising…")
        self._status_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:12px;")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body_lay.addWidget(self._status_lbl)

        # Stats row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(40)
        stats_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._elapsed_lbl  = _stat_label("Elapsed", "0:00")
        self._frames_lbl   = _stat_label("Frames",  "0 / 0")

        for w in (self._elapsed_lbl, self._frames_lbl):
            stats_row.addWidget(w)

        body_lay.addLayout(stats_row)
        root.addWidget(body, 1)

        # ── Footer ────────────────────────────────────────────────────
        footer = QWidget()
        footer.setFixedHeight(58)
        footer.setStyleSheet(f"background:{_C_SURFACE}; border-top:1px solid {_C_BORDER};")
        foot_lay = QHBoxLayout(footer)
        foot_lay.setContentsMargins(20, 0, 20, 0)
        foot_lay.addStretch()

        self._cancel_btn = QPushButton("■  Cancel")
        self._cancel_btn.setProperty("class", "danger")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.setMinimumWidth(110)
        self._cancel_btn.clicked.connect(self._cancel)
        foot_lay.addWidget(self._cancel_btn)

        root.addWidget(footer)

    # ------------------------------------------------------------------
    def on_enter(self) -> None:
        """Start the DIC analysis thread."""
        self._pbar.setValue(0)
        self._status_lbl.setText("Starting…")
        self._cancel_btn.setEnabled(True)
        self._last_thumb_idx = -1
        self._last_shown_frame = -1
        self._t_start = time.perf_counter()
        self._timer.start()

        analysis = self._wizard.analysis
        seed_xy = getattr(self._wizard, "seed_xy", None)
        use_gpu = getattr(self._wizard, "use_gpu", False)

        self._thread = QThread()
        self._worker = _Worker(analysis, seed_xy, use_gpu)  # <-- Pass to Worker
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @pyqtSlot(float, str)
    def _on_progress(self, frac: float, msg: str) -> None:
        self._pbar.setValue(int(frac * 1000))
        self._status_lbl.setText(msg)

        import re
        m = re.search(r"\[(\d+)/(\d+)\]", msg)
        if m:
            cur_f, tot_f = int(m.group(1)), int(m.group(2))
            self._frames_lbl.findChild(QLabel, "value").setText(f"{cur_f} / {tot_f}")
            frame_idx = cur_f - 1  # 0-based
        else:
            frame_idx = self._last_shown_frame

        n_results = len(self._wizard.analysis.results)
        
        # We always want to show the LATEST frame that has a computed ROI.
        # This means while Frame 2 is tracking, we show Frame 1 (with its ROI).
        if n_results > 0:
            show_idx = n_results - 1
        else:
            show_idx = frame_idx

        # Only redraw if the frame we want to show changed, OR if a new result became available for it
        if show_idx != self._last_shown_frame or n_results != self._last_thumb_idx:
            self._last_shown_frame = show_idx
            self._last_thumb_idx = n_results
            
            if show_idx >= 0:
                self._show_frame_thumbnail(show_idx)

    def _show_frame_thumbnail(self, idx: int) -> None:
        """Draw the deformed-image thumbnail with a cyan ROI overlay."""
        import cv2
        import os

        analysis = self._wizard.analysis
        paths = analysis.def_paths

        if idx < 0 or idx >= len(paths):
            return

        try:
            img = cv2.imread(paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is None:
                return
            H, W = img.shape

            # --- compute thumbnail dimensions ---
            thumb_w = min(W, self._preview_lbl.width() - 20)
            thumb_h = int(thumb_w * H / W)
            if thumb_h > self._preview_lbl.height() - 20:
                thumb_h = self._preview_lbl.height() - 20
                thumb_w = int(thumb_h * W / H)
            thumb_w = max(1, thumb_w)
            thumb_h = max(1, thumb_h)

            img_small = cv2.resize(img, (thumb_w, thumb_h))
            rgb = cv2.cvtColor(img_small, cv2.COLOR_GRAY2RGB)

            # --- Draw live ROI overlay ---
            try:
                if idx < len(analysis.results):
                    result = analysis.results[idx]
                    if result is not None and hasattr(result, 'u'):
                        valid_mask = np.isfinite(result.u)
                        if valid_mask.any():
                            s = max(1, analysis.params.subset_spacing)

                            # Dilate the sparse subset grid into a solid mask
                            kernel = np.ones((s + 1, s + 1), np.uint8)
                            dense_mask = cv2.dilate(
                                valid_mask.astype(np.uint8), kernel
                            )

                            # Fill enclosed holes using flood-fill from borders:
                            # 1. Pad with a 1px border of zeros
                            # 2. Flood-fill from (0,0) to find exterior background
                            # 3. Everything NOT exterior is ROI (fills interior holes)
                            padded = np.zeros(
                                (dense_mask.shape[0] + 2, dense_mask.shape[1] + 2),
                                dtype=np.uint8,
                            )
                            padded[1:-1, 1:-1] = dense_mask
                            flood = padded.copy()
                            cv2.floodFill(flood, None, (0, 0), 255)
                            # Invert: exterior=0, interior holes=255
                            interior_holes = 255 - flood[1:-1, 1:-1]
                            # Merge: original ROI + filled holes
                            filled_mask = np.maximum(dense_mask * 255, interior_holes)

                            # Resize to thumbnail dimensions
                            thumb_mask = cv2.resize(
                                filled_mask, (thumb_w, thumb_h),
                                interpolation=cv2.INTER_NEAREST,
                            )

                            # Blend cyan tint onto valid pixels
                            roi_pixels = thumb_mask > 0
                            if roi_pixels.any():
                                # Direct alpha blend (no cv2.addWeighted needed)
                                cyan = np.array([0, 150, 255], dtype=np.float32)
                                rgb[roi_pixels] = (
                                    rgb[roi_pixels].astype(np.float32) * 0.55
                                    + cyan * 0.45
                                ).astype(np.uint8)
            except Exception:
                import traceback
                traceback.print_exc()

            # --- Push to screen ---
            h, w = rgb.shape[:2]
            self._current_thumb_array = np.ascontiguousarray(rgb)
            qimg = QImage(
                self._current_thumb_array.data, w, h, w * 3,
                QImage.Format.Format_RGB888,
            )
            self._preview_lbl.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            import traceback
            traceback.print_exc()

    @pyqtSlot()
    def _on_finished(self) -> None:
        self._timer.stop()
        self._pbar.setValue(1000)
        self._cancel_btn.setEnabled(False)
        n = len(self._wizard.analysis.results)
        self._status_lbl.setText(
            f"Analysis complete - {n} frame{'s' if n != 1 else ''} processed."
        )
        self._status_lbl.setStyleSheet(f"color:{_C_SUCCESS}; font-size:13px; font-weight:600;")
        self._wizard.go_results()

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._timer.stop()
        self._cancel_btn.setEnabled(False)
        self._status_lbl.setText(f"Error: {msg}")
        self._status_lbl.setStyleSheet(f"color:{_C_DANGER}; font-size:12px;")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Analysis Error", msg)

    def _cancel(self) -> None:
        self._wizard.analysis.cancel()
        self._cancel_btn.setEnabled(False)
        self._status_lbl.setText("Cancelling…")

    def _tick_elapsed(self) -> None:
        s = int(time.perf_counter() - self._t_start)
        self._elapsed_lbl.findChild(QLabel, "value").setText(
            f"{s // 60}:{s % 60:02d}"
        )


# ---------------------------------------------------------------------------
# Tiny stat display widget
# ---------------------------------------------------------------------------

def _stat_label(title: str, value: str) -> QWidget:
    w = QWidget()
    w.setStyleSheet(
        f"background:{_C_CARD}; border:1px solid {_C_BORDER}; "
        f"border-radius:8px; padding:8px 20px;"
    )
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)

    t = QLabel(title.upper())
    t.setStyleSheet(
        f"color:{_C_TEXT2}; font-size:9px; font-weight:700; "
        f"letter-spacing:0.8px; background:transparent; border:none;"
    )
    t.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(t)

    v = QLabel(value)
    v.setObjectName("value")
    v.setStyleSheet(
        f"color:{_C_TEXT}; font-size:18px; font-weight:700; "
        f"font-family:'Fira Code','JetBrains Mono',monospace; "
        f"background:transparent; border:none;"
    )
    v.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(v)

    return w
