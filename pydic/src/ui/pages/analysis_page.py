"""
analysis_page.py — Step 5: Live progress during DIC analysis.
"""
from __future__ import annotations
import queue
import threading
import time
from typing import TYPE_CHECKING, Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QFrame, QSizePolicy,
)

# Palette comes from the single source of truth in theme.py. These were
# duplicated literals, which is why re-theming previously left pages behind.
from src.ui.theme import C_ACCENT, C_BORDER, C_CARD, C_DANGER, C_SUCCESS, C_SURFACE, C_TEXT, C_TEXT2

_C_ACCENT = C_ACCENT
_C_BORDER = C_BORDER
_C_CARD = C_CARD
_C_DANGER = C_DANGER
_C_SUCCESS = C_SUCCESS
_C_SURFACE = C_SURFACE
_C_TEXT = C_TEXT
_C_TEXT2 = C_TEXT2


if TYPE_CHECKING:
    from src.ui.wizard import Wizard




# ---------------------------------------------------------------------------
# Analysis page
# ---------------------------------------------------------------------------

class AnalysisPage(QWidget):
    """Step 5 — runs DIC in background, shows live progress."""

    def __init__(self, wizard: "Wizard") -> None:
        super().__init__()
        self._wizard = wizard
        self._worker = None
        self._thread: Optional[threading.Thread] = None
        self._progress_queue: queue.SimpleQueue[tuple[float, str]] = (
            queue.SimpleQueue())
        self._worker_error: Optional[str] = None
        self._worker_poll = QTimer(self)
        self._worker_poll.setInterval(30)
        self._worker_poll.timeout.connect(self._poll_worker)
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

        self._title = QLabel("Step 5  ·  Running Analysis")
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
            f"border-radius:3px; color:{_C_TEXT2}; font-size:13px;"
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
    def on_before_show(self) -> None:
        """Blank the previous run's log, thumbnail and progress before showing."""
        try:
            self._pbar.setValue(0)
            self._status_lbl.setText("")
            self._last_thumb_idx = -1
            self._last_shown_frame = -1
            if hasattr(self, "_log") and hasattr(self._log, "clear"):
                self._log.clear()
            if hasattr(self, "_canvas"):
                self._canvas.clear_result_overlay()
                self._canvas.set_streaklines(None)
            for attr in ("_thumb", "_preview"):
                w = getattr(self, attr, None)
                if w is not None and hasattr(w, "clear"):
                    w.clear()
        except Exception:
            pass

    def on_enter(self) -> None:
        """Start the DIC analysis thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._pbar.setValue(0)
        self._status_lbl.setText("Starting…")
        self._cancel_btn.setEnabled(True)
        self._last_thumb_idx = -1
        self._last_shown_frame = -1
        self._worker_error = None
        self._progress_queue = queue.SimpleQueue()
        self._t_start = time.perf_counter()
        self._timer.start()

        analysis = self._wizard.analysis
        seed_xy = getattr(self._wizard, "seed_xy", None)
        use_gpu = getattr(self._wizard, "use_gpu", False)

        # The worker makes no Qt calls. A GUI-owned timer drains its thread-safe
        # queue and detects completion, so navigation can never depend on a Qt
        # timer or signal accidentally dispatched from the Python worker thread.
        self._thread = threading.Thread(
            target=self._run_analysis,
            args=(analysis, seed_xy, use_gpu),
            name="PyDIC-analysis",
            daemon=True,
        )
        self._worker_poll.start()
        self._thread.start()

    def _run_analysis(self, analysis, seed_xy, use_gpu: bool) -> None:
        try:
            analysis.run(
                progress_cb=lambda f, m: self._progress_queue.put(
                    (float(f), str(m))),
                seed_xy=seed_xy,
                use_gpu=use_gpu,
            )
        except BaseException as exc:
            self._worker_error = str(exc) or type(exc).__name__

    def _poll_worker(self) -> None:
        """Drain progress and complete the run, always on the GUI thread."""
        latest = None
        while True:
            try:
                latest = self._progress_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._on_progress(*latest)

        thread = self._thread
        if thread is None or thread.is_alive():
            return
        thread.join(0)
        self._thread = None
        self._worker_poll.stop()
        if self._worker_error is not None:
            self._on_error(self._worker_error)
        else:
            self._on_finished()

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
                dense_mask = None
                source_mask = self._thumbnail_source_mask(idx)
                if idx == 0:
                    # Keep frame 1 identical to the mask approved on the
                    # Dynamic ROI page. Later thumbnails continue to show the
                    # measured per-frame validity as before.
                    if source_mask is not None:
                        dense_mask = source_mask.astype(np.uint8) * 255
                elif source_mask is not None:
                    valid_mask = source_mask
                    if valid_mask.any():
                        s = max(1, analysis.params.subset_spacing)

                        # Dilate the sparse subset grid into a solid mask
                        kernel = np.ones((s + 1, s + 1), np.uint8)
                        dense_mask = cv2.dilate(
                            valid_mask.astype(np.uint8), kernel
                        )

                        # Fill enclosed holes using flood-fill from borders.
                        padded = np.zeros(
                            (dense_mask.shape[0] + 2, dense_mask.shape[1] + 2),
                            dtype=np.uint8,
                        )
                        padded[1:-1, 1:-1] = dense_mask
                        flood = padded.copy()
                        cv2.floodFill(flood, None, (0, 0), 255)
                        interior_holes = 255 - flood[1:-1, 1:-1]
                        dense_mask = np.maximum(
                            dense_mask * 255, interior_holes)

                if dense_mask is not None and dense_mask.any():
                    thumb_mask = cv2.resize(
                        dense_mask, (thumb_w, thumb_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    roi_pixels = thumb_mask > 0
                    if roi_pixels.any():
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

    def _thumbnail_source_mask(self, idx: int) -> Optional[np.ndarray]:
        """Authoritative mask before thumbnail-only densification."""
        analysis = self._wizard.analysis
        if idx == 0:
            return analysis.reference_analysis_mask()
        if idx < 0 or idx >= len(analysis.results):
            return None
        result = analysis.results[idx]
        if result is None or not hasattr(result, "u"):
            return None
        valid = np.isfinite(result.u) & np.isfinite(result.v)
        if result.valid is not None:
            valid &= result.valid
        return valid

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
        # This method is called only by the GUI-owned poll timer after the
        # native worker has stopped; navigation is therefore safe and immediate.
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

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        """Cancel and join the worker before the Qt widget tree is destroyed."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            self._worker_poll.stop()
            return True
        self._wizard.analysis.cancel()
        thread.join(max(0, int(timeout_ms)) / 1000.0)
        stopped = not thread.is_alive()
        if stopped:
            self._worker_poll.stop()
        return stopped

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
        f"border-radius:3px; padding:8px 20px;"
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
