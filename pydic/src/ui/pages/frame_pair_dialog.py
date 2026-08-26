"""
frame_pair_dialog.py — build a playable sequence of smoothed frame intervals.

A "pair" is two displayed frames treated as one longer measurement interval.
Sliding pairs such as 1→4, 2→5, 3→6 retain temporal playback while reducing the
share of each measurement contributed by adjacent-frame correlation noise.
"""
from __future__ import annotations

from typing import List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSpinBox, QListWidget, QListWidgetItem, QGroupBox, QFrame, QMessageBox,
    QAbstractItemView,
)

# Palette comes from the single source of truth in theme.py. These were
# duplicated literals, which is why re-theming previously left pages behind.
from src.ui.theme import C_ACCENT, C_BORDER, C_CARD, C_RAISED, C_SURFACE, C_TEXT, C_TEXT2, C_TEXT3, C_WARNING

_C_ACCENT = C_ACCENT
_C_BORDER = C_BORDER
_C_CARD = C_CARD
_C_RAISED = C_RAISED
_C_SURFACE = C_SURFACE
_C_TEXT = C_TEXT
_C_TEXT2 = C_TEXT2
_C_TEXT3 = C_TEXT3
_C_WARN = C_WARNING



_BTN = (
    f"QPushButton {{ background:{_C_CARD}; color:{_C_TEXT}; "
    f"border:1px solid {_C_BORDER}; border-radius:3px; font-size:11px; "
    f"padding:5px 12px; }} "
    f"QPushButton:hover {{ background:{_C_RAISED}; }} "
    f"QPushButton:disabled {{ color:{_C_TEXT3}; }}"
)
_BTN_ACCENT = (
    f"QPushButton {{ background:{_C_ACCENT}; color:#fff; border:none; "
    f"border-radius:3px; font-size:12px; font-weight:700; padding:6px 18px; }} "
    f"QPushButton:hover {{ background:#74a6c9; }} "
    f"QPushButton:disabled {{ background:{_C_CARD}; color:{_C_TEXT3}; }}"
)
_SPIN = (
    f"QSpinBox {{ background:{_C_CARD}; color:{_C_TEXT}; "
    f"border:1px solid {_C_BORDER}; border-radius:3px; padding:4px 6px; "
    f"font-size:11px; }}"
)


class FramePairDialog(QDialog):
    """Build a list of (frame_a, frame_b) pairs. Frames are 0-based on the way
    out; everything shown to the user is 1-based to match the results view."""

    def __init__(self, n_frames: int, fps: float = 1.0,
                 existing: List[Tuple[int, int]] | None = None,
                 strain_window: int = 5, grid_spacing: int = 1,
                 existing_mode: str = "custom",
                 parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Smoothed Frame-Pair Sequence")
        self.setMinimumWidth(520)
        self.setModal(True)
        self.setStyleSheet(
            f"QDialog {{ background:{_C_SURFACE}; }} "
            f"QLabel {{ color:{_C_TEXT}; font-size:11px; }} "
            f"QGroupBox {{ color:{_C_TEXT2}; font-size:10px; font-weight:700; "
            f"  border:1px solid {_C_BORDER}; border-radius:3px; "
            f"  margin-top:8px; padding-top:12px; }} "
            f"QGroupBox::title {{ subcontrol-origin:margin; left:8px; }} "
            f"QListWidget {{ background:{_C_CARD}; color:{_C_TEXT}; "
            f"  border:1px solid {_C_BORDER}; border-radius:3px; font-size:11px; }} "
            f"QListWidget::item {{ padding:4px 6px; }} "
            f"QListWidget::item:selected {{ background:{_C_RAISED}; "
            f"  border-left:2px solid {_C_ACCENT}; }}"
        )

        self._n = int(n_frames)
        self._fps = float(fps) if fps and fps > 0 else 1.0
        self._pairs: List[Tuple[int, int]] = list(existing or [])
        self._sequence_mode = (existing_mode if existing_mode in
                               ("sliding", "non_overlapping") else "custom")
        self._initial_strain_window = max(0, int(strain_window))
        self._grid_spacing = max(1, int(grid_spacing))

        self._build_ui()
        self._refresh_list()

    # -- construction ---------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("Build a smoothed frame-pair sequence")
        title.setStyleSheet(f"color:{_C_TEXT}; font-size:15px; font-weight:700;")
        root.addWidget(title)

        sub = QLabel(
            "Each pair becomes one item on the Results timeline. A wider pair "
            "averages motion over a longer interval, reducing frame-to-frame "
            "noise. Green–Lagrange strain is recomputed from that pair's composed "
            "displacement rather than copied from either endpoint."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        root.addWidget(sub)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background:{_C_BORDER}; max-height:1px;")
        root.addWidget(line)

        # ── Add a single pair ──────────────────────────────────────────
        add_grp = QGroupBox("Add a pair")
        ag = QGridLayout(add_grp)
        ag.setSpacing(8)

        ag.addWidget(QLabel("From frame"), 0, 0)
        self._a_spin = QSpinBox()
        self._a_spin.setRange(1, max(1, self._n))
        self._a_spin.setValue(1)
        self._a_spin.setStyleSheet(_SPIN)
        self._a_spin.valueChanged.connect(self._sync_add_note)
        ag.addWidget(self._a_spin, 0, 1)

        ag.addWidget(QLabel("to frame"), 0, 2)
        self._b_spin = QSpinBox()
        self._b_spin.setRange(1, max(1, self._n))
        self._b_spin.setValue(min(2, self._n))
        self._b_spin.setStyleSheet(_SPIN)
        self._b_spin.valueChanged.connect(self._sync_add_note)
        ag.addWidget(self._b_spin, 0, 3)

        add_btn = QPushButton("Add")
        add_btn.setStyleSheet(_BTN)
        add_btn.clicked.connect(self._add_pair)
        ag.addWidget(add_btn, 0, 4)

        self._add_note = QLabel("")
        self._add_note.setStyleSheet(f"color:{_C_TEXT2}; font-size:10px;")
        ag.addWidget(self._add_note, 1, 0, 1, 5)
        root.addWidget(add_grp)

        # ── Bulk add ───────────────────────────────────────────────────
        # Typing out twenty consecutive pairs by hand is the common case and
        # the most annoying one, so it gets a one-click path.
        bulk_grp = QGroupBox("Add many at once")
        bg = QGridLayout(bulk_grp)
        bg.setSpacing(8)

        bg.addWidget(QLabel("Frames"), 0, 0)
        self._bulk_from = QSpinBox()
        self._bulk_from.setRange(1, max(1, self._n))
        self._bulk_from.setValue(1)
        self._bulk_from.setStyleSheet(_SPIN)
        bg.addWidget(self._bulk_from, 0, 1)

        bg.addWidget(QLabel("to"), 0, 2)
        self._bulk_to = QSpinBox()
        self._bulk_to.setRange(1, max(1, self._n))
        self._bulk_to.setValue(self._n)
        self._bulk_to.setStyleSheet(_SPIN)
        bg.addWidget(self._bulk_to, 0, 3)

        bg.addWidget(QLabel("span"), 0, 4)
        self._bulk_span = QSpinBox()
        self._bulk_span.setRange(1, max(1, self._n - 1))
        self._bulk_span.setValue(1)
        self._bulk_span.setToolTip(
            "Frames between the two ends of each pair.\n"
            "1 pairs every frame with the next one.")
        self._bulk_span.setStyleSheet(_SPIN)
        bg.addWidget(self._bulk_span, 0, 5)

        seq_btn = QPushButton("Add sliding")
        seq_btn.setStyleSheet(_BTN)
        seq_btn.setToolTip(
            "Advance one frame at a time: (1→4), (2→5), (3→6) …")
        seq_btn.clicked.connect(lambda: self._add_bulk(overlap=True))
        bg.addWidget(seq_btn, 1, 0, 1, 3)

        blk_btn = QPushButton("Add non-overlapping")
        blk_btn.setStyleSheet(_BTN)
        blk_btn.setToolTip(
            "Advance by the span without gaps: (1→4), (4→7), (7→10) …")
        blk_btn.clicked.connect(lambda: self._add_bulk(overlap=False))
        bg.addWidget(blk_btn, 1, 3, 1, 3)
        root.addWidget(bulk_grp)

        # Temporal smoothing and spatial derivative smoothing are independent.
        # Keeping this value local to the pair sequence lets a user recalculate
        # pair strains with more/less spatial support without changing the
        # strain window used by the original single-frame analysis.
        strain_grp = QGroupBox("Spatial strain recalculation")
        sg = QGridLayout(strain_grp)
        sg.setSpacing(8)
        sg.addWidget(QLabel("Pair strain half-window"), 0, 0)
        self._strain_window = QSpinBox()
        self._strain_window.setRange(0, 2000)
        self._strain_window.setValue(self._initial_strain_window)
        self._strain_window.setSuffix(" px")
        self._strain_window.setStyleSheet(_SPIN)
        self._strain_window.setToolTip(
            "Spatial least-squares radius used only when recomputing strain for\n"
            "this temporal pair sequence. It does not alter single-frame strain.")
        self._strain_window.valueChanged.connect(self._sync_strain_note)
        sg.addWidget(self._strain_window, 0, 1)
        self._strain_note = QLabel("")
        self._strain_note.setWordWrap(True)
        self._strain_note.setStyleSheet(f"color:{_C_TEXT2}; font-size:10px;")
        sg.addWidget(self._strain_note, 1, 0, 1, 2)
        root.addWidget(strain_grp)

        # ── Current pairs ──────────────────────────────────────────────
        root.addWidget(QLabel("Selected pairs"))
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.setMinimumHeight(150)
        root.addWidget(self._list, 1)

        row = QHBoxLayout()
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(f"color:{_C_TEXT2}; font-size:11px;")
        row.addWidget(self._count_lbl)
        row.addStretch()
        rm_btn = QPushButton("Remove selected")
        rm_btn.setStyleSheet(_BTN)
        rm_btn.clicked.connect(self._remove_selected)
        row.addWidget(rm_btn)
        clr_btn = QPushButton("Clear all")
        clr_btn.setStyleSheet(_BTN)
        clr_btn.clicked.connect(self._clear)
        row.addWidget(clr_btn)
        root.addLayout(row)

        # ── Dialog buttons ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(_BTN)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)
        self._ok_btn = QPushButton("Show sequence")
        self._ok_btn.setStyleSheet(_BTN_ACCENT)
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._accept_if_valid)
        btn_row.addWidget(self._ok_btn)
        root.addLayout(btn_row)

        self._sync_add_note()
        self._sync_strain_note()

    # -- helpers --------------------------------------------------------

    def _interval_text(self, a: int, b: int) -> str:
        """Elapsed time across a pair, in whatever unit reads cleanly."""
        dt = abs(b - a) / self._fps
        if dt >= 1.0:
            return f"{dt:.4g} s"
        if dt >= 1e-3:
            return f"{dt * 1e3:.4g} ms"
        return f"{dt * 1e6:.4g} µs"

    def _sync_add_note(self) -> None:
        a, b = self._a_spin.value(), self._b_spin.value()
        if a == b:
            self._add_note.setText(
                f"<span style='color:{_C_WARN}'>A pair needs two different frames.</span>")
        else:
            self._add_note.setText(
                f"Δt = {self._interval_text(a, b)}  ·  spans {abs(b - a)} frame(s)")

    def _sync_strain_note(self) -> None:
        requested = self._strain_window.value()
        minimum = self._grid_spacing
        effective = max(requested, minimum)
        points = 2 * (effective // self._grid_spacing) + 1
        note = (f"Effective radius {effective} px · {points} nominal grid points "
                "per axis. Temporal pairing and this spatial fit are both applied.")
        if requested < minimum:
            note += f" Values below {minimum} px are safely clamped."
        self._strain_note.setText(note)

    def _add(self, a0: int, b0: int) -> bool:
        """Add one 0-based pair, normalised and de-duplicated."""
        if a0 == b0:
            return False
        key = (min(a0, b0), max(a0, b0))
        if key in self._pairs:
            return False
        self._pairs.append(key)
        return True

    def _add_pair(self) -> None:
        a, b = self._a_spin.value() - 1, self._b_spin.value() - 1
        if a == b:
            QMessageBox.information(self, "PyDIC", "A pair needs two different frames.")
            return
        if not self._add(a, b):
            QMessageBox.information(self, "PyDIC", "That pair is already in the list.")
            return
        self._sequence_mode = "custom"
        self._refresh_list()

    def _add_bulk(self, overlap: bool) -> None:
        lo = self._bulk_from.value() - 1
        hi = self._bulk_to.value() - 1
        span = self._bulk_span.value()
        if hi <= lo:
            QMessageBox.information(
                self, "PyDIC", "The end frame must come after the start frame.")
            return

        # Sliding windows move by one displayed frame. Disjoint windows share
        # only their boundary frame, so their *time intervals* do not overlap:
        # 1→4 covers intervals 1→2, 2→3, 3→4 and the next begins at frame 4.
        # The previous span/span*2 rule produced 1→4, 4→7 for "sequential" and
        # 1→4, 7→10 for "non-overlapping", which both skipped requested starts.
        stride = 1 if overlap else span
        was_empty = not self._pairs
        added = 0
        start = lo
        while start + span <= hi:
            if self._add(start, start + span):
                added += 1
            start += stride

        if added == 0:
            QMessageBox.information(
                self, "PyDIC",
                "No new pairs — that range and span produced nothing not already listed.")
            return
        self._sequence_mode = (
            "sliding" if overlap else "non_overlapping") if was_empty else "custom"
        self._refresh_list()

    def _remove_selected(self) -> None:
        rows = sorted((self._list.row(i) for i in self._list.selectedItems()), reverse=True)
        for r in rows:
            if 0 <= r < len(self._pairs):
                del self._pairs[r]
        self._sequence_mode = "custom"
        self._refresh_list()

    def _clear(self) -> None:
        self._pairs.clear()
        self._sequence_mode = "custom"
        self._refresh_list()

    def _refresh_list(self) -> None:
        self._pairs.sort()
        self._list.clear()
        for a, b in self._pairs:
            it = QListWidgetItem(
                f"  frame {a + 1}  →  frame {b + 1}      "
                f"Δt = {self._interval_text(a, b)}")
            self._list.addItem(it)

        n = len(self._pairs)
        if n == 0:
            self._count_lbl.setText("No pairs selected")
        else:
            self._count_lbl.setText(
                f"{n} timeline pair{'s' if n != 1 else ''}")
        self._ok_btn.setEnabled(n > 0)

    def _accept_if_valid(self) -> None:
        if not self._pairs:
            QMessageBox.information(self, "PyDIC", "Add at least one frame pair.")
            return
        self.accept()

    # -- result ---------------------------------------------------------

    def pairs(self) -> List[Tuple[int, int]]:
        """Selected pairs as 0-based (a, b) frame indices."""
        return list(self._pairs)

    def strain_window(self) -> int:
        """Requested spatial half-window for pair-only strain recalculation."""
        return int(self._strain_window.value())

    def sequence_mode(self) -> str:
        """How the current list was generated: bulk temporal or hand-picked."""
        return self._sequence_mode
