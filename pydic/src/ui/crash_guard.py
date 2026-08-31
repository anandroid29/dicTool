"""
Keep an unhandled exception from taking the window with it.

PyQt calls into Python from C++. When a slot raises and nothing catches it,
the exception cannot propagate back through the C++ stack, so Qt aborts the
process: the window disappears with no message and no chance to save. Every
crash reported against this application so far has had that shape -- a
TypeError or AttributeError in a signal handler, fatal only because of where
it happened.

Reporting the fault and carrying on is the right trade here. The state after a
failed handler is a redraw or a click that did not complete, not a corrupted
analysis, and losing an hour of correlation to a mistyped attribute is far
worse than a stale panel.
"""
from __future__ import annotations

import sys
import traceback
from typing import Optional

_installed = False
_seen: set[str] = set()
_suppressed = 0


def install(parent=None) -> None:
    """Route unhandled exceptions to stderr and a dialog instead of abort()."""
    global _installed
    if _installed:
        return
    _installed = True
    previous = sys.excepthook

    def hook(exc_type, exc, tb) -> None:
        # Ctrl+C and an explicit exit are requests, not faults.
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            previous(exc_type, exc, tb)
            return

        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        sys.stderr.flush()

        # A fault inside a paint or timer handler repeats every frame. Report
        # each distinct one once; a dialog per repaint would be its own denial
        # of service, and the stderr log above keeps the full record.
        global _suppressed
        key = f"{exc_type.__name__}:{_origin(tb)}"
        if key in _seen:
            _suppressed += 1
            return
        _seen.add(key)

        try:
            _show(exc_type, exc, tb, parent)
        except Exception:
            # Never let the reporter become the crash.
            pass

    sys.excepthook = hook


def _origin(tb) -> str:
    """Innermost application frame, so the same bug keys the same way."""
    last = ""
    while tb is not None:
        frame = tb.tb_frame
        last = f"{frame.f_code.co_filename}:{tb.tb_lineno}"
        tb = tb.tb_next
    return last


def _show(exc_type, exc, tb, parent) -> None:
    from PyQt6.QtWidgets import QApplication, QMessageBox

    if QApplication.instance() is None:
        return
    summary = "".join(traceback.format_exception_only(exc_type, exc)).strip()
    where = _origin(tb)

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Something went wrong")
    box.setText("An action did not complete.")
    box.setInformativeText(
        f"{summary}\n\nThe application is still running and your analysis is "
        f"intact. If the panel looks wrong, change frame or field to redraw it.")
    box.setDetailedText("".join(traceback.format_exception(exc_type, exc, tb))
                        + f"\n\nOrigin: {where}")
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def suppressed_count() -> int:
    """Repeats folded away after the first report of each distinct fault."""
    return _suppressed
