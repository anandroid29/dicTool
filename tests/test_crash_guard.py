"""
An exception in a Qt slot must not end the process.

PyQt calls slots from C++, so an exception that escapes one cannot propagate
and Qt aborts. Both crashes reported against this application were that shape:
an ordinary TypeError or AttributeError in a signal handler, fatal only for
where it happened. The guard reports and continues.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

ROOT = os.path.dirname(os.path.dirname(__file__))
PYDIC = os.path.join(ROOT, "pydic")
if PYDIC not in sys.path:
    sys.path.insert(0, PYDIC)

import src.ui.crash_guard as crash_guard  # noqa: E402

APP = QApplication.instance() or QApplication([])


class _Emitter(QObject):
    fired = pyqtSignal()


@pytest.fixture
def guard(monkeypatch):
    """Install the guard with the dialog stubbed out, isolated per test."""
    shown = []
    monkeypatch.setattr(crash_guard, "_installed", False)
    monkeypatch.setattr(crash_guard, "_seen", set())
    monkeypatch.setattr(crash_guard, "_suppressed", 0)
    monkeypatch.setattr(crash_guard, "_show",
                        lambda t, e, tb, parent: shown.append(t))
    original = sys.excepthook
    crash_guard.install()
    yield shown
    sys.excepthook = original


def test_raising_slot_is_reported_and_survives(guard, capsys):
    emitter = _Emitter()

    def bad():
        raise TypeError("'list' object is not callable")

    emitter.fired.connect(bad)
    emitter.fired.emit()          # would abort the process unguarded
    APP.processEvents()

    assert guard == [TypeError], "the fault was not reported"
    assert "not callable" in capsys.readouterr().err, (
        "the traceback must still reach stderr for diagnosis")


def test_repeated_faults_report_once(guard):
    """A fault in a repaint recurs every frame; one dialog, not hundreds."""
    emitter = _Emitter()
    emitter.fired.connect(lambda: (_ for _ in ()).throw(ValueError("boom")))
    for _ in range(25):
        emitter.fired.emit()
    APP.processEvents()

    assert len(guard) == 1, f"expected one report, got {len(guard)}"
    assert crash_guard.suppressed_count() == 24


def test_distinct_faults_are_each_reported(guard):
    a, b = _Emitter(), _Emitter()
    a.fired.connect(lambda: (_ for _ in ()).throw(ValueError("first")))
    b.fired.connect(lambda: (_ for _ in ()).throw(AttributeError("second")))
    a.fired.emit()
    b.fired.emit()
    APP.processEvents()

    assert guard == [ValueError, AttributeError]


def test_exit_and_interrupt_are_not_swallowed(guard):
    """Quitting is a request, not a fault; it must reach the default hook."""
    passed = []
    original = sys.excepthook

    try:
        sys.excepthook = lambda *a: passed.append(a[0])
        crash_guard._installed = False
        crash_guard.install()
        sys.excepthook(SystemExit, SystemExit(0), None)
    finally:
        sys.excepthook = original

    assert passed == [SystemExit]
    assert guard == [], "SystemExit must not raise a fault dialog"
