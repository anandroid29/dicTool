"""
Sidebar layout, the export menu, and the marker history plot.

The sidebar holds more than fits on a short screen, so it scrolls rather than
clipping its lower sections. The exports are behind one menu instead of a
column of buttons. The plot is the temporal counterpart to the field views.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QScrollArea

ROOT = os.path.dirname(os.path.dirname(__file__))
PYDIC = os.path.join(ROOT, "pydic")
if PYDIC not in sys.path:
    sys.path.insert(0, PYDIC)

from src.core.analysis import DICAnalysis  # noqa: E402
from src.ui.pages.marker_plot_dialog import MarkerPlotDialog, PLOTTABLE  # noqa: E402
from src.ui.pages.results_page import ResultsPage  # noqa: E402
from src.ui.wizard import Wizard  # noqa: E402

APP = QApplication.instance() or QApplication([])

H = W = 120
SPACING = 4
FPS = 10.0


class _Res:
    pass


def _analysis(n_frames=25, rate=lambda i: float(i)):
    a = object.__new__(DICAnalysis)

    class _P:
        subset_spacing = SPACING

    a.params = _P()
    a.fps = FPS
    a._ref_image = np.zeros((H, W), np.uint8)
    ys, xs = np.mgrid[0:H:SPACING, 0:W:SPACING]
    results = []
    for i in range(n_frames):
        r = _Res()
        for name, _, _ in PLOTTABLE:
            setattr(r, name, None)
        u = np.full((H, W), np.nan, np.float32)
        v = np.full((H, W), np.nan, np.float32)
        e = np.full((H, W), np.nan, np.float32)
        u[ys, xs] = 1.0
        v[ys, xs] = 0.0
        e[ys, xs] = rate(i)
        r.u, r.v, r.Eeff_rate = u, v, e
        r.Veff = np.where(np.isfinite(u), 10.0, np.nan).astype(np.float32)
        r.valid = np.isfinite(u)
        r.image_path = "frame.tif"
        results.append(r)
    a.results = results
    return a


# --- sidebar ---------------------------------------------------------------

def _page():
    wizard = Wizard()
    page = ResultsPage(wizard)
    return wizard, page


def test_sidebar_scrolls_instead_of_clipping():
    """
    Every sidebar section must stay reachable at a short window height.

    A plain layout answers an overflow by squeezing widgets below their minimum
    and clipping the rest, which is how the export section became unreadable.
    """
    wizard, page = _page()
    try:
        areas = page.findChildren(QScrollArea)
        assert areas, "sidebar is not scrollable"
        area = areas[0]
        assert area.widgetResizable()
        # The panel must never scroll sideways; that would hide control labels.
        from PyQt6.QtCore import Qt
        assert (area.horizontalScrollBarPolicy()
                == Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        page.resize(900, 380)   # deliberately short
        APP.processEvents()
        body = area.widget()
        # Content taller than the viewport is the scrollable case; either way
        # the body keeps its natural height rather than being compressed.
        assert body.sizeHint().height() >= body.minimumSizeHint().height()
    finally:
        page.deleteLater()
        wizard.deleteLater()
        APP.processEvents()


def test_exports_live_behind_one_menu():
    wizard, page = _page()
    try:
        actions = [a.text() for a in page._export_menu.actions() if a.text()]
        assert any("CSV" in t for t in actions)
        assert any("HDF5" in t for t in actions)
        assert any("Video" in t for t in actions)
        assert any("time series" in t for t in actions)
        assert any("Plot" in t for t in actions)
        assert page._export_btn.menu() is page._export_menu
    finally:
        page.deleteLater()
        wizard.deleteLater()
        APP.processEvents()


def test_colour_scale_defaults_are_stable_across_frames():
    """
    A colour must mean the same thing on every frame by default, and a value
    past the trimmed limits must not be painted as if it were an error.
    """
    wizard, page = _page()
    try:
        assert page._scale_global_rb.isChecked(), (
            "per-frame rescaling as the default makes features appear to pulse")
        assert not page._clip_chk.isChecked(), (
            "clip marks on by default paint ordinary hot spots magenta")
    finally:
        page.deleteLater()
        wizard.deleteLater()
        APP.processEvents()


def test_plot_and_marker_actions_do_not_crash_without_markers():
    """Both entry points must explain themselves rather than raise."""
    wizard, page = _page()
    try:
        from PyQt6.QtWidgets import QMessageBox
        seen = []
        for name in ("information", "warning"):
            setattr(QMessageBox, name,
                    staticmethod(lambda *a, **k: seen.append(a[1:3])))
        page._plot_marker_timeseries()
        page._export_marker_timeseries()
        assert len(seen) == 2, "expected an explanation, not an exception"
    finally:
        page.deleteLater()
        wizard.deleteLater()
        APP.processEvents()


# --- plot ------------------------------------------------------------------

def test_plot_series_follow_the_material():
    """Curves are sampled along the advected path, not at a fixed pixel."""
    a = _analysis(rate=lambda i: float(i))
    dlg = MarkerPlotDialog(a, [(20.0, 60.0)], labels=["P1"])
    try:
        y = dlg._series[0]["Eeff_rate"]
        assert np.isfinite(y).all()
        assert np.allclose(y, np.arange(len(y), dtype=float))
    finally:
        dlg.deleteLater()
        APP.processEvents()


def test_plot_axis_falls_back_to_frame_without_a_capture_rate():
    a = _analysis()
    a.fps = 0.0
    dlg = MarkerPlotDialog(a, [(20.0, 60.0)])
    try:
        assert not dlg._have_time
        assert "frame number" in dlg._status.text()
    finally:
        dlg.deleteLater()
        APP.processEvents()


def test_plot_smoothing_keeps_gaps_as_gaps():
    """A dropout must not be smoothed into a plausible-looking value."""
    y = np.array([1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, 1.0])
    out = MarkerPlotDialog._smooth(y, window=3)
    assert np.isnan(out[4]), "smoothing invented a value across a wide gap"
    assert np.isfinite(out[0])


def test_plot_csv_matches_the_plotted_curve(tmp_path, monkeypatch):
    a = _analysis(n_frames=12, rate=lambda i: 2.0 * i)
    dlg = MarkerPlotDialog(a, [(20.0, 60.0)], labels=["P1"])
    out = tmp_path / "hist.csv"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        staticmethod(lambda *args, **kw: (str(out), "")))
    try:
        dlg._save_csv()
        rows = out.read_text(encoding="utf-8").splitlines()
        assert rows[0].startswith("frame,time_s,")
        assert len(rows) == 13
        # Frame 3 at 2*i, and time is frame / fps.
        frame, t, value = rows[4].split(",")
        assert int(frame) == 3
        assert float(t) == pytest.approx(3 / FPS)
        assert float(value) == pytest.approx(6.0)
    finally:
        dlg.deleteLater()
        APP.processEvents()


def test_plot_reports_markers_that_lose_tracking():
    """A marker that leaves the field is named, not silently dropped."""
    a = _analysis(n_frames=20)
    for res in a.results[10:]:
        res.u = np.full((H, W), np.nan, np.float32)
        res.v = np.full((H, W), np.nan, np.float32)
    dlg = MarkerPlotDialog(a, [(20.0, 60.0)], labels=["P1"])
    try:
        y = dlg._series[0]["Eeff_rate"]
        assert np.isfinite(y[:10]).all()
        assert not np.isfinite(y[12:]).any()
        assert "P1" in dlg._status.text()
    finally:
        dlg.deleteLater()
        APP.processEvents()


# --- pair colour range -----------------------------------------------------

class _FakePairStore:
    """Minimal stand-in for a computed pair sequence on disk."""

    def __init__(self, fields):
        self._fields = fields

    def __len__(self):
        return len(self._fields)

    def has(self, index):
        return 0 <= index < len(self._fields)

    def __getitem__(self, index):
        holder = _Res()
        holder.u = self._fields[index]
        return holder


def test_pair_range_covers_the_whole_sequence_not_just_the_first_pair():
    """
    The fixed pair scale is seeded from pairs sampled across the sequence.

    Seeding from the first pair alone under-covers whenever the field
    intensifies later, which is the normal case in cutting: every subsequent
    pair then sits past the top of the scale and reads as clipped.
    """
    wizard, page = _page()
    try:
        # Pair 0 is quiet, a later pair is ten times hotter.
        fields = []
        for i in range(10):
            arr = np.full((20, 20), 1.0 if i < 5 else 10.0, np.float32)
            fields.append(arr)
        page._pair_store = _FakePairStore(fields)
        page._field = "u"

        from src.ui.render import RangeSpec
        spec = RangeSpec(mode="auto", symmetric=False, percentile=100.0)
        rng = page._pair_sequence_range(spec, 1.0)

        assert rng is not None
        assert rng[1] == pytest.approx(10.0), (
            "range stopped at the first pair and would clip every later one")
    finally:
        page.deleteLater()
        wizard.deleteLater()
        APP.processEvents()


def test_pair_range_falls_back_when_nothing_is_precomputed():
    """With no computed pairs the caller must fall back, not fail."""
    wizard, page = _page()
    try:
        page._pair_store = None
        from src.ui.render import RangeSpec
        spec = RangeSpec(mode="auto", symmetric=False, percentile=100.0)
        assert page._pair_sequence_range(spec, 1.0) is None
    finally:
        page.deleteLater()
        wizard.deleteLater()
        APP.processEvents()
