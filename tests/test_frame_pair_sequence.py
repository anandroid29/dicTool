import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QPushButton


ROOT = os.path.dirname(os.path.dirname(__file__))
STRAINX = os.path.join(ROOT, "src")
if STRAINX not in sys.path:
    sys.path.insert(0, STRAINX)

from strainx.core.analysis import DICAnalysis, PairResult
from strainx.core.rg_dic import DICParams
from strainx.core.temporal import TemporalResultSequence, save_temporal_result
from strainx.ui.pages.frame_pair_dialog import FramePairDialog
from strainx.ui.pages.results_page import ResultsPage
from strainx.ui.pages.video_export_dialog import VideoExportDialog
from strainx.ui.render import PanelSpec
from strainx.ui.video_export import ExportSpec, ViewRenderer, export_video
from strainx.ui.wizard import Wizard


APP = QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    APP.processEvents()
    return bool(predicate())


def test_bulk_sliding_pairs_advance_one_frame():
    dialog = FramePairDialog(10, fps=1000.0)
    dialog._bulk_from.setValue(1)
    dialog._bulk_to.setValue(10)
    dialog._bulk_span.setValue(3)

    dialog._add_bulk(overlap=True)

    assert dialog.pairs() == [
        (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9)
    ]
    assert dialog.sequence_mode() == "sliding"
    dialog.close()


def test_bulk_non_overlapping_pairs_are_contiguous_without_skips():
    dialog = FramePairDialog(10, fps=1000.0)
    dialog._bulk_from.setValue(1)
    dialog._bulk_to.setValue(10)
    dialog._bulk_span.setValue(3)

    dialog._add_bulk(overlap=False)

    assert dialog.pairs() == [(0, 3), (3, 6), (6, 9)]
    assert dialog.sequence_mode() == "non_overlapping"
    dialog.close()


def test_pair_dialog_keeps_an_independent_spatial_strain_window():
    dialog = FramePairDialog(
        10, fps=1000.0, strain_window=12, grid_spacing=3)

    assert dialog.strain_window() == 12
    dialog._strain_window.setValue(21)
    assert dialog.strain_window() == 21
    assert "15 nominal grid points" in dialog._strain_note.text()
    dialog.close()


def _increment(shape, radius, spacing, u_at_grid, v_at_grid):
    u = np.full(shape, np.nan, dtype=np.float32)
    v = np.full(shape, np.nan, dtype=np.float32)
    ys = np.arange(radius, shape[0] - radius, spacing)
    xs = np.arange(radius, shape[1] - radius, spacing)
    gx, gy = np.meshgrid(xs, ys)
    u[gy, gx] = u_at_grid(gx, gy)
    v[gy, gx] = v_at_grid(gx, gy)
    valid = np.isfinite(u) & np.isfinite(v)
    nan = np.full(shape, np.nan, dtype=np.float32)
    return PairResult(
        image_path="synthetic",
        u=u,
        v=v,
        Exx=nan,
        Exy=nan,
        Eyy=nan,
        Eeff=nan,
        du_dx=nan,
        du_dy=nan,
        dv_dx=nan,
        dv_dy=nan,
        corr=nan,
        valid=valid,
    )


def test_pair_green_lagrange_uses_selected_history_endpoint(monkeypatch):
    shape, radius, spacing = (81, 81), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=10)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis.fps = 1000.0

    stretch = 0.01
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: stretch * x,
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame, frame]
    captured = {}

    def selected_history(_window, use_gpu=False, progress_cb=None,
                         cancel_flag=None, min_frame=None, max_frame=None):
        captured["range"] = (min_frame, max_frame)
        return tuple({
            name: np.full(shape, 0.1 * index + 0.01 * offset,
                          dtype=np.float32)
            for offset, name in enumerate(
                ("Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"))
        } for index in range(max_frame + 1))

    monkeypatch.setattr(
        analysis, "_temporal_accumulated_strains", selected_history)

    pair = analysis.pair_kinematics(0, 3)
    assert captured["range"] == (1, 3)
    assert np.nanmedian(pair.Exx_gl) == pytest.approx(0.3)


def test_pair_rigid_translation_has_zero_green_lagrange_strain():
    shape, radius, spacing = (81, 81), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=10)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis.fps = 1000.0
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.4, dtype=float),
        lambda _x, y: np.full_like(y, -0.2, dtype=float),
    )
    analysis.results = [frame, frame, frame, frame]

    pair = analysis.pair_kinematics(0, 3)

    assert np.nanmax(np.abs(pair.Exx_gl)) < 1e-6
    assert np.nanmax(np.abs(pair.Exy_gl)) < 1e-6
    assert np.nanmax(np.abs(pair.Eyy_gl)) < 1e-6


def test_pair_rate_comes_from_composed_end_to_end_motion():
    from strainx.core.strain import compute_velocity_strains

    shape, radius, spacing = (81, 81), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=10)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis.fps = 1000.0

    frames = []
    stretches = (0.0, 0.01, -0.01, 0.004)
    for stretch in stretches:
        frame = _increment(
            shape, radius, spacing,
            lambda x, _y, amount=stretch: amount * x,
            lambda _x, y: np.zeros_like(y, dtype=float),
        )
        frames.append(frame)
    analysis.results = frames

    pair = analysis.pair_kinematics(0, 3)
    values = pair.Exx_rate[np.isfinite(pair.Exx_rate)]

    # Pair 0→3 consumes increments 1..3.  Rate is Grad(U_total)/dt, where the
    # total material map is composed multiplicatively.
    total_f11 = np.prod([1.0 + value for value in stretches[1:]])
    expected = (total_f11 - 1.0) / pair.elapsed
    assert np.median(values) == pytest.approx(expected, rel=2e-3, abs=2e-3)
    direct = compute_velocity_strains(
        np.asarray(pair.Vx), np.asarray(pair.Vy), np.asarray(pair.valid),
        10, spacing)
    for name in ("Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate"):
        assert np.allclose(
            np.asarray(getattr(pair, name)), np.asarray(direct[name]),
            rtol=3e-3, atol=2e-5, equal_nan=True)


def test_changed_pair_window_refits_spatial_strain_and_rate(monkeypatch):
    from strainx.core import strain as strain_module

    shape, radius, spacing = (61, 61), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis.fps = 1000.0
    analysis.results = [
        _increment(
            shape, radius, spacing,
            lambda x, _y: 0.001 * x,
            lambda _x, y: np.zeros_like(y, dtype=float),
        ) for _ in range(3)
    ]
    original = strain_module.compute_velocity_strains
    calls = []

    def recording_fit(*args, **kwargs):
        calls.append((args[3], kwargs.get("use_gpu", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(strain_module, "compute_velocity_strains", recording_fit)
    result = analysis.pair_kinematics(0, 2, strain_window=15)

    # Pair 0→2 has two selected history increments. A third fit differentiates
    # the temporally averaged velocity itself.
    assert calls == [
        (15, False), (15, False), (15, False)]
    assert np.isfinite(result.Exx_rate).any()
    assert np.isfinite(result.Exx_gl).any()


def test_temporal_history_stops_at_selected_pair_endpoint():
    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis.results = [
        _increment(
            shape, radius, spacing,
            lambda x, _y: 0.001 * x,
            lambda _x, y: np.zeros_like(y, dtype=float),
        ) for _ in range(8)
    ]
    messages = []

    analysis.pair_kinematics(
        0, 2, strain_window=9,
        progress_cb=lambda _fraction, message: messages.append(message))

    history = [message for message in messages
               if message.startswith("Strain history")]
    assert history == [
        "Strain history 1/2", "Strain history 2/2"]


def test_bulk_hides_rates_until_each_averaged_velocity_is_derived(monkeypatch):
    from strainx.core import strain as strain_module

    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: 0.001 * x,
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame, frame]
    events = []
    original_pair = analysis.pair_kinematics
    original_fit = strain_module.compute_velocity_strains
    first_started = threading.Event()
    allow_first = threading.Event()
    second_started = threading.Event()
    allow_second = threading.Event()
    fit_count = [0]

    def record_pair(a, b, **kwargs):
        events.append(("velocity", a, b))
        return original_pair(a, b, **kwargs)

    def controlled_fit(*args, **kwargs):
        fit_count[0] += 1
        events.append(("rate", fit_count[0]))
        if fit_count[0] == 1:
            first_started.set()
            assert allow_first.wait(2.0)
        elif fit_count[0] == 2:
            second_started.set()
            assert allow_second.wait(2.0)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(analysis, "pair_kinematics", record_pair)
    monkeypatch.setattr(strain_module, "compute_velocity_strains", controlled_fit)
    page = ResultsPage(SimpleNamespace(analysis=analysis, new_session=lambda: None))
    page._pair_list = [(0, 2), (1, 3)]
    page._pair_mode = True
    page._pair_sequence_mode = "sliding"
    page._pair_strain_window = 9
    page._begin_pair_bulk()

    try:
        assert _wait_until(first_started.is_set)
        assert len(page._pair_rate_ready) == 2
        assert page._pair_store[0].Exx_rate.values.size == 0
        assert page._pair_store[1].Exx_rate.values.size == 0

        allow_first.set()
        assert _wait_until(second_started.is_set)
        assert _wait_until(lambda: 0 in page._pair_bulk_ready)
        saved_first = page._pair_store[0]
        assert np.isfinite(saved_first.Exx_rate.values).any()
        assert page._pair_store[1].Exx_rate.values.size == 0
        assert page._pair_avg is not None
        assert np.allclose(
            np.asarray(page._pair_avg.Exx_rate),
            np.asarray(saved_first.Exx_rate), equal_nan=True)
        direct = original_fit(
            np.asarray(saved_first.Vx), np.asarray(saved_first.Vy),
            np.asarray(saved_first.valid), 9, spacing)
        assert np.allclose(
            np.asarray(saved_first.Exx_rate), direct["Exx_rate"],
            rtol=3e-3, atol=2e-5, equal_nan=True)

        allow_second.set()
        assert _wait_until(lambda: len(page._pair_bulk_ready) == 2)
        assert [event[0] for event in events] == [
            "velocity", "velocity", "rate", "rate"]
    finally:
        allow_first.set()
        allow_second.set()
        page._invalidate_pair_jobs()
        page._pair_pool.waitForDone(3000)
        page.close()


def test_results_pair_sequence_keeps_timeline_and_restores_single_frame():
    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis._ref_image = np.zeros(shape, dtype=float)
    analysis.fps = 1000.0
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.1, dtype=float),
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame, frame]
    analysis.def_paths = []
    page = ResultsPage(SimpleNamespace(analysis=analysis, new_session=lambda: None))
    page._single_frame_before_pairs = 2
    page._pair_list = [(0, 2), (1, 3)]
    page._pair_mode = True
    page._slider.setMaximum(1)
    page._sync_pair_ui()

    page._slider.setValue(1)
    page._scrub_timer.stop()
    page._render_scrubbed_frame()

    assert page._slider.isEnabled()
    assert page._play_btn.isEnabled()
    assert _wait_until(lambda: page._pair_avg is not None)
    assert page._pair_avg.pair_start == 1
    assert page._pair_avg.pair_end == 3
    assert "Pair 2 / 2" in page._frame_lbl.text()

    generation = page._pair_generation
    retained_pairs = list(page._pair_list)
    page._clear_pair_average()

    assert not page._pair_mode
    assert page._pair_generation == generation
    assert page._pair_list == retained_pairs
    assert page._slider.maximum() == 3
    # Leaving temporal mode stays on the endpoint image of the selected pair.
    assert page._slider.value() == 3

    page._resume_pair_sequence()
    assert page._pair_mode
    assert page._slider.value() == 1
    assert _wait_until(lambda: page._pair_avg is not None)
    assert page._pair_avg.pair_start == 1

    page._discard_pair_data()
    assert not page._pair_list
    assert page._pair_generation == generation + 1
    page._pair_pool.waitForDone(3000)
    page.close()


def test_temporal_mode_streakline_uses_the_pair_endpoint():
    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis._ref_image = np.zeros(shape, dtype=float)
    analysis.fps = 1000.0
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.1, dtype=float),
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame, frame]
    analysis.def_paths = []
    page = ResultsPage(SimpleNamespace(analysis=analysis, new_session=lambda: None))
    page._pair_list = [(0, 2), (1, 3)]
    page._pair_mode = True
    page._frame = 1
    page._canvas.set_markers([(radius, radius)])
    page._streak_chk.blockSignals(True)
    page._streak_chk.setChecked(True)
    page._streak_chk.blockSignals(False)
    page._render_trajectories(1)

    assert page._trajectory_source_index(1) == 3
    assert _wait_until(lambda: len(page._canvas._streak_paths) == 1)
    trajectories, _draw_points = next(iter(page._traj_cache.values()))
    assert len(trajectories[0]["points"]) == 5
    assert trajectories[0]["points"][-1][0] == pytest.approx(radius + 0.4)
    page._pair_pool.waitForDone(3000)
    page.close()


def test_sidebar_export_buttons_and_marker_order_are_compact_but_unclipped():
    analysis = DICAnalysis()
    page = ResultsPage(SimpleNamespace(analysis=analysis, new_session=lambda: None))
    video_button = next(
        button for button in page.findChildren(QPushButton)
        if button.text() == "Video / image sequence…")

    assert video_button.height() == 34
    assert video_button.parentWidget().layout().spacing() == 0
    sidebar_layout = page._marker_panel.parentWidget().layout()
    assert sidebar_layout.indexOf(page._marker_panel) > sidebar_layout.indexOf(
        page._pair_clear_btn)
    page.close()


def test_video_export_uses_source_dimensions_without_streakline_only_panel():
    dialog = VideoExportDialog(
        {"u": ("Horizontal displacement", "px")}, ["turbo"], 20,
        "u", "turbo", source_size=(1920, 1080))

    assert (dialog.cell_w.value(), dialog.cell_h.value()) == (1920, 1080)
    choices = [dialog._editors[0].content.itemData(i)
               for i in range(dialog._editors[0].content.count())]
    assert "streaklines" not in choices
    assert "image" in choices
    assert dialog._editors[0].streaks.text() == "Overlay streaklines"
    dialog.close()


def test_streakline_toggle_never_traces_on_the_ui_thread(monkeypatch):
    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis._ref_image = np.zeros(shape, dtype=float)
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.1, dtype=float),
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame]
    analysis.def_paths = []
    original = analysis.get_trajectories_from_seeds

    def delayed_trace(*args, **kwargs):
        time.sleep(0.15)
        return original(*args, **kwargs)

    monkeypatch.setattr(analysis, "get_trajectories_from_seeds", delayed_trace)
    page = ResultsPage(SimpleNamespace(analysis=analysis, new_session=lambda: None))
    page._frame = 2
    page._canvas.set_markers([(radius, radius)])

    started = time.monotonic()
    page._streak_chk.setChecked(True)
    returned_in = time.monotonic() - started

    assert returned_in < 0.1
    assert len(page._canvas._streak_paths) == 0
    assert _wait_until(lambda: len(page._canvas._streak_paths) == 1)

    # Clearing while a different path is still running must not let that stale
    # worker completion put a deleted marker back on the canvas.
    page._canvas.set_markers([(radius + spacing, radius)])
    page._canvas.clear_markers()
    assert _wait_until(lambda: page._traj_active_request is None)
    assert len(page._canvas._streak_paths) == 0
    page._pair_pool.waitForDone(3000)
    page.close()


def test_pair_calculation_is_queued_without_blocking_the_ui(monkeypatch):
    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis._ref_image = np.zeros(shape, dtype=float)
    analysis.fps = 1000.0
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.1, dtype=float),
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame]
    analysis.def_paths = []
    original = analysis.pair_kinematics
    received_windows = []

    def delayed_pair(a, b, strain_window=None, use_gpu=False,
                     progress_cb=None, cancel_flag=None,
                     include_strain=True, include_rate=True):
        received_windows.append(strain_window)
        time.sleep(0.15)
        return original(
            a, b, strain_window=strain_window, use_gpu=use_gpu,
            progress_cb=progress_cb, cancel_flag=cancel_flag,
            include_strain=include_strain, include_rate=include_rate)

    monkeypatch.setattr(analysis, "pair_kinematics", delayed_pair)
    page = ResultsPage(SimpleNamespace(analysis=analysis, new_session=lambda: None))
    page._pair_list = [(0, 2)]
    page._pair_mode = True
    page._pair_strain_window = 17

    started = time.monotonic()
    page._show_pair_average()
    returned_in = time.monotonic() - started

    assert returned_in < 0.1
    assert page._pair_avg is None
    assert _wait_until(lambda: page._pair_avg is not None)
    assert received_windows == [17]
    page._clear_pair_average()
    page._pair_pool.waitForDone(3000)
    page.close()


def test_new_session_cancels_temporal_worker_without_render_reentry(monkeypatch):
    wizard = Wizard()
    analysis = wizard.analysis
    shape, radius, spacing = (41, 41), 5, 5
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.1, dtype=float),
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame]

    started = threading.Event()

    def waits_for_cancel(_a, _b, strain_window=None, use_gpu=False,
                         progress_cb=None, cancel_flag=None,
                         include_strain=True, include_rate=True):
        started.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not cancel_flag[0]:
            time.sleep(0.005)
        raise RuntimeError("Temporal calculation cancelled.")

    monkeypatch.setattr(analysis, "pair_kinematics", waits_for_cancel)
    page = wizard._results
    page._pair_list = [(0, 2)]
    page._pair_mode = True
    page._pair_sequence_mode = "sliding"
    page._pair_strain_window = 9
    page._begin_pair_bulk()
    assert started.wait(1.0)

    then = time.monotonic()
    wizard.new_session()
    elapsed = time.monotonic() - then

    assert elapsed < 0.1
    assert wizard._stack.currentIndex() == 0
    assert wizard.analysis is not analysis
    assert page._canvas.markers == []
    assert page._pair_pool.waitForDone(1000)
    wizard.close()


def test_pair_fixed_colour_range_does_not_jump_between_pairs():
    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis._ref_image = np.zeros(shape, dtype=float)
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.1, dtype=float),
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame]
    page = ResultsPage(SimpleNamespace(analysis=analysis, new_session=lambda: None))
    page._pair_mode = True
    page._pair_list = [(0, 1)]
    page._field = "u"
    page._scale_global_rb.setChecked(True)
    page._pair_avg = frame

    first = np.where(frame.valid, frame.u, np.nan)
    page._apply_overlay(first)
    saved = dict(page._pair_fixed_ranges)
    page._apply_overlay(first * 100.0)

    assert saved
    assert page._pair_fixed_ranges == saved
    page._clear_pair_average()
    page.close()


def test_generated_temporal_sequence_bulk_precomputes_for_seeking():
    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis._ref_image = np.zeros(shape, dtype=float)
    analysis.fps = 1000.0
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.1, dtype=float),
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame, frame]
    analysis.def_paths = []
    page = ResultsPage(SimpleNamespace(analysis=analysis, new_session=lambda: None))
    page._pair_list = [(0, 2), (1, 3)]
    page._pair_sequence_mode = "sliding"
    page._pair_mode = True
    page._slider.setMaximum(1)
    page._invalidate_pair_jobs()
    page._begin_pair_bulk()
    store = page._pair_store
    generation = page._pair_generation
    page._clear_pair_average()

    assert _wait_until(lambda: len(page._pair_bulk_ready) == 2, timeout=5.0)
    assert not page._pair_mode
    assert page._pair_generation == generation
    assert page._pair_store is store
    assert page._pair_store.completed_count() == 2
    assert page._export_progress.value() == 100

    page._resume_pair_sequence()
    page._pair_cache.clear()
    page._frame = 1
    loaded = page._pair_result_at(1)
    assert loaded is not None
    assert loaded.pair_start == 1
    assert loaded.pair_end == 3
    page._discard_pair_data()
    page._pair_pool.waitForDone(3000)
    page.close()


def test_hdf5_export_contains_complete_temporal_sequence(tmp_path):
    import h5py

    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis.fps = 1000.0
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.1, dtype=float),
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame]
    pairs = [(0, 2)]
    store_dir = tmp_path / "temporal"
    store_dir.mkdir()
    sequence = TemporalResultSequence(str(store_dir), pairs)
    result = analysis.pair_kinematics(0, 2)
    save_temporal_result(sequence.path_for(0), result)
    output = tmp_path / "with_temporal.h5"

    analysis.export_hdf5(
        str(output), temporal_results=sequence, temporal_pairs=pairs,
        temporal_metadata={"mode": "sliding", "strain_window": 11})

    with h5py.File(output, "r") as handle:
        group = handle["temporal_sequence"]
        assert bool(group.attrs["complete"])
        assert int(group.attrs["schema"]) == 4
        assert group.attrs["strain_semantics"] == (
            "green_lagrange_history_of_temporally_averaged_frames")
        assert group.attrs["mode"] == "sliding"
        assert int(group.attrs["strain_window"]) == 11
        assert group["pairs"][:].tolist() == [[0, 2]]
        pair = group["pair_000000"]
        assert "u" in pair
        assert "Exx_rate" in pair
        assert "Exx_gl" in pair

    reopened = DICAnalysis()
    reopened.load_hdf5(str(output))
    assert reopened.temporal_pairs == pairs
    assert reopened.temporal_metadata["mode"] == "sliding"
    restored = reopened.temporal_results[0]
    assert restored.pair_start == 0
    assert restored.pair_end == 2
    assert np.isfinite(restored.Exx_gl.values).any()


def test_video_renderer_reads_temporal_results_and_pair_endpoint(tmp_path):
    shape, radius, spacing = (41, 41), 5, 5
    analysis = DICAnalysis()
    analysis.params = DICParams(
        subset_radius=radius, subset_spacing=spacing, strain_window=5)
    analysis._roi_mask = np.ones(shape, dtype=bool)
    analysis._ref_image = np.zeros(shape, dtype=float)
    analysis.fps = 1000.0
    frame = _increment(
        shape, radius, spacing,
        lambda x, _y: np.full_like(x, 0.2, dtype=float),
        lambda _x, y: np.zeros_like(y, dtype=float),
    )
    analysis.results = [frame, frame, frame]
    pair = analysis.pair_kinematics(0, 2)
    store_dir = tmp_path / "video_temporal"
    store_dir.mkdir()
    sequence = TemporalResultSequence(str(store_dir), [(0, 2)])
    save_temporal_result(sequence.path_for(0), pair)

    renderer = ViewRenderer(
        analysis, results=sequence, pairs=[(0, 2)])
    values, _unit = renderer.field_array(0, "u")

    assert renderer._source_index(0) == 2
    stored = np.nanmedian(np.asarray(sequence[0].u))
    factor, _ = analysis.calibration.factor_and_unit("u", "px")
    assert stored == pytest.approx(0.4, rel=1e-5)
    assert np.nanmedian(np.asarray(values)) == pytest.approx(
        stored * factor, rel=1e-5)

    spec = ExportSpec(
        panels=[PanelSpec(field="u", background="Reference frame")],
        cell_w=96, cell_h=72, codec="PNG image sequence",
        first=0, last=0)
    output = export_video(
        analysis, spec, str(tmp_path / "temporal.png"),
        results=sequence, pairs=[(0, 2)])
    assert os.path.isfile(os.path.join(output, "frame_00000.png"))
