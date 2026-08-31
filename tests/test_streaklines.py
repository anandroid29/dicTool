"""Streakline / pathline tracing: correctness, caching, and cost.

Trajectories are traced by accumulating the measured displacement at a material
point frame by frame. Two things can go wrong and both are silent:

  * the traced path drifts away from the motion actually encoded in the fields,
    which produces a plausible-looking streakline that is simply wrong;
  * tracing is re-run from frame 0 on every render, which is correct but makes
    the viewer unusable on a long sequence.

These tests pin both. The synthetic flows have closed-form trajectories, so
accuracy is measured against an exact answer rather than against a previous
run's output.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.core.analysis import DICAnalysis, PairResult, _compact_field


# --------------------------------------------------------------------------
# Synthetic sequences with analytically known trajectories
# --------------------------------------------------------------------------

def _sequence(shape, n_frames, u_of_xy, v_of_xy, spacing=3, fps=1000.0):
    """A DICAnalysis whose per-frame displacement is a known function of position.

    Fields are stored exactly as the solver stores them -- packed at subset
    centres -- so the tests exercise the same code path the application does.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    valid = np.zeros(shape, bool)
    valid[spacing // 2::spacing, spacing // 2::spacing] = True
    indices = np.flatnonzero(valid.reshape(-1)).astype(np.uint32)

    u = np.where(valid, u_of_xy(xx, yy), np.nan)
    v = np.where(valid, v_of_xy(xx, yy), np.nan)
    zero = np.where(valid, 0.0, np.nan)

    analysis = DICAnalysis()
    analysis.fps = fps
    analysis._roi_mask = valid
    analysis.results = [
        PairResult(
            image_path=f"frame_{k:04d}",
            u=_compact_field(u, valid, indices),
            v=_compact_field(v, valid, indices),
            Exx=None, Exy=None, Eyy=None, Eeff=None,
            du_dx=_compact_field(zero, valid, indices),
            du_dy=_compact_field(zero, valid, indices),
            dv_dx=_compact_field(zero, valid, indices),
            dv_dy=_compact_field(zero, valid, indices),
            corr=_compact_field(zero, valid, indices),
        )
        for k in range(n_frames)
    ]
    analysis.def_paths = [f"frame_{k:04d}" for k in range(n_frames)]
    return analysis


def _uniform(shape=(120, 160), n=40, dx=2.0, dy=0.5):
    """Rigid translation: every point moves (dx, dy) per frame."""
    return _sequence(shape, n, lambda x, y: np.full_like(x, dx, float),
                     lambda x, y: np.full_like(y, dy, float))


def _shear(shape=(120, 160), n=40, rate=0.02):
    """Simple shear: horizontal speed proportional to height.

    A point keeps its y and advances in x by rate*y each frame, so the exact
    position after k frames is (x0 + k*rate*y0, y0).
    """
    return _sequence(shape, n, lambda x, y: rate * y,
                     lambda x, y: np.zeros_like(y, float))


# --------------------------------------------------------------------------
# Accuracy against the closed-form trajectory
# --------------------------------------------------------------------------

def test_uniform_translation_matches_exactly():
    dx, dy, n = 2.0, 0.5, 40
    a = _uniform(n=n, dx=dx, dy=dy)
    seed = (60.0, 45.0)

    traj = a.get_trajectories_from_seeds([seed], n - 1, 0)[0]
    pts = np.asarray(traj["points"])

    assert traj["lost_at"] is None
    assert len(pts) == n + 1, "one point for the seed plus one per frame"

    k = np.arange(len(pts))
    expected = np.column_stack([seed[0] + k * dx, seed[1] + k * dy])
    assert np.allclose(pts, expected, atol=1e-6), (
        f"max deviation {np.abs(pts - expected).max():.3e} px")


def test_shear_flow_matches_exactly():
    rate, n = 0.02, 40
    a = _shear(n=n, rate=rate)

    for seed in [(40.0, 30.0), (80.0, 60.0), (100.0, 90.0)]:
        traj = a.get_trajectories_from_seeds([seed], n - 1, 0)[0]
        pts = np.asarray(traj["points"])
        k = np.arange(len(pts))
        expected = np.column_stack([seed[0] + k * rate * seed[1],
                                    np.full(len(pts), seed[1])])
        err = np.abs(pts - expected).max()
        # Result fields are stored as float32, and tracing sums one sample per
        # frame, so the floor here is n * eps32 * |value| ~ 1e-5 px. That is the
        # storage precision, not the interpolator: for the method's own accuracy
        # see test_interpolator_is_linear_exact, which works in float64.
        assert err < 1e-5, f"seed {seed}: max deviation {err:.3e} px"


def test_interpolator_is_linear_exact():
    """The sampler must reproduce a linear field exactly.

    Displacement is sampled at points that fall between subset centres, and the
    result is then summed over every frame of a trajectory. An interpolator that
    is merely close introduces a bias, and a bias accumulates: inverse-distance
    weighting was off by 1.6% at the midpoint between grid rows, which over a
    thousand frames walks the traced path several pixels away from the material
    point it is meant to follow.
    """
    from src.core.analysis import DICAnalysis

    # A regular grid of subset centres carrying an exactly linear field.
    gy, gx = np.mgrid[1:60:3, 1:60:3]
    xs, ys = gx.ravel().astype(float), gy.ravel().astype(float)

    for (a0, ax, ay) in [(0.0, 1.0, 0.0),      # varies in x only
                         (0.0, 0.0, 1.0),      # varies in y only
                         (2.5, -0.3, 0.7)]:    # general plane
        vals = a0 + ax * xs + ay * ys
        for (qx, qy) in [(30.0, 30.0),         # on a centre
                         (31.5, 31.5),         # between centres, both axes
                         (30.0, 31.5),         # between rows only
                         (14.2, 47.9)]:        # arbitrary
            got = DICAnalysis._interpolate_at(xs, ys, vals, qx, qy)
            exact = a0 + ax * qx + ay * qy
            assert abs(got - exact) < 1e-9, (
                f"plane ({a0}, {ax}, {ay}) at ({qx}, {qy}): "
                f"got {got:.9f}, exact {exact:.9f}")


def test_interpolator_falls_back_when_a_plane_is_undetermined():
    """Too few or degenerate points must still yield a usable value."""
    from src.core.analysis import DICAnalysis

    # Two points cannot determine a plane.
    got = DICAnalysis._interpolate_at(
        np.array([0.0, 10.0]), np.array([0.0, 0.0]), np.array([0.0, 10.0]), 5.0, 0.0)
    assert np.isfinite(got) and 0.0 <= got <= 10.0

    # All points on one row: degenerate in y.
    xs = np.arange(0.0, 30.0, 3.0)
    got = DICAnalysis._interpolate_at(xs, np.zeros_like(xs), xs, 13.5, 0.0)
    assert np.isfinite(got)

    # An exact hit returns the stored value untouched.
    assert DICAnalysis._interpolate_at(
        np.array([4.0, 7.0]), np.array([4.0, 7.0]),
        np.array([42.0, 99.0]), 4.0, 4.0) == 42.0


def test_marker_position_is_the_end_of_its_own_path():
    """A marker and the tip of its trail must never disagree."""
    a = _shear(n=30)
    seeds = [(40.0, 30.0), (90.0, 75.0)]
    for frame in (0, 7, 18, 29):
        ends = a.marker_positions(seeds, frame)
        trajs = a.get_trajectories_from_seeds(seeds, frame, 0)
        for end, traj in zip(ends, trajs):
            assert end == pytest.approx(traj["points"][-1], abs=1e-9)


# --------------------------------------------------------------------------
# Tracing from any starting position, and from anywhere in the sequence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [
    (10.0, 10.0),        # near the origin corner
    (150.0, 110.0),      # near the far corner
    (79.5, 45.5),        # between subset centres
    (3.0, 3.0),          # inside the first subset spacing
])
def test_traces_from_any_position(seed):
    a = _uniform(n=25, dx=1.5, dy=-0.75)
    traj = a.get_trajectories_from_seeds([seed], 24, 0)[0]
    pts = np.asarray(traj["points"])
    assert traj["lost_at"] is None, f"lost the point seeded at {seed}"
    step = np.diff(pts, axis=0)
    # Uniform flow: every step is the same regardless of where tracing began.
    assert np.allclose(step, step[0], atol=1e-6)
    assert step[0] == pytest.approx([1.5, -0.75], abs=1e-6)


def test_requesting_any_frame_gives_a_prefix_of_the_full_path():
    a = _shear(n=30)
    seed = (70.0, 50.0)
    full = a.get_trajectories_from_seeds([seed], 29, 0)[0]["points"]
    for frame in (0, 1, 9, 20, 29):
        partial = a.get_trajectories_from_seeds([seed], frame, 0)[0]["points"]
        assert partial == pytest.approx(np.asarray(full[:len(partial)]), abs=1e-9)
        assert len(partial) == frame + 2


def test_trail_keeps_the_most_recent_segments():
    a = _uniform(n=30)
    seed = (60.0, 45.0)
    full = a.get_trajectories_from_seeds([seed], 29, 0)[0]["points"]
    for trail in (1, 5, 12):
        cut = a.get_trajectories_from_seeds([seed], 29, trail)[0]["points"]
        assert len(cut) == trail + 1
        assert cut == pytest.approx(np.asarray(full[-(trail + 1):]), abs=1e-9)


# --------------------------------------------------------------------------
# The cache must not change any answer
# --------------------------------------------------------------------------

def test_cached_result_equals_a_cold_trace():
    a = _shear(n=35)
    seeds = [(40.0, 30.0), (95.0, 80.0)]

    warm = [a.get_trajectories_from_seeds(seeds, f, 0) for f in range(35)]
    cold = []
    for f in range(35):
        a._path_cache.clear()                    # force a full re-trace
        cold.append(a.get_trajectories_from_seeds(seeds, f, 0))

    for f, (w, c) in enumerate(zip(warm, cold)):
        for tw, tc in zip(w, c):
            assert tw["lost_at"] == tc["lost_at"], f"frame {f}"
            assert np.asarray(tw["points"]) == pytest.approx(
                np.asarray(tc["points"]), abs=1e-12), f"frame {f}"


def test_scrubbing_backwards_and_forwards_is_consistent():
    a = _shear(n=25)
    seed = [(55.0, 40.0)]
    forward = {f: a.get_trajectories_from_seeds(seed, f, 0)[0]["points"]
               for f in range(25)}
    backward = {f: a.get_trajectories_from_seeds(seed, f, 0)[0]["points"]
                for f in reversed(range(25))}
    for f in range(25):
        assert np.asarray(forward[f]) == pytest.approx(
            np.asarray(backward[f]), abs=1e-12)


def test_reanalysis_invalidates_cached_paths():
    """A cached path must never survive the data it was traced through."""
    a = _uniform(n=20, dx=2.0, dy=0.0)
    seed = [(60.0, 45.0)]
    before = a.get_trajectories_from_seeds(seed, 19, 0)[0]["points"][-1]
    assert before == pytest.approx((60.0 + 20 * 2.0, 45.0), abs=1e-6)

    # Replace the sequence with a different flow, the way a re-run would.
    b = _uniform(n=20, dx=-1.0, dy=0.0)
    a.results.clear()                 # bumps the epoch
    a.results.extend(b.results)

    after = a.get_trajectories_from_seeds(seed, 19, 0)[0]["points"][-1]
    assert after == pytest.approx((60.0 - 20 * 1.0, 45.0), abs=1e-6), (
        "stale path served after the results were replaced")


# --------------------------------------------------------------------------
# Dropout
# --------------------------------------------------------------------------

def test_point_lost_mid_sequence_reports_where():
    a = _uniform(n=20)
    # Blank the whole field at frame 12: nothing can be tracked through it.
    from src.core.compact_field import CompactField
    blank = CompactField.empty(a.results[12].u.shape)
    a.results[12].u = blank
    a.results[12].v = blank

    traj = a.get_trajectories_from_seeds([(60.0, 45.0)], 19, 0)[0]
    assert traj["lost_at"] == 12
    assert len(traj["points"]) == 13, "seed plus frames 0..11"

    # Before the dropout the path is still reported as intact.
    early = a.get_trajectories_from_seeds([(60.0, 45.0)], 5, 0)[0]
    assert early["lost_at"] is None
    assert a.marker_positions([(60.0, 45.0)], 15) == [None]


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------

def test_scrubbing_cost_is_not_quadratic():
    """Tracing must extend, not replay.

    Re-walking the history on every frame is O(frames^2) over a scrub, which is
    what made the viewer stop responding. Compare the cost of the second half of
    a scrub against the first: with incremental tracing the two are comparable,
    while a replaying implementation makes the second half several times dearer.
    """
    n = 160
    a = _sequence((200, 260), n,
                  lambda x, y: np.full_like(x, 1.0, float),
                  lambda x, y: np.full_like(y, 0.25, float))
    seeds = [(60.0 + 10 * i, 90.0) for i in range(5)]
    a._path_cache.clear()

    def scrub(lo, hi):
        t0 = time.perf_counter()
        for f in range(lo, hi):
            a.get_trajectories_from_seeds(seeds, f, 0)
            a.marker_positions(seeds, f)
        return time.perf_counter() - t0

    first = scrub(0, n // 2)
    second = scrub(n // 2, n)

    # Each half traces the same number of new frames, so the times should be of
    # the same order. Generous bound: this is catching an O(n^2) regression,
    # not measuring performance.
    assert second < first * 4 + 0.05, (
        f"second half {second*1000:.0f} ms vs first {first*1000:.0f} ms — "
        "tracing looks like it is replaying the history")


def test_revisiting_a_frame_is_free():
    a = _uniform(n=80)
    seeds = [(60.0, 45.0), (70.0, 55.0)]
    a.get_trajectories_from_seeds(seeds, 79, 0)      # warm

    t0 = time.perf_counter()
    for _ in range(50):
        a.get_trajectories_from_seeds(seeds, 79, 0)
        a.marker_positions(seeds, 79)
    per_call = (time.perf_counter() - t0) / 50

    assert per_call < 0.01, (
        f"{per_call*1000:.2f} ms to re-serve an already-traced path")


# --------------------------------------------------------------------------
# The API-shape regression that actually crashed the application
# --------------------------------------------------------------------------

def test_canvas_marker_accessors_are_used_correctly():
    """`ImageCanvas.markers` is a property; calling it aborts the process.

    It was converted from a method to a property, and seven call sites were
    left invoking it. Every one raised TypeError inside a Qt slot, and PyQt
    turns an unhandled slot exception into an abort -- so enabling streaklines,
    placing a marker, dragging one, or exporting video all killed the window
    with no message and nothing written to the log.

    A source-level check is crude, but it catches exactly the mistake that was
    made, in the places it was made, and needs no display to run.
    """
    import inspect
    import re

    from src.ui import image_canvas
    from src.ui.pages import results_page

    assert isinstance(inspect.getattr_static(image_canvas.ImageCanvas, "markers"),
                      property), "markers is expected to be a property"

    offenders = []
    for module in (image_canvas, results_page):
        source = inspect.getsource(module)
        for lineno, line in enumerate(source.splitlines(), start=1):
            if re.search(r"\.markers\s*\(", line):
                offenders.append(f"{module.__name__}:{lineno}: {line.strip()}")

    assert not offenders, (
        "markers is a property but is being called:\n  " + "\n  ".join(offenders))
