"""
Marker trajectories must follow the material, not the seed.

Displacement fields are measured on the immediately preceding frame, so
``results[i].u[y, x]`` describes whatever material occupies pixel (x, y) *in
frame i*. Following a material point therefore means re-sampling at its current
position every step. Sampling at the fixed seed instead reads the velocity of
whatever later flows past the seed -- in orthogonal cutting, the steady incoming
feed -- which drives markers straight through the shear zone instead of turning
them up the chip. These tests use analytic flows with closed-form pathlines so
the expected geometry is known exactly rather than asserted from a golden run.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydic.core.analysis import DICAnalysis  # noqa: E402
from pydic.core.compact_field import CompactField  # noqa: E402

H = W = 200
GRID = 4  # subset spacing of the synthetic measurement grid


class _Res:
    """The subset of PairResult that marker tracing touches."""

    def __init__(self, u, v):
        self.u, self.v = u, v
        self.valid = np.isfinite(u)
        self.image_path = "synthetic.tif"


class _Params:
    subset_spacing = GRID


def _build(velocity, n_frames, hole=None):
    """
    An analysis whose per-frame fields sample ``velocity`` on a regular grid.

    ``velocity(x, y)`` is Eulerian: it returns the displacement of whatever sits
    at (x, y) during that interval, which is exactly what DIC measures here.
    ``hole``, if given, marks grid points that failed to correlate.
    """
    a = object.__new__(DICAnalysis)
    a.params = _Params()
    a.fps = 1.0
    a._ref_image = np.zeros((H, W), np.uint8)
    ys, xs = np.mgrid[0:H:GRID, 0:W:GRID]
    xf, yf = xs.astype(float), ys.astype(float)
    du, dv = velocity(xf, yf)
    keep = np.ones_like(xf, dtype=bool) if hole is None else ~hole(xf, yf)
    results = []
    for _ in range(n_frames):
        u = np.full((H, W), np.nan, np.float32)
        v = np.full((H, W), np.nan, np.float32)
        u[ys[keep], xs[keep]] = du[keep]
        v[ys[keep], xs[keep]] = dv[keep]
        results.append(_Res(u, v))
    a.results = results
    return a


def _truth(velocity, seed, n_steps):
    """Closed-form pathline: integrate the Eulerian field at the current point."""
    x, y = seed
    out = [(x, y)]
    for _ in range(n_steps):
        dx, dy = velocity(np.array(x), np.array(y))
        x, y = x + float(dx), y + float(dy)
        out.append((x, y))
    return out


# --- flows -----------------------------------------------------------------

SHEAR_X = 100.0


def cutting(x, y):
    """Feed to the right, then turn up the chip past the shear plane."""
    in_chip = x >= SHEAR_X
    return (np.where(in_chip, 1.0, 2.0), np.where(in_chip, -1.7, 0.0))


def rotation(x, y):
    om = 0.02
    return (-om * (y - 100.0), om * (x - 100.0))


def uniform(x, y):
    return (np.full_like(x, 1.5), np.full_like(y, -0.5))


# --- tests -----------------------------------------------------------------

def test_marker_turns_up_the_chip_past_the_shear_zone():
    """The reported bug: trajectories ran straight instead of following the chip."""
    a = _build(cutting, 80)
    seed = (20.0, 150.0)
    pts = a.get_trajectories_from_seeds([seed], 79)[0]["points"]

    # It must actually leave the feed line -- a fixed-seed tracer stays at y=150.
    assert pts[-1][1] < 100.0, "marker did not turn up the chip"

    expected = _truth(cutting, seed, 80)
    err = max(np.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(pts, expected))
    assert err < 2.0, f"pathline deviates from closed form by {err:.2f} px"


def test_trajectory_is_straight_only_where_the_flow_is():
    """Before the shear plane the same path must stay on the feed line."""
    a = _build(cutting, 80)
    pts = a.get_trajectories_from_seeds([(20.0, 150.0)], 79)[0]["points"]
    before = [p for p in pts if p[0] < SHEAR_X - GRID]
    assert before, "expected some travel before the shear plane"
    assert max(abs(p[1] - 150.0) for p in before) < 0.5


def test_rotation_preserves_radius_and_angle():
    """A rigid rotation has an exactly circular pathline."""
    a = _build(rotation, 80)
    seed = (160.0, 100.0)
    pts = a.get_trajectories_from_seeds([seed], 79)[0]["points"]
    radii = [np.hypot(px - 100.0, py - 100.0) for px, py in pts]
    assert max(abs(r - 60.0) for r in radii) < 1.5
    angle = np.arctan2(pts[-1][1] - 100.0, pts[-1][0] - 100.0)
    assert abs(angle - 80 * 0.02) < 0.02


def test_uniform_flow_is_exact():
    """With a spatially constant field, advection and accumulation agree exactly."""
    a = _build(uniform, 40)
    pts = a.get_trajectories_from_seeds([(50.0, 120.0)], 39)[0]["points"]
    for k, (px, py) in enumerate(pts):
        assert px == pytest.approx(50.0 + 1.5 * k, abs=1e-6)
        assert py == pytest.approx(120.0 - 0.5 * k, abs=1e-6)


def test_compact_field_marker_sampling_matches_dense_field():
    """The fast sorted-index path must retain trajectory interpolation semantics."""
    dense = _build(uniform, 12)
    compact = _build(uniform, 12)
    for res in compact.results:
        res.u = CompactField.from_dense(res.u)
        res.v = CompactField.from_dense(res.v)
    seed = (51.25, 119.5)
    expected = dense.get_trajectories_from_seeds([seed], 11)[0]
    actual = compact.get_trajectories_from_seeds([seed], 11)[0]
    assert actual["lost_at"] == expected["lost_at"]
    assert actual["points"] == pytest.approx(expected["points"], abs=1e-9)


def test_marker_positions_match_trajectory_endpoint():
    """The drawn marker and the end of its trail must be the same point."""
    a = _build(cutting, 60)
    seed = (20.0, 150.0)
    for frame in (0, 1, 17, 40, 59):
        end = a.marker_positions([seed], frame)[0]
        pts = a.get_trajectories_from_seeds([seed], frame)[0]["points"]
        assert end is not None
        assert end == pytest.approx(pts[-1], abs=1e-9)


def test_trail_truncation_keeps_the_recent_end():
    a = _build(cutting, 60)
    seed = (20.0, 150.0)
    full = a.get_trajectories_from_seeds([seed], 59)[0]["points"]
    for trail in (1, 5, 20):
        cut = a.get_trajectories_from_seeds([seed], 59, trail)[0]["points"]
        assert len(cut) == trail + 1
        assert cut == pytest.approx(full[-(trail + 1):], abs=1e-9)


def test_prefix_property():
    """Tracing to a later frame extends the earlier path, never rewrites it."""
    a = _build(cutting, 60)
    seed = (20.0, 150.0)
    short = a.get_trajectories_from_seeds([seed], 20)[0]["points"]
    long = a.get_trajectories_from_seeds([seed], 59)[0]["points"]
    assert long[:len(short)] == pytest.approx(short, abs=1e-9)


def test_survives_a_gap_in_the_correlated_grid():
    """
    A band of dropouts along the shear plane must not truncate the trajectory.

    This is why the sampler widens its search before giving up: the marker has
    to coast across the gap and pick the field up again on the far side.
    """
    def band(x, y):
        return (x >= SHEAR_X - GRID) & (x <= SHEAR_X + GRID)

    a = _build(cutting, 80, hole=band)
    traj = a.get_trajectories_from_seeds([(20.0, 150.0)], 79)[0]
    assert traj["lost_at"] is None, "trajectory truncated at the dropout band"
    assert traj["points"][-1][1] < 100.0, "did not reach the chip"


def test_marker_lost_outside_the_measured_region():
    """A seed with no data anywhere still reports loss rather than inventing a path."""
    a = _build(cutting, 20, hole=lambda x, y: np.ones_like(x, dtype=bool))
    traj = a.get_trajectories_from_seeds([(20.0, 150.0)], 19)[0]
    assert traj["lost_at"] == 0
    assert traj["points"] == [(20.0, 150.0)]
    assert a.marker_positions([(20.0, 150.0)], 19)[0] is None


@pytest.mark.parametrize("flow", [cutting, rotation, uniform])
@pytest.mark.parametrize("frame", [0, 15, 45])
def test_reference_from_current_inverts_the_forward_map(flow, frame):
    """
    Clicking where a marker is drawn must recover the seed that put it there.

    reference_from_current is the inverse of the tracing above, so it has to be
    built on the same advection; summing per-frame fields at a fixed grid index
    adds displacements measured on different configurations.
    """
    a = _build(flow, 60)
    seed = (60.0, 120.0)
    landed = a.marker_positions([seed], frame)[0]
    assert landed is not None
    got = a.reference_from_current(landed[0], landed[1], frame)
    assert got is not None
    (rx, ry), residual = got
    assert residual < 1.0, f"inverse missed by {residual:.2f} px"
    # Round-tripping the recovered seed must land back under the cursor.
    back = a.marker_positions([(rx, ry)], frame)[0]
    assert back == pytest.approx(landed, abs=1.0)


def test_fields_are_sampled_at_the_moving_position():
    """
    Guard the specific regression: a flow that is zero at the seed but nonzero
    downstream must still move the marker, and one that is nonzero only at the
    seed must not carry it forever.
    """
    def only_downstream(x, y):
        return (np.where(x < 50.0, 1.0, 3.0), np.zeros_like(y))

    a = _build(only_downstream, 40)
    pts = a.get_trajectories_from_seeds([(20.0, 100.0)], 39)[0]["points"]
    # Sampling at the fixed seed reads 1 px/frame forever: 20 + 40 = 60.
    # Advecting, the point covers x=20..50 at 1 px/frame (30 frames) and the
    # remaining 10 frames at 3 px/frame, ending at 80.
    assert pts[-1][0] == pytest.approx(80.0, abs=1.0)
