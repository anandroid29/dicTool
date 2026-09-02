"""Stage-by-stage numerical parity checks for the native CUDA DIC port."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates, shift, spline_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydic.core.cuda_native import (
    NativeCudaError, NativeCudaSolver, native_cuda_available,
    native_cuda_diagnostic)
from pydic.core.ncc import ncc_initial_guess
from pydic.core.rg_dic import DICParams, run_rg_dic


pytestmark = pytest.mark.skipif(
    not native_cuda_available(), reason=native_cuda_diagnostic())


def _texture(seed: int = 41, shape=(112, 120)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fine = gaussian_filter(rng.random(shape), 0.7)
    broad = gaussian_filter(rng.random(shape), 3.0)
    image = 0.75 * fine + 0.25 * broad
    image -= image.min()
    return image / image.max()


def _safe_texture(seed: int = 41, shape=(112, 120)) -> np.ndarray:
    """Texture away from the solver's documented occlusion thresholds."""
    return 0.15 + 0.70 * _texture(seed, shape)


def _compose_inverse_affine(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    a1, b1, c1 = 1.0 + p[2], p[3], p[0]
    d1, e1, f1 = p[4], 1.0 + p[5], p[1]
    a2, b2, c2 = 1.0 + dp[2], dp[3], dp[0]
    d2, e2, f2 = dp[4], 1.0 + dp[5], dp[1]
    det = a2 * e2 - b2 * d2
    assert abs(det) >= 1e-12
    ia, ib = e2 / det, -b2 / det
    ic = (b2 * f2 - c2 * e2) / det
    id_, ie = -d2 / det, a2 / det
    iff = (c2 * d2 - a2 * f2) / det
    return np.array([
        a1 * ic + b1 * iff + c1,
        d1 * ic + e1 * iff + f1,
        a1 * ia + b1 * id_ - 1.0,
        a1 * ib + b1 * ie,
        d1 * ia + e1 * id_,
        d1 * ib + e1 * ie - 1.0,
    ])


def _cupy_port_reference_icgn(ref: np.ndarray, current: np.ndarray,
                              center_x: int, center_y: int, radius: int,
                              initial: np.ndarray, max_iter: int,
                              conv_tol: float, corr_cutoff: float,
                              spacing: int):
    """NumPy/SciPy oracle for the former CuPy single-subset equations."""
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    inside = xx * xx + yy * yy <= radius * radius
    dx = xx[inside].astype(np.float64)
    dy = yy[inside].astype(np.float64)
    xs = center_x + dx
    ys = center_y + dy

    ref_coeff = spline_filter(ref, order=3, mode="mirror", output=np.float64)
    cur_coeff = spline_filter(current, order=3, mode="mirror", output=np.float64)

    def sample(coeff, x, y):
        return map_coordinates(
            coeff, np.vstack((np.ravel(y), np.ravel(x))), order=3,
            mode="mirror", prefilter=False).reshape(np.shape(x))

    f = sample(ref_coeff, xs, ys)
    f_centered = f - f.mean()
    sigma_f = np.sqrt(np.sum(f_centered * f_centered))
    f_norm = f_centered / sigma_f
    h = 1e-3
    grad_x = (sample(ref_coeff, xs + h, ys) -
              sample(ref_coeff, xs - h, ys)) / (2.0 * h)
    grad_y = (sample(ref_coeff, xs, ys + h) -
              sample(ref_coeff, xs, ys - h)) / (2.0 * h)
    raw = np.column_stack((
        grad_x, grad_y, grad_x * dx, grad_x * dy,
        grad_y * dx, grad_y * dy))
    correction = f_norm @ raw
    sd = (raw - np.outer(f_norm, correction)) / sigma_f
    hessian = sd.T @ sd + np.eye(6) * 1e-6
    hessian_inverse = np.linalg.inv(hessian)

    p = np.asarray(initial, dtype=np.float64).copy()
    start = p.copy()
    best = p.copy()
    best_score = np.inf

    def evaluate(candidate):
        x = xs + candidate[0] + candidate[2] * dx + candidate[3] * dy
        y = ys + candidate[1] + candidate[4] * dx + candidate[5] * dy
        if (x.min() < -0.5 or x.max() > ref.shape[1] - 0.5 or
                y.min() < -0.5 or y.max() > ref.shape[0] - 0.5):
            return np.inf, None
        g = sample(cur_coeff, x, y)
        centered = g - g.mean()
        sigma_g = np.sqrt(np.sum(centered * centered))
        if sigma_g <= 1e-12:
            return np.inf, None
        residual = centered / sigma_g - f_norm
        return float(residual @ residual), residual

    for _ in range(max_iter):
        score, residual = evaluate(p)
        if score < best_score:
            best_score, best = score, p.copy()
        if not np.isfinite(score):
            break
        dp = hessian_inverse @ (sd.T @ residual)
        norm = np.linalg.norm(dp)
        if (not np.isfinite(norm) or np.hypot(dp[0], dp[1]) > 5.0 or
                np.linalg.norm(dp[2:]) > 8.0):
            break
        p = _compose_inverse_affine(p, dp)
        if norm < conv_tol:
            break

    score, _ = evaluate(p)
    if score < best_score:
        best_score, best = score, p.copy()
    limit = max(spacing * 1.5, 10.0)
    accepted = (np.isfinite(best_score) and best_score < corr_cutoff and
                abs(best[0] - start[0]) < limit and
                abs(best[1] - start[1]) < limit)
    return best, best_score, accepted


@pytest.mark.parametrize(
    "translation,guess",
    [((4, -3), (0.0, 0.0)),
     ((-5, 2), (-2.2, 1.1)),
     ((1, 6), (2.6, 3.2))],
)
def test_stage_1_native_ncc_matches_reference_zncc(translation, guess):
    """Native NCC must reproduce the existing square-template CPU search."""
    ref = _texture()
    u_true, v_true = translation
    current = shift(ref, shift=(v_true, u_true), order=3, mode="mirror")
    # ZNCC must be insensitive to a global affine intensity change.
    current = current * 1.17 + 0.031
    params = DICParams(
        subset_radius=9, subset_spacing=11, search_radius=8,
        max_iter=20, corr_cutoff=0.5, rescue_radius=3)
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, np.ones(ref.shape, dtype=bool))
        candidates = [
            int(np.argmin((solver.gx_flat - x) ** 2 +
                          (solver.gy_flat - y) ** 2))
            for x, y in ((35, 35), (60, 55), (85, 75))
        ]
        for index in candidates:
            x, y = int(solver.gx_flat[index]), int(solver.gy_flat[index])
            expected = ncc_initial_guess(
                ref, current, x, y, params.subset_radius,
                params.search_radius, *guess)
            actual = solver.ncc_guess(current, index, *guess)
            assert actual[:2] == expected[:2]
            assert actual[0] == pytest.approx(u_true, abs=0.0)
            assert actual[1] == pytest.approx(v_true, abs=0.0)
            assert actual[2] == pytest.approx(expected[2], abs=2e-6)
    finally:
        solver.close()


@pytest.mark.parametrize(
    "center,guess",
    [((20, 20), (0.5, 0.5)),
     ((20, 20), (-0.5, -0.5)),
     ((21, 21), (1.5, 1.5)),
     ((21, 21), (-0.5, -0.5))],
)
def test_stage_1_ncc_matches_python_half_tie_rounding(center, guess):
    ref = _texture(shape=(64, 64))
    params = DICParams(
        subset_radius=7, subset_spacing=1, search_radius=0,
        max_iter=10, corr_cutoff=0.5, rescue_radius=2)
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, np.ones(ref.shape, dtype=bool))
        cx, cy = center
        index = int(np.flatnonzero(
            (solver.gx_flat == cx) & (solver.gy_flat == cy))[0])
        expected = ncc_initial_guess(
            ref, ref, cx, cy, params.subset_radius, 0, *guess)
        actual = solver.ncc_guess(ref, index, *guess)
        assert actual[:2] == expected[:2]
        assert actual[2] == pytest.approx(expected[2], abs=2e-6)
    finally:
        solver.close()


@pytest.mark.parametrize("case", ("fractional_translation", "small_affine"))
def test_stage_2_isolated_native_icgn_matches_cupy_equations(case):
    ref = _safe_texture(shape=(128, 136))
    if case == "fractional_translation":
        current = shift(ref, shift=(-1.35, 2.40), order=3, mode="mirror")
        initial = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        transform_xy = np.array([[1.006, -0.004], [0.003, 0.994]])
        translation_xy = np.array([1.25, -0.75])
        inverse = np.linalg.inv(transform_xy)
        current = affine_transform(
            ref, inverse[::-1, ::-1],
            offset=(-inverse @ translation_xy)[::-1],
            order=3, mode="mirror")
        initial = np.array([1.0, -1.0, 0.004, -0.002, 0.001, -0.004])
    params = DICParams(
        subset_radius=11, subset_spacing=9, search_radius=5,
        max_iter=35, conv_tol=1e-5, corr_cutoff=0.8, rescue_radius=3)
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, np.ones(ref.shape, dtype=bool))
        index = int(np.argmin(
            (solver.gx_flat - 68) ** 2 + (solver.gy_flat - 64) ** 2))
        cx, cy = int(solver.gx_flat[index]), int(solver.gy_flat[index])
        expected = _cupy_port_reference_icgn(
            ref, current, cx, cy, params.subset_radius, initial,
            params.max_iter, params.conv_tol, params.corr_cutoff,
            params.subset_spacing)
        actual = solver.solve_subset(current, index, initial)
        assert actual[2] and bool(expected[2])
        np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=2e-7)
        assert actual[1] == pytest.approx(expected[1], rel=0, abs=2e-8)
    finally:
        solver.close()


@pytest.mark.parametrize("translation", ((3.35, -2.20), (-4.15, 2.65)))
def test_stage_3_production_ncc_then_icgn_matches_reference(translation):
    """Exercise fresh-mode wiring with propagation disabled by a one-point ROI."""
    ref = _safe_texture(seed=67, shape=(128, 136))
    u_true, v_true = translation
    current = shift(ref, shift=(v_true, u_true), order=3, mode="mirror")
    params = DICParams(
        subset_radius=11, subset_spacing=9, search_radius=7,
        max_iter=35, conv_tol=1e-5, corr_cutoff=0.8, rescue_radius=3,
        mask_subsets_to_roi=False)
    # Select a grid centre first, then make it the only propagation-eligible
    # point. The full circular subset remains available, matching CUDA mode's
    # historical ROI convention for this stage.
    r, spacing = params.subset_radius, params.subset_spacing
    xs = np.arange(r, ref.shape[1] - r, spacing)
    ys = np.arange(r, ref.shape[0] - r, spacing)
    gx, gy = np.meshgrid(xs, ys)
    seed = int(np.argmin((gx.ravel() - 68) ** 2 + (gy.ravel() - 64) ** 2))
    cx, cy = int(gx.ravel()[seed]), int(gy.ravel()[seed])
    roi = np.zeros(ref.shape, dtype=bool)
    roi[cy, cx] = True

    ncc = ncc_initial_guess(
        ref, current, cx, cy, params.subset_radius,
        params.search_radius, 0.0, 0.0)
    initial = np.array([ncc[0], ncc[1], 0.0, 0.0, 0.0, 0.0])
    expected = _cupy_port_reference_icgn(
        ref, current, cx, cy, params.subset_radius, initial,
        params.max_iter, params.conv_tol, params.corr_cutoff,
        params.subset_spacing)

    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, roi)
        fields = solver.solve_frame(
            current, seed_idx=seed, seed_guess=(0.0, 0.0), warm_start=False)
        assert sum(np.count_nonzero(np.isfinite(field)) for field in fields[:2]) == 2
        actual_p = np.array([field[cy, cx] for field in fields[:6]])
        actual_score = fields[6][cy, cx]
        assert bool(expected[2])
        np.testing.assert_allclose(actual_p, expected[0], rtol=0, atol=2e-7)
        assert actual_score == pytest.approx(expected[1], rel=0, abs=2e-8)
    finally:
        solver.close()


@pytest.mark.parametrize("poor_prior", (False, True))
def test_stage_4_warm_start_and_reference_promotion_match_fresh_pair(poor_prior):
    """Warm state is only a guess; the pair result must match a fresh solve."""
    ref = _safe_texture(seed=93, shape=(112, 120))
    if poor_prior:
        first_increment = (6.2, -4.7)
        second_increment = (-3.6, 3.1)
    else:
        first_increment = (2.2, -1.4)
        second_increment = (2.6, -1.1)
    frame_1 = shift(
        ref, shift=(first_increment[1], first_increment[0]),
        order=3, mode="mirror")
    frame_2 = shift(
        frame_1, shift=(second_increment[1], second_increment[0]),
        order=3, mode="mirror")
    params = DICParams(
        subset_radius=9, subset_spacing=9, search_radius=9,
        max_iter=35, conv_tol=1e-5, corr_cutoff=0.8, rescue_radius=12,
        mask_subsets_to_roi=False)
    roi = np.zeros(ref.shape, dtype=bool)
    roi[18:-18, 18:-18] = True

    warm_solver = NativeCudaSolver(params)
    cold_solver = NativeCudaSolver(params)
    try:
        warm_solver.precompute_reference(ref, roi)
        seed = int(np.argmin(
            (warm_solver.gx_flat - 60) ** 2 +
            (warm_solver.gy_flat - 56) ** 2))
        warm_solver.solve_frame(
            frame_1, seed_idx=seed, seed_guess=(0.0, 0.0),
            warm_start=False)
        warm_solver.update_reference_image(frame_1)
        warm = warm_solver.solve_frame(frame_2, warm_start=True)

        cold_solver.precompute_reference(frame_1, roi)
        cold = cold_solver.solve_frame(
            frame_2, seed_idx=seed, seed_guess=(0.0, 0.0),
            warm_start=False)

        warm_valid = np.logical_and.reduce([np.isfinite(field) for field in warm])
        cold_valid = np.logical_and.reduce([np.isfinite(field) for field in cold])
        assert warm_valid.sum() >= 50
        common = warm_valid & cold_valid
        union = warm_valid | cold_valid
        assert common.sum() / union.sum() > 0.90
        assert np.quantile(np.abs(warm[0][common] - cold[0][common]), 0.95) < 0.025
        assert np.quantile(np.abs(warm[1][common] - cold[1][common]), 0.95) < 0.025
        assert np.median(warm[0][warm_valid]) == pytest.approx(
            second_increment[0], abs=0.03)
        assert np.median(warm[1][warm_valid]) == pytest.approx(
            second_increment[1], abs=0.03)
        assert np.quantile(
            np.abs(warm[0][warm_valid] - second_increment[0]), 0.95) < 0.05
        assert np.quantile(
            np.abs(warm[1][warm_valid] - second_increment[1]), 0.95) < 0.05

        # Reference promotion swaps both the raw image and its cubic-spline
        # coefficients. At a fixed initial guess it must be indistinguishable
        # from precomputing a new solver explicitly on frame_1.
        initial = np.array([
            second_increment[0], second_increment[1], 0.0, 0.0, 0.0, 0.0])
        promoted_subset = warm_solver.solve_subset(frame_2, seed, initial)
        explicit_subset = cold_solver.solve_subset(frame_2, seed, initial)
        assert promoted_subset[2] == explicit_subset[2]
        np.testing.assert_allclose(
            promoted_subset[0], explicit_subset[0], rtol=0, atol=2e-7)
        assert promoted_subset[1] == pytest.approx(
            explicit_subset[1], rel=0, abs=2e-8)
    finally:
        warm_solver.close()
        cold_solver.close()


def test_stage_4_constant_increment_sequence_does_not_alternate():
    """Every promoted pair must measure the same immediate displacement."""
    ref = _safe_texture(seed=97, shape=(112, 120))
    increment = np.array([2.4, -1.7])
    frames = [
        shift(ref, shift=(step * increment[1], step * increment[0]),
              order=3, mode="mirror")
        for step in range(1, 7)
    ]
    params = DICParams(
        subset_radius=9, subset_spacing=9, search_radius=8,
        max_iter=35, conv_tol=1e-5, corr_cutoff=0.8, rescue_radius=6,
        mask_subsets_to_roi=False)
    roi = np.zeros(ref.shape, dtype=bool)
    roi[18:-18, 18:-18] = True
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, roi)
        seed = int(np.argmin(
            (solver.gx_flat - 60) ** 2 + (solver.gy_flat - 56) ** 2))
        medians = []
        for frame_index, frame in enumerate(frames):
            fields = solver.solve_frame(
                frame, seed_idx=seed, seed_guess=tuple(increment),
                warm_start=frame_index > 0)
            valid = np.isfinite(fields[0]) & np.isfinite(fields[1])
            assert valid.sum() >= 50
            medians.append((
                float(np.median(fields[0][valid])),
                float(np.median(fields[1][valid]))))
            solver.update_reference_image(frame)
        np.testing.assert_allclose(
            medians, np.broadcast_to(increment, (len(frames), 2)),
            rtol=0, atol=0.03)
    finally:
        solver.close()


def test_stage_4_full_gpu_analysis_correlates_adjacent_files(tmp_path, monkeypatch):
    """The app path must report k-1 -> k, never frame 0 -> k or k-2 -> k."""
    monkeypatch.setenv("PYDIC_SETTINGS_PATH", str(tmp_path / "settings.json"))
    from pydic.core.analysis import DICAnalysis

    ref = _safe_texture(seed=101, shape=(112, 120))
    increment = np.array([2.0, -1.5])
    images = [ref] + [
        shift(ref, shift=(step * increment[1], step * increment[0]),
              order=3, mode="mirror")
        for step in range(1, 7)
    ]
    paths = []
    for index, image in enumerate(images):
        path = tmp_path / f"frame_{index:03d}.png"
        assert cv2.imwrite(str(path), np.rint(image * 255.0).astype(np.uint8))
        paths.append(path)

    analysis = DICAnalysis()
    analysis.set_reference(str(paths[0]))
    for path in paths[1:]:
        analysis.add_deformed(str(path))
    roi = np.zeros(ref.shape, dtype=bool)
    roi[18:-18, 18:-18] = True
    analysis.set_roi_mask(roi)
    analysis.params.dynamic_roi = "None"
    analysis.params.subset_radius = 9
    analysis.params.subset_spacing = 9
    analysis.params.search_radius = 8
    analysis.params.rescue_radius = 6
    analysis.params.max_iter = 35
    analysis.params.conv_tol = 1e-5
    analysis.params.corr_cutoff = 0.8
    analysis.params.mask_subsets_to_roi = False
    analysis.run(seed_xy=(60, 56), use_gpu=True)

    medians = []
    for result in analysis.results:
        u, v = np.asarray(result.u), np.asarray(result.v)
        valid = np.isfinite(u) & np.isfinite(v)
        assert valid.sum() >= 50
        medians.append((float(np.median(u[valid])), float(np.median(v[valid]))))
    np.testing.assert_allclose(
        medians, np.broadcast_to(increment, (len(images) - 1, 2)),
        rtol=0, atol=0.04)


def test_stage_5_roi_clipping_prevents_background_motion_bias():
    ref = _safe_texture(seed=109, shape=(112, 120))
    params = DICParams(
        subset_radius=11, subset_spacing=9, search_radius=5,
        max_iter=35, conv_tol=1e-5, corr_cutoff=0.8, rescue_radius=4,
        mask_subsets_to_roi=True)
    # Material on the left is stationary; everything outside its ROI follows a
    # different displacement. Boundary subsets must not ingest that background.
    roi = np.zeros(ref.shape, dtype=bool)
    boundary_x = 65
    roi[12:-12, 12:boundary_x + 1] = True
    moved_background = shift(ref, shift=(3.0, -5.0), order=3, mode="mirror")
    current = moved_background
    current[roi] = ref[roi]

    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, roi)
        seed = int(np.argmin(
            (solver.gx_flat - 38) ** 2 + (solver.gy_flat - 56) ** 2 +
            (~solver.valid_mask) * 1e9))
        fields = solver.solve_frame(
            current, seed_idx=seed, seed_guess=(0.0, 0.0), warm_start=False)
        valid = np.logical_and.reduce([np.isfinite(field) for field in fields])
        assert valid.sum() >= 35
        assert np.quantile(np.abs(fields[0][valid]), 0.95) < 2e-5
        assert np.quantile(np.abs(fields[1][valid]), 0.95) < 2e-5
        # Explicitly include the rightmost eligible grid column, whose circular
        # subsets straddle the material/background boundary.
        eligible_x = solver.gx_flat[solver.valid_mask]
        edge_x = int(eligible_x.max())
        edge = valid[:, edge_x]
        assert edge.any()
        assert np.max(np.abs(fields[0][:, edge_x][edge])) < 2e-5
        assert np.max(np.abs(fields[1][:, edge_x][edge])) < 2e-5
    finally:
        solver.close()


def test_stage_5_native_icgn_is_invariant_to_8_bit_scaling():
    ref = _safe_texture(seed=131, shape=(112, 120))
    current = shift(ref, shift=(-1.35, 2.40), order=3, mode="mirror")
    params = DICParams(
        subset_radius=11, subset_spacing=9, search_radius=5,
        max_iter=35, conv_tol=1e-5, corr_cutoff=0.8, rescue_radius=3,
        mask_subsets_to_roi=False)
    outputs = []
    for scale in (1.0, 255.0):
        solver = NativeCudaSolver(params)
        try:
            solver.precompute_reference(
                ref * scale, np.ones(ref.shape, dtype=bool))
            index = int(np.argmin(
                (solver.gx_flat - 60) ** 2 +
                (solver.gy_flat - 56) ** 2))
            outputs.append(solver.solve_subset(
                current * scale, index,
                np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])))
        finally:
            solver.close()
    assert outputs[0][2] and outputs[1][2]
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=0, atol=2e-9)
    assert outputs[0][1] == pytest.approx(outputs[1][1], rel=0, abs=2e-10)


def test_stage_5_recovery_seeds_each_disconnected_failed_component():
    ref = _safe_texture(seed=141, shape=(112, 140))
    current = shift(ref, shift=(-1.25, 2.5), order=3, mode="mirror")
    roi = np.zeros(ref.shape, dtype=bool)
    roi[18:94, 14:53] = True
    roi[18:94, 87:126] = True
    params = DICParams(
        subset_radius=9, subset_spacing=9, search_radius=6,
        max_iter=35, conv_tol=1e-5, corr_cutoff=0.8, rescue_radius=4,
        mask_subsets_to_roi=True)
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, roi)
        left_candidates = np.flatnonzero(
            solver.valid_mask & (solver.gx_flat < 60))
        seed = int(left_candidates[len(left_candidates) // 2])
        initial_fields = solver.solve_frame(
            current, seed_idx=seed, seed_guess=(0.0, 0.0), warm_start=False)
        initial_valid = np.logical_and.reduce(
            [np.isfinite(field) for field in initial_fields])
        # A fresh solve deliberately preserves the former CuPy pipeline's one
        # global NCC seed; a wavefront cannot cross the gap between components.
        assert not np.any(initial_valid & roi &
                          (np.indices(ref.shape)[1] > 80))

        fields = solver.recover_failed(0.0, 0.0)
        valid = np.logical_and.reduce([np.isfinite(field) for field in fields])
        left = valid & roi & (np.indices(ref.shape)[1] < 60)
        right = valid & roi & (np.indices(ref.shape)[1] > 80)
        assert left.sum() >= 20
        assert right.sum() >= 20
        for region in (left, right):
            assert np.median(fields[0][region]) == pytest.approx(2.5, abs=0.03)
            assert np.median(fields[1][region]) == pytest.approx(-1.25, abs=0.03)
    finally:
        solver.close()


def test_stage_5_standalone_ncc_keeps_promotable_frame_state_coherent():
    ref = _safe_texture(seed=147, shape=(112, 120))
    frame_1 = shift(ref, shift=(-1.0, 2.0), order=3, mode="mirror")
    frame_2 = shift(frame_1, shift=(0.75, -1.25), order=3, mode="mirror")
    params = DICParams(
        subset_radius=9, subset_spacing=9, search_radius=5,
        max_iter=30, conv_tol=1e-5, corr_cutoff=0.8, rescue_radius=3,
        mask_subsets_to_roi=False)
    roi = np.ones(ref.shape, dtype=bool)
    staged = NativeCudaSolver(params)
    explicit = NativeCudaSolver(params)
    try:
        staged.precompute_reference(ref, roi)
        index = int(np.argmin(
            (staged.gx_flat - 60) ** 2 + (staged.gy_flat - 56) ** 2))
        staged.ncc_guess(frame_1, index)
        staged.update_reference_image(frame_1)
        initial = np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        promoted = staged.solve_subset(frame_2, index, initial)

        explicit.precompute_reference(frame_1, roi)
        direct = explicit.solve_subset(frame_2, index, initial)
        assert promoted[2] == direct[2]
        np.testing.assert_allclose(promoted[0], direct[0], rtol=0, atol=2e-7)
        assert promoted[1] == pytest.approx(direct[1], rel=0, abs=2e-8)
    finally:
        staged.close()
        explicit.close()


def test_stage_5_cuda_rejects_unsupported_quadratic_shape_order():
    params = DICParams(shape_order=2)
    with pytest.raises(NativeCudaError, match="first-order affine"):
        NativeCudaSolver(params)


@pytest.mark.parametrize("case", ("translation", "affine"))
def test_stage_5_complete_native_field_agrees_with_cpu(case):
    ref = _safe_texture(seed=151, shape=(112, 120))
    if case == "translation":
        current = shift(ref, shift=(-1.7, 2.3), order=3, mode="mirror")
    else:
        transform_xy = np.array([[1.006, -0.003], [0.002, 0.995]])
        translation_xy = np.array([1.4, -0.9])
        inverse = np.linalg.inv(transform_xy)
        current = affine_transform(
            ref, inverse[::-1, ::-1],
            offset=(-inverse @ translation_xy)[::-1],
            order=3, mode="mirror")
    roi = np.zeros(ref.shape, dtype=bool)
    roi[18:-18, 18:-18] = True
    params = DICParams(
        subset_radius=11, subset_spacing=9, search_radius=6,
        max_iter=40, conv_tol=1e-5, corr_cutoff=0.8, rescue_radius=4,
        mask_subsets_to_roi=True, shape_order=1)
    seed_xy = (60, 56)
    cpu = run_rg_dic(ref, current, roi, params, seed_xy=seed_xy)

    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, roi)
        seed = int(np.argmin(
            (solver.gx_flat - seed_xy[0]) ** 2 +
            (solver.gy_flat - seed_xy[1]) ** 2 +
            (~solver.valid_mask) * 1e9))
        native = solver.solve_frame(
            current, seed_idx=seed, seed_guess=(0.0, 0.0), warm_start=False)
    finally:
        solver.close()
    native_valid = np.logical_and.reduce([np.isfinite(field) for field in native])
    common = native_valid & cpu.analyzed
    union = native_valid | cpu.analyzed
    assert common.sum() >= 45
    assert common.sum() / union.sum() > 0.90
    displacement_delta = np.hypot(
        native[0][common] - cpu.u[common],
        native[1][common] - cpu.v[common])
    # CPU intentionally uses quintic interpolation while the port preserves
    # the former GPU cubic convention; sub-hundredth differences are typical.
    assert np.median(displacement_delta) < 0.02
    assert np.quantile(displacement_delta, 0.95) < 0.05
