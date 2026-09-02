import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strainx.core.cuda_native import (
    NativeCudaSolver, native_cuda_available, native_cuda_diagnostic,
    native_plane_fit,
)
from strainx.core.rg_dic import DICParams


pytestmark = pytest.mark.skipif(
    not native_cuda_available(), reason=native_cuda_diagnostic())


def test_native_plane_fit_is_exact_for_an_affine_velocity_field():
    yy, xx = np.mgrid[:96, :112]
    vx = 1.2 + 0.03 * xx - 0.02 * yy
    vy = -0.4 + 0.01 * xx + 0.04 * yy
    mask = np.ones(vx.shape, dtype=bool)

    fitted = native_plane_fit(vx, vy, mask, 7)
    core = np.s_[8:-8, 8:-8]
    for actual, expected in zip(fitted, (0.03, -0.02, 0.01, 0.04)):
        np.testing.assert_allclose(actual[core], expected, rtol=0, atol=2e-12)


def test_native_solver_translation_warm_start_and_reference_promotion():
    from scipy.ndimage import gaussian_filter, shift

    rng = np.random.default_rng(8)
    ref = gaussian_filter(rng.random((96, 104)), 1.0)
    cur = shift(ref, shift=(1.0, 2.0), order=3, mode="mirror")
    params = DICParams(
        subset_radius=7, subset_spacing=8, search_radius=4,
        max_iter=20, corr_cutoff=0.5, rescue_radius=2)
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, np.ones(ref.shape, dtype=bool))
        seed = int(np.argmin(
            (solver.gx_flat - 52) ** 2 + (solver.gy_flat - 48) ** 2))
        fields = solver.solve_frame(
            cur, seed_idx=seed, seed_guess=(0.0, 0.0), warm_start=False)
        u, v = fields[:2]
        valid = np.isfinite(u) & np.isfinite(v)
        assert valid.sum() >= 70
        assert np.median(u[valid]) == pytest.approx(2.0, abs=0.05)
        assert np.median(v[valid]) == pytest.approx(1.0, abs=0.05)

        solver.update_reference_image(cur)
        next_frame = shift(cur, shift=(-1.0, 1.0), order=3, mode="mirror")
        next_fields = solver.solve_frame(next_frame, warm_start=True)
        valid = np.isfinite(next_fields[0]) & np.isfinite(next_fields[1])
        assert valid.sum() >= 70
        assert np.median(next_fields[0][valid]) == pytest.approx(1.0, abs=0.05)
        assert np.median(next_fields[1][valid]) == pytest.approx(-1.0, abs=0.05)
    finally:
        solver.close()


def test_native_solver_recovers_affine_displacement_gradients():
    from scipy.ndimage import affine_transform, gaussian_filter

    rng = np.random.default_rng(2)
    ref = gaussian_filter(rng.random((144, 144)), 1.0)
    transform_xy = np.array([[1.01, 0.004], [-0.003, 0.992]])
    translation_xy = np.array([1.5, -0.8])
    inverse = np.linalg.inv(transform_xy)
    cur = affine_transform(
        ref, inverse[::-1, ::-1],
        offset=(-inverse @ translation_xy)[::-1], order=3, mode="mirror")
    params = DICParams(
        subset_radius=11, subset_spacing=8, search_radius=5,
        max_iter=30, corr_cutoff=0.5, rescue_radius=2)
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, np.ones(ref.shape, dtype=bool))
        seed = len(solver.gx_flat) // 2
        fields = solver.solve_frame(
            cur, seed_idx=seed, seed_guess=(2.0, -1.0))
        valid = np.logical_and.reduce([np.isfinite(field) for field in fields[:6]])
        valid[:24] = False; valid[-24:] = False
        valid[:, :24] = False; valid[:, -24:] = False
        expected = (0.01, 0.004, -0.003, -0.008)
        for actual, truth in zip(fields[2:6], expected):
            assert np.median(actual[valid]) == pytest.approx(truth, abs=5e-4)
    finally:
        solver.close()
