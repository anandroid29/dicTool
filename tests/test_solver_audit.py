"""Physical and recovery regressions discovered during the solver audit."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from strainx.core.bspline import BSplineInterpolator, circular_subset, image_gradient
from strainx.core.cuda_native import NativeCudaSolver, native_cuda_available
from strainx.core.icgn import precompute_subset
from strainx.core.rg_dic import DICParams, run_rg_dic


def test_session_preserves_solver_parameters(tmp_path, monkeypatch):
    from strainx.core.analysis import DICAnalysis, PairResult
    monkeypatch.setenv('STRAINX_SETTINGS_PATH', str(tmp_path / 'settings.json'))
    analysis = DICAnalysis()
    analysis.params = DICParams(max_iter=37, conv_tol=0.004, corr_cutoff=0.21,
                               search_radius=19, rescue_radius=6,
                               shape_order=2, mask_subsets_to_roi=False)
    zero = np.zeros((4, 4), dtype=np.float32)
    analysis.results = [PairResult(image_path='frame', u=zero, v=zero,
                                   Exx=zero, Exy=zero, Eyy=zero, Eeff=zero,
                                   du_dx=None, du_dy=None, dv_dx=None,
                                   dv_dy=None, corr=None)]
    path = str(tmp_path / 'session.h5')
    analysis.export_hdf5(path)
    restored = DICAnalysis()
    restored.load_hdf5(path)
    for name in ('max_iter', 'conv_tol', 'corr_cutoff', 'search_radius',
                 'rescue_radius', 'shape_order', 'mask_subsets_to_roi'):
        assert getattr(restored.params, name) == getattr(analysis.params, name)


def test_znssd_jacobian_includes_derivative_of_subset_mean():
    y, x = np.mgrid[:80, :80]
    ref = 0.25 + 0.004 * x + 0.08 * np.sin(x / 3) * np.cos(y / 4)
    dx, dy = circular_subset(9)
    gx, gy = image_gradient(ref)
    subset = precompute_subset(ref, gx, gy, 40, 40, dx, dy, intensity_scale=1)
    interp = BSplineInterpolator(ref)

    def normalized(translation):
        values = interp.eval(40.0 + dx + translation, 40.0 + dy)
        values -= values.mean()
        return values / np.linalg.norm(values)

    derivative = (normalized(1e-4) - normalized(-1e-4)) / 2e-4
    np.testing.assert_allclose(subset.sd[:, 0], derivative, atol=1e-8)


@pytest.mark.skipif(not native_cuda_available(), reason="Native CUDA unavailable")
def test_ncc_recovery_does_not_repeat_an_unusable_component_center():
    rng = np.random.default_rng(2026)
    ref = 0.2 + 0.6 * gaussian_filter(rng.random((96, 96)), 0.7)
    # The geometric centre of the only failed region cannot correlate, but
    # surrounding material is textured and exactly stationary.
    ref[37:60, 37:60] *= 0.01
    roi = np.zeros(ref.shape, bool)
    roi[15:82, 15:82] = True
    params = DICParams(subset_radius=7, subset_spacing=3, search_radius=4,
                       rescue_radius=2, hole_recovery="ncc")
    cpu = run_rg_dic(ref, ref, roi, params, seed_xy=(49, 49))
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, roi)
        seed = int(np.argmin((solver.gx_flat - 49)**2 + (solver.gy_flat - 49)**2))
        solver.solve_frame(ref, seed_idx=seed)
        result = solver.recover_failed(strategy="ncc")
        gpu_valid = np.isfinite(result[0])
        assert cpu.analyzed.sum() > 40
        assert gpu_valid.sum() >= cpu.analyzed.sum()
        np.testing.assert_allclose(result[0][gpu_valid], 0, atol=1e-6)
    finally:
        solver.close()


@pytest.mark.skipif(not native_cuda_available(), reason="Native CUDA unavailable")
def test_ncc_recovery_does_not_starve_regions_after_32_bad_seeds():
    rng = np.random.default_rng(2027)
    ref = 0.2 + 0.6 * gaussian_filter(rng.random((128, 160)), 0.7)
    roi = np.zeros(ref.shape, bool)
    # One good component is deliberately ranked behind 32 permanently bad
    # equal-sized components by the previous recovery implementation.
    centers = [(x, y) for y in range(12, 113, 20) for x in range(12, 153, 20)][:33]
    for x, y in centers:
        roi[y, x] = True
    for x, y in centers[1:]:
        ref[y-4:y+5, x-4:x+5] *= 0.01
    params = DICParams(subset_radius=2, subset_spacing=5, search_radius=2,
                       rescue_radius=0, mask_subsets_to_roi=False)
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, roi)
        x, y = centers[-1]
        seed = int(np.flatnonzero((solver.gx_flat == x) & (solver.gy_flat == y))[0])
        solver.solve_frame(ref, seed_idx=seed)
        recovered = solver.recover_failed(strategy="ncc")
        assert np.isfinite(recovered[0][centers[0][1], centers[0][0]])
    finally:
        solver.close()


@pytest.mark.skipif(not native_cuda_available(), reason="Native CUDA unavailable")
def test_gpu_restarts_after_total_tracking_dropout():
    ref = np.full((72, 72), 0.5)
    rng = np.random.default_rng(2028)
    textured = 0.2 + 0.6 * gaussian_filter(rng.random(ref.shape), 0.7)
    params = DICParams(subset_radius=7, subset_spacing=7, search_radius=3,
                       rescue_radius=2, hole_recovery='off')
    solver = NativeCudaSolver(params)
    try:
        solver.precompute_reference(ref, np.ones(ref.shape, bool))
        blank = solver.solve_frame(textured, seed_idx=20)
        assert not np.isfinite(blank[0]).any()
        solver.update_reference_image(textured)
        result = solver.solve_frame(textured, warm_start=True)
        assert np.isfinite(result[0]).sum() == solver.valid_mask.sum()
        np.testing.assert_allclose(result[0][np.isfinite(result[0])], 0, atol=1e-6)
    finally:
        solver.close()
