"""Independent local least-squares checks for noise-sensitive strain windows."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strainx.core.cuda_native import native_cuda_available, native_cuda_diagnostic
from strainx.core.strain import (
    compute_velocity_strains, iter_velocity_strains_multi_window,
)


@pytest.mark.parametrize("backend", ["cpu", "gpu", "sweep"])
@pytest.mark.parametrize("spacing", [1, 3])
@pytest.mark.parametrize("motion", ["translation", "affine", "nonlinear"])
def test_minimum_window_matches_independent_lstsq(backend, spacing, motion):
    if backend == "gpu" and not native_cuda_available():
        pytest.skip(native_cuda_diagnostic())
    yy, xx = np.indices((27, 33), dtype=float)
    valid = np.zeros(xx.shape, bool)
    valid[2:-2:spacing, 2:-2:spacing] = True
    valid[10:14, 13:17] = False
    u, v = np.full(xx.shape, 2.4), np.full(xx.shape, -0.35)
    if motion != "translation":
        u += 0.013 * xx - 0.007 * yy
        v += 0.009 * xx + 0.021 * yy
    if motion == "nonlinear":
        u += 0.04 * np.sin(xx * 0.7) * np.cos(yy * 0.4)
        v += 0.03 * np.cos(xx * 0.3) * np.sin(yy * 0.6)
    u[~valid] = v[~valid] = np.nan
    radius = spacing  # Minimum nominal 3-by-3 neighbourhood.
    if backend == "sweep":
        result = dict(iter_velocity_strains_multi_window(
            u, v, valid, (radius,), spacing))[radius]
    else:
        result = compute_velocity_strains(
            u, v, valid, radius, spacing, use_gpu=backend == "gpu")

    names = ("dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy")
    for y, x in zip(*np.nonzero(valid)):
        local = valid & (abs(xx - x) <= radius) & (abs(yy - y) <= radius)
        design = np.column_stack((np.ones(local.sum()),
                                  xx[local] - x, yy[local] - y))
        actual = np.array([result[name][y, x] for name in names])
        if local.sum() < 6 or np.linalg.matrix_rank(design) < 3:
            assert np.isnan(actual).all()
            continue
        coefficients = np.linalg.lstsq(
            design, np.column_stack((u[local], v[local])), rcond=None)[0]
        expected = coefficients[1:, :].T.reshape(-1)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-11)
        if motion == "translation":
            assert abs(result["Eeff_rate"][y, x]) < 2e-11
