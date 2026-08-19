import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pydic"))

from src.core.analysis import DICAnalysis, PairResult
from src.core.rg_dic import DICParams
from src.core.strain_accum import StrainAccumulator, _neighbour_estimate
from src.core.units import Calibration
from src.ui.render import field_to_rgba


def full(shape, value):
    return np.full(shape, value, dtype=float)


class StrainWindowValidationTests(unittest.TestCase):
    def test_exact_grid_count_uses_floor_before_doubling(self):
        params = DICParams(strain_window=3, subset_spacing=2)
        self.assertEqual(params.strain_points_per_axis(), 3)
        self.assertEqual(params.effective_strain_window(warn=False), 3)

    def test_window_is_clamped_only_below_three_points_per_axis(self):
        params = DICParams(strain_window=3, subset_spacing=4)
        self.assertEqual(params.strain_points_per_axis(), 1)
        self.assertEqual(params.effective_strain_window(warn=False), 4)

    def test_window_equal_to_spacing_is_valid(self):
        params = DICParams(strain_window=5, subset_spacing=5)
        self.assertEqual(params.strain_points_per_axis(), 3)
        self.assertEqual(params.effective_strain_window(warn=False), 5)


class StrainAccumulatorTests(unittest.TestCase):
    @staticmethod
    def _xy(shape=(9, 9)):
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        return xx.astype(float), yy.astype(float)

    def test_accumulates_formulations_and_equivalent_magnitudes_separately(self):
        shape = (7, 7)
        acc = StrainAccumulator(shape, strain_window=2)
        zeros = full(shape, 0.0)
        x = np.broadcast_to(np.arange(shape[1], dtype=float), shape)

        # Two identical 10% x-stretch increments.
        for _ in range(2):
            acc.add_frame(0.1 * x, zeros, zeros, zeros, zeros, zeros)

        out = acc.results()
        self.assertTrue(np.allclose(out["Exx_inf"], 0.2))
        # Per increment Green-Lagrange Exx = ((1.1)^2 - 1)/2 = 0.105.
        self.assertTrue(np.allclose(out["Exx_gl"], 0.21))
        self.assertTrue(np.allclose(out["Gxy_inf"], 2.0 * out["Exy_inf"]))
        self.assertTrue(np.allclose(out["Gxy_gl"], 2.0 * out["Exy_gl"]))

        one_inf_eq = np.sqrt((2.0 / 3.0) * (0.1**2 + 0.0 + (-0.1)**2))
        one_gl_eq = np.sqrt((2.0 / 3.0) * (0.105**2 + 0.0 + (-0.105)**2))
        self.assertTrue(np.allclose(out["Eeff_inf"], 2.0 * one_inf_eq))
        self.assertTrue(np.allclose(out["Eeff_gl"], 2.0 * one_gl_eq))

    def test_dropout_is_hidden_then_can_recover(self):
        shape = (7, 7)
        acc = StrainAccumulator(shape, strain_window=2)
        zeros = full(shape, 0.0)
        x = np.broadcast_to(np.arange(shape[1], dtype=float), shape)
        ramp = 0.01 * x
        nan = full(shape, np.nan)

        acc.add_frame(ramp, zeros, zeros, zeros, zeros, zeros)
        first = acc.results()["Exx_inf"].copy()
        acc.add_frame(nan, nan, nan, nan, nan, nan)
        self.assertFalse(acc.valid.any())
        acc.add_frame(ramp, zeros, zeros, zeros, zeros, zeros)

        self.assertTrue(acc.valid.all())
        self.assertTrue(np.allclose(acc.results()["Exx_inf"], first + 0.01))

    def test_simple_shear_keeps_tensor_and_engineering_names_distinct(self):
        shape = (9, 9)
        x, y = self._xy(shape)
        z = np.zeros(shape)
        gamma = 0.08
        acc = StrainAccumulator(shape, strain_window=3)
        acc.add_frame(gamma * y, z, z, z, z, z)
        out = acc.results()
        self.assertTrue(np.allclose(out["Exy_inf"], gamma / 2.0))
        self.assertTrue(np.allclose(out["Gxy_inf"], gamma))
        self.assertTrue(np.allclose(out["Exy_gl"], gamma / 2.0))
        self.assertTrue(np.allclose(out["Eyy_gl"], gamma**2 / 2.0))

    def test_rigid_rotation_is_zero_green_lagrange_without_special_correction(self):
        shape = (11, 11)
        x, y = self._xy(shape)
        z = np.zeros(shape)
        angle = np.deg2rad(7.0)
        c, s = np.cos(angle), np.sin(angle)
        u = (c - 1.0) * x - s * y
        v = s * x + (c - 1.0) * y
        acc = StrainAccumulator(shape, strain_window=4)
        acc.add_frame(u, v, z, z, z, z)
        out = acc.results()
        self.assertLess(float(np.nanmax(np.abs(out["Exx_gl"]))), 1e-10)
        self.assertLess(float(np.nanmax(np.abs(out["Eyy_gl"]))), 1e-10)
        self.assertLess(float(np.nanmax(out["Eeff_gl"])), 1e-10)
        # Infinitesimal strain is intentionally not rotation-corrected.
        self.assertGreater(float(np.nanmax(out["Eeff_inf"])), 1e-4)

    def test_compression_preserves_negative_components_and_positive_equivalent(self):
        shape = (9, 9)
        x, _ = self._xy(shape)
        z = np.zeros(shape)
        acc = StrainAccumulator(shape, strain_window=3)
        acc.add_frame(-0.05 * x, z, z, z, z, z)
        out = acc.results()
        self.assertTrue(np.allclose(out["Exx_inf"], -0.05))
        self.assertTrue(np.allclose(out["Exx_gl"], -0.04875))
        self.assertTrue(np.all(out["Eeff_inf"] > 0.0))
        self.assertTrue(np.all(out["Eeff_gl"] > 0.0))

    def test_reversal_cancels_infinitesimal_component_but_not_equivalent_magnitude(self):
        shape = (9, 9)
        x, _ = self._xy(shape)
        z = np.zeros(shape)
        acc = StrainAccumulator(shape, strain_window=3)
        acc.add_frame(0.05 * x, z, z, z, z, z)
        acc.add_frame(-0.05 * x, z, z, z, z, z)
        out = acc.results()
        self.assertLess(float(np.nanmax(np.abs(out["Exx_inf"]))), 1e-10)
        self.assertGreater(float(np.nanmin(out["Eeff_inf"])), 0.1)

    def test_disconnected_components_do_not_share_a_strain_fit(self):
        """A physical cut separates independently translating bodies."""
        shape = (15, 15)
        u = np.zeros(shape)
        v = np.zeros(shape)
        u[:, 8:] = 2.0
        u[:, 7] = np.nan  # physical cut between independently moving bodies
        v[:, 7] = np.nan
        z = np.zeros(shape)
        acc = StrainAccumulator(shape, strain_window=4)
        acc.add_frame(u, v, z, z, z, z)
        grad = acc.results()["du_dx"]
        near_cut = grad[:, 4:11]
        self.assertLess(float(np.nanmax(np.abs(near_cut))), 1e-8)

    def test_infinity_is_invalid_for_displacement_and_every_strain_field(self):
        shape = (9, 9)
        x, _ = self._xy(shape)
        u = 0.02 * x
        v = np.zeros(shape)
        u[4, 4] = np.inf
        acc = StrainAccumulator(shape, strain_window=3)
        acc.add_frame(u, v, v, v, v, v)
        out = acc.results()

        self.assertFalse(out["valid"][4, 4])
        for name in ("du_dx", "du_dy", "dv_dx", "dv_dy",
                     "Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"):
            self.assertFalse(np.isinf(out[name]).any(), name)
            self.assertTrue(np.isnan(out[name][4, 4]), name)

    def test_recovery_smoothing_does_not_cross_an_invalid_cut(self):
        shape = (7, 9)
        eligible = np.ones(shape, dtype=bool)
        eligible[:, 4] = False
        field = np.where(np.indices(shape)[1] < 4, 1.0, 100.0)
        live = eligible.copy()
        live[3, 3] = False

        estimate = _neighbour_estimate(
            field, live, eligible=eligible, grid_spacing=1, radii=(3,))
        self.assertAlmostEqual(float(estimate[3, 3]), 1.0)

    def test_rendering_masks_nonfinite_and_outside_roi_before_interpolation(self):
        arr = np.array([[0.0, np.nan, 10.0],
                        [0.0, np.inf, 10.0],
                        [0.0, np.nan, 10.0]])
        roi = np.zeros(arr.shape, dtype=bool)
        roi[:, :2] = True
        rgba = field_to_rgba(arr, 0.0, 10.0, "turbo", roi, spacing=1)
        self.assertIsNotNone(rgba)
        self.assertTrue(np.all(rgba[:, 2, 3] == 0))
        self.assertTrue(np.all(rgba[:, 1, 3] == 0))


class InstantaneousKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.analysis = object.__new__(DICAnalysis)
        self.analysis.results = []
        self.analysis.fps = 20.0
        self.analysis._roi_mask = np.ones((2, 2), dtype=bool)
        self.analysis.params = DICParams(strain_window=3, subset_spacing=1)

    def _result(self, displacement):
        a = full((2, 2), displacement)
        z = full((2, 2), 0.0)
        return PairResult(
            image_path="frame", u=a, v=z,
            Exx=z.copy(), Exy=z.copy(), Eyy=z.copy(), Eeff=z.copy(),
            du_dx=z.copy(), du_dy=z.copy(), dv_dx=z.copy(), dv_dy=z.copy(),
            corr=z.copy(), valid=np.ones((2, 2), dtype=bool),
        )

    def test_displacement_alias_is_not_differenced_again(self):
        self.analysis.results = [self._result(1.0), self._result(1.5)]
        self.analysis._compute_incremental_displacements()
        self.assertTrue(np.allclose(self.analysis.results[0].u_inc, 1.0))
        self.assertTrue(np.allclose(self.analysis.results[1].u_inc, 1.5))

    def test_velocity_is_interval_displacement_over_dt(self):
        self.analysis.results = [self._result(1.5)]
        result = self.analysis.results[0]
        result.valid[0, 0] = False
        self.analysis._compute_velocities_and_rates()
        self.assertTrue(np.allclose(result.Vx[1:, :], 30.0))
        self.assertTrue(np.allclose(result.Vy[1:, :], 0.0))
        self.assertEqual(result.Vx.dtype, np.float32)
        self.assertEqual(result.Eeff_rate.dtype, np.float32)
        self.assertIs(result.Exx_rate, result.dVx_dx)
        self.assertIs(result.Eyy_rate, result.dVy_dy)

    def test_displacement_aliases_share_compact_storage(self):
        result = self._result(1.5)
        result.valid[0, 0] = False
        self.analysis.results = [result]
        self.analysis._compute_incremental_displacements()
        self.assertEqual(result.u.dtype, np.float32)
        self.assertEqual(result.v.dtype, np.float32)
        self.assertEqual(result.mag_inc.dtype, np.float32)
        self.assertIs(result.u_inc, result.u)
        self.assertIs(result.v_inc, result.v)
        self.assertTrue(np.isnan(result.u[0, 0]))

    def test_schema_three_hdf5_round_trip_preserves_explicit_fields(self):
        result = self._result(1.25)
        result.u_inc = result.u.copy()
        result.v_inc = result.v.copy()
        result.mag_inc = np.abs(result.u)
        result.Exx_inf = full((2, 2), 0.1)
        result.Eyy_inf = full((2, 2), -0.02)
        result.Exy_inf = full((2, 2), 0.03)
        result.Gxy_inf = full((2, 2), 0.06)
        result.Eeff_inf = full((2, 2), 0.15)
        result.Exx_gl = full((2, 2), 0.11)
        result.Eyy_gl = full((2, 2), -0.01)
        result.Exy_gl = full((2, 2), 0.04)
        result.Gxy_gl = full((2, 2), 0.08)
        result.Eeff_gl = full((2, 2), 0.17)

        self.analysis.results = [result]
        self.analysis.ref_path = None
        self.analysis.calibration = Calibration()

        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "roundtrip.h5")
            self.analysis.export_hdf5(path)

            loaded = object.__new__(DICAnalysis)
            loaded.results = []
            loaded.def_paths = []
            loaded.params = DICParams()
            loaded.calibration = Calibration()
            loaded._roi_mask = None
            loaded._ref_image = None
            loaded.ref_path = None
            loaded.fps = 1.0
            loaded.load_hdf5(path)

        self.assertEqual(len(loaded.results), 1)
        restored = loaded.results[0]
        self.assertTrue(np.allclose(restored.u, 1.25))
        self.assertTrue(np.allclose(restored.u_inc, restored.u))
        self.assertTrue(np.allclose(restored.Gxy_inf, 2.0 * restored.Exy_inf))
        self.assertTrue(np.allclose(restored.Eeff_gl, 0.17))


class StatePersistenceTests(unittest.TestCase):
    def test_calibration_gpu_and_rescue_radius_are_cached(self):
        with tempfile.TemporaryDirectory() as td:
            settings = str(Path(td) / "settings.json")
            with patch.dict("os.environ", {"PYDIC_SETTINGS_PATH": settings}):
                saved = DICAnalysis()
                saved.calibration = Calibration.from_pixel_size(12.5, "µm")
                saved.prefer_gpu = False
                saved.params.rescue_radius = 7
                saved.save_settings()

                loaded = DICAnalysis()

        self.assertAlmostEqual(loaded.calibration.pixel_size_in("µm"), 12.5)
        self.assertEqual(loaded.calibration.display_unit, "µm")
        self.assertFalse(loaded.prefer_gpu)
        self.assertEqual(loaded.params.rescue_radius, 7)

    def test_new_reference_clears_spatial_state_and_old_results(self):
        import cv2
        with tempfile.TemporaryDirectory() as td:
            image_path = str(Path(td) / "reference.png")
            cv2.imwrite(image_path, np.zeros((8, 9), np.uint8))
            analysis = object.__new__(DICAnalysis)
            analysis.results = [object()]
            analysis.dynamic_include_mask = np.ones((8, 9), bool)
            analysis.dynamic_exclude_mask = np.ones((8, 9), bool)
            analysis._roi_mask = np.ones((8, 9), bool)
            analysis.set_reference(image_path)

        self.assertEqual(analysis.results, [])
        self.assertIsNone(analysis.roi_mask)
        self.assertIsNone(analysis.dynamic_include_mask)
        self.assertIsNone(analysis.dynamic_exclude_mask)


if __name__ == "__main__":
    unittest.main()
