import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydic.core.analysis import DICAnalysis, PairResult
from pydic.core.rg_dic import DICParams
from pydic.core.strain_accum import StrainPathTracker
from pydic.core.units import Calibration
from pydic.ui.render import field_to_rgba


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


class StrainPathTrackerTests(unittest.TestCase):
    shape = (17, 41)
    radius = 2
    spacing = 2

    def _field(self, value):
        arr = np.full(self.shape, np.nan, dtype=float)
        ys = np.arange(self.radius, self.shape[0] - self.radius, self.spacing)
        xs = np.arange(self.radius, self.shape[1] - self.radius, self.spacing)
        arr[np.ix_(ys, xs)] = value
        return arr

    def _tracker(self):
        domain = np.ones(self.shape, dtype=bool)
        origin = np.zeros(self.shape, dtype=bool)
        origin[:, self.radius:self.radius + 1] = True
        return StrainPathTracker(
            self.shape, origin, domain, self.radius, self.spacing)

    def test_continuous_origin_replenishes_material_as_it_moves(self):
        tracker = self._tracker()
        u, z = self._field(2.0), self._field(0.0)
        valid = np.isfinite(u)

        tracker.seed(valid)
        first_count = tracker.count
        tracker.advance(u, z, z, z, z, z)
        tracker.seed(valid)
        self.assertGreater(tracker.count, first_count)
        tracker.advance(u, z, z, z, z, z)
        out = tracker.snapshot()["Exx_inf"]

        # One newly seeded column and one older downstream column coexist.
        self.assertTrue(np.isfinite(out[:, 4]).any())
        self.assertTrue(np.isfinite(out[:, 6]).any())

    def test_image_edge_origin_snaps_to_first_measurable_subset_layer(self):
        domain = np.ones(self.shape, dtype=bool)
        origin = np.zeros(self.shape, dtype=bool)
        origin[:, 0] = True
        tracker = StrainPathTracker(
            self.shape, origin, domain, self.radius, self.spacing)
        valid = np.isfinite(self._field(0.0))
        seeded = tracker.seed(valid)
        self.assertGreater(seeded, 0)
        self.assertTrue(np.all(tracker.x == self.radius))

    def test_green_lagrange_uses_composed_deformation_gradient(self):
        tracker = self._tracker()
        u, z, stretch = self._field(2.0), self._field(0.0), self._field(0.1)
        valid = np.isfinite(u)
        for _ in range(2):
            tracker.seed(valid)
            tracker.advance(u, z, stretch, z, z, z)
        out = tracker.snapshot()

        self.assertTrue(np.allclose(out["Exx_inf"][np.isfinite(out["Exx_inf"])].max(), 0.2))
        expected_gl = 0.5 * ((1.1 ** 2) ** 2 - 1.0)
        self.assertAlmostEqual(float(np.nanmax(out["Exx_gl"])), expected_gl)

    def test_signed_reversal_can_cancel_but_equivalent_never_decreases(self):
        tracker = self._tracker()
        u, z = self._field(2.0), self._field(0.0)
        valid = np.isfinite(u)
        tracker.seed(valid)
        tracker.advance(u, z, self._field(0.1), z, z, z)
        first_equivalent = float(np.nanmax(tracker.snapshot()["Eeff_gl"]))
        tracker.seed(valid)
        tracker.advance(u, z, self._field(-0.1), z, z, z)
        out = tracker.snapshot()

        oldest = np.nanmax(out["Eeff_gl"])
        self.assertGreater(oldest, first_equivalent)
        self.assertLess(float(np.nanmin(np.abs(out["Exx_inf"]))), 1e-12)

    def test_snapshot_ignores_live_points_inside_image_but_outside_subset_grid(self):
        tracker = self._tracker()
        tracker._append_zeros(
            np.array([self.radius, self.shape[1] - 1.0]),
            np.array([self.radius, self.radius]))

        out = tracker.snapshot()["Exx_inf"]

        self.assertEqual(np.count_nonzero(np.isfinite(out)), 1)
        self.assertEqual(float(out[self.radius, self.radius]), 0.0)

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

    def test_selected_start_frame_is_zero_then_continuous_strain_advects(self):
        shape = (17, 41)
        radius, spacing = 2, 2
        ys = np.arange(radius, shape[0] - radius, spacing)
        xs = np.arange(radius, shape[1] - radius, spacing)

        def sparse(value):
            arr = np.full(shape, np.nan, dtype=float)
            arr[np.ix_(ys, xs)] = value
            return arr

        u, z = sparse(2.0), sparse(0.0)
        # The last interval has a different strain. Its newly seeded path later
        # revisits x=radius+2 and must not repaint the first path's value there.
        stretch_history = [0.05, 0.05, 0.05, 0.20]
        results = []
        for stretch_value in stretch_history:
            stretch = sparse(stretch_value)
            results.append(PairResult(
                image_path="frame", u=u.copy(), v=z.copy(),
                Exx=z.copy(), Exy=z.copy(), Eyy=z.copy(), Eeff=z.copy(),
                du_dx=stretch.copy(), du_dy=z.copy(),
                dv_dx=z.copy(), dv_dy=z.copy(), corr=z.copy(),
                valid=np.isfinite(u)))

        analysis = object.__new__(DICAnalysis)
        analysis.results = results
        analysis._roi_mask = np.ones(shape, dtype=bool)
        analysis._strain_origin_mask = np.zeros(shape, dtype=bool)
        analysis._strain_origin_mask[:, radius] = True
        analysis.strain_start_frame = 2
        analysis.params = DICParams(
            subset_radius=radius, subset_spacing=spacing, strain_window=2)
        analysis._transport_accumulated_strain()

        self.assertFalse(np.isfinite(results[0].Exx_gl).any())
        self.assertAlmostEqual(float(np.nanmax(results[1].Exx_gl)), 0.0)
        self.assertGreater(float(np.nanmax(results[2].Exx_gl)), 0.0)
        coverage = [np.isfinite(result.Exx_gl) for result in results[1:]]
        for previous, current in zip(coverage[:-1], coverage[1:]):
            self.assertTrue(np.all(current[previous]))
        self.assertGreater(int(coverage[-1].sum()), int(coverage[0].sum()))
        # The inlet/source element remains coloured after its material path has
        # moved downstream; the reached region grows instead of translating.
        self.assertTrue(np.isfinite(results[-1].Exx_gl[:, radius]).any())
        self.assertTrue(np.isfinite(results[-1].Exx_gl[:, radius + 4]).any())
        # First arrival wins. Frame 3 first populates radius+2 with 0.05; the
        # new path arriving there under the later 0.20 interval cannot change it.
        first = float(results[2].Exx_gl[radius, radius + 2])
        later = float(results[3].Exx_gl[radius, radius + 2])
        self.assertGreater(first, 0.0)
        self.assertEqual(later, first)
        # More generally, every finite strain value from an earlier frame is
        # immutable in every later swept snapshot.
        for previous, current in zip(results[1:-1], results[2:]):
            for name in ("Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"):
                old = getattr(previous, name)
                new = getattr(current, name)
                fixed = np.isfinite(old)
                self.assertTrue(np.array_equal(new[fixed], old[fixed]))

    def test_current_hdf5_round_trip_preserves_explicit_fields_and_strain_origin(self):
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
        self.analysis._strain_origin_mask = np.array(
            [[True, False], [True, False]], dtype=bool)
        self.analysis.strain_start_frame = 1

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
        self.assertIsNone(restored.Exy_inf)
        self.assertIsNone(restored.Gxy_inf)
        self.assertIsNone(restored.corr)
        self.assertIsNone(restored.du_dx)
        self.assertIsNone(restored.Gxy_gl)
        self.assertTrue(np.allclose(restored.Eeff_gl, 0.17))
        self.assertEqual(loaded.strain_start_frame, 1)
        self.assertTrue(np.array_equal(
            loaded.strain_origin_mask, self.analysis._strain_origin_mask))


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
            analysis._strain_origin_mask = np.ones((8, 9), bool)
            analysis._roi_mask = np.ones((8, 9), bool)
            analysis.set_reference(image_path)

        self.assertEqual(analysis.results, [])
        self.assertEqual(analysis.reference_image.dtype, np.float32)
        self.assertIsNone(analysis.roi_mask)
        self.assertIsNone(analysis.dynamic_include_mask)
        self.assertIsNone(analysis.dynamic_exclude_mask)
        self.assertIsNone(analysis.strain_origin_mask)


if __name__ == "__main__":
    unittest.main()
