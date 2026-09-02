import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydic.core.analysis import DICAnalysis, DynamicROI, _dynamic_measurement_mask


class DynamicROICoordinateTests(unittest.TestCase):
    def test_exact_frame_override_precedence_and_threshold(self):
        analysis = object.__new__(DICAnalysis)
        analysis._roi_mask = np.ones((5, 7), dtype=bool)
        analysis.params = type("Params", (), {
            "dynamic_roi_threshold": 0.40,
        })()
        include = np.zeros((5, 7), dtype=bool)
        exclude = np.zeros((5, 7), dtype=bool)
        exclude[2, 2:5] = True
        include[2, 3] = True
        analysis.dynamic_frame_overrides = {
            3: {"threshold": 0.72, "exclude": exclude, "include": include},
        }

        base = np.ones((5, 7), dtype=bool)
        frame_two = analysis.apply_dynamic_frame_override(base, 2)
        frame_three = analysis.apply_dynamic_frame_override(base, 3)

        self.assertTrue(frame_two.all())
        self.assertFalse(frame_three[2, 2])
        self.assertTrue(frame_three[2, 3])  # frame Include wins last
        self.assertFalse(frame_three[2, 4])
        self.assertAlmostEqual(analysis.dynamic_threshold_for_frame(2), 0.40)
        self.assertAlmostEqual(analysis.dynamic_threshold_for_frame(3), 0.72)

        analysis.dynamic_frame_overrides[4] = {
            "replace": True, "include": include,
        }
        replaced = analysis.apply_dynamic_frame_override(base, 4)
        self.assertTrue(np.array_equal(replaced, include))

        analysis.dynamic_future_overrides = {
            5: {"threshold": 0.55, "replace": True, "include": include},
        }
        self.assertAlmostEqual(analysis.dynamic_threshold_for_frame(5), 0.55)
        self.assertTrue(np.array_equal(
            analysis.apply_dynamic_frame_override(base, 6), include))
        analysis.dynamic_frame_overrides[6] = {"replace": False}
        self.assertTrue(
            analysis.apply_dynamic_frame_override(base, 6).all())
        analysis.dynamic_future_overrides[7] = {"reset": True}
        self.assertAlmostEqual(analysis.dynamic_threshold_for_frame(7), 0.40)
        self.assertTrue(analysis.apply_dynamic_frame_override(base, 7).all())

    def test_temporal_hysteresis_prevents_borderline_mask_flicker(self):
        shape = (24, 32)
        roi = DynamicROI(
            "Edge Detection", threshold=0.5, fill_holes=False,
            keep_min_area_frac=0.0, hysteresis=0.03)
        if not roi.enabled:
            self.skipTest("OpenCV is required for Dynamic ROI")
        roi.thresh = 127.5
        metric = np.full(shape, 130, dtype=np.uint8)
        roi._metric8 = lambda _image, _metric=None: metric.copy()
        image = np.zeros(shape, dtype=float)

        self.assertTrue(roi.mask(image, reference_frame=False).all())
        metric[:] = 124  # below threshold, but still inside the exit band
        self.assertTrue(roi.mask(image, reference_frame=False).all())
        metric[:] = 110  # a real texture loss still exits immediately
        self.assertFalse(roi.mask(image, reference_frame=False).any())

    def test_current_mask_is_not_clipped_to_stationary_reference_roi(self):
        shape = (24, 32)
        static = np.zeros(shape, dtype=bool)
        static[7:17, 3:12] = True
        metric = np.zeros(shape, dtype=np.uint8)
        metric[7:17, 19:28] = 255

        roi = DynamicROI(
            "Edge Detection", threshold=0.5, roi_mask=static,
            fill_holes=False)
        if not roi.enabled:
            self.skipTest("OpenCV is required for Dynamic ROI")
        roi.thresh = 127.5
        roi._metric8 = lambda _image, _metric=None: metric.copy()

        image = np.zeros(shape, dtype=float)
        current = roi.mask(image, reference_frame=False)
        preview = roi.mask(image, reference_frame=True)

        self.assertTrue(current[10, 22])
        self.assertFalse(preview[10, 22])
        self.assertFalse(np.any(preview & ~static))

    def test_each_pair_samples_from_its_own_previous_frame_grid(self):
        shape = (9, 24)
        points = np.zeros(shape, dtype=bool)
        points[4, 3:9] = True
        inc_u = np.where(points, 2.0, np.nan)
        inc_v = np.where(points, 0.0, np.nan)
        for frame in range(6):
            current = np.zeros(shape, dtype=bool)
            # Every iteration represents a new previous->current pair. The
            # source centres are spatially fresh, so only this pair's +2 px
            # displacement belongs in the coordinate transform.
            start = 5
            current[4, start:start + 6] = True
            valid = _dynamic_measurement_mask(
                points, current, inc_u, inc_v)
            self.assertTrue(valid[points].all(), frame)

    def test_source_grid_overrides_apply_each_pair_and_include_wins(self):
        shape = (7, 12)
        points = np.zeros(shape, dtype=bool)
        points[3, 2:5] = True
        inc_u = np.where(points, 4.0, np.nan)
        inc_v = np.where(points, 0.0, np.nan)
        current = np.zeros(shape, dtype=bool)
        current[3, 6:9] = True
        current[3, 7] = False

        include = np.zeros(shape, dtype=bool)
        exclude = np.zeros(shape, dtype=bool)
        include[3, 3] = True
        exclude[3, 2:4] = True

        valid = _dynamic_measurement_mask(
            points, current, inc_u, inc_v,
            include_mask=include, exclude_mask=exclude)

        self.assertFalse(valid[3, 2])
        self.assertTrue(valid[3, 3])
        self.assertTrue(valid[3, 4])

    def test_exact_frame_destination_override_beats_global_base(self):
        shape = (7, 12)
        points = np.zeros(shape, dtype=bool)
        points[3, 2:5] = True
        inc_u = np.where(points, 4.0, np.nan)
        inc_v = np.where(points, 0.0, np.nan)
        current = np.zeros(shape, dtype=bool)
        current[3, 6:9] = True

        global_include = np.zeros(shape, dtype=bool)
        global_exclude = np.zeros(shape, dtype=bool)
        global_include[3, 2] = True
        global_exclude[3, 3] = True
        frame_exclude = np.zeros(shape, dtype=bool)
        frame_include = np.zeros(shape, dtype=bool)
        frame_exclude[3, 6] = True   # beats global Include at source x=2
        frame_include[3, 7] = True   # beats global Exclude at source x=3

        valid = _dynamic_measurement_mask(
            points, current, inc_u, inc_v,
            include_mask=global_include, exclude_mask=global_exclude,
            frame_include_mask=frame_include,
            frame_exclude_mask=frame_exclude)

        self.assertFalse(valid[3, 2])
        self.assertTrue(valid[3, 3])
        self.assertTrue(valid[3, 4])

        replaced = _dynamic_measurement_mask(
            points, current, inc_u, inc_v,
            frame_include_mask=frame_include, replace_base=True)
        self.assertFalse(replaced[3, 2])
        self.assertTrue(replaced[3, 3])
        self.assertFalse(replaced[3, 4])


if __name__ == "__main__":
    unittest.main()
