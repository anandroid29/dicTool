import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pydic"))

from src.core.analysis import DynamicROI, _dynamic_measurement_mask


class DynamicROICoordinateTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
