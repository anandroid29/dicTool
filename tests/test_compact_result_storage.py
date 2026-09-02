import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pydic"))

from pydic.core.analysis import DICAnalysis, PairResult
from pydic.core.compact_field import CompactField, CompactMask
from pydic.core.rg_dic import DICParams
from pydic.core.units import Calibration


class CompactResultStorageTests(unittest.TestCase):
    @staticmethod
    def _field(shape, ys, xs, value):
        dense = np.full(shape, np.nan, dtype=np.float32)
        dense[np.ix_(ys, xs)] = value
        return dense

    def _compact_result(self):
        shape = (17, 41)
        radius = spacing = 2
        ys = np.arange(radius, shape[0] - radius, spacing)
        xs = np.arange(radius, shape[1] - radius, spacing)
        valid = np.isfinite(self._field(shape, ys, xs, 0.0))
        indices = np.flatnonzero(valid.reshape(-1)).astype(np.uint32)

        def packed(value):
            return CompactField.from_dense(
                self._field(shape, ys, xs, value), indices=indices)

        result = PairResult(
            image_path="frame.png", u=packed(2.0), v=packed(0.0),
            Exx=None, Exy=None, Eyy=None, Eeff=None,
            du_dx=packed(0.05), du_dy=packed(0.0),
            dv_dx=packed(0.0), dv_dy=packed(0.0),
            corr=packed(0.99), valid=CompactMask(shape, indices))
        return result, shape, radius, spacing

    def test_compact_postprocessing_keeps_only_finite_subset_samples(self):
        result, shape, radius, spacing = self._compact_result()
        analysis = object.__new__(DICAnalysis)
        analysis.results = [result]
        analysis.fps = 20.0
        analysis.params = DICParams(
            subset_radius=radius, subset_spacing=spacing, strain_window=2)
        analysis._roi_mask = np.ones(shape, dtype=bool)
        analysis._strain_origin_mask = np.zeros(shape, dtype=bool)
        analysis._strain_origin_mask[:, radius] = True
        analysis.strain_start_frame = 0

        analysis._compute_incremental_displacements()
        analysis._compute_velocities_and_rates()
        analysis._transport_accumulated_strain()

        for name in ("u", "v", "mag_inc", "Vx", "Vy", "Veff",
                     "Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate",
                     "Exx_gl", "Exy_gl", "Eyy_gl", "Eeff_gl"):
            field = getattr(result, name)
            self.assertIsInstance(field, CompactField, name)
            self.assertFalse(np.isnan(field.values).any(), name)
        self.assertIsInstance(result.valid, CompactMask)
        self.assertIsNone(result.corr)
        self.assertIsNone(result.du_dx)
        seen = set()
        retained = 0
        for value in vars(result).values():
            arrays = ((value.indices, value.values)
                      if isinstance(value, CompactField) else
                      ((value.indices,) if isinstance(value, CompactMask) else ()))
            for array in arrays:
                if id(array) not in seen:
                    seen.add(id(array))
                    retained += array.nbytes
        self.assertLess(retained, shape[0] * shape[1] * 20)

    def test_schema_five_hdf5_contains_no_dense_or_derived_frame_fields(self):
        result, shape, radius, spacing = self._compact_result()
        analysis = object.__new__(DICAnalysis)
        analysis.results = [result]
        analysis.fps = 20.0
        analysis.params = DICParams(
            subset_radius=radius, subset_spacing=spacing, strain_window=2)
        analysis._roi_mask = np.ones(shape, dtype=bool)
        analysis._strain_origin_mask = np.zeros(shape, dtype=bool)
        analysis._strain_origin_mask[:, radius] = True
        analysis.dynamic_include_mask = None
        analysis.dynamic_exclude_mask = None
        analysis.strain_start_frame = 0
        analysis.ref_path = None
        analysis.calibration = Calibration()
        analysis._compute_incremental_displacements()
        analysis._compute_velocities_and_rates()
        analysis._transport_accumulated_strain()

        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "compact.h5")
            analysis.export_hdf5(path)
            with h5py.File(path, "r") as handle:
                self.assertEqual(int(handle.attrs["result_schema"]), 5)
                frame = handle["frame_0000"]
                self.assertNotIn("Vx", frame)
                self.assertNotIn("mag_inc", frame)
                self.assertNotIn("corr", frame)
                self.assertNotIn("du_dx", frame)
                for name in ("u", "v", "Exx_rate", "Exy_rate", "Eyy_rate",
                             "Eeff_rate", "Exx_gl", "Exy_gl", "Eyy_gl",
                             "Eeff_gl"):
                    self.assertEqual(frame[name].ndim, 1)
                    self.assertFalse(np.isnan(frame[name][:]).any(), name)


if __name__ == "__main__":
    unittest.main()
