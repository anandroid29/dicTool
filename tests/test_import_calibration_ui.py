from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pydic"))

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QCheckBox

from src.core.units import Calibration
from src.core.analysis import DICAnalysis
from src.core.rg_dic import DICParams
from src.ui.pages.params_page import ParamsPage
from src.ui.pages.dynamic_roi_page import DynamicROIPage
from src.ui.pages.analysis_page import AnalysisPage
from src.ui.pages.results_page import FIELDS, FIELD_GROUPS
from src.ui.image_canvas import ROITool
from src.ui.pages.welcome_page import ImageLoadSettingsDialog
from src.ui.video_importer import VideoImporterDialog


class ImportCalibrationUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_video_importer_prefills_and_writes_pixel_mapping(self):
        calibration = Calibration.from_pixel_size(8.25, "µm")
        dialog = VideoImporterDialog(initial_calibration=calibration)
        self.assertAlmostEqual(dialog.get_calibration().pixel_size_in("µm"), 8.25)

        with tempfile.TemporaryDirectory() as td:
            frame = Path(td) / "frame_000000.png"
            frame.write_bytes(b"")
            dialog._capture_fps_spin.setValue(2000.0)
            dialog._step_spin.setValue(4)
            dialog._on_done([str(frame)])
            metadata = json.loads((Path(td) / "dic_metadata.json").read_text())

        restored = Calibration.from_dict(metadata["calibration"])
        self.assertAlmostEqual(metadata["fps"], 500.0)
        self.assertAlmostEqual(restored.pixel_size_in("µm"), 8.25)
        dialog.close()

    def test_image_importer_prefills_cached_pixel_mapping(self):
        calibration = Calibration.from_pixel_size(0.04, "mm")
        dialog = ImageLoadSettingsDialog(
            ["reference.png", "deformed.png"], ".",
            initial_calibration=calibration)
        restored = dialog.get_calibration()
        self.assertAlmostEqual(restored.pixel_size_in("mm"), 0.04)
        self.assertEqual(restored.display_unit, "mm")
        dialog.close()

    def test_results_exposes_one_strain_family_without_gdot(self):
        self.assertIn("Strain", FIELD_GROUPS)
        self.assertNotIn("Infinitesimal strain", FIELD_GROUPS)
        self.assertNotIn("Green-Lagrange strain", FIELD_GROUPS)
        self.assertNotIn("Gxy_rate", FIELDS)
        self.assertNotIn("Gxy_gl", FIELDS)
        self.assertEqual(
            FIELD_GROUPS["Strain"],
            ["Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"])

    def test_all_parameter_controls_update_model_and_reentry_restores_it(self):
        analysis = object.__new__(DICAnalysis)
        analysis.params = DICParams()
        analysis.results = [object()]
        analysis._ref_image = None
        analysis._roi_mask = None
        analysis.dynamic_include_mask = None
        analysis.dynamic_exclude_mask = None

        class Wizard:
            use_gpu = False
            seed_xy = None
            def __init__(self, model): self.analysis = model
            def go_before_params(self): pass
            def go_dynamic_roi(self): pass
            def go_analysis(self): pass

        wizard = Wizard(analysis)
        page = ParamsPage(wizard)
        page._sp_strain.setValue(31)
        page._sp_maxiter.setValue(65)
        page._sp_tol.setValue(0.002)
        page._sp_cutoff.setValue(0.45)
        page._sp_search.setValue(90)
        self.assertEqual(analysis.params.strain_window, 31)
        self.assertEqual(analysis.params.max_iter, 65)
        self.assertAlmostEqual(analysis.params.conv_tol, 0.002)
        self.assertAlmostEqual(analysis.params.corr_cutoff, 0.45)
        self.assertEqual(analysis.params.search_radius, 90)
        self.assertEqual(analysis.results, [])

        analysis.params.strain_window = 43
        analysis.params.max_iter = 80
        page.on_enter()
        self.assertEqual(page._sp_strain.value(), 43)
        self.assertEqual(page._sp_maxiter.value(), 80)
        page.close()

    def test_strain_window_warning_matches_actual_grid_support(self):
        analysis = object.__new__(DICAnalysis)
        analysis.params = DICParams(strain_window=3, subset_spacing=2)
        analysis.results = []
        analysis._ref_image = None
        analysis._roi_mask = None
        analysis.dynamic_include_mask = None
        analysis.dynamic_exclude_mask = None

        class Wizard:
            use_gpu = False
            seed_xy = None
            def __init__(self, model): self.analysis = model
            def go_before_params(self): pass
            def go_dynamic_roi(self): pass
            def go_analysis(self): pass

        page = ParamsPage(Wizard(analysis))
        page.on_enter()
        self.assertTrue(page._strain_warn.isHidden())

        page._sp_spacing.setValue(4)
        self.assertFalse(page._strain_warn.isHidden())
        self.assertIn("spans only 1 grid point", page._strain_warn.text())
        self.assertIn("4 px will be used instead", page._strain_warn.text())
        page.close()

    def test_strain_warning_updates_during_keyboard_entry(self):
        analysis = object.__new__(DICAnalysis)
        analysis.params = DICParams(strain_window=5, subset_spacing=4)
        analysis.results = [object()]
        analysis._ref_image = None
        analysis._roi_mask = None
        analysis.dynamic_include_mask = None
        analysis.dynamic_exclude_mask = None

        class Wizard:
            use_gpu = False
            seed_xy = None
            def __init__(self, model): self.analysis = model
            def go_before_params(self): pass
            def go_dynamic_roi(self): pass
            def go_analysis(self): pass

        page = ParamsPage(Wizard(analysis))
        page.on_enter()
        editor = page._sp_strain.lineEdit()
        editor.setFocus()
        QTest.keyClick(editor, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClicks(editor, "3")
        self.assertFalse(page._strain_warn.isHidden())
        self.assertIn("3 px spans only 1 grid point", page._strain_warn.text())

        QTest.keyClick(editor, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClicks(editor, "5")
        self.assertTrue(page._strain_warn.isHidden())
        page.close()

    def test_dynamic_roi_choices_are_persistent_and_default_to_include_rect(self):
        analysis = object.__new__(DICAnalysis)
        analysis.params = DICParams(dynamic_roi="Edge Detection")
        analysis.results = []
        analysis._ref_image = None
        analysis._roi_mask = None
        analysis.dynamic_include_mask = None
        analysis.dynamic_exclude_mask = None

        class Wizard:
            def __init__(self, model): self.analysis = model
            def go_roi(self): pass
            def go_params(self): pass

        page = DynamicROIPage(Wizard(analysis))
        self.assertIsInstance(page._fill_chk, QCheckBox)
        self.assertTrue(page._fill_chk.isChecked())
        self.assertTrue(page._btn_inc.isChecked())
        self.assertFalse(page._btn_exc.isChecked())
        self.assertTrue(page._tool_buttons[ROITool.RECTANGLE].isChecked())

        page._tool_buttons[ROITool.POLYGON].click()
        page._btn_exc.click()
        self.assertTrue(page._btn_exc.isChecked())
        self.assertFalse(page._btn_inc.isChecked())
        self.assertTrue(page._tool_buttons[ROITool.POLYGON].isChecked())
        self.assertEqual(page._canvas._tool, ROITool.POLYGON)
        page.close()

    def test_parameters_preview_uses_dynamic_reference_mask(self):
        analysis = object.__new__(DICAnalysis)
        analysis.params = DICParams(dynamic_roi="Edge Detection")
        analysis.results = []
        analysis._ref_image = np.zeros((12, 14), dtype=float)
        analysis._roi_mask = np.ones((12, 14), dtype=bool)
        analysis.dynamic_include_mask = None
        analysis.dynamic_exclude_mask = None
        preview = np.zeros((12, 14), dtype=bool)
        preview[3:9, 4:11] = True
        analysis.reference_analysis_mask = lambda: preview.copy()

        class Wizard:
            use_gpu = False
            seed_xy = None
            def __init__(self, model): self.analysis = model
            def go_before_params(self): pass
            def go_dynamic_roi(self): pass
            def go_analysis(self): pass

        page = ParamsPage(Wizard(analysis))
        page.on_enter()
        self.assertTrue(np.array_equal(page._canvas.roi_mask, preview))
        self.assertTrue(np.array_equal(page._preview_roi_mask, preview))
        page.close()

    def test_analysis_first_thumbnail_uses_reference_dynamic_mask(self):
        preview = np.zeros((8, 9), dtype=bool)
        preview[2:6, 3:7] = True
        later_u = np.zeros((8, 9), dtype=float)
        later_v = np.zeros((8, 9), dtype=float)
        later_u[4, 4] = np.inf
        later_valid = np.ones((8, 9), dtype=bool)
        later_valid[5, 5] = False
        analysis = SimpleNamespace(
            results=[None, SimpleNamespace(
                u=later_u, v=later_v, valid=later_valid)],
            reference_analysis_mask=lambda: preview.copy())

        class Wizard:
            def __init__(self, model): self.analysis = model

        page = AnalysisPage(Wizard(analysis))
        self.assertTrue(np.array_equal(page._thumbnail_source_mask(0), preview))
        later = page._thumbnail_source_mask(1)
        self.assertFalse(later[4, 4])
        self.assertFalse(later[5, 5])
        self.assertTrue(later[3, 3])
        page.close()


if __name__ == "__main__":
    unittest.main()
