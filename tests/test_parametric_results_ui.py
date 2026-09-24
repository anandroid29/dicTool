from __future__ import annotations

import os
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication

from strainx.core.compact_field import CompactField, finite_values
from strainx.core.parametric import (
    ParametricCase, ParametricSweep, build_derived_cache,
)
from strainx.core.strain import (
    compute_velocity_strains, iter_velocity_strains_multi_window,
)
from strainx.core.units import Calibration
from strainx.ui import render
from strainx.ui.image_canvas import ImageCanvas
from strainx.ui.components import ResultColorBar
from strainx.ui.parametric_plot_3d import ParametricPlot3D
from strainx.ui.pages.parametric_results_page import ParametricResultsPage
from strainx.ui.pages.welcome_page import _HDF5LoadWorker
from strainx.ui.wizard import Wizard


class _FakeSweep:
    STRAIN_FIELDS = ParametricSweep.STRAIN_FIELDS
    DERIVED_FIELDS = ParametricSweep.DERIVED_FIELDS

    def __init__(self):
        self.cases = [
            SimpleNamespace(
                label=f"r={5 + i} px  ·  grid=2 px",
                path=Path(f"case_{i}.h5"), subset_radius=5 + i,
                grid_spacing=2,
                mean_valid_fraction=0.9 - i * 0.1, median_corr=0.02 + i * 0.01)
            for i in range(2)
        ]
        self.strain_windows = (1, 2, 3)
        self.frame_count = 3
        self.calls = []

    def frame_data(self, case, frame, field, window, *, temporal_span=1):
        self.calls.append((case, frame, field, window))
        shape = (24, 32)
        indices = np.array([8 * 32 + 8, 8 * 32 + 12,
                            12 * 32 + 8, 12 * 32 + 12], np.uint32)
        values = np.arange(1, 5, dtype=np.float32) * (case + 1)
        return CompactField(shape, indices, values), ""


class ParametricResultsUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_uses_readable_field_families_and_interactive_case_canvases(self):
        wizard = SimpleNamespace(new_session=lambda: None)
        page = ParametricResultsPage(wizard)
        sweep = _FakeSweep()
        page.set_sweep(sweep)

        self.assertEqual(page._cat_combo.currentText(), "Displacement")
        self.assertEqual(page.current_field, "u")
        self.assertTrue(all(call[2] == "u" for call in sweep.calls),
                        "Opening the viewer must not derive strain rates")
        self.assertEqual(
            [page._cat_combo.itemText(i) for i in range(page._cat_combo.count())],
            ["Displacement", "Velocity", "Strain rate", "Strain", "Quality"])
        self.assertTrue(all("..." not in button.text()
                            for button in page._field_btns.values()))
        self.assertIn("Equivalent strain rate",
                      page._field_btns["Eeff_rate"].toolTip())
        self.assertEqual(len(page._panels), 2)
        self.assertEqual(len(page.findChildren(ResultColorBar)), 1)
        self.assertTrue(all(isinstance(panel.canvas, ImageCanvas)
                            for panel in page._panels.values()))
        first_panel = next(iter(page._panels.values()))
        first_panel._show_probe(8, 8, float("nan"))
        self.assertIn("x=8", first_panel.probe.text())
        self.assertIsNone(first_panel.canvas.probe_annotation,
                          "The probe must not repeat over the image")
        first_panel._show_probe(99_999, 99_999, float("nan"))
        self.assertEqual(first_panel.probe.text(), "Outside image")
        self.assertIsNone(first_panel.canvas._widget_to_image(
            QPointF(-10_000, -10_000)))

        calls = len(sweep.calls)
        page._cmap.setCurrentText("viridis")
        self.assertEqual(len(sweep.calls), calls,
                         "Changing display options must reuse loaded field data")

        page._cat_combo.setCurrentText("Velocity")
        self.assertEqual(page.current_field, "Vx")
        self.assertFalse(page._fixed_rows["window"].isHidden())
        self.assertTrue(page._fixed_rows["radius"].isHidden())
        self.assertFalse(hasattr(page, "_window"))
        self.assertFalse(hasattr(page, "_span"))
        page.close()

    def test_parametric_values_follow_editable_physical_calibration(self):
        page = ParametricResultsPage(SimpleNamespace(new_session=lambda: None))
        sweep = _FakeSweep()
        sweep.calibration = Calibration.from_pixel_size(0.04, "mm")
        page.set_sweep(sweep)
        panel = next(iter(page._panels.values()))
        self.assertEqual(panel.unit, "mm")
        self.assertEqual(page._colorbar_unit.text(), "mm")
        self.assertAlmostEqual(float(finite_values(panel.values)[0]), 0.04)
        self.assertEqual(page._pixel_size.value(), 0.04)

        page._length_unit.setCurrentText("µm")
        self.assertEqual(panel.unit, "µm")
        self.assertEqual(page._colorbar_unit.text(), "µm")
        self.assertAlmostEqual(float(finite_values(panel.values)[0]), 40.0)
        self.assertAlmostEqual(page._pixel_size.value(), 40.0)
        self.assertAlmostEqual(float(finite_values(page._loaded[0][1])[0]), 1.0,
                               msg="Cached solver values must remain in pixels")

        page._pixel_size.setValue(8.0)
        self.assertAlmostEqual(float(finite_values(panel.values)[0]), 8.0)
        page._summary_mode.setCurrentText("ROI mean")
        self.assertAlmostEqual(
            float(page._plot_axes.lines[0].get_ydata()[0]), 20.0)
        page._axis_y.setCurrentText("Grid spacing")
        self.assertAlmostEqual(page._plot_3d._points[0][2], 20.0)
        page._cat_combo.setCurrentText("Velocity")
        self.assertEqual(next(iter(page._panels.values())).unit, "µm/s")
        page._cat_combo.setCurrentText("Strain rate")
        self.assertEqual(next(iter(page._panels.values())).unit, "s⁻¹")
        self.assertEqual(page._shared_colorbar._vmin, 0.0)
        self.assertAlmostEqual(float(finite_values(
            next(iter(page._panels.values())).values)[0]), 1.0)
        page.close()

    def test_sweep_recovers_and_saves_display_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "_sweep_source.h5"
            path = root / "correlation_r02_g01.h5"
            with h5py.File(source, "w") as handle:
                handle.attrs["metres_per_pixel"] = 4e-5
                handle.attrs["display_unit"] = "mm"
            with h5py.File(path, "w") as handle:
                handle.attrs.update({
                    "sweep_cache_schema": 1, "sweep_complete": True,
                    "subset_radius": 2, "subset_spacing": 1,
                    "frame_count": 1, "strain_windows_requested": [1],
                    "source_hdf5": str(source),
                })
            sweep = ParametricSweep.from_hdf5(path)
            self.assertAlmostEqual(sweep.calibration.pixel_size_in("mm"), 0.04)
            sweep.calibration = Calibration.from_pixel_size(8, "µm")
            sweep.save_display_calibration()
            reopened = ParametricSweep.from_hdf5(path)
            self.assertAlmostEqual(reopened.calibration.pixel_size_in("µm"), 8)
            self.assertEqual(reopened.calibration.display_unit, "µm")

    def test_comparison_controls_collapse_and_fit_all_panels(self):
        page = ParametricResultsPage(SimpleNamespace(new_session=lambda: None))
        page.resize(1200, 800)
        page.show()
        page.set_sweep(_FakeSweep())
        self.app.processEvents()
        self.assertEqual(
            [page._fixed_labels[key].text() for key in
             ("radius", "grid", "window", "temporal")],
            ["Subset radius", "Grid spacing", "Strain window", "Temporal span"])
        for key, row in page._fixed_rows.items():
            self.assertIs(row.layout().itemAt(0).widget(), page._fixed_labels[key])
            self.assertIs(row.layout().itemAt(1).widget(), page._fixed[key])
        self.assertTrue(page._fixed_rows["radius"].isHidden())
        self.assertFalse(page._fixed_rows["grid"].isHidden())
        self.assertLess(page._fixed_rows["grid"].height(), 65)
        page._axis_y.setCurrentText("Grid spacing")
        self.assertTrue(page._fixed_rows["grid"].isHidden())
        page._axis_x.setCurrentText("Strain window")
        self.assertTrue(page._fixed_rows["window"].isHidden())
        self.assertFalse(page._fixed_rows["radius"].isHidden())

        page._sidebar_toggle.setChecked(True)
        page._plot_toggle.setChecked(True)
        self.assertTrue(page._sidebar.isHidden())
        self.assertTrue(page._plot_stack.isHidden())
        page._sidebar_toggle.setChecked(False)
        page._plot_toggle.setChecked(False)
        self.app.processEvents()

        page._linked_view_changed(0.7, 0.4, 2.0)
        page._columns.setValue(1)
        self.app.processEvents()
        for key in page._visible_keys():
            panel = page._panels.get(key)
            if panel is not None:
                x, y, zoom = panel.canvas.view_state()
                self.assertAlmostEqual(x, 0.5, delta=0.02)
                self.assertAlmostEqual(y, 0.5, delta=0.02)
                self.assertAlmostEqual(zoom, 1.0, delta=0.02)
        page.close()

    def test_axis_ranges_select_inclusive_computed_values(self):
        page = ParametricResultsPage(SimpleNamespace(new_session=lambda: None))
        page.set_sweep(_FakeSweep())
        self.assertEqual((page._axis_x_min.value(), page._axis_x_max.value()),
                         (5, 6))
        self.assertEqual(page._axis_x_min.suffix(), " px")
        self.assertEqual(len(page._selected()), 2)

        page._axis_x_min.setValue(6)
        self.assertEqual(len(page._selected()), 1)
        page._axis_x_max.setValue(5)
        self.assertEqual((page._axis_x_min.value(), page._axis_x_max.value()),
                         (5, 5))
        self.assertEqual(len(page._selected()), 1)

        page._axis_y.setCurrentText("Strain window")
        self.assertFalse(page._axis_y_range.isHidden())
        self.assertEqual((page._axis_y_min.value(), page._axis_y_max.value()),
                         (1, 3))
        self.assertEqual(len(page._selected()), 3)
        page._axis_y_min.setValue(2)
        self.assertEqual(len(page._selected()), 2)

        page._axis_x.setCurrentText("Temporal span")
        self.assertEqual(page._axis_x_min.suffix(), " frames")
        self.assertEqual((page._axis_x_min.value(), page._axis_x_max.value()),
                         (1, 1))
        page._axis_y.setCurrentIndex(0)
        self.assertTrue(page._axis_y_range.isHidden())
        page.close()

    def test_velocity_is_derived_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs["fps"] = 4.0
                frames = handle.create_group("correlations")
                frame = frames.create_group("frame_000000")
                frame.attrs["field_shape"] = (3, 3)
                frame.attrs["image_path"] = ""
                frame.create_dataset("valid_indices", data=np.array([4], np.uint32))
                frame.create_dataset("u", data=np.array([3.0], np.float32))
                frame.create_dataset("v", data=np.array([4.0], np.float32))
                frame.create_dataset("corr", data=np.array([0.1], np.float32))
            case = ParametricCase(path, 5, 2, 1, 1.0, 0.1)
            sweep = ParametricSweep(
                [case], manifest_path=None, reference_image="",
                strain_windows=(1,))

            velocity, _ = sweep.frame_data(0, 0, "Veff")

        self.assertAlmostEqual(float(velocity.values[0]), 20.0)

    def test_temporal_view_loads_without_blocking_the_ui(self):
        wizard = SimpleNamespace(new_session=lambda: None)
        page = ParametricResultsPage(wizard)
        sweep = _FakeSweep()
        sweep.temporal_spans = (1, 2)
        page.set_sweep(sweep)
        page._axis_x.setCurrentIndex(3)  # Temporal span

        self.assertIsNotNone(page._compute_thread)
        deadline = time.monotonic() + 3.0
        while page._compute_thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertIsNone(page._compute_thread)
        self.assertEqual(len(page._loaded), 2)
        self.assertEqual(page._compute_percent, 100)
        page.close()

    def test_3d_plot_rotation_survives_playback(self):
        page = ParametricResultsPage(SimpleNamespace(new_session=lambda: None))
        page.set_sweep(_FakeSweep())
        page._summary_mode.setCurrentText("ROI mean")
        page._axis_y.setCurrentText("Grid spacing")
        plot = page._plot_3d
        self.assertIs(page._plot_stack.currentWidget(), plot)
        self.assertEqual(len(plot._points), 2)
        self.assertEqual(plot._x_label, "Subset radius r")
        self.assertEqual(plot._y_label, "Grid spacing")
        self.assertIn("Horizontal displacement u", plot._z_label)
        self.assertNotIn("Value", plot._z_label)
        event = lambda x, y: SimpleNamespace(
            position=lambda: QPointF(x, y), accept=lambda: None,
            button=lambda: Qt.MouseButton.LeftButton)
        initial_yaw = plot._yaw
        plot.mousePressEvent(event(20, 20))
        plot.mouseMoveEvent(event(65, 35))
        plot.mouseReleaseEvent(event(65, 35))
        self.assertNotEqual(plot._yaw, initial_yaw)
        rotated_yaw = plot._yaw
        page._playback_rate.setCurrentText("20 fps")
        page._play.setChecked(True)
        deadline = time.monotonic() + 3.0
        while page._play.isChecked() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertFalse(page._play.isChecked())
        self.assertEqual(page._frame.value(), 2)
        self.assertIs(page._plot_stack.currentWidget(), plot)
        self.assertAlmostEqual(plot._yaw, rotated_yaw)
        page._cat_combo.setCurrentText("Strain rate")
        self.assertIn("Equivalent strain rate", plot._z_label)
        page._axis_y.setCurrentIndex(0)
        self.assertIn("Equivalent strain rate", page._plot_axes.get_ylabel())
        page.close()

    def test_3d_surface_and_empty_state_paint_without_matplotlib(self):
        plot = ParametricPlot3D()
        plot.resize(640, 250)
        plot.show()
        plot.set_data([(1, 1, 0.2), (2, 1, 0.4),
                       (1, 2, 0.5), (2, 2, 0.8)],
                      "Value by parameters", "Radius", "Grid", "Mean")
        self.assertEqual(len(plot._points), 4)
        self.assertFalse(plot.grab().isNull())
        plot.set_data([], "Empty", "Radius", "Grid", "Mean")
        self.assertFalse(plot.grab().isNull())
        plot.close()

    def test_graph_uses_shared_colorbar_colours_and_limits(self):
        self.assertEqual(render.sample_colorbar(
            5, 0, 10, [(0, 0, 0), (200, 100, 0)]), (100, 50, 0))
        page = ParametricResultsPage(SimpleNamespace(new_session=lambda: None))
        page.set_sweep(_FakeSweep())
        page._summary_mode.setCurrentText("ROI mean")
        page._axis_y.setCurrentText("Grid spacing")
        page._cmap.setCurrentText("viridis")
        bar = page._shared_colorbar
        plot = page._plot_3d
        self.assertEqual(plot._colors, tuple(bar._colors))
        self.assertEqual(plot._color_limits, (bar._vmin, bar._vmax))
        value = plot._points[0][2]
        self.assertEqual(plot._colour(value).getRgb()[:3],
                         render.sample_colorbar(
                             value, bar._vmin, bar._vmax, bar._colors,
                             page._flag_clipped.isChecked()))
        self.assertEqual(plot._colour(bar._vmin - 1).getRgb()[:3],
                         render.UNDER_RANGE_RGB)

        page._axis_y.setCurrentIndex(0)
        curve = page._plot_axes.lines[0]
        marker = page._plot_axes.collections[0]
        first_value = float(curve.get_ydata()[0])
        expected = render.sample_colorbar(
            first_value, bar._vmin, bar._vmax, bar._colors,
            page._flag_clipped.isChecked())
        self.assertEqual(tuple(round(channel * 255)
                               for channel in marker.get_facecolors()[0][:3]),
                         expected)
        page.close()

    def test_large_3d_plot_loads_next_frame_off_the_ui_thread(self):
        class SlowSweep(_FakeSweep):
            def frame_data(self, *args, **kwargs):
                time.sleep(0.02)
                return super().frame_data(*args, **kwargs)

        sweep = SlowSweep()
        sweep.cases = [SimpleNamespace(
            label=f"r={i + 1} px  ·  grid=2 px", path=Path(f"case_{i}.h5"),
            subset_radius=i + 1, grid_spacing=2,
            mean_valid_fraction=0.9, median_corr=0.02) for i in range(8)]
        page = ParametricResultsPage(SimpleNamespace(new_session=lambda: None))
        page.set_sweep(sweep)
        self.assertIsNotNone(page._compute_thread, "The 2D plot should load off-thread too")
        page._summary_mode.setCurrentText("ROI mean")
        page._axis_y.setCurrentText("Grid spacing")
        deadline = time.monotonic() + 3.0
        while page._compute_thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertIsNone(page._compute_thread)

        started = time.monotonic()
        page._frame.setValue(1)
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertIsNotNone(page._compute_thread)
        deadline = time.monotonic() + 3.0
        while page._compute_thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertIsNone(page._compute_thread)
        self.assertEqual(page._loaded_key[1], 1)
        page.close()

    def test_start_menu_opens_each_parametric_hdf5_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "_sweep_source.h5"
            correlation = root / "correlation_r02_g01.h5"
            derived = root / "derived_r02_g01.h5"
            temporal = root / "correlation_r02_g01.temporal.h5"
            with h5py.File(source, "w"):
                pass
            with h5py.File(correlation, "w") as handle:
                handle.attrs.update({
                    "sweep_cache_schema": 1, "sweep_complete": True,
                    "subset_radius": 2, "subset_spacing": 1,
                    "frame_count": 1, "strain_windows_requested": [1],
                })
                frame = handle.create_group("correlations/frame_000000")
                frame.attrs["field_shape"] = (2, 2)
                frame.attrs["image_path"] = ""
                frame.create_dataset("valid_indices", data=np.array([0], np.uint32))
                frame.create_dataset("u", data=np.array([1], np.float32))
                frame.create_dataset("v", data=np.array([0], np.float32))
                frame.create_dataset("corr", data=np.array([0.1], np.float32))
            with h5py.File(derived, "w") as handle:
                handle.attrs["derived_cache_schema"] = 2
                handle.attrs["correlation_file"] = correlation.name
            with h5py.File(temporal, "w") as handle:
                handle.attrs["temporal_cache_schema"] = 1
            (root / "manifest.json").write_text(json.dumps({
                "correlations": [{"file": correlation.name,
                                  "status": "complete"}],
                "source": {"reference_image": ""},
                "strain_windows_requested_px": [1],
            }), encoding="utf-8")

            for path in (source, correlation, derived, temporal):
                loaded, failed = [], []
                worker = _HDF5LoadWorker(str(path))
                worker.loaded.connect(lambda sweep, _path: loaded.append(sweep))
                worker.failed.connect(failed.append)
                worker.run()
                self.assertFalse(failed, f"{path.name}: {failed}")
                self.assertEqual(len(loaded[0].cases), 1, path.name)
                wizard = Wizard()
                wizard._welcome._on_hdf5_loaded(loaded[0], str(path))
                self.assertEqual(wizard._stack.currentIndex(), 7, path.name)
                self.assertEqual(len(wizard._parametric_results._loaded), 1)
                wizard.close()

    def test_temporal_fields_are_reused_from_persistent_hdf5_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correlation_r02_g01.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs["frame_count"] = 2
            case = ParametricCase(path, 2, 1, 2, 1.0, 0.01)
            sweep = ParametricSweep(
                [case], manifest_path=None, reference_image="",
                strain_windows=(1,), temporal_spans=(2,))
            expected = CompactField(
                (3, 3), np.array([4, 5], np.uint32),
                np.array([1.25, 2.5], np.float32))
            sweep._save_temporal_disk(case, 1, 1, 2, {"Veff": expected})
            loaded = sweep._load_temporal_disk(case, 1, "Veff", 1, 2)

            self.assertTrue(np.array_equal(loaded.indices, expected.indices))
            self.assertTrue(np.array_equal(loaded.values, expected.values))
            self.assertTrue(sweep._temporal_cache_path(case).is_file())

    def test_precomputed_sidecar_serves_velocity_rate_and_strain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correlation_r02_g01.h5"
            shape = (12, 12)
            yy, xx = np.meshgrid(np.arange(2, 10), np.arange(2, 10),
                                 indexing="ij")
            indices = (yy * shape[1] + xx).reshape(-1).astype(np.uint32)
            with h5py.File(path, "w") as handle:
                handle.attrs.update({
                    "sweep_cache_schema": 1,
                    "sweep_complete": True,
                    "subset_radius": 2,
                    "subset_spacing": 1,
                    "frame_count": 2,
                    "fps": 4.0,
                    "source_hdf5": "",
                    "mean_valid_fraction": 1.0,
                    "median_frame_corr": 0.01,
                    "strain_windows_requested": np.array([1, 3], np.int16),
                    "strain_start_frame": 0,
                })
                handle.create_dataset("roi_mask", data=np.ones(shape, bool))
                origin = np.zeros(shape, bool)
                origin[:, 2] = True
                handle.create_dataset("strain_origin_mask", data=origin)
                frames = handle.create_group("correlations")
                for frame_index in range(2):
                    frame = frames.create_group(f"frame_{frame_index:06d}")
                    frame.attrs["field_shape"] = shape
                    frame.attrs["image_path"] = ""
                    frame.create_dataset("valid_indices", data=indices)
                    frame.create_dataset(
                        "u", data=np.ones(indices.size, np.float32))
                    frame.create_dataset(
                        "v", data=np.zeros(indices.size, np.float32))
                    frame.create_dataset(
                        "corr", data=np.full(indices.size, 0.01, np.float32))

            build_derived_cache(path, (1, 3), use_gpu=False)
            sweep = ParametricSweep.from_hdf5(path)
            velocity, _ = sweep.frame_data(0, 0, "Veff", 1)
            rate, _ = sweep.frame_data(0, 0, "Eeff_rate", 1)
            strain, _ = sweep.frame_data(0, 1, "Eeff_gl", 3)
            sweep.precompute_temporal((1,), (2,))
            temporal_path = sweep._temporal_cache_path(sweep.cases[0])
            temporal_cached = temporal_path.is_file()
            sweep._temporal_analysis = lambda _index: self.fail(
                "Resume should reuse the saved temporal result")
            sweep.precompute_temporal((1,), (2,))
            reloaded = ParametricSweep.from_hdf5(path)
            temporal_rate, _ = reloaded.frame_data(
                0, 1, "Eeff_rate", 1, temporal_span=2)
            with h5py.File(sweep.cases[0].derived_path, "r") as derived:
                has_velocity_dataset = "velocity" in derived

        self.assertEqual(sweep.strain_windows, (1, 3))
        self.assertTrue(np.allclose(velocity.values, 4.0))
        self.assertTrue(finite_values(rate).size)
        self.assertTrue(np.allclose(finite_values(rate), 0.0, atol=1e-6))
        self.assertTrue(finite_values(strain).size)
        self.assertTrue(np.allclose(finite_values(strain), 0.0, atol=1e-6))
        self.assertFalse(has_velocity_dataset)
        self.assertTrue(temporal_cached)
        self.assertTrue(finite_values(temporal_rate).size)

    def test_multi_window_fit_matches_single_window_equations(self):
        shape = (31, 35)
        yy, xx = np.indices(shape)
        valid = np.zeros(shape, bool)
        valid[3:27:2, 4:30:2] = True
        valid[13:17, 16:20] = False
        vx = np.where(valid, 0.04 * xx - 0.02 * yy + 0.001 * xx * yy,
                      np.nan)
        vy = np.where(valid, -0.03 * xx + 0.05 * yy - 0.0007 * yy ** 2,
                      np.nan)
        # Windows 2 and 3 contain the same spacing-2 lattice neighbours.  Both
        # still have to match the public single-window implementation exactly.
        windows = (2, 3, 4, 6)
        batched = dict(iter_velocity_strains_multi_window(
            vx, vy, valid, windows, grid_spacing=2))

        for window in windows:
            direct = compute_velocity_strains(
                vx, vy, valid, window, grid_spacing=2, use_gpu=False)
            for name in ("dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                         "Eeff_rate"):
                expected, actual = direct[name], batched[window][name]
                self.assertTrue(np.array_equal(
                    np.isfinite(expected), np.isfinite(actual)),
                    f"finite support differs for {name}, window {window}")
                finite = np.isfinite(expected)
                self.assertTrue(np.allclose(
                    expected[finite], actual[finite], rtol=1e-9, atol=1e-10),
                    f"values differ for {name}, window {window}")


if __name__ == "__main__":
    unittest.main()
