import os
import numpy as np
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from src.core.units import Calibration
from src.ui.pages.results_page import ResultsPage
from src.ui.video_export import ViewRenderer


def test_three_digit_micrometre_velocity_promotes_to_millimetres():
    calibration = Calibration.from_pixel_size(1.0, "µm")

    factor, unit = calibration.compact_factor_and_unit(
        "Vx", magnitude=100.0, base_unit="px/s")

    assert unit == "mm/s"
    assert np.isclose(100.0 * factor, 0.1)


def test_two_digit_value_keeps_the_selected_unit():
    calibration = Calibration.from_pixel_size(1.0, "µm")

    factor, unit = calibration.compact_factor_and_unit(
        "Vx", magnitude=99.0, base_unit="px/s")

    assert unit == "µm/s"
    assert np.isclose(99.0 * factor, 99.0)


def test_large_value_can_promote_across_multiple_units():
    calibration = Calibration.from_pixel_size(1.0, "µm")

    factor, unit = calibration.compact_factor_and_unit(
        "u", magnitude=1_000_000.0, base_unit="px")

    assert unit == "m"
    assert np.isclose(1_000_000.0 * factor, 1.0)


def test_dimensionless_fields_are_never_rescaled():
    calibration = Calibration.from_pixel_size(1.0, "µm")

    factor, unit = calibration.compact_factor_and_unit(
        "Exx_gl", magnitude=1000.0, base_unit="")

    assert factor == 1.0
    assert unit == ""


def _analysis_with_micrometre_velocity():
    result = SimpleNamespace(Vx=np.array([[0.0, 100.0]]))
    analysis = SimpleNamespace(
        calibration=Calibration.from_pixel_size(1.0, "µm"),
        results=[result],
    )
    analysis.get_global_range = lambda _field, _coverage=100.0: (0.0, 100.0)
    return analysis


def test_results_viewer_uses_compact_sequence_unit():
    analysis = _analysis_with_micrometre_velocity()
    app = QApplication.instance() or QApplication([])
    page = ResultsPage(SimpleNamespace(analysis=analysis, new_session=lambda: None))
    page._field = "Vx"

    factor, unit = page._unit_factor()

    assert unit == "mm/s"
    assert np.isclose(100.0 * factor, 0.1)
    page.close()


def test_video_renderer_matches_results_compact_unit():
    analysis = _analysis_with_micrometre_velocity()
    renderer = ViewRenderer(analysis)

    values, unit = renderer.field_array(0, "Vx")

    assert unit == "mm/s"
    assert np.isclose(values[0, 1], 0.1)
