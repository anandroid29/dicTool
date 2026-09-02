import numpy as np

from strainx.ui.pages.results_page import _interpolate_between_subset_centres


def _field():
    arr = np.full((10, 10), np.nan)
    arr[2, 2] = 0.0
    arr[2, 6] = 4.0
    arr[6, 2] = 8.0
    arr[6, 6] = 12.0
    return arr


def test_hover_bilinearly_interpolates_four_surrounding_centres():
    value = _interpolate_between_subset_centres(
        _field(), x=4, y=4, spacing=4, origin=2)
    assert np.isclose(value, 6.0)


def test_hover_interpolates_between_two_centres_on_grid_line():
    value = _interpolate_between_subset_centres(
        _field(), x=4, y=2, spacing=4, origin=2)
    assert np.isclose(value, 2.0)


def test_hover_does_not_hide_failed_surrounding_subset():
    arr = _field()
    arr[6, 6] = np.nan
    assert _interpolate_between_subset_centres(
        arr, x=4, y=4, spacing=4, origin=2) is None


def test_hover_does_not_replace_missing_value_at_subset_centre():
    arr = _field()
    arr[2, 2] = np.nan
    assert _interpolate_between_subset_centres(
        arr, x=2, y=2, spacing=4, origin=2) is None
