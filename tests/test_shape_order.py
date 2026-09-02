import numpy as np

from pydic.core.shape_order import shape_order_report


def _textured_image(height=96, width=112):
    y, x = np.mgrid[:height, :width]
    image = (0.45 + 0.18 * np.sin(x * 0.43) + 0.16 * np.cos(y * 0.37)
             + 0.12 * np.sin((x + y) * 0.71))
    return np.clip(image, 0.05, 0.95)


def test_shape_order_report_accepts_normalised_application_images():
    image = _textured_image()
    mask = np.zeros_like(image, dtype=bool)
    mask[12:-12, 12:-12] = True

    report = shape_order_report(image, mask, radius=8, n_samples=24,
                                verbose=False)

    assert np.isfinite(report["sigma_u_order1"])
    assert np.isfinite(report["sigma_u_order2"])
    assert np.isfinite(report["penalty_px"])
    assert report["noise_sigma"] > 0.0
    assert report["valid_samples_order1"] > 0
    assert report["valid_samples_order2"] > 0


def test_shape_order_report_is_invariant_to_8_bit_scaling():
    image = _textured_image()
    mask = np.zeros_like(image, dtype=bool)
    mask[12:-12, 12:-12] = True

    normalised = shape_order_report(image, mask, radius=8, n_samples=24,
                                    verbose=False)
    eight_bit = shape_order_report(image * 255.0, mask, radius=8, n_samples=24,
                                   verbose=False)

    for key in ("noise_sigma", "mean_gradient", "sigma_u_order1",
                "sigma_u_order2", "penalty_px"):
        assert np.isclose(normalised[key], eight_bit[key], rtol=1e-6, atol=1e-9)
