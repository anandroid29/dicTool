import numpy as np

from strainx.ui import render


def test_export_annotations_do_not_use_ascii_only_opencv_text(monkeypatch):
    def reject_opencv_text(*_args, **_kwargs):
        raise AssertionError("Unicode export text must not use cv2.putText")

    monkeypatch.setattr(render.cv2, "putText", reject_opencv_text)
    canvas = np.zeros((120, 520, 3), dtype=np.uint8)

    labelled = render.draw_label(
        canvas.copy(), "Tensor Shear Strain Rate  Ėxy")
    finished = render.draw_colorbar(
        labelled, "turbo", -5.0, 10.0, "s⁻¹")

    assert np.any(finished != 0)
    assert not np.array_equal(finished, canvas)


def test_unicode_velocity_unit_has_nonzero_text_extent():
    width, height = render._unicode_text_size("125 µm/s", 13)

    assert width > 0
    assert height > 0
