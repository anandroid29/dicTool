"""
A loaded session must be able to draw every field family, not just displacement.

Velocity is deliberately absent from the session file -- export_hdf5 stores only
"independent, user-facing data", and Vx/Vy/Veff are a deterministic view of u/v
and the frame interval. Nothing rebuilt them on load for schema-3 files, so
every Velocity field came back as None and its screen rendered blank while
displacement, which is stored, drew normally.
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pydic"))

import src.core.analysis as analysis_module
from src.core.analysis import DICAnalysis

FPS = 10.0
N = 3


def _write_session(path: Path, n_frames: int = N) -> None:
    """A minimal schema-3 file: displacement and rates stored, velocity not."""
    with h5py.File(path, "w") as handle:
        handle.attrs["result_schema"] = 3
        handle.attrs["reference_image"] = ""
        handle.attrs["subset_radius"] = 8
        handle.attrs["subset_spacing"] = 2
        handle.attrs["strain_window"] = 4
        handle.attrs["fps"] = FPS
        handle.create_dataset("roi_mask", data=np.ones((4, 4), dtype=bool))
        for index in range(n_frames):
            group = handle.create_group(f"frame_{index:04d}")
            group.attrs["image_path"] = ""
            group.attrs["elapsed_s"] = 0.1
            group.create_dataset(
                "u", data=np.full((4, 4), 3.0 * (index + 1), np.float32))
            group.create_dataset(
                "v", data=np.full((4, 4), -4.0 * (index + 1), np.float32))
            for name in ("Exx", "Exy", "Eyy", "Eeff",
                         "du_dx", "du_dy", "dv_dx", "dv_dy", "corr"):
                group.create_dataset(
                    name, data=np.full((4, 4), float(index), np.float32))
            group.create_dataset("valid", data=np.ones((4, 4), dtype=bool))


@pytest.mark.parametrize("lazy", [False, True])
def test_velocity_is_available_after_loading_a_session(tmp_path, monkeypatch, lazy):
    monkeypatch.setenv("PYDIC_SETTINGS_PATH", str(tmp_path / "settings.json"))
    # Threshold 0 forces lazy; a huge threshold forces eager in-memory loading.
    monkeypatch.setattr(analysis_module, "HDF5_LAZY_THRESHOLD_BYTES",
                        0 if lazy else 1 << 40)
    path = tmp_path / "session.h5"
    _write_session(path)

    analysis = DICAnalysis()
    try:
        analysis.load_hdf5(str(path))
        assert analysis.hdf5_lazy is lazy

        for index, res in enumerate(analysis.results):
            for name in ("Vx", "Vy", "Veff"):
                assert getattr(res, name) is not None, (
                    f"{name} missing on frame {index} -- its screen would be blank")

            # Velocity is displacement over the frame interval.
            vx = np.asarray(res.Vx)
            vy = np.asarray(res.Vy)
            veff = np.asarray(res.Veff)
            assert np.allclose(vx, 3.0 * (index + 1) * FPS)
            assert np.allclose(vy, -4.0 * (index + 1) * FPS)
            # 3-4-5 triangle, so the magnitude is exact.
            assert np.allclose(veff, 5.0 * (index + 1) * FPS)
    finally:
        analysis._release_loaded_hdf5()


def test_lazy_velocity_does_not_read_the_whole_field(tmp_path, monkeypatch):
    """A lazy session must stay lazy: slicing velocity reads only that slice."""
    monkeypatch.setenv("PYDIC_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(analysis_module, "HDF5_LAZY_THRESHOLD_BYTES", 0)
    path = tmp_path / "session.h5"
    _write_session(path)

    analysis = DICAnalysis()
    try:
        analysis.load_hdf5(str(path))
        res = analysis.results[0]
        assert isinstance(res.u, h5py.Dataset), "precondition: displacement is lazy"
        assert not isinstance(res.Vx, np.ndarray), (
            "velocity was materialised, defeating lazy loading")

        block = res.Vx[0:2, 0:2]
        assert block.shape == (2, 2)
        assert np.allclose(block, 3.0 * FPS)
        assert res.Vx.shape == res.u.shape
        assert np.asarray(res.Veff).shape == (4, 4)
    finally:
        analysis._release_loaded_hdf5()


def test_stored_rates_are_not_overwritten_on_load(tmp_path, monkeypatch):
    """Rebuilding velocity must not disturb the strain rates read from file."""
    monkeypatch.setenv("PYDIC_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(analysis_module, "HDF5_LAZY_THRESHOLD_BYTES", 1 << 40)
    path = tmp_path / "session.h5"
    _write_session(path)
    with h5py.File(path, "a") as handle:
        for index in range(N):
            handle[f"frame_{index:04d}"].create_dataset(
                "Eeff_rate", data=np.full((4, 4), 7.0 + index, np.float32))

    analysis = DICAnalysis()
    try:
        analysis.load_hdf5(str(path))
        for index, res in enumerate(analysis.results):
            assert np.allclose(np.asarray(res.Eeff_rate), 7.0 + index)
            assert res.Vx is not None
    finally:
        analysis._release_loaded_hdf5()


def test_strain_rates_are_rebuilt_when_absent_from_file(tmp_path, monkeypatch):
    """
    A file with no stored rates but intact gradients must still fill the
    strain-rate screens rather than leaving them blank.
    """
    monkeypatch.setenv("PYDIC_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(analysis_module, "HDF5_LAZY_THRESHOLD_BYTES", 1 << 40)
    path = tmp_path / "session.h5"
    _write_session(path)  # writes gradients, no *_rate datasets

    analysis = DICAnalysis()
    try:
        analysis.load_hdf5(str(path))
        for res in analysis.results:
            assert res.Eeff_rate is not None, "strain-rate screen would be blank"
            assert np.isfinite(np.asarray(res.Eeff_rate)).any()
    finally:
        analysis._release_loaded_hdf5()
