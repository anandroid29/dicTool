from pathlib import Path

import h5py
import numpy as np

import src.core.analysis as analysis_module
from src.core.analysis import DICAnalysis


def _write_minimal_session(path: Path, n_frames: int = 3) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["result_schema"] = 3
        handle.attrs["reference_image"] = ""
        handle.attrs["subset_radius"] = 8
        handle.attrs["subset_spacing"] = 2
        handle.attrs["strain_window"] = 4
        handle.attrs["fps"] = 10.0
        handle.create_dataset("roi_mask", data=np.ones((4, 4), dtype=bool))
        for index in range(n_frames):
            group = handle.create_group(f"frame_{index:04d}")
            group.attrs["image_path"] = ""
            group.attrs["elapsed_s"] = 0.1
            for name in ("u", "v", "Exx", "Exy", "Eyy", "Eeff",
                         "du_dx", "du_dy", "dv_dx", "dv_dy", "corr"):
                group.create_dataset(
                    name, data=np.full((4, 4), float(index), np.float32))
            group.create_dataset("valid", data=np.ones((4, 4), dtype=bool))


def test_hdf5_loader_reports_frame_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("PYDIC_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(analysis_module, "HDF5_LAZY_THRESHOLD_BYTES", 0)
    path = tmp_path / "session.h5"
    _write_minimal_session(path)
    updates = []

    analysis = DICAnalysis()
    try:
        analysis.load_hdf5(
            str(path), progress_cb=lambda fraction, message: updates.append(
                (fraction, message)))

        assert len(analysis.results) == 3
        assert updates[0][0] == 0.0
        assert updates[-1] == (1.0, "Loaded 3 frames.")
        assert any("Loading frame 2 of 3" in message for _, message in updates)
        assert all(a[0] <= b[0] for a, b in zip(updates, updates[1:]))
        assert analysis.hdf5_lazy
        assert isinstance(analysis.results[0].u, h5py.Dataset)
        assert np.allclose(np.asarray(analysis.results[2].u), 2.0)
    finally:
        analysis._release_loaded_hdf5()


def test_frame_dynamic_roi_overrides_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("PYDIC_SETTINGS_PATH", str(tmp_path / "settings.json"))
    path = tmp_path / "frame_overrides.h5"
    include = np.zeros((4, 4), dtype=bool)
    exclude = np.zeros((4, 4), dtype=bool)
    include[1, 2] = True
    exclude[2, 1] = True

    analysis = DICAnalysis()
    analysis._roi_mask = np.ones((4, 4), dtype=bool)
    analysis.dynamic_frame_overrides = {
        7: {"threshold": 0.63, "replace": True,
            "include": include, "exclude": exclude},
    }
    analysis.dynamic_future_overrides = {
        8: {"threshold": 0.58, "replace": True, "include": include},
        12: {"reset": True},
    }
    analysis.export_hdf5(str(path))

    loaded = DICAnalysis()
    try:
        loaded.load_hdf5(str(path))
        entry = loaded.dynamic_frame_overrides[7]
        assert entry["threshold"] == 0.63
        assert entry["replace"] is True
        assert np.array_equal(entry["include"], include)
        assert np.array_equal(entry["exclude"], exclude)
        future = loaded.dynamic_future_overrides[8]
        assert future["threshold"] == 0.58
        assert future["replace"] is True
        assert np.array_equal(future["include"], include)
        assert loaded.dynamic_future_overrides[12] == {"reset": True}
    finally:
        loaded._release_loaded_hdf5()
