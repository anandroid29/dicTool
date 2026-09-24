"""Lazy access to resumable parametric DIC correlation caches."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Optional

import h5py
import numpy as np

from .compact_field import CompactField
from .units import Calibration


DERIVED_CACHE_SCHEMA = 2
RATE_FIELDS = ("Eeff_rate",)
STRAIN_FIELDS = ("Eeff_gl",)


def derived_cache_path(correlation_path: str | Path) -> Path:
    """Return the sidecar used for precomputed velocity and strain fields."""
    path = Path(correlation_path)
    name = (path.name.replace("correlation_", "derived_", 1)
            if path.name.startswith("correlation_")
            else f"{path.stem}_derived{path.suffix}")
    return path.with_name(name)


def _source_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def derived_cache_complete(path: str | Path, correlation_path: str | Path,
                           strain_windows: tuple[int, ...]) -> bool:
    """Check that a derived sidecar exactly matches its correlation cache."""
    path, source = Path(path), Path(correlation_path)
    if not path.is_file() or not source.is_file():
        return False
    try:
        size, mtime = _source_signature(source)
        with h5py.File(path, "r") as handle:
            cached = tuple(int(v) for v in handle.attrs.get(
                "strain_windows_requested", ()))
            return bool(
                handle.attrs.get("derived_complete", False)
                and int(handle.attrs.get("derived_cache_schema", -1))
                == DERIVED_CACHE_SCHEMA
                and int(handle.attrs.get("correlation_size", -1)) == size
                and int(handle.attrs.get("correlation_mtime_ns", -1)) == mtime
                and cached == tuple(int(v) for v in strain_windows))
    except (OSError, KeyError, ValueError):
        return False


def _write_aligned(group: h5py.Group, name: str, field: np.ndarray,
                   indices: np.ndarray) -> None:
    """Store values against the correlation frame's existing compact indices."""
    values = np.asarray(field).reshape(-1)[indices].astype(
        np.float32, copy=False)
    group.create_dataset(name, data=values, compression="gzip",
                         compression_opts=4)


class _AccumulatedWindow:
    """Stream one strain window through the normal material-path semantics."""

    def __init__(self, shape, origin, roi, radius, spacing, start_frame):
        from .strain_accum import StrainPathTracker
        self.tracker = StrainPathTracker(
            shape, origin, roi, radius, spacing)
        self.roi = np.asarray(roi, dtype=bool)
        self.start = int(start_frame)
        self.names = (
            "Exx_inf", "Eyy_inf", "Exy_inf", "Eeff_inf",
            "Exx_gl", "Eyy_gl", "Exy_gl")
        self.effective = np.full(shape, np.nan, np.float32)
        self.encountered = np.zeros(shape, dtype=bool)

    def _deposit(self) -> None:
        snapshot = self.tracker.snapshot()
        new = self.roi & ~self.encountered
        for name in self.names:
            new &= np.isfinite(snapshot[name])
        if not new.any():
            return
        self.effective[new] = snapshot["Eeff_inf"][new]
        self.encountered[new] = True

    def advance(self, frame: int, valid: np.ndarray, u: np.ndarray,
                v: np.ndarray, gradients: dict[str, np.ndarray]
                ) -> dict[str, np.ndarray]:
        if frame < self.start - 1:
            return {"Eeff_gl": np.full(u.shape, np.nan, np.float32)}
        if self.start > 0 and frame == self.start - 1:
            self.tracker.seed()
            self._deposit()
        else:
            self.tracker.seed(valid)
            self._deposit()
            self.tracker.advance(
                u, v, gradients["dVx_dx"], gradients["dVx_dy"],
                gradients["dVy_dx"], gradients["dVy_dy"])
            self._deposit()
        return {"Eeff_gl": self.effective}


def _strain_start_frame(correlation: h5py.File) -> int:
    if "strain_start_frame" in correlation.attrs:
        return int(correlation.attrs["strain_start_frame"])
    source = Path(str(correlation.attrs.get("source_hdf5", "")))
    if source.is_file():
        try:
            with h5py.File(source, "r") as handle:
                return int(handle.attrs.get("strain_start_frame", 0))
        except OSError:
            pass
    return 0


def build_derived_cache(
    correlation_path: str | Path,
    strain_windows: tuple[int, ...],
    *,
    use_gpu: bool = False,
    progress=None,
    should_stop=None,
) -> Path:
    """Precompute effective rate and strain without duplicating displacement."""
    source = Path(correlation_path)
    target = derived_cache_path(source)
    windows = tuple(dict.fromkeys(int(value) for value in strain_windows))
    if not windows:
        raise ValueError("Select at least one strain window to cache.")
    if derived_cache_complete(target, source, windows):
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    started = time.perf_counter()
    try:
        with h5py.File(source, "r") as correlation, h5py.File(
                temporary, "w") as output:
            radius = int(correlation.attrs["subset_radius"])
            spacing = int(correlation.attrs["subset_spacing"])
            fps = float(correlation.attrs.get("fps", 1.0))
            frame_count = int(correlation.attrs["frame_count"])
            roi = correlation["roi_mask"][:].astype(bool)
            origin = (correlation["strain_origin_mask"][:].astype(bool)
                      if "strain_origin_mask" in correlation else roi)
            start_frame = _strain_start_frame(correlation)
            effective_windows = tuple(sorted({max(value, spacing)
                                              for value in windows}))
            canonical_by_bin = {}
            window_aliases = {}
            for window in effective_windows:
                canonical = canonical_by_bin.setdefault(
                    window // spacing, window)
                window_aliases[window] = canonical
            canonical_windows = tuple(canonical_by_bin.values())
            size, mtime = _source_signature(source)
            output.attrs.update({
                "derived_cache_schema": DERIVED_CACHE_SCHEMA,
                "derived_complete": False,
                "correlation_file": source.name,
                "correlation_size": size,
                "correlation_mtime_ns": mtime,
                "subset_radius": radius,
                "subset_spacing": spacing,
                "frame_count": frame_count,
                "fps": fps,
                "strain_start_frame": start_frame,
                "strain_windows_requested": np.asarray(windows, np.int16),
                "strain_windows_effective": np.asarray(
                    effective_windows, np.int16),
                "strain_windows_computed": np.asarray(
                    canonical_windows, np.int16),
                "velocity_semantics": "derived_from_compact_u_v",
                "rate_fields": ",".join(RATE_FIELDS),
                "strain_fields": ",".join(STRAIN_FIELDS),
            })
            window_groups = {
                window: output.create_group(f"windows/w{window:04d}/frames")
                for window in canonical_windows}
            accumulators = {
                window: _AccumulatedWindow(
                    roi.shape, origin, roi, radius, spacing, start_frame)
                for window in canonical_windows}

            from .strain import iter_velocity_strains_multi_window
            frames = correlation["correlations"]
            for frame in range(frame_count):
                if should_stop is not None and should_stop():
                    raise KeyboardInterrupt
                raw = frames[f"frame_{frame:06d}"]
                shape = tuple(int(v) for v in raw.attrs["field_shape"])
                indices = raw["valid_indices"][:].astype(
                    np.uint32, copy=False)
                u_values = raw["u"][:].astype(np.float32, copy=False)
                v_values = raw["v"][:].astype(np.float32, copy=False)
                u = np.full(shape, np.nan, np.float32)
                v = np.full(shape, np.nan, np.float32)
                u.reshape(-1)[indices], v.reshape(-1)[indices] = u_values, v_values
                valid = np.zeros(shape, dtype=bool)
                valid.reshape(-1)[indices] = True
                for window, gradients in iter_velocity_strains_multi_window(
                        u, v, valid, canonical_windows, spacing,
                        use_gpu=use_gpu and spacing == 1):
                    strains = accumulators[window].advance(
                        frame, valid, u, v, gradients)
                    derived = window_groups[window].create_group(
                        f"frame_{frame:06d}")
                    _write_aligned(
                        derived, "Eeff_rate",
                        gradients["Eeff_rate"] * fps, indices)
                    _write_aligned(
                        derived, "Eeff_gl", strains["Eeff_gl"], indices)
                if progress is not None:
                    progress((frame + 1) / max(1, frame_count),
                             f"Caching effective strain {frame + 1}/{frame_count}")
            windows_group = output["windows"]
            for window, canonical in window_aliases.items():
                if window != canonical:
                    windows_group[f"w{window:04d}"] = windows_group[
                        f"w{canonical:04d}"]
            output.attrs["elapsed_s"] = time.perf_counter() - started
            output.attrs["derived_complete"] = True
            output.flush()
        os.replace(temporary, target)
        return target
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def write_sweep_source(analysis, path: str | Path) -> Path:
    """Write a small restart descriptor consumed by run_parameter_sweep.py.

    It contains paths and masks, not result fields, so creating it is quick and
    does not duplicate any correlation computation.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not analysis.ref_path or not analysis.def_paths:
        raise ValueError("A reference and at least one deformed frame are required.")
    if analysis.roi_mask is None:
        raise ValueError("Draw an ROI before starting a parameter sweep.")
    p = analysis.params
    with h5py.File(target, "w") as handle:
        handle.attrs.update({
            "reference_image": str(analysis.ref_path),
            "strain_start_frame": int(getattr(analysis, "strain_start_frame", 0)),
            "fps": float(analysis.fps),
            "metres_per_pixel": float(
                analysis.calibration.metres_per_pixel
                if analysis.calibration.calibrated else 0.0),
            "display_unit": str(analysis.calibration.display_unit),
            "dynamic_roi": str(getattr(p, "dynamic_roi", "None")),
            "dynamic_roi_threshold": (
                np.nan if getattr(p, "dynamic_roi_threshold", None) is None
                else float(p.dynamic_roi_threshold)),
            "dynamic_roi_min_area_frac": float(getattr(
                p, "dynamic_roi_min_area_frac", 0.02)),
            "max_iter": int(p.max_iter),
            "conv_tol": float(p.conv_tol),
            "corr_cutoff": float(p.corr_cutoff),
            "search_radius": int(p.search_radius),
            "rescue_radius": int(p.rescue_radius),
            "shape_order": int(p.shape_order),
            "mask_subsets_to_roi": bool(p.mask_subsets_to_roi),
            "hole_recovery": str(getattr(p, "hole_recovery", "neighbour")),
            "hole_recovery_passes": int(getattr(
                p, "hole_recovery_passes", 3)),
        })
        handle.create_dataset("roi_mask", data=np.asarray(
            analysis.roi_mask, dtype=bool), compression="gzip",
            compression_opts=4)
        origin = analysis.strain_origin_mask
        if origin is not None:
            handle.create_dataset("strain_origin_mask", data=np.asarray(
                origin, dtype=bool), compression="gzip", compression_opts=4)
        for name in ("dynamic_include_mask", "dynamic_exclude_mask"):
            mask = getattr(analysis, name, None)
            if mask is not None:
                handle.create_dataset(name, data=np.asarray(mask, dtype=bool),
                                      compression="gzip", compression_opts=4)
        for index, image_path in enumerate(analysis.def_paths):
            group = handle.create_group(f"frame_{index:06d}")
            group.attrs["image_path"] = str(image_path)
    return target


@dataclass(frozen=True)
class ParametricCase:
    path: Path
    subset_radius: int
    grid_spacing: int
    frame_count: int
    mean_valid_fraction: float
    median_corr: float
    derived_path: Optional[Path] = None

    @property
    def label(self) -> str:
        return f"r={self.subset_radius} px  ·  grid={self.grid_spacing} px"


class ParametricSweep:
    """A collection of HDF5 caches with precomputed frame fields."""

    RAW_FIELDS = {
        "u": "Horizontal displacement",
        "v": "Vertical displacement",
        "magnitude": "Displacement magnitude",
        "corr": "ZNSSD correlation cost",
        "valid": "Valid coverage",
    }
    VELOCITY_FIELDS = {
        "Vx": "Horizontal velocity",
        "Vy": "Vertical velocity",
        "Veff": "Velocity magnitude",
    }
    RATE_FIELDS = {"Eeff_rate": "Equivalent strain rate"}
    STRAIN_FIELDS = {"Eeff_gl": "Accumulated equivalent strain"}
    DERIVED_FIELDS = {**RATE_FIELDS, **STRAIN_FIELDS}

    def __init__(self, cases: list[ParametricCase], *, manifest_path: Path | None,
                 reference_image: str, strain_windows: tuple[int, ...],
                 temporal_spans: tuple[int, ...] = (1,),
                 calibration: Calibration | None = None):
        if not cases:
            raise ValueError("The parametric sweep contains no completed cases.")
        self.cases = cases
        self.manifest_path = manifest_path
        self.calibration = calibration or Calibration()
        self.reference_image = reference_image
        self.requested_strain_windows = strain_windows or tuple(range(1, 16))
        self.temporal_spans = tuple(temporal_spans) or (1,)
        self.frame_count = min(case.frame_count for case in cases)
        self._refresh_cache_capabilities()
        self._temporal_analyses = OrderedDict()
        self._temporal_results = OrderedDict()
        self._temporal_disk_lock = threading.RLock()

    @property
    def display_calibration_path(self) -> Path:
        if self.manifest_path is not None:
            return self.manifest_path.with_name("display_calibration.json")
        return self.cases[0].path.with_suffix(".display_calibration.json")

    def save_display_calibration(self) -> None:
        """Persist a viewing override without changing correlation signatures."""
        path = self.display_calibration_path
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_text(json.dumps(self.calibration.to_dict(), indent=2),
                             encoding="utf-8")
        os.replace(temporary, path)

    def _refresh_cache_capabilities(self) -> None:
        cached_windows = []
        self._derived_cases = set()
        for index, case in enumerate(self.cases):
            path = case.derived_path
            if (path is None or not derived_cache_complete(
                    path, case.path, self.requested_strain_windows)):
                cached_windows.append(set())
                continue
            try:
                with h5py.File(path, "r") as handle:
                    if not bool(handle.attrs.get("derived_complete", False)):
                        raise ValueError("incomplete derived cache")
                    cached_windows.append({int(v) for v in handle.attrs.get(
                        "strain_windows_requested", ())})
                    self._derived_cases.add(index)
            except (OSError, ValueError):
                cached_windows.append(set())
        common = (set.intersection(*cached_windows)
                  if cached_windows and all(cached_windows) else set())
        self.strain_windows = tuple(
            value for value in self.requested_strain_windows if value in common)
        self.available_fields = set(self.RAW_FIELDS) | set(self.VELOCITY_FIELDS)
        if self.strain_windows:
            self.available_fields |= set(self.DERIVED_FIELDS)

    @classmethod
    def from_hdf5(cls, selected_path: str | Path) -> "ParametricSweep":
        selected = Path(selected_path).resolve()
        with h5py.File(selected, "r") as handle:
            if "derived_cache_schema" in handle.attrs:
                correlation_file = str(handle.attrs.get("correlation_file", ""))
                if not correlation_file:
                    raise ValueError("This derived cache does not name its correlation file.")
                selected = selected.with_name(Path(correlation_file).name)
            elif "temporal_cache_schema" in handle.attrs:
                if not selected.name.endswith(".temporal.h5"):
                    raise ValueError("Cannot locate the correlation cache for this temporal file.")
                selected = selected.with_name(
                    selected.name.removesuffix(".temporal.h5") + ".h5")
            elif "sweep_cache_schema" not in handle.attrs:
                if selected.name != "_sweep_source.h5":
                    raise ValueError("This is not a parametric sweep cache.")
                manifest = selected.parent / "manifest.json"
                if not manifest.is_file():
                    raise ValueError("No parameter sweep manifest was found beside this source file.")
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                selected = next((
                    selected.parent / str(row["file"])
                    for row in payload.get("correlations", [])
                    if row.get("status") == "complete" and
                    (selected.parent / str(row.get("file", ""))).is_file()
                ), None)
                if selected is None:
                    raise ValueError("This parameter sweep has no completed correlation files.")
        if not selected.is_file():
            raise ValueError(f"The correlation cache is missing: {selected.name}")
        with h5py.File(selected, "r") as handle:
            if "sweep_cache_schema" not in handle.attrs:
                raise ValueError("The linked file is not a parametric correlation cache.")
            selected_source = str(handle.attrs.get("source_hdf5", ""))
            metres_per_pixel = float(handle.attrs.get(
                "metres_per_pixel", 0.0) or 0.0)
            display_unit = str(handle.attrs.get("display_unit", "mm"))
            selected_windows = tuple(int(v) for v in handle.attrs.get(
                "strain_windows_requested", np.arange(1, 16)))

        calibration = Calibration(
            metres_per_pixel if metres_per_pixel > 0 else None, display_unit)
        source_path = Path(selected_source) if selected_source else None
        if source_path is not None and not source_path.is_absolute():
            source_path = selected.parent / source_path
        if source_path is None or not source_path.is_file():
            candidate = selected.parent / "_sweep_source.h5"
            source_path = candidate if candidate.is_file() else None
        if source_path is not None:
            try:
                with h5py.File(source_path, "r") as source:
                    metres_per_pixel = float(source.attrs.get(
                        "metres_per_pixel", 0.0) or 0.0)
                    display_unit = str(source.attrs.get("display_unit", "mm"))
                calibration = Calibration(
                    metres_per_pixel if metres_per_pixel > 0 else None,
                    display_unit)
            except OSError:
                pass

        manifest_path = selected.parent / "manifest.json"
        rows = []
        reference = ""
        windows = selected_windows
        temporal_spans = (1,)
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = [row for row in payload.get("correlations", [])
                    if row.get("status") == "complete"]
            reference = str(payload.get("source", {}).get(
                "reference_image", ""))
            windows = tuple(int(v) for v in payload.get(
                "strain_windows_requested_px", selected_windows))
            temporal_spans = tuple(int(v) for v in payload.get(
                "temporal_spans_requested_frames", (1,)))
        if not rows:
            rows = [{"file": selected.name, "status": "complete"}]
            manifest_path = None

        cases = []
        for row in rows:
            path = selected.parent / str(row["file"])
            if not path.is_file():
                continue
            with h5py.File(path, "r") as handle:
                if not bool(handle.attrs.get("sweep_complete", False)):
                    continue
                cases.append(ParametricCase(
                    path=path,
                    subset_radius=int(handle.attrs["subset_radius"]),
                    grid_spacing=int(handle.attrs["subset_spacing"]),
                    frame_count=int(handle.attrs["frame_count"]),
                    mean_valid_fraction=float(handle.attrs.get(
                        "mean_valid_fraction", np.nan)),
                    median_corr=float(handle.attrs.get(
                        "median_frame_corr", np.nan)),
                    derived_path=(
                        derived_cache_path(path)
                        if derived_cache_path(path).is_file() else None),
                ))
                if not reference:
                    source = str(handle.attrs.get("source_hdf5", selected_source))
                    if source and Path(source).is_file():
                        with h5py.File(source, "r") as original:
                            reference = str(original.attrs.get("reference_image", ""))
        cases.sort(key=lambda item: (item.subset_radius, item.grid_spacing))
        sweep = cls(cases, manifest_path=manifest_path,
                    reference_image=reference, strain_windows=windows,
                    temporal_spans=temporal_spans, calibration=calibration)
        try:
            override = json.loads(sweep.display_calibration_path.read_text(
                encoding="utf-8"))
            sweep.calibration = Calibration.from_dict(override)
        except (OSError, ValueError, TypeError):
            pass
        return sweep

    def add_hdf5(self, path: str | Path) -> None:
        other = self.from_hdf5(path)
        existing = {case.path for case in self.cases}
        self.cases.extend(case for case in other.cases if case.path not in existing)
        self.frame_count = min(case.frame_count for case in self.cases)
        self._temporal_analyses.clear()
        self._temporal_results.clear()
        self._refresh_cache_capabilities()

    @staticmethod
    def _compact(group, name: str, shape: tuple[int, int],
                 indices: np.ndarray) -> CompactField:
        return CompactField(shape, indices, group[name][:].astype(
            np.float32, copy=False))

    def frame_data(self, case_index: int, frame_index: int,
                   field: str, strain_window: int = 1, *,
                   temporal_span: int = 1, cancel_flag: list | None = None,
                   progress_cb=None):
        if int(temporal_span) > 1:
            return self._temporal_frame_data(
                case_index, frame_index, field, strain_window,
                temporal_span, cancel_flag=cancel_flag,
                progress_cb=progress_cb)
        case = self.cases[int(case_index)]
        frame = max(0, min(int(frame_index), case.frame_count - 1))
        with h5py.File(case.path, "r") as handle:
            group = handle["correlations"][f"frame_{frame:06d}"]
            shape = tuple(int(v) for v in group.attrs["field_shape"])
            indices = group["valid_indices"][:].astype(np.uint32, copy=False)
            image_path = str(group.attrs.get("image_path", ""))
            if field in ("u", "v", "corr"):
                values = self._compact(group, field, shape, indices)
                return values, image_path
            u = self._compact(group, "u", shape, indices)
            v = self._compact(group, "v", shape, indices)
            fps = float(handle.attrs.get("fps", 1.0))
            strain_start = int(handle.attrs.get("strain_start_frame", 0))

        if field == "magnitude":
            return CompactField(
                shape, indices,
                np.hypot(u.values, v.values).astype(np.float32, copy=False)), image_path
        if field == "valid":
            return CompactField(
                shape, indices, np.ones(indices.size, np.float32)), image_path
        if field in self.VELOCITY_FIELDS:
            # Velocity is a scalar transform of compact displacement. Keeping
            # it virtual avoids duplicating u/v for every correlation case and
            # remains far cheaper than reading another HDF5 dataset.
            source = {"Vx": u.values, "Vy": v.values}.get(
                field, np.hypot(u.values, v.values))
            return CompactField(
                shape, indices,
                (source * np.float32(fps)).astype(np.float32, copy=False)), image_path
        if field not in self.DERIVED_FIELDS:
            raise KeyError(field)
        if case_index not in self._derived_cases:
            raise ValueError(
                "This sweep has no cached strain fields. Resume the parameter "
                "sweep once to build them without rerunning correlation.")
        effective = max(int(strain_window), case.grid_spacing)
        with h5py.File(case.derived_path, "r") as derived:
            cached = derived[
                f"windows/w{effective:04d}/frames/frame_{frame:06d}"]
            cached_values = cached[field][:].astype(np.float32, copy=False)
        return CompactField(shape, indices, cached_values), image_path

    def _temporal_analysis(self, case_index: int):
        """Build the existing material-path calculator over a lazy cache.

        Fields stay compact; the analysis only expands values it samples while
        composing a temporal window.
        """
        if case_index in self._temporal_analyses:
            self._temporal_analyses.move_to_end(case_index)
            return self._temporal_analyses[case_index]
        from types import SimpleNamespace
        from .analysis import DICAnalysis
        from .compact_field import CompactMask
        from .rg_dic import DICParams

        case = self.cases[case_index]
        results = []
        with h5py.File(case.path, "r") as handle:
            roi = handle["roi_mask"][:].astype(bool)
            origin = (handle["strain_origin_mask"][:].astype(bool)
                      if "strain_origin_mask" in handle else roi)
            fps = float(handle.attrs.get("fps", 1.0))
            strain_start = int(handle.attrs.get("strain_start_frame", 0))
            frames = handle["correlations"]
            first = frames["frame_000000"]
            shape = tuple(int(v) for v in first.attrs["field_shape"])
            empty = CompactField.empty(shape)
            results.append(SimpleNamespace(u=empty, v=empty,
                                           valid=CompactMask(shape, np.zeros(0, np.uint32)),
                                           elapsed=1.0 / max(fps, 1e-9)))
            for frame_index in range(case.frame_count):
                group = frames[f"frame_{frame_index:06d}"]
                indices = group["valid_indices"][:].astype(
                    np.uint32, copy=False)
                results.append(SimpleNamespace(
                    u=self._compact(group, "u", shape, indices),
                    v=self._compact(group, "v", shape, indices),
                    valid=CompactMask(shape, indices),
                    elapsed=float(group.attrs.get(
                        "elapsed_s", 1.0 / max(fps, 1e-9)))))
        analysis = DICAnalysis()
        analysis.results = results
        analysis._roi_mask = roi
        analysis._strain_origin_mask = origin
        analysis.fps = fps
        analysis.strain_start_frame = strain_start
        analysis.params = DICParams(
            subset_radius=case.subset_radius,
            subset_spacing=case.grid_spacing,
            strain_window=max(int(getattr(case, "grid_spacing", 1)), 1))
        self._temporal_analyses[case_index] = analysis
        while len(self._temporal_analyses) > 1:
            self._temporal_analyses.popitem(last=False)
        return analysis

    def _temporal_frame_data(self, case_index: int, frame_index: int,
                             field: str, strain_window: int, span: int, *,
                             cancel_flag: list | None = None,
                             progress_cb=None):
        case = self.cases[int(case_index)]
        frame = max(0, min(int(frame_index), case.frame_count - 1))
        if frame + 1 < int(span):
            raise ValueError("Insufficient temporal history for this frame.")
        key = (int(case_index), frame, int(strain_window), int(span), field)
        cached = self._temporal_results.get(key)
        if cached is not None:
            self._temporal_results.move_to_end(key)
            return cached[field], self._temporal_image_path(case, frame)
        disk_cached = self._load_temporal_disk(case, frame, field,
                                                strain_window, span)
        if disk_cached is not None:
            self._temporal_results[key] = {field: disk_cached}
            self._trim_temporal_memory_cache()
            return disk_cached, self._temporal_image_path(case, frame)
        analysis = self._temporal_analysis(int(case_index))
        result = analysis.pair_kinematics(
            frame + 1 - int(span), frame + 1,
            strain_window=int(strain_window),
            include_strain=(field == "Eeff_gl"),
            include_rate=(field == "Eeff_rate"),
            cancel_flag=cancel_flag, progress_cb=progress_cb)
        fields = self._compact_temporal_fields(result)
        if field not in fields:
            raise ValueError(f"{field} is unavailable for temporal averages.")
        self._save_temporal_disk(case, frame, strain_window, span, fields)
        self._temporal_results[key] = fields
        self._trim_temporal_memory_cache()
        return fields[field], self._temporal_image_path(case, frame)

    @staticmethod
    def _compact_temporal_fields(result) -> dict[str, CompactField]:
        names = {
            "u": "u", "v": "v", "magnitude": "mag_inc",
            "Vx": "Vx", "Vy": "Vy", "Veff": "Veff",
            "Eeff_rate": "Eeff_rate", "Eeff_gl": "Eeff_gl",
            "valid": "valid",
        }
        fields = {alias: getattr(result, target) for alias, target in names.items()
                  if getattr(result, target, None) is not None}
        if "valid" in fields:
            mask = fields["valid"]
            indices = (mask.indices if hasattr(mask, "indices") else
                       np.flatnonzero(np.asarray(mask, dtype=bool))
                       .astype(np.uint32, copy=False))
            fields["valid"] = CompactField(
                mask.shape, indices,
                np.ones(indices.size, np.float32))
        for name, value in tuple(fields.items()):
            if value is not None and not isinstance(value, CompactField):
                array = np.asarray(value)
                if array.ndim == 2:
                    fields[name] = CompactField.from_dense(array)
        return fields

    def _trim_temporal_memory_cache(self) -> None:
        def size(entry):
            return sum(value.nbytes for value in entry.values()
                       if isinstance(value, CompactField))
        total_bytes = sum(size(entry)
                          for entry in self._temporal_results.values())
        while total_bytes > 64 * 1024 * 1024 and len(self._temporal_results) > 1:
            oldest = next(iter(self._temporal_results))
            total_bytes -= size(self._temporal_results[oldest])
            self._temporal_results.popitem(last=False)

    def precompute_temporal(self, windows: tuple[int, ...],
                            spans: tuple[int, ...], *, progress=None,
                            should_stop=None) -> None:
        """Fill missing temporal sidecars for all requested configurations."""
        total = len(windows) * sum(
            max(0, case.frame_count - int(span) + 1)
            for case in self.cases for span in spans if int(span) > 1)
        done = 0
        for case_index, case in enumerate(self.cases):
            for span_value in spans:
                span = int(span_value)
                if span <= 1:
                    continue
                for window_value in windows:
                    window = int(window_value)
                    for frame in range(span - 1, case.frame_count):
                        if should_stop is not None and should_stop():
                            raise KeyboardInterrupt
                        if not self._temporal_disk_has_fields(
                                case, frame, window, span,
                                ("u", "v", "magnitude", "Vx", "Vy", "Veff",
                                 "Eeff_rate", "Eeff_gl", "valid")):
                            if progress is not None:
                                progress(done / max(1, total),
                                         f"Computing temporal span {span}, "
                                         f"window {window}, frame "
                                         f"{frame + 1}/{case.frame_count}")
                            analysis = self._temporal_analysis(case_index)
                            def report_inner(fraction: float, message: str) -> None:
                                if progress is not None:
                                    progress(
                                        (done + 0.95 * max(0.0, min(
                                            1.0, float(fraction)))) /
                                        max(1, total), message)
                            result = analysis.pair_kinematics(
                                frame + 1 - span, frame + 1,
                                strain_window=window, include_strain=True,
                                include_rate=True, progress_cb=report_inner)
                            fields = self._compact_temporal_fields(result)
                            self._save_temporal_disk(
                                case, frame, window, span, fields, strict=True)
                        done += 1
                        if progress is not None:
                            progress(done / max(1, total),
                                     f"Temporal span {span}, window {window}, "
                                     f"frame {frame + 1}/{case.frame_count}")

    @staticmethod
    def _temporal_cache_path(case: ParametricCase) -> Path:
        return case.path.with_name(case.path.stem + ".temporal.h5")

    def _temporal_disk_has_fields(self, case: ParametricCase, frame: int,
                                  window: int, span: int,
                                  names: tuple[str, ...]) -> bool:
        path = self._temporal_cache_path(case)
        if not path.is_file():
            return False
        key = (f"span_{int(span):05d}/window_{int(window):05d}/"
               f"frame_{int(frame):06d}")
        try:
            with self._temporal_disk_lock, h5py.File(path, "r") as handle:
                stat = case.path.stat()
                if (int(handle.attrs.get("source_size", -1)) != stat.st_size or
                        int(handle.attrs.get("source_mtime_ns", -1)) !=
                        stat.st_mtime_ns or key not in handle):
                    return False
                group = handle[key]
                return all(name in group and "indices" in group[name] and
                           "values" in group[name] for name in names)
        except (OSError, KeyError, ValueError):
            return False

    def _load_temporal_disk(self, case: ParametricCase, frame: int,
                            field: str, window: int, span: int):
        path = self._temporal_cache_path(case)
        if not path.is_file():
            return None
        key = (f"span_{int(span):05d}/window_{int(window):05d}/"
               f"frame_{int(frame):06d}")
        try:
            with self._temporal_disk_lock, h5py.File(path, "r") as handle:
                stat = case.path.stat()
                if (int(handle.attrs.get("source_size", -1)) != stat.st_size or
                        int(handle.attrs.get("source_mtime_ns", -1)) !=
                        stat.st_mtime_ns):
                    return None
                group = handle[key]
                if field not in group:
                    return None
                data = group[field]
                shape = tuple(int(value) for value in data.attrs["shape"])
                return CompactField(shape, data["indices"][:],
                                    data["values"][:])
        except (OSError, KeyError, ValueError):
            return None

    def _save_temporal_disk(self, case: ParametricCase, frame: int,
                            window: int, span: int, fields: dict, *,
                            strict: bool = False) -> None:
        path = self._temporal_cache_path(case)
        key = (f"span_{int(span):05d}/window_{int(window):05d}/"
               f"frame_{int(frame):06d}")
        try:
            with self._temporal_disk_lock:
                stat = case.path.stat()
                mode = "a"
                if path.is_file():
                    try:
                        with h5py.File(path, "r") as existing:
                            stale = (
                                int(existing.attrs.get("source_size", -1)) !=
                                stat.st_size or
                                int(existing.attrs.get("source_mtime_ns", -1)) !=
                                stat.st_mtime_ns)
                        if stale:
                            path.unlink()
                    except OSError:
                        path.unlink(missing_ok=True)
                with h5py.File(path, mode) as handle:
                    handle.attrs["temporal_cache_schema"] = 1
                    handle.attrs["source_size"] = stat.st_size
                    handle.attrs["source_mtime_ns"] = stat.st_mtime_ns
                    parent = handle.require_group(key)
                    for name, value in fields.items():
                        if not isinstance(value, CompactField) or name in parent:
                            continue
                        data = parent.create_group(name)
                        data.attrs["shape"] = np.asarray(value.shape, np.int64)
                        data.create_dataset("indices", data=value.indices,
                                            compression="gzip", compression_opts=4)
                        data.create_dataset("values", data=value.values,
                                            compression="gzip", compression_opts=4)
                    handle.flush()
        except (OSError, ValueError):
            if strict:
                raise
            # The result remains usable for this session if the cache volume
            # becomes read-only or runs out of space.
            return

    @staticmethod
    def _temporal_image_path(case: ParametricCase, frame: int) -> str:
        with h5py.File(case.path, "r") as handle:
            return str(handle["correlations"][
                f"frame_{frame:06d}"].attrs.get("image_path", ""))

    @property
    def field_choices(self) -> dict[str, str]:
        return {**self.RAW_FIELDS, **self.VELOCITY_FIELDS,
                **self.RATE_FIELDS, **self.STRAIN_FIELDS}
