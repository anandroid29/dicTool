"""Resumable, streamed DIC parameter sweep.

Each (subset radius, grid spacing) pair is one atomic HDF5 checkpoint.  Strain
windows are recorded as downstream parameters because they do not participate
in correlation; their fields can be derived from these lossless u/v caches
without rerunning the expensive CUDA solve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

import cv2
import h5py
import numpy as np

from strainx.core.analysis import DICAnalysis, DynamicROI, _load_image
from strainx.core.compact_field import CompactField, CompactMask
from strainx.core.parametric import (
    ParametricCase, ParametricSweep, build_derived_cache,
    derived_cache_complete, derived_cache_path,
)
from strainx.core.rg_dic import DICParams
from strainx.core.units import Calibration


SWEEP_SCHEMA = 1
RADIUS_VALUES = tuple(range(5, 13))
SPACING_VALUES = tuple(range(5, 0, -1))  # cheap checkpoints first
STRAIN_WINDOWS = tuple(range(1, 16))


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def fixed_dynamic_roi(handle: h5py.File, reference: np.ndarray) -> np.ndarray:
    """Recreate the saved frame-zero Dynamic ROI and contour-fill it once."""
    static = handle["roi_mask"][:].astype(bool)
    include = (handle["dynamic_include_mask"][:].astype(bool)
               if "dynamic_include_mask" in handle else None)
    exclude = (handle["dynamic_exclude_mask"][:].astype(bool)
               if "dynamic_exclude_mask" in handle else None)
    threshold = float(handle.attrs.get("dynamic_roi_threshold", np.nan))
    roi = DynamicROI(
        _text(handle.attrs.get("dynamic_roi", "None")),
        keep_min_area_frac=float(handle.attrs.get(
            "dynamic_roi_min_area_frac", 0.02)),
        threshold=(threshold if np.isfinite(threshold) else None),
        include_mask=include, exclude_mask=exclude, roi_mask=static,
        fill_holes=True, hysteresis=0.0)
    roi.calibrate(reference)
    dynamic = roi.mask(reference, reference_frame=True)
    if dynamic is None:
        dynamic = static

    binary = np.asarray(dynamic, dtype=np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool) & static


def load_source(path: Path, max_frames: int | None = None):
    with h5py.File(path, "r") as source:
        reference_path = _text(source.attrs["reference_image"])
        reference = _load_image(reference_path)
        static_roi = source["roi_mask"][:].astype(bool)
        roi = fixed_dynamic_roi(source, reference)
        frame_keys = sorted(k for k in source if k.startswith("frame_"))
        if max_frames is not None:
            frame_keys = frame_keys[:max_frames]
        deformed = [_text(source[key].attrs.get("image_path", ""))
                    for key in frame_keys]
        origin = (source["strain_origin_mask"][:].astype(bool) & roi
                  if "strain_origin_mask" in source else None)
        attrs = {str(key): (value.item() if isinstance(value, np.generic)
                            else _text(value) if isinstance(value, bytes)
                            else value)
                 for key, value in source.attrs.items()}

    missing = [item for item in (reference_path, *deformed)
               if not os.path.isfile(item)]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} source frame(s): {missing[0]}")
    signature = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "frame_count": len(deformed),
        "reference_image": reference_path,
        "reference_sha256": _sha256(reference),
        "static_roi_sha256": _sha256(static_roi),
        "roi_sha256": _sha256(roi),
    }
    return reference, deformed, roi, origin, attrs, signature


def make_analysis(reference_path: str, reference: np.ndarray,
                  deformed: list[str], roi: np.ndarray,
                  origin: np.ndarray | None, attrs: dict[str, Any],
                  radius: int, spacing: int) -> DICAnalysis:
    analysis = DICAnalysis()
    analysis._release_loaded_hdf5()
    analysis.ref_path = reference_path
    analysis._ref_image = reference
    analysis.def_paths = list(deformed)
    analysis._roi_mask = roi
    analysis._strain_origin_mask = origin
    analysis.strain_start_frame = int(attrs.get("strain_start_frame", 0))
    analysis.fps = float(attrs.get("fps", 1.0))
    mpp = float(attrs.get("metres_per_pixel", 0.0) or 0.0)
    analysis.calibration = Calibration(
        mpp if mpp > 0 else None, _text(attrs.get("display_unit", "mm")))
    analysis.dynamic_include_mask = None
    analysis.dynamic_exclude_mask = None
    analysis.dynamic_frame_overrides = {}
    analysis.dynamic_future_overrides = {}

    params = DICParams()
    for name, cast in (
        ("max_iter", int), ("conv_tol", float), ("corr_cutoff", float),
        ("search_radius", int), ("rescue_radius", int), ("shape_order", int),
        ("mask_subsets_to_roi", bool), ("hole_recovery", str),
        ("hole_recovery_passes", int),
    ):
        if name in attrs:
            setattr(params, name, cast(attrs[name]))
    params.subset_radius = radius
    params.subset_spacing = spacing
    params.strain_window = spacing
    # The saved Dynamic ROI was frozen into analysis._roi_mask above. Applying
    # it again per frame would change the experimental domain between cases.
    params.dynamic_roi = "None"
    analysis.params = params
    return analysis


def _aligned_values(field: CompactField, indices: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(field.indices, indices)
    if (positions.size and
            (np.any(positions >= field.indices.size) or
             not np.array_equal(field.indices[positions], indices))):
        raise ValueError("Compact result fields do not share the validity grid.")
    return field.values[positions]


class CorrelationCacheWriter:
    def __init__(self, temporary: Path, final: Path, *, radius: int,
                 spacing: int, roi: np.ndarray, origin: np.ndarray | None,
                 source_signature: dict[str, Any], attrs: dict[str, Any],
                 backend: str, frame_count: int,
                 strain_windows: tuple[int, ...] = STRAIN_WINDOWS):
        self.temporary = temporary
        self.final = final
        self.handle = h5py.File(temporary, "w")
        h = self.handle
        h.attrs.update({
            "sweep_cache_schema": SWEEP_SCHEMA,
            "sweep_complete": False,
            "source_hdf5": source_signature["path"],
            "source_size": source_signature["size"],
            "source_mtime_ns": source_signature["mtime_ns"],
            "source_roi_sha256": source_signature["roi_sha256"],
            "source_reference_sha256": source_signature["reference_sha256"],
            "fixed_roi_sha256": _sha256(roi),
            "frame_count": frame_count,
            "subset_radius": radius,
            "subset_spacing": spacing,
            "correlation_backend": backend,
            "strain_windows_requested": np.asarray(strain_windows, np.int16),
            "strain_window_semantics": "downstream_lazy_derivation",
            "fps": float(attrs.get("fps", 1.0)),
            "metres_per_pixel": float(attrs.get("metres_per_pixel", 0.0) or 0.0),
            "display_unit": _text(attrs.get("display_unit", "mm")),
            "strain_start_frame": int(attrs.get("strain_start_frame", 0)),
            "corr_measure": "ZNSSD_lower_is_better",
            "corr_cutoff": float(attrs.get("corr_cutoff", 0.30)),
        })
        h.create_dataset("roi_mask", data=roi, compression="gzip",
                         compression_opts=4)
        if origin is not None:
            h.create_dataset("strain_origin_mask", data=origin,
                             compression="gzip", compression_opts=4)
        self.expected = int(roi[radius:-radius or None:spacing,
                                radius:-radius or None:spacing].sum())
        h.attrs["eligible_subset_centres"] = self.expected
        self.frames = h.create_group("correlations")
        self.valid_counts: list[int] = []
        self.corr_medians: list[float] = []

    def write(self, index: int, result) -> None:
        if not (isinstance(result.u, CompactField) and
                isinstance(result.v, CompactField) and
                isinstance(result.corr, CompactField) and
                isinstance(result.valid, CompactMask)):
            raise TypeError("Streamed sweep results must use compact storage.")
        indices = result.valid.indices
        group = self.frames.create_group(f"frame_{index:06d}")
        group.attrs["image_path"] = result.image_path
        group.attrs["elapsed_s"] = float(result.elapsed)
        group.attrs["field_shape"] = np.asarray(result.u.shape, np.int64)
        group.attrs["valid_count"] = int(indices.size)
        group.attrs["valid_fraction"] = (
            float(indices.size / self.expected) if self.expected else 0.0)
        group.create_dataset("valid_indices", data=indices,
                             compression="gzip", compression_opts=4)
        group.create_dataset("u", data=_aligned_values(result.u, indices),
                             compression="gzip", compression_opts=4)
        group.create_dataset("v", data=_aligned_values(result.v, indices),
                             compression="gzip", compression_opts=4)
        corr = _aligned_values(result.corr, indices)
        group.create_dataset("corr", data=corr, compression="gzip",
                             compression_opts=4)
        self.valid_counts.append(int(indices.size))
        self.corr_medians.append(float(np.median(corr)) if corr.size else np.nan)
        self.handle.flush()

    def finish(self, elapsed: float) -> None:
        h = self.handle
        h.attrs["elapsed_s"] = float(elapsed)
        h.attrs["mean_valid_fraction"] = (
            float(np.mean(self.valid_counts) / self.expected)
            if self.expected and self.valid_counts else 0.0)
        finite = np.asarray(self.corr_medians, dtype=float)
        finite = finite[np.isfinite(finite)]
        h.attrs["median_frame_corr"] = (
            float(np.median(finite)) if finite.size else np.nan)
        h.attrs["sweep_complete"] = True
        h.flush()
        h.close()
        os.replace(self.temporary, self.final)

    def abort(self) -> None:
        try:
            self.handle.close()
        except Exception:
            pass


def cache_is_complete(path: Path, signature: dict[str, Any],
                      radius: int, spacing: int) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            stored_reference = _text(handle.attrs.get(
                "source_reference_sha256", ""))
            reference_matches = (
                not stored_reference or
                stored_reference == signature["reference_sha256"])
            return bool(
                handle.attrs.get("sweep_complete", False)
                and int(handle.attrs.get("sweep_cache_schema", -1)) == SWEEP_SCHEMA
                and int(handle.attrs.get("subset_radius", -1)) == radius
                and int(handle.attrs.get("subset_spacing", -1)) == spacing
                and int(handle.attrs.get("frame_count", -1)) == signature["frame_count"]
                and _text(handle.attrs.get("source_roi_sha256", ""))
                == signature["roi_sha256"]
                and reference_matches
                and len(handle.get("correlations", {})) == signature["frame_count"])
    except (OSError, KeyError, ValueError):
        return False


def write_manifest(output: Path, source: dict[str, Any], rows: list[dict],
                   radii: tuple[int, ...] = RADIUS_VALUES,
                   spacings: tuple[int, ...] = SPACING_VALUES,
                   strain_windows: tuple[int, ...] = STRAIN_WINDOWS,
                   temporal_spans: tuple[int, ...] = (1,)) -> None:
    payload = {
        "schema": SWEEP_SCHEMA,
        "source": source,
        "subset_radii_px": list(radii),
        "grid_spacings_px": sorted(spacings),
        "strain_windows_requested_px": list(strain_windows),
        "temporal_spans_requested_frames": list(temporal_spans),
        "independent_correlation_count": len(radii) * len(spacings),
        "logical_parameter_combinations": (
            len(radii) * len(spacings) * len(strain_windows) *
            len(temporal_spans)),
        "strain_window_mapping": {
            str(spacing): [
                {
                    "requested_px": requested,
                    "effective_px": max(requested, spacing),
                    "canonical_equivalent_px": (
                        max(requested, spacing) // spacing * spacing),
                    "grid_points_per_axis": (
                        2 * (max(requested, spacing) // spacing) + 1),
                }
                for requested in strain_windows
            ]
            for spacing in sorted(spacings)
        },
        "design": (
            "Correlation caches store u, v, validity and ZNSSD once per "
            "radius/spacing pair. Velocity is derived directly from compact "
            "u/v. Atomic sidecars precompute only effective strain rate and "
            "accumulated effective strain for the selected windows without "
            "rerunning correlation."),
        "correlations": rows,
    }
    temporary = output / "manifest.json.part"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("good_vid_result_1.h5"))
    parser.add_argument("--output", type=Path, default=Path("dic_parameter_sweep"))
    parser.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--max-correlations", type=int)
    parser.add_argument("--max-frames", type=int,
                        help="Testing only: truncate the source sequence.")
    parser.add_argument(
        "--stop-file", type=Path,
        help=("Stop before the next correlation when this file exists. "
              "Defaults to OUTPUT/STOP_AFTER_CURRENT."))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--radii", default="5:12:1",
                        help="Inclusive START:STOP:STEP integer range.")
    parser.add_argument("--spacings", default="1:5:1",
                        help="Inclusive START:STOP:STEP integer range.")
    parser.add_argument("--strain-windows", default="1:15:1",
                        help="Inclusive cached START:STOP:STEP window range.")
    parser.add_argument("--temporal-spans", default="1:1:1",
                        help="Inclusive temporal frame-interval range.")
    parser.add_argument("--precompute-temporal", action="store_true",
                        help="Compute and save every selected temporal result.")
    return parser.parse_args()


def _parse_inclusive_range(spec: str) -> tuple[int, ...]:
    try:
        start, stop, step = (int(part) for part in spec.split(":"))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Expected START:STOP:STEP, got {spec!r}") from exc
    if step <= 0 or stop < start:
        raise argparse.ArgumentTypeError(
            f"Range must have STOP >= START and STEP > 0: {spec!r}")
    return tuple(range(start, stop + 1, step))


def main() -> int:
    args = parse_args()
    radii = _parse_inclusive_range(args.radii)
    # Descending spacing order gives cheap/sparse checkpoints first while the
    # manifest still presents the user's ascending range.
    spacings = tuple(reversed(_parse_inclusive_range(args.spacings)))
    strain_windows = _parse_inclusive_range(args.strain_windows)
    temporal_spans = _parse_inclusive_range(args.temporal_spans)
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    stop_file = (args.stop_file.resolve() if args.stop_file is not None
                 else (args.output / "STOP_AFTER_CURRENT").resolve())
    reference, deformed, roi, origin, attrs, signature = load_source(
        args.source, args.max_frames)
    reference_path = signature["reference_image"]
    if max(temporal_spans) > len(deformed):
        parser = argparse.ArgumentParser()
        parser.error("Temporal spans cannot exceed the source frame count.")

    if args.backend == "auto":
        from strainx.core.cuda_native import native_cuda_available
        backend = "gpu" if native_cuda_available(refresh=True) else "cpu"
    else:
        backend = args.backend
    print(
        f"Source: {args.source}\nFrames: {len(deformed)}\n"
        f"Fixed ROI: {roi.sum():,} px ({signature['roi_sha256'][:12]})\n"
        f"Backend: {backend}\nOutput: {args.output.resolve()}\n"
        f"Graceful stop file: {stop_file}", flush=True)

    stop = {"after_current": False, "signals": 0}
    current = {"analysis": None}

    def request_stop(_signum, _frame):
        stop["signals"] += 1
        if stop["signals"] == 1:
            stop["after_current"] = True
            print(
                "\nStop requested: finishing this radius/spacing checkpoint. "
                "Press Ctrl+C again to abort it immediately.", flush=True)
        else:
            if current["analysis"] is not None:
                current["analysis"].cancel()
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    combinations = [(radius, spacing)
                    for radius in radii for spacing in spacings]
    rows: list[dict] = []
    for index, (radius, spacing) in enumerate(combinations, start=1):
        name = f"correlation_r{radius:02d}_g{spacing:02d}.h5"
        rows.append({
            "index": index, "subset_radius_px": radius,
            "grid_spacing_px": spacing, "file": name,
            "status": ("complete" if cache_is_complete(
                args.output / name, signature, radius, spacing) else "pending"),
            "derived_file": derived_cache_path(args.output / name).name,
            "derived_status": ("complete" if derived_cache_complete(
                derived_cache_path(args.output / name), args.output / name,
                strain_windows) else "pending"),
        })
    write_manifest(args.output, signature, rows, radii, spacings,
                   strain_windows, temporal_spans)
    started = 0
    total = len(radii) * len(spacings)

    for row, (radius, spacing) in zip(rows, combinations):
        name = row["file"]
        final = args.output / name
        complete = cache_is_complete(final, signature, radius, spacing)
        if complete:
            print(f"[{row['index']}/{total}] skip complete correlation {name}",
                  flush=True)
            derived = derived_cache_path(final)
            if not derived_cache_complete(
                    derived, final, strain_windows):
                if args.dry_run:
                    print(f"[{row['index']}/{total}] would build {derived.name}",
                          flush=True)
                    continue
                print(f"[{row['index']}/{total}] start {derived.name}",
                      flush=True)
                derive_report = {"time": 0.0}

                def derive_progress(fraction: float, message: str) -> None:
                    now = time.monotonic()
                    if now - derive_report["time"] < 5.0 and fraction < 1.0:
                        return
                    print(
                        f"[{row['index']}/{total}] r={radius} g={spacing} "
                        f"{100.0 * fraction:5.1f}% {message}", flush=True)
                    derive_report["time"] = now

                try:
                    build_derived_cache(
                        final, strain_windows, use_gpu=(backend == "gpu"),
                        progress=derive_progress,
                        should_stop=lambda: stop["after_current"] or
                        stop_file.exists())
                except KeyboardInterrupt:
                    row["derived_status"] = "interrupted"
                    write_manifest(args.output, signature, rows, radii,
                                   spacings, strain_windows, temporal_spans)
                    return 130
            row["derived_status"] = "complete"
            write_manifest(args.output, signature, rows, radii, spacings,
                           strain_windows, temporal_spans)
            print(f"[{row['index']}/{total}] complete {derived.name}",
                  flush=True)
            continue
        if stop["after_current"] or stop_file.exists():
            if stop_file.exists():
                print(f"Stop file found: {stop_file}", flush=True)
            write_manifest(args.output, signature, rows, radii, spacings,
                           strain_windows, temporal_spans)
            return 130
        if args.max_correlations is not None and started >= args.max_correlations:
            write_manifest(args.output, signature, rows, radii, spacings,
                           strain_windows, temporal_spans)
            return 0
        if args.dry_run:
            print(f"[{row['index']}/{total}] would run {name}", flush=True)
            continue

        temporary = final.with_suffix(final.suffix + ".part")
        analysis = make_analysis(
            reference_path, reference, deformed, roi, origin, attrs,
            radius, spacing)
        current["analysis"] = analysis
        writer = CorrelationCacheWriter(
            temporary, final, radius=radius, spacing=spacing,
            roi=roi, origin=origin, source_signature=signature, attrs=attrs,
            backend=backend, frame_count=len(deformed),
            strain_windows=strain_windows)
        unit_start = time.perf_counter()
        last_report = {"time": 0.0, "message": ""}

        def report_case(fraction: float, message: str) -> None:
            now = time.monotonic()
            if now - last_report["time"] >= 5.0 or fraction >= 1.0:
                print(
                    f"[{row['index']}/{total}] r={radius} g={spacing} "
                    f"{100.0 * fraction:5.1f}% {message}", flush=True)
                last_report.update(time=now, message=message)

        def progress(fraction: float, message: str) -> None:
            report_case(0.85 * fraction, message)

        print(f"[{row['index']}/{total}] start {name}", flush=True)
        try:
            analysis.run(
                progress_cb=progress, use_gpu=(backend == "gpu"),
                postprocess=False, result_cb=writer.write,
                retain_results=False)
            if analysis._cancel[0]:
                raise KeyboardInterrupt
            elapsed = time.perf_counter() - unit_start
            writer.finish(elapsed)
            row["status"] = "complete"
            row["elapsed_s"] = elapsed
            print(
                f"[{row['index']}/{total}] complete {name} in "
                f"{elapsed / 60.0:.1f} min", flush=True)
            derived = derived_cache_path(final)
            print(f"[{row['index']}/{total}] start {derived.name}", flush=True)
            build_derived_cache(
                final, strain_windows, use_gpu=(backend == "gpu"),
                progress=lambda fraction, message: report_case(
                    0.85 + 0.15 * fraction, message),
                should_stop=lambda: stop["after_current"] or stop_file.exists())
            row["derived_status"] = "complete"
            print(f"[{row['index']}/{total}] complete {derived.name}",
                  flush=True)
            started += 1
            write_manifest(args.output, signature, rows, radii, spacings,
                           strain_windows, temporal_spans)
        except KeyboardInterrupt:
            writer.abort()
            if cache_is_complete(final, signature, radius, spacing):
                row["status"] = "complete"
                row["derived_status"] = "interrupted"
            else:
                row["status"] = "interrupted"
            write_manifest(args.output, signature, rows, radii, spacings,
                           strain_windows, temporal_spans)
            print(
                f"Interrupted during {name}; completed .h5 checkpoints are "
                "unchanged and this .part file will be restarted next run.",
                flush=True)
            return 130
        finally:
            current["analysis"] = None

        if stop["after_current"] or stop_file.exists():
            if stop_file.exists():
                print(f"Stop file found: {stop_file}", flush=True)
            print("Stopped cleanly between correlation checkpoints.", flush=True)
            return 130

    write_manifest(args.output, signature, rows, radii, spacings,
                   strain_windows, temporal_spans)
    if args.precompute_temporal and any(span > 1 for span in temporal_spans):
        print("Precomputing temporal spans into HDF5 sidecars...", flush=True)
        for position, row in enumerate(rows, 1):
            if stop["after_current"] or stop_file.exists():
                print("Stopped cleanly between temporal checkpoints.", flush=True)
                return 130
            path = args.output / row["file"]
            case = ParametricCase(
                path=path, subset_radius=int(row["subset_radius_px"]),
                grid_spacing=int(row["grid_spacing_px"]),
                frame_count=len(deformed), mean_valid_fraction=np.nan,
                median_corr=np.nan,
                derived_path=derived_cache_path(path))
            sweep = ParametricSweep(
                [case], manifest_path=None, reference_image=reference_path,
                strain_windows=strain_windows, temporal_spans=temporal_spans)
            print(f"TEMPORAL [{position}/{total}] 0.0% start {path.name}",
                  flush=True)
            last_report = {"time": 0.0}

            def temporal_progress(fraction: float, message: str) -> None:
                now = time.monotonic()
                if now - last_report["time"] >= 5.0 or fraction >= 1.0:
                    print(f"TEMPORAL [{position}/{total}] "
                          f"{100.0 * fraction:5.1f}% {message}", flush=True)
                    last_report["time"] = now

            try:
                sweep.precompute_temporal(
                    strain_windows, temporal_spans,
                    progress=temporal_progress,
                    should_stop=lambda: stop["after_current"] or
                    stop_file.exists())
            except KeyboardInterrupt:
                row["temporal_status"] = "interrupted"
                write_manifest(args.output, signature, rows, radii, spacings,
                               strain_windows, temporal_spans)
                print("Stopped during temporal precomputation; saved results "
                      "will be reused on resume.", flush=True)
                return 130
            row["temporal_status"] = "complete"
            write_manifest(args.output, signature, rows, radii, spacings,
                           strain_windows, temporal_spans)
            print(f"TEMPORAL [{position}/{total}] 100.0% complete {path.name}",
                  flush=True)
    print(("All correlation, strain, and temporal checkpoints are complete."
           if args.precompute_temporal and any(span > 1 for span in temporal_spans)
           else "All correlation and derived-field checkpoints are complete."),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
