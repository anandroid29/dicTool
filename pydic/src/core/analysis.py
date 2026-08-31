# src/core/analysis.py
"""
analysis.py — DICAnalysis with strain rate computation, frame-sync support, and batched GPU execution.
Fixed: Survival rate denominator uses valid ROI subset count to prevent false Auto-Fallback triggers.
"""
from __future__ import annotations
import importlib.util
import os, time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

try:
    from PIL import Image as PILImage; _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False
try:
    import cv2; _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

# Importing CuPy creates a CUDA context on some installations. Merely opening
# the UI must not reserve hundreds of MB before the operator starts analysis.
# The real import and driver validation happen in the GPU analysis path.
_HAS_CUPY = importlib.util.find_spec("cupy") is not None

from .rg_dic import DICParams
from .compact_field import CompactField, CompactMask, finite_values
from .roi_loader import load_roi_mask
from .stats import field_summary, robust_limits
from .units import Calibration


# Native (uncalibrated) unit of each exportable field, used to label exports.
# Kept beside the data rather than in the UI so a headless export says the same
# thing the results view does.
_FIELD_BASE_UNIT = {
    "u": "px", "v": "px",
    "u_inc": "px", "v_inc": "px", "mag_inc": "px",
    "Vx": "px/s", "Vy": "px/s", "Veff": "px/s",
    "Exx": "", "Exy": "", "Eyy": "", "Eeff": "",
    "Exx_inf": "", "Exy_inf": "", "Gxy_inf": "", "Eyy_inf": "", "Eeff_inf": "",
    "Exx_gl": "", "Exy_gl": "", "Gxy_gl": "", "Eyy_gl": "", "Eeff_gl": "",
    "Exx_rate": "1/s", "Exy_rate": "1/s", "Gxy_rate": "1/s",
    "Eyy_rate": "1/s", "Eeff_rate": "1/s",
    "dVx_dx": "1/s", "dVx_dy": "1/s", "dVy_dx": "1/s", "dVy_dy": "1/s",
    "du_dx": "", "du_dy": "", "dv_dx": "", "dv_dy": "",
    "corr": "ZNSSD",
}

# Keep ordinary result files self-contained in memory, but avoid expanding long
# compressed sessions into many gigabytes of RAM merely to open the viewer.
# This threshold is based on the datasets' logical (uncompressed) size.
HDF5_LAZY_THRESHOLD_BYTES = 512 * 1024 * 1024


def _result_f32(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Compact a completed result field without changing solver precision.

    Correlation, interpolation, gradient fitting and accumulation continue in
    float64.  Once a frame is finished, keeping every display/export field in
    float64 doubles long-sequence RAM for no useful precision: HDF5 already
    writes float32, and the UI cannot display the extra digits.  ``asarray``
    avoids a second copy when a field is already compact.
    """
    if arr is None:
        return None
    if isinstance(arr, CompactField):
        return arr
    out = np.asarray(arr, dtype=np.float32)
    infinite = np.isinf(out)
    if infinite.any():
        out = out.copy()
        out[infinite] = np.nan
    return out


def _compact_field(arr: np.ndarray, valid: np.ndarray,
                   indices: Optional[np.ndarray] = None) -> CompactField:
    """Pack one completed solver field at its valid subset centres."""
    if indices is None:
        indices = np.flatnonzero(np.asarray(valid, dtype=bool).reshape(-1)).astype(
            np.uint32, copy=False)
    return CompactField.from_dense(arr, indices=indices)


def _finite_measurement_mask(base: np.ndarray, *fields: np.ndarray) -> np.ndarray:
    """One validity rule for solver output, derivatives, display and export."""
    valid = np.asarray(base, dtype=bool).copy()
    for field in fields:
        valid &= np.isfinite(field)
    return valid


def _mask_invalid(valid: np.ndarray, *fields: np.ndarray) -> None:
    """Replace every rejected or non-finite measurement with NaN in-place."""
    for field in fields:
        if field is not None:
            field[~valid | ~np.isfinite(field)] = np.nan


def _dynamic_measurement_mask(
    base_valid: np.ndarray,
    current_mask: Optional[np.ndarray],
    inc_u: np.ndarray,
    inc_v: np.ndarray,
    include_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    frame_include_mask: Optional[np.ndarray] = None,
    frame_exclude_mask: Optional[np.ndarray] = None,
    replace_base: bool = False,
) -> np.ndarray:
    """Filter one adjacent-frame pair using the current-frame texture mask.

    The source points and overrides are image-space labels on the immediately
    previous frame. They meet ``current_mask`` after this pair's displacement;
    no frame-0 position history is involved.

    Keeping this transform in one backend-neutral helper prevents the CPU and
    GPU paths from quietly using different ROI semantics. Include overrides win
    over excludes and the automatic texture decision, but cannot resurrect a
    failed or off-frame correlation.
    """
    valid = np.asarray(base_valid, dtype=bool).copy()
    if current_mask is None or not valid.any():
        return valid

    current = np.asarray(current_mask, dtype=bool)
    if current.shape != valid.shape:
        raise ValueError(
            f"Dynamic ROI shape {current.shape} does not match result shape "
            f"{valid.shape}.")

    y_ref, x_ref = np.where(valid)
    du = np.asarray(inc_u)[y_ref, x_ref]
    dv = np.asarray(inc_v)[y_ref, x_ref]

    x_pos = x_ref + du
    y_pos = y_ref + dv
    h, w = current.shape
    in_bounds = (np.isfinite(x_pos) & np.isfinite(y_pos) &
                 (x_pos >= 0) & (x_pos <= w - 1) &
                 (y_pos >= 0) & (y_pos <= h - 1))

    kept = np.zeros(len(x_ref), dtype=bool)
    ib = np.where(in_bounds)[0]
    if ib.size:
        x_cur = np.rint(x_pos[ib]).astype(np.intp)
        y_cur = np.rint(y_pos[ib]).astype(np.intp)
        kept[ib] = current[y_cur, x_cur]

    def _source_override(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != valid.shape:
            raise ValueError(
                f"Dynamic ROI override shape {arr.shape} does not match result "
                f"shape {valid.shape}.")
        return arr[y_ref, x_ref]

    excluded = _source_override(exclude_mask)
    included = _source_override(include_mask)
    if excluded is not None:
        kept &= ~excluded
    if included is not None:
        kept = in_bounds & (kept | included)

    def _destination_override(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != valid.shape:
            raise ValueError(
                "Frame Dynamic ROI override shape does not match result shape.")
        selected = np.zeros(len(x_ref), dtype=bool)
        if ib.size:
            selected[ib] = arr[y_cur, x_cur]
        return selected

    # Exact-frame decisions are destination-image masks and take precedence
    # over every global source-space decision. Include still cannot resurrect a
    # failed solver measurement or a point that moved off-frame.
    frame_excluded = _destination_override(frame_exclude_mask)
    frame_included = _destination_override(frame_include_mask)
    if replace_base:
        kept = (in_bounds & frame_included
                if frame_included is not None else np.zeros_like(kept))
    if frame_excluded is not None:
        kept &= ~frame_excluded
    if frame_included is not None:
        kept = in_bounds & (kept | frame_included)

    valid[y_ref[~kept], x_ref[~kept]] = False
    return valid


@dataclass
class PairResult:
    image_path: str
    u:     np.ndarray
    v:     np.ndarray
    Exx:   np.ndarray
    Exy:   np.ndarray
    Eyy:   np.ndarray
    Eeff:  np.ndarray
    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray
    corr:  np.ndarray
    # u/v are immediate previous-frame -> current-frame displacement.  These
    # aliases are retained for file/API compatibility and carry the same values.
    u_inc:   Optional[np.ndarray] = None
    v_inc:   Optional[np.ndarray] = None
    mag_inc: Optional[np.ndarray] = None
    Vx:    Optional[np.ndarray] = None
    Vy:    Optional[np.ndarray] = None
    Veff:  Optional[np.ndarray] = None
    dVx_dx: Optional[np.ndarray] = None
    dVx_dy: Optional[np.ndarray] = None
    dVy_dx: Optional[np.ndarray] = None
    dVy_dy: Optional[np.ndarray] = None
    Exx_rate:  Optional[np.ndarray] = None
    Exy_rate:  Optional[np.ndarray] = None
    Gxy_rate:  Optional[np.ndarray] = None
    Eyy_rate:  Optional[np.ndarray] = None
    Eeff_rate: Optional[np.ndarray] = None
    # True where the current interval displacement is trustworthy on the
    # pair's source grid. Accumulated strain is transported separately and
    # remeshed at its destination position in the current frame.
    valid: Optional[np.ndarray] = None
    elapsed: float = 0.0
    # Explicit accumulated strain formulations. Exy is tensor shear. Gxy is a
    # legacy optional compatibility field and is not generated by new runs.
    # Equivalent fields integrate a non-negative rate magnitude along a path.
    Exx_inf: Optional[np.ndarray] = None
    Exy_inf: Optional[np.ndarray] = None
    Gxy_inf: Optional[np.ndarray] = None
    Eyy_inf: Optional[np.ndarray] = None
    Eeff_inf: Optional[np.ndarray] = None
    Exx_gl: Optional[np.ndarray] = None
    Exy_gl: Optional[np.ndarray] = None
    Gxy_gl: Optional[np.ndarray] = None
    Eyy_gl: Optional[np.ndarray] = None
    Eeff_gl: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _CroppedTemporalField:
    """Dense finite-support crop used by the sliding gradient LRU cache."""
    shape: tuple[int, int]
    y0: int
    x0: int
    values: np.ndarray


class DICAnalysis:
    # Class-level defaults for the trajectory cache.
    #
    # An instance is not always built through __init__ -- state-clearing methods
    # are exercised against object.__new__(DICAnalysis), and those methods bump
    # the epoch. Defaulting on the class means the bump resolves and creates the
    # instance attribute, instead of raising inside a method whose whole job is
    # to reset state. The dict default is None rather than {} because a mutable
    # class attribute would be shared by every analysis in the process.
    _results_epoch: int = 0
    _path_cache: Optional[dict] = None

    def __init__(self) -> None:
        self.ref_path:  Optional[str]      = None
        self.def_paths: List[str]          = []
        self._ref_image: Optional[np.ndarray] = None
        # One compact UI frame cache. A non-zero strain start frame is queried
        # repeatedly by the ROI, dynamic-ROI and parameters previews; rereading
        # and renormalising it on every control event made those screens lag.
        self._preview_frame_index: Optional[int] = None
        self._preview_frame_image: Optional[np.ndarray] = None
        self._roi_mask:  Optional[np.ndarray] = None
        # Accumulated strain is continuously seeded through this spatial
        # region from strain_start_frame onward. It is deliberately separate
        # from the analysis ROI used by every adjacent-frame DIC solve.
        self._strain_origin_mask: Optional[np.ndarray] = None
        self.strain_start_frame: int = 0
        self.params:  DICParams      = DICParams()
        self.results: List[PairResult] = []
        self.temporal_results = None
        self.temporal_pairs: list[tuple[int, int]] = []
        self.temporal_metadata: dict = {}
        self.fps: float = 1.0
        # Spatial calibration. Uncalibrated by default: the solver is pixel-native
        # and stays that way -- this only affects how results are presented.
        self.calibration: Calibration = Calibration()
        self.prefer_gpu: bool = True
        self.last_backend: str = "unknown"
        self._cancel: list = [False]

        # Incremental trajectory tracing. A path traced to frame N is a prefix
        # of the path to frame N+1, so scrubbing forward should extend the
        # accumulation rather than replay it from frame 0. Without this every
        # frame change re-walked the whole history for every marker, which is
        # what made streaklines unusable on a long sequence.
        # The epoch is bumped whenever the result sequence is rebuilt, so a
        # cached path can never outlive the data it was traced through.
        self._results_epoch = 0
        self._path_cache = {}
        # Schema-3 sessions keep datasets on disk and materialise only fields
        # that are actually viewed or exported. These proxies require the file
        # handle to remain open for the lifetime of the loaded session.
        self._hdf5_handle = None
        self.hdf5_lazy: bool = False

        # Human-readable notes about anything load_settings had to correct. The
        # console print alone was too easy to miss -- a silently repaired
        # setting is exactly the kind of thing that should be surfaced, since
        # the stored value was producing bad results. The UI drains this once at
        # startup.
        self.settings_notices: List[str] = []
        # Cleared if the settings file exists but cannot be parsed, which blocks
        # save_settings from overwriting it with defaults.
        self._settings_readable: bool = True

        # Operator overrides for the dynamic ROI, in reference-frame coordinates.
        # Arrays, so they live here rather than in the JSON-serialised params.
        self.dynamic_include_mask: Optional[np.ndarray] = None
        self.dynamic_exclude_mask: Optional[np.ndarray] = None
        # Exact displayed-frame overrides. Each sparse entry may contain an
        # ``include`` mask, an ``exclude`` mask and/or a normalised ``threshold``.
        # Frame 0 is the imported reference; result i enters displayed frame i+1.
        self.dynamic_frame_overrides: dict[int, dict[str, object]] = {}
        # Sparse keyframes used as defaults from their displayed frame onward.
        # Exact-frame entries above are layered on top, so local corrections win.
        self.dynamic_future_overrides: dict[int, dict[str, object]] = {}

        self.last_video_directory: str = os.path.expanduser("~")
        self.last_image_directory: str = os.path.expanduser("~")
        self.last_hdf5_directory: str = os.path.expanduser("~")

        self.load_settings()

    def _release_loaded_hdf5(self) -> None:
        handle = getattr(self, "_hdf5_handle", None)
        self._hdf5_handle = None
        self.hdf5_lazy = False
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def __del__(self):
        self._release_loaded_hdf5()

    def make_dynamic_roi(self) -> "DynamicROI":
        """Build the dynamic ROI from params + operator overrides.

        Single construction point so the CPU and GPU paths cannot drift apart
        in how they interpret the threshold and include/exclude masks.
        """
        return DynamicROI(
            self.params.dynamic_roi,
            keep_min_area_frac=getattr(self.params, "dynamic_roi_min_area_frac", 0.02),
            threshold=getattr(self.params, "dynamic_roi_threshold", None),
            include_mask=self.dynamic_include_mask,
            exclude_mask=self.dynamic_exclude_mask,
            roi_mask=self._roi_mask,
            fill_holes=getattr(self.params, "dynamic_roi_fill_holes", True),
        )

    def dynamic_threshold_for_frame(self, frame_index: int) -> Optional[float]:
        """Effective normalised threshold for one displayed sequence frame."""
        entry = self.dynamic_override_for_frame(frame_index)
        if "threshold" in entry and entry["threshold"] is not None:
            return float(np.clip(entry["threshold"], 0.0, 1.0))
        return getattr(self.params, "dynamic_roi_threshold", None)

    def dynamic_override_for_frame(self, frame_index: int) -> dict[str, object]:
        """Return inherited future defaults with the exact frame layered last."""
        index = int(frame_index)
        future = getattr(self, "dynamic_future_overrides", {})
        starts = [start for start in future if int(start) <= index]
        merged = dict(future[max(starts)]) if starts else {}
        merged.update(getattr(self, "dynamic_frame_overrides", {}).get(index, {}))
        return merged

    def apply_dynamic_frame_override(
            self, mask: Optional[np.ndarray], frame_index: int,
            *, clip_static: bool = False) -> Optional[np.ndarray]:
        """Apply exact-frame decisions after the global Dynamic ROI base."""
        if mask is None:
            return None
        out = np.asarray(mask, dtype=bool).copy()
        entry = self.dynamic_override_for_frame(frame_index)
        exclude = entry.get("exclude")
        include = entry.get("include")
        if bool(entry.get("replace", False)):
            out[:] = False
        if exclude is not None:
            arr = np.asarray(exclude, dtype=bool)
            if arr.shape != out.shape:
                raise ValueError("Frame Dynamic ROI exclude shape mismatch.")
            out &= ~arr
        if include is not None:
            arr = np.asarray(include, dtype=bool)
            if arr.shape != out.shape:
                raise ValueError("Frame Dynamic ROI include shape mismatch.")
            out |= arr
        if clip_static and self._roi_mask is not None:
            out &= self._roi_mask
        return out

    def reference_analysis_mask(self) -> Optional[np.ndarray]:
        """Mask shown by the dynamic-ROI editor on the zero-strain image.

        This is also the authoritative preview mask for the parameters page and
        zero-strain preview, so those screens cannot show different ROIs for
        the same configured frame.
        """
        base = self.strain_reference_image()
        if base is None:
            return None
        static = (self._roi_mask if self._roi_mask is not None
                  else np.ones(base.shape, dtype=bool))
        roi = self.make_dynamic_roi()
        roi.calibrate(base)
        frame_index = int(getattr(self, "strain_start_frame", 0))
        roi.set_threshold_normalised(
            self.dynamic_threshold_for_frame(frame_index))
        dynamic = roi.mask(base, reference_frame=True)
        if dynamic is None:
            return static.copy()
        return self.apply_dynamic_frame_override(
            dynamic, frame_index, clip_static=True)

    def frame_image(self, frame_index: int) -> Optional[np.ndarray]:
        """Load a full-sequence frame where 0 is the imported reference."""
        if self._ref_image is None:
            return None
        paths = getattr(self, "def_paths", [])
        index = int(frame_index)
        if index <= 0:
            return self._ref_image
        if index > len(paths):
            raise IndexError(f"Frame {index} is outside 0..{len(paths)}.")
        if (self._preview_frame_index == index and
                self._preview_frame_image is not None):
            return self._preview_frame_image
        image = _load_image(paths[index - 1])
        self._preview_frame_index = index
        self._preview_frame_image = image
        return image

    def strain_reference_image(self) -> Optional[np.ndarray]:
        """The image on which the strain origin and analysis ROI are drawn."""
        if self._ref_image is None:
            return None
        paths = getattr(self, "def_paths", [])
        index = min(max(0, int(getattr(self, "strain_start_frame", 0))),
                    len(paths))
        return self.frame_image(index)

    def set_reference(self, path: str) -> None:
        # A reference owns every spatially dependent state below it. Keeping an
        # old ROI override or completed result after selecting new footage made
        # later pages display a plausible mixture of two sessions.
        self._release_loaded_hdf5()
        self.results.clear()
        self._results_epoch += 1
        self.ref_path = path
        self._ref_image = _load_image(path)
        self._preview_frame_index = None
        self._preview_frame_image = None
        self._roi_mask = None
        self._strain_origin_mask = None
        self.strain_start_frame = 0
        self.dynamic_include_mask = None
        self.dynamic_exclude_mask = None
        self.dynamic_frame_overrides = {}
        self.dynamic_future_overrides = {}

    def add_deformed(self, path: str) -> None:
        # The existing result sequence no longer describes the input list once
        # a frame is added.
        self._release_loaded_hdf5()
        self.results.clear()
        self._results_epoch += 1
        self.def_paths.append(path)

    def clear_deformed(self) -> None:
        self._release_loaded_hdf5()
        self.def_paths.clear()
        self._preview_frame_index = None
        self._preview_frame_image = None
        self.results.clear()
        self._results_epoch += 1
        self.dynamic_frame_overrides = {}
        self.dynamic_future_overrides = {}

    def set_roi_mask(self, mask: np.ndarray) -> None:
        if self._ref_image is not None and mask.shape != self._ref_image.shape:
            raise ValueError(f"ROI mask shape {mask.shape} != reference {self._ref_image.shape}")
        self._release_loaded_hdf5()
        self._roi_mask = mask.astype(bool)
        if self._strain_origin_mask is not None:
            self._strain_origin_mask &= self._roi_mask
        self.dynamic_frame_overrides = {}
        self.dynamic_future_overrides = {}
        self.results.clear()
        self._results_epoch += 1

    def set_strain_origin_mask(self, mask: np.ndarray) -> None:
        if self._ref_image is not None and mask.shape != self._ref_image.shape:
            raise ValueError(
                f"Strain-origin mask shape {mask.shape} != reference "
                f"{self._ref_image.shape}")
        self._release_loaded_hdf5()
        origin = np.asarray(mask, dtype=bool)
        if self._roi_mask is not None:
            origin &= self._roi_mask
        self._strain_origin_mask = origin
        self.results.clear()
        self._results_epoch += 1

    def set_strain_origin_from_file(self, path: str) -> None:
        if self._ref_image is None:
            raise RuntimeError("Load reference image before setting strain origin.")
        mask = load_roi_mask(path, expected_shape=self._ref_image.shape)
        self.set_strain_origin_mask(mask)

    def clear_strain_origin(self) -> None:
        self._release_loaded_hdf5()
        self._strain_origin_mask = None
        self.results.clear()
        self._results_epoch += 1

    def set_roi_from_file(self, path: str) -> None:
        if self._ref_image is None:
            raise RuntimeError("Load reference image before setting ROI from file.")
        mask = load_roi_mask(path, expected_shape=self._ref_image.shape)
        self.set_roi_mask(mask)

    def set_full_roi(self) -> None:
        if self._ref_image is None:
            raise RuntimeError("Load reference first.")
        self.set_roi_mask(np.ones(self._ref_image.shape, dtype=bool))

    def clear_roi(self) -> None:
        """Clear ROI state and every result/override that depends on it."""
        self._release_loaded_hdf5()
        self._roi_mask = None
        self._strain_origin_mask = None
        self.dynamic_include_mask = None
        self.dynamic_exclude_mask = None
        self.dynamic_frame_overrides = {}
        self.dynamic_future_overrides = {}
        self.results.clear()
        self._results_epoch += 1

    @property
    def reference_image(self) -> Optional[np.ndarray]:
        return self._ref_image

    @property
    def roi_mask(self) -> Optional[np.ndarray]:
        return self._roi_mask

    @property
    def strain_origin_mask(self) -> Optional[np.ndarray]:
        return self._strain_origin_mask

    def cancel(self) -> None:
        self._cancel[0] = True

    def run(
            self,
            progress_cb: Optional[Callable[[float, str], None]] = None,
            seed_xy: Optional[tuple] = None,
            use_gpu: bool = False
    ) -> None:

        if use_gpu:
            if not _HAS_CUPY:
                raise RuntimeError("GPU acceleration requested but CuPy is not installed or NVIDIA drivers are missing.")
            self.last_backend = "gpu"
            self._run_gpu(progress_cb, seed_xy)
            return

        self.last_backend = "cpu"
        self._cancel[0] = False
        self._release_loaded_hdf5()
        self.results.clear()
        self._results_epoch += 1
        if self._ref_image is None:
            raise RuntimeError("No reference image.")
        if not self.def_paths:
            raise RuntimeError("No deformed images.")
        if self._roi_mask is None:
            self.set_full_roi()

        ref = self._ref_image
        prev_image = ref
        mask = self._roi_mask
        n = len(self.def_paths)

        guess_u, guess_v = 0.0, 0.0

        # Calibrate the texture threshold ONCE on the selected zero-strain frame. The old
        # per-frame _compute_dynamic_mask recalibrated its scale and Otsu
        # threshold on every image, so "enough texture to correlate" drifted
        # frame to frame and the mask edge flickered -- see DynamicROI's
        # docstring, which the GPU path already follows.
        dyn_roi = self.make_dynamic_roi()
        dyn_roi.calibrate(self.strain_reference_image())

        for i, def_path in enumerate(self.def_paths):
            if self._cancel[0]:
                break

            def pair_cb(frac, msg, _i=i, _n=n):
                if progress_cb:
                    progress_cb(0.90 * (_i / _n) + frac * (0.90 / _n), f"[{_i + 1}/{_n}] {msg}")

            pair_cb(0.0, f"Loading {os.path.basename(def_path)}…")
            cur = _load_image(def_path)
            if cur.shape != ref.shape:
                raise ValueError(f"Shape mismatch: {def_path}")

            t0 = time.perf_counter()
            # The CPU solver imports SciPy's interpolation/linear-algebra stack;
            # defer that sizeable runtime until analysis genuinely starts.
            from .rg_dic import run_rg_dic
            dic = run_rg_dic(
                prev_image, cur, mask, self.params,
                seed_xy=seed_xy, progress_cb=pair_cb, cancel_flag=self._cancel,
                guess_u=guess_u, guess_v=guess_v,
                use_gpu=use_gpu,
            )
            elapsed = time.perf_counter() - t0

            valid = _finite_measurement_mask(
                dic.analyzed, dic.u, dic.v, dic.corr)

            # This grid belongs to the immediately previous image, not frame 0.
            # Sample the current dynamic mask at x+du,y+dv only. Accumulated
            # material positions belong exclusively to strain transport and
            # must never decide whether a fresh pairwise measurement survives.
            frame_index = i + 1
            dyn_roi.set_threshold_normalised(
                self.dynamic_threshold_for_frame(frame_index))
            d_mask = dyn_roi.mask(cur, reference_frame=False)
            frame_override = self.dynamic_override_for_frame(frame_index)
            valid = _dynamic_measurement_mask(
                valid, d_mask, dic.u, dic.v,
                include_mask=self.dynamic_include_mask,
                exclude_mask=self.dynamic_exclude_mask,
                frame_include_mask=frame_override.get("include"),
                frame_exclude_mask=frame_override.get("exclude"),
                replace_base=bool(frame_override.get("replace", False)))

            _mask_invalid(valid, dic.u, dic.v, dic.du_dx, dic.du_dy,
                          dic.dv_dx, dic.dv_dy, dic.corr)

            if valid.any():
                guess_u = float(np.median(dic.u[valid]))
                guess_v = float(np.median(dic.v[valid]))

            u_out = _result_f32(np.where(valid, dic.u, np.nan))
            v_out = _result_f32(np.where(valid, dic.v, np.nan))
            gradients = self._pair_displacement_gradients(u_out, v_out, valid)
            indices = np.flatnonzero(valid.reshape(-1)).astype(np.uint32, copy=False)

            self.results.append(PairResult(
                image_path=def_path,
                u=_compact_field(u_out, valid, indices),
                v=_compact_field(v_out, valid, indices),
                Exx=None, Exy=None, Eyy=None, Eeff=None,
                du_dx=_compact_field(gradients["du_dx"], valid, indices),
                du_dy=_compact_field(gradients["du_dy"], valid, indices),
                dv_dx=_compact_field(gradients["dv_dx"], valid, indices),
                dv_dy=_compact_field(gradients["dv_dy"], valid, indices),
                corr=_compact_field(dic.corr, valid, indices),
                valid=CompactMask(ref.shape, indices), elapsed=elapsed,
            ))

            # Immediate-frame analysis: the current image becomes the next
            # reference. Previous displacement is used only as a seed hint.
            prev_image = cur

        if not self._cancel[0] and self.results:
            self._compute_incremental_displacements()
            self._compute_velocities_and_rates(progress_cb)
            self._transport_accumulated_strain(progress_cb)

        if progress_cb:
            progress_cb(1.0, "Complete.")

    def _run_gpu(
            self,
            progress_cb: Optional[Callable[[float, str], None]] = None,
            seed_xy: Optional[tuple] = None
    ) -> None:
        """
        Executes the Wavefront GPU pipeline with intelligent Global Seed Tracking and Auto-Fallback.
        """
        self._cancel[0] = False
        self._release_loaded_hdf5()
        self.results.clear()
        self._results_epoch += 1

        if self._ref_image is None or not self.def_paths:
            raise RuntimeError("Missing reference or deformed images.")
        if self._roi_mask is None:
            self.set_full_roi()

        n_frames = len(self.def_paths)

        if progress_cb:
            progress_cb(0.0, "Initializing GPU solver and precomputing reference...")

        try:
            from .icgn_gpu import GPUWavefrontDIC
            gpu_solver = GPUWavefrontDIC(self.params)
            gpu_solver.precompute_reference(self._ref_image, self._roi_mask)
        except Exception as e:
            raise RuntimeError(f"GPU initialization failed: {e}")

        if seed_xy is None:
            ys_roi, xs_roi = np.where(self._roi_mask)
            if len(xs_roi) == 0:
                raise ValueError("ROI mask is empty.")
            seed_xy = (int(xs_roi.mean()), int(ys_roi.mean()))

        dist_sq = (gpu_solver.gx_flat - seed_xy[0])**2 + (gpu_solver.gy_flat - seed_xy[1])**2
        seed_idx = int(np.argmin(dist_sq))
        actual_seed_x = int(gpu_solver.gx_flat[seed_idx])
        actual_seed_y = int(gpu_solver.gy_flat[seed_idx])

        # CORRECTED EXPECTED SUBSETS: Count only subsets strictly inside the ROI
        expected_subsets = int(gpu_solver.valid_mask.sum())

        from .ncc import ncc_initial_guess

        warm_start_active = False
        guess_u, guess_v = 0.0, 0.0

        prev_image = self._ref_image  # first reference is frame 0
        dyn_roi = self.make_dynamic_roi()
        dyn_roi.calibrate(self.strain_reference_image())

        H_img, W_img = self._ref_image.shape
        self.H_ref, self.W_ref = H_img, W_img

        for i, def_path in enumerate(self.def_paths):
            if self._cancel[0]: break
            t0 = time.perf_counter()

            if progress_cb:
                progress_cb(0.90 * (i / n_frames), f"[{i + 1}/{n_frames}] Loading {os.path.basename(def_path)}...")

            cur_image = _load_image(def_path)

            # Pairwise DIC is spatially reinitialised on the previous frame.
            # Keep the seed in that image-space ROI; accumulated material
            # coordinates are strain state, not solver geometry.
            current_seed_x = actual_seed_x
            current_seed_y = actual_seed_y

            if not warm_start_active:
                if progress_cb:
                    progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.3, f"[{i + 1}/{n_frames}] Global NCC Search...")

                guess_u, guess_v, _ = ncc_initial_guess(
                    prev_image, cur_image, current_seed_x, current_seed_y,
                    self.params.subset_radius, self.params.search_radius,
                    guess_u, guess_v
                )
                seed_p = np.array([guess_u, guess_v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

                if progress_cb:
                    progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.6, f"[{i + 1}/{n_frames}] Growing Wavefront...")

                inc_u, inc_v, inc_du_dx, inc_du_dy, inc_dv_dx, inc_dv_dy, corr_f = gpu_solver.solve_frame(
                    cur_image, seed_idx=seed_idx, seed_p=seed_p, warm_start=False
                )
                warm_start_active = True

            else:
                if progress_cb:
                    progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.5, f"[{i + 1}/{n_frames}] Batched temporal tracking...")

                inc_u, inc_v, inc_du_dx, inc_du_dy, inc_dv_dx, inc_dv_dy, corr_f = gpu_solver.solve_frame(
                    cur_image, warm_start=True
                )

                valid_count = np.count_nonzero(
                    np.isfinite(inc_u[self._roi_mask]) &
                    np.isfinite(inc_v[self._roi_mask]) &
                    np.isfinite(corr_f[self._roi_mask]))
                survival_rate = valid_count / max(1, expected_subsets)

                if survival_rate < 0.60:
                    print(f"\n[AUTO-FALLBACK] Frame {i+1} collapsed (Survival: {survival_rate*100:.1f}%). Re-running with targeted NCC...")

                    if progress_cb:
                        progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.7, f"[{i + 1}/{n_frames}] Jolt detected. Repairing via NCC...")

                    guess_u, guess_v, _ = ncc_initial_guess(
                        prev_image, cur_image, current_seed_x, current_seed_y,
                        self.params.subset_radius, self.params.search_radius,
                        guess_u, guess_v
                    )
                    seed_p = np.array([guess_u, guess_v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

                    inc_u, inc_v, inc_du_dx, inc_du_dy, inc_dv_dx, inc_dv_dy, corr_f = gpu_solver.solve_frame(
                        cur_image, seed_idx=seed_idx, seed_p=seed_p, warm_start=False
                    )

            # Dynamic ROI rejection must happen BEFORE strain accumulation. In
            # the old order, a subset that left the frame could contribute one
            # huge affine increment permanently and was only hidden afterwards.
            frame_index = i + 1
            dyn_roi.set_threshold_normalised(
                self.dynamic_threshold_for_frame(frame_index))
            d_mask = dyn_roi.mask(cur_image, reference_frame=False)
            frame_override = self.dynamic_override_for_frame(frame_index)
            measurement_valid = _finite_measurement_mask(
                np.ones(inc_u.shape, dtype=bool), inc_u, inc_v, corr_f)
            measurement_valid = _dynamic_measurement_mask(
                measurement_valid, d_mask, inc_u, inc_v,
                include_mask=self.dynamic_include_mask,
                exclude_mask=self.dynamic_exclude_mask,
                frame_include_mask=frame_override.get("include"),
                frame_exclude_mask=frame_override.get("exclude"),
                replace_base=bool(frame_override.get("replace", False)))

            _mask_invalid(measurement_valid, inc_u, inc_v, inc_du_dx,
                          inc_du_dy, inc_dv_dx, inc_dv_dy, corr_f)

            frame_valid = measurement_valid

            # Track seed displacement for NCC initial guess (incremental, small)
            if (np.isfinite(inc_u[actual_seed_y, actual_seed_x]) and
                    np.isfinite(inc_v[actual_seed_y, actual_seed_x])):
                guess_u = float(inc_u[actual_seed_y, actual_seed_x])
                guess_v = float(inc_v[actual_seed_y, actual_seed_x])

            # --- Updated Lagrangian: swap reference to current frame for next iteration ---
            gpu_solver.update_reference_image(cur_image)
            prev_image = cur_image

            elapsed = time.perf_counter() - t0

            u_out = _result_f32(np.where(frame_valid, inc_u, np.nan))
            v_out = _result_f32(np.where(frame_valid, inc_v, np.nan))
            gradients = self._pair_displacement_gradients(
                u_out, v_out, frame_valid)
            indices = np.flatnonzero(frame_valid.reshape(-1)).astype(
                np.uint32, copy=False)
            self.results.append(PairResult(
                image_path=def_path,
                u=_compact_field(u_out, frame_valid, indices),
                v=_compact_field(v_out, frame_valid, indices),
                Exx=None, Exy=None, Eyy=None, Eeff=None,
                du_dx=_compact_field(gradients["du_dx"], frame_valid, indices),
                du_dy=_compact_field(gradients["du_dy"], frame_valid, indices),
                dv_dx=_compact_field(gradients["dv_dx"], frame_valid, indices),
                dv_dy=_compact_field(gradients["dv_dy"], frame_valid, indices),
                corr=_compact_field(corr_f, frame_valid, indices),
                valid=CompactMask(self._ref_image.shape, indices), elapsed=elapsed,
            ))

            # CuPy's pool deliberately caches freed rescue/IC-GN workspaces.
            # Their shapes vary with each frame's active subset count, so a
            # long sequence can retain many large, unusable blocks and appear
            # to leak gigabytes. Persistent solver arrays remain referenced;
            # this releases only blocks that are no longer in use.
            gpu_solver.release_temporary_memory()

        # The GPU solver owns persistent reference coefficients and warm-start
        # state. Release it before CPU velocity/strain post-processing, whose
        # host-memory peak otherwise overlaps the still-live CUDA allocations
        # and pinned-memory pool on long sequences.
        gpu_solver.release_temporary_memory()
        del gpu_solver
        try:
            import cupy as cp
            cp.cuda.get_current_stream().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

        if not self._cancel[0] and self.results:
            self._compute_incremental_displacements()
            self._compute_velocities_and_rates(progress_cb)
            self._transport_accumulated_strain(progress_cb)

        if progress_cb:
            progress_cb(1.0, "Complete.")

    def _pair_displacement_gradients(
            self, u: np.ndarray, v: np.ndarray,
            valid: np.ndarray) -> dict[str, np.ndarray]:
        """Fit one pair's displacement gradient with the shared validity rule."""
        from .strain import compute_velocity_strains
        fitted = compute_velocity_strains(
            u, v, np.asarray(valid, dtype=bool),
            self.params.effective_strain_window(),
            self.params.subset_spacing)
        return {
            "du_dx": _result_f32(fitted["dVx_dx"]),
            "du_dy": _result_f32(fitted["dVx_dy"]),
            "dv_dx": _result_f32(fitted["dVy_dx"]),
            "dv_dy": _result_f32(fitted["dVy_dy"]),
        }

    @staticmethod
    def _set_strain_fields(res: PairResult,
                           fields: dict[str, np.ndarray]) -> None:
        # Only the selected Green-Lagrange strain convention is retained.  The
        # tracker still computes its infinitesimal state internally, but keeping
        # both formulations for every frame doubled the largest result family.
        names = ("Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl")
        if all(isinstance(fields[name], CompactField) for name in names):
            for name in names:
                setattr(res, name, fields[name])
        else:
            dense = {name: np.asarray(fields[name]) for name in names}
            complete = np.logical_and.reduce(
                [np.isfinite(dense[name]) for name in names])
            indices = np.flatnonzero(complete.reshape(-1)).astype(
                np.uint32, copy=False)
            for name in names:
                setattr(res, name,
                        CompactField.from_dense(dense[name], indices=indices))
        res.Exx_inf = res.Eyy_inf = res.Exy_inf = res.Eeff_inf = None
        res.Gxy_inf = res.Gxy_gl = None
        res.Exx = res.Eyy = res.Exy = res.Eeff = None

    @staticmethod
    def _release_transport_work_fields(res: PairResult) -> None:
        """Drop solver-only fields after their one strain-transport use."""
        res.du_dx = res.du_dy = res.dv_dx = res.dv_dy = None
        res.corr = None
        # Diagonal names are aliases of the displayed rates. Cross velocity
        # gradients and engineering shear are not displayed or exported.
        res.dVx_dx = res.Exx_rate
        res.dVy_dy = res.Eyy_rate
        res.dVx_dy = res.dVy_dx = None
        res.Gxy_rate = None

    def _transport_accumulated_strain(
            self,
            progress_cb: Optional[Callable[[float, str], None]] = None
            ) -> None:
        """Advect paths and freeze each spatial strain cell on first arrival."""
        if not self.results or self._roi_mask is None:
            return
        origin = getattr(self, "_strain_origin_mask", None)
        # Headless/legacy callers that predate the origin control retain useful
        # behaviour. The GUI requires an explicit origin before analysis.
        if origin is None or not np.any(origin):
            origin = self._roi_mask
        # SciPy-backed path tracking is analysis-only. Keep it out of the idle
        # UI process until accumulated strain is actually computed.
        from .strain_accum import StrainPathTracker

        tracker = StrainPathTracker(
            self._roi_mask.shape, origin, self._roi_mask,
            self.params.subset_radius, self.params.subset_spacing)
        start = int(np.clip(getattr(self, "strain_start_frame", 0),
                            0, len(self.results)))
        blank = {name: CompactField.empty(self._roi_mask.shape)
                 for name in ("Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl")}
        # The displayed accumulated-strain region is a first-arrival history:
        # once a valid path reaches a DIC grid element, its complete strain state
        # is frozen there. Later particles may only expand into unseen elements;
        # they never repaint a position that has already been encountered.
        persistent_names = (
            "Exx_inf", "Eyy_inf", "Exy_inf", "Eeff_inf",
            "Exx_gl", "Eyy_gl", "Exy_gl")
        swept = {name: np.full(self._roi_mask.shape, np.nan, dtype=np.float32)
                 for name in persistent_names}
        encountered = np.zeros(self._roi_mask.shape, dtype=bool)

        def deposit(snapshot: dict[str, np.ndarray]) -> None:
            unseen = self._roi_mask & ~encountered
            if not unseen.any():
                return
            # Treat the strain tensor/equivalent value as one state. A cell is
            # marked encountered only when every stored component is finite, so
            # partially invalid data cannot permanently block a later valid hit.
            new = unseen.copy()
            for name in persistent_names:
                new &= np.isfinite(np.asarray(snapshot[name]))
            if not new.any():
                return
            for name in persistent_names:
                current = np.asarray(snapshot[name])
                swept[name][new] = current[new]
            encountered[new] = True

        def swept_copy() -> dict[str, np.ndarray]:
            # _set_strain_fields immediately packs the finite samples, producing
            # an immutable snapshot without copying seven full-screen arrays.
            return {
                "Exx_gl": swept["Exx_gl"],
                "Eyy_gl": swept["Eyy_gl"],
                "Exy_gl": swept["Exy_gl"],
                "Eeff_gl": swept["Eeff_inf"],
            }

        for i, res in enumerate(self.results):
            if progress_cb:
                progress_cb(
                    0.97 + 0.025 * (i / max(1, len(self.results))),
                    f"[{i + 1}/{len(self.results)}] Transporting accumulated strain…")
            if i < start - 1:
                self._set_strain_fields(res, blank)
                self._release_transport_work_fields(res)
                continue
            if start > 0 and i == start - 1:
                # Result i is displayed on full-sequence frame i+1, which is
                # the selected zero-strain frame when i == start-1.
                tracker.seed()
                deposit(tracker.snapshot())
                self._set_strain_fields(res, swept_copy())
                self._release_transport_work_fields(res)
                continue

            pair_valid = (res.valid if res.valid is not None else
                          (np.isfinite(res.u) & np.isfinite(res.v)))
            tracker.seed(pair_valid)
            # Deposit the source position before it moves as well as the
            # destination afterwards. This preserves the initial line on a
            # start-frame-0 run and makes the coloured region expand rather
            # than merely translate with the live particles.
            deposit(tracker.snapshot())
            tracker.advance(
                np.asarray(res.u), np.asarray(res.v),
                np.asarray(res.du_dx), np.asarray(res.du_dy),
                np.asarray(res.dv_dx), np.asarray(res.dv_dy))
            deposit(tracker.snapshot())
            self._set_strain_fields(res, swept_copy())
            self._release_transport_work_fields(res)

    def _compute_incremental_displacements(self) -> None:
        """Populate compatibility aliases for immediate displacement.

        u/v already mean previous-frame -> current-frame motion.  Subtracting
        adjacent result fields here would incorrectly produce acceleration-like
        data, so u_inc/v_inc are direct copies.
        """
        for res in self.results:
            if isinstance(res.u, CompactField) and isinstance(res.v, CompactField):
                res.u_inc = res.u
                res.v_inc = res.v
                if np.array_equal(res.u.indices, res.v.indices):
                    values = np.hypot(res.u.values, res.v.values).astype(
                        np.float32, copy=False)
                    res.mag_inc = CompactField(
                        res.u.shape, res.u.indices, values)
                else:
                    dense = np.hypot(np.asarray(res.u), np.asarray(res.v))
                    res.mag_inc = CompactField.from_dense(dense)
                continue
            valid = np.isfinite(res.u) & np.isfinite(res.v)
            if res.valid is not None:
                valid &= res.valid
            # u/v already carry the interval displacement. Normalise their mask
            # once, then share the exact arrays with the compatibility aliases
            # instead of retaining two additional full-frame copies per result.
            if not np.all(valid == (np.isfinite(res.u) & np.isfinite(res.v))):
                res.u = _result_f32(np.where(valid, res.u, np.nan))
                res.v = _result_f32(np.where(valid, res.v, np.nan))
            else:
                res.u = _result_f32(res.u)
                res.v = _result_f32(res.v)
            res.u_inc = res.u
            res.v_inc = res.v
            with np.errstate(invalid="ignore"):
                res.mag_inc = _result_f32(np.sqrt(res.u ** 2 + res.v ** 2))

    def _compute_velocities_and_rates(self, progress_cb: Optional[Callable[[float, str], None]] = None) -> None:
        N = len(self.results)
        if N < 1: return
        dt = 1.0 / max(self.fps, 1e-9)

        # Each result is one measured interval, so its instantaneous mean
        # velocity is displacement divided by that interval's duration.
        for res in self.results:
            if isinstance(res.u, CompactField) and isinstance(res.v, CompactField):
                common = np.intersect1d(
                    res.u.indices, res.v.indices, assume_unique=True)
                up = np.searchsorted(res.u.indices, common)
                vp = np.searchsorted(res.v.indices, common)
                vx = res.u.values[up] / np.float32(dt)
                vy = res.v.values[vp] / np.float32(dt)
                res.Vx = CompactField(res.u.shape, common, vx)
                res.Vy = CompactField(res.u.shape, common, vy)
                res.Veff = CompactField(
                    res.u.shape, common,
                    np.hypot(vx, vy).astype(np.float32, copy=False))
                continue
            valid = np.isfinite(res.u) & np.isfinite(res.v)
            if res.valid is not None:
                valid &= res.valid
            res.Vx = _result_f32(np.where(valid, res.u / dt, np.nan))
            res.Vy = _result_f32(np.where(valid, res.v / dt, np.nan))
            with np.errstate(invalid="ignore"):
                res.Veff = _result_f32(np.sqrt(res.Vx ** 2 + res.Vy ** 2))

        from .strain import von_mises_equivalent

        for i, res in enumerate(self.results):
            if progress_cb:
                p = 0.90 + 0.07 * (i / max(1, N))
                progress_cb(p, f"[{i + 1}/{N}] Computing strain rates…")

            if (isinstance(res.Vx, CompactField) and
                    all(isinstance(getattr(res, name, None), CompactField)
                        for name in ("du_dx", "du_dy", "dv_dx", "dv_dy"))):
                sources = [res.Vx, res.Vy, res.du_dx, res.du_dy,
                           res.dv_dx, res.dv_dy]
                common = sources[0].indices
                for source in sources[1:]:
                    common = np.intersect1d(
                        common, source.indices, assume_unique=True)

                def packed(source):
                    return source.values[np.searchsorted(source.indices, common)]

                h11 = packed(res.du_dx) / np.float32(dt)
                h12 = packed(res.du_dy) / np.float32(dt)
                h21 = packed(res.dv_dx) / np.float32(dt)
                h22 = packed(res.dv_dy) / np.float32(dt)
                exy = np.float32(0.5) * (h12 + h21)
                eeff = von_mises_equivalent(h11, h22, exy).astype(
                    np.float32, copy=False)
                res.dVx_dx = CompactField(res.u.shape, common, h11)
                res.dVx_dy = CompactField(res.u.shape, common, h12)
                res.dVy_dx = CompactField(res.u.shape, common, h21)
                res.dVy_dy = CompactField(res.u.shape, common, h22)
                res.Exx_rate = res.dVx_dx
                res.Exy_rate = CompactField(res.u.shape, common, exy)
                res.Gxy_rate = res.Exy_rate.scaled(2.0)
                res.Eyy_rate = res.dVy_dy
                res.Eeff_rate = CompactField(res.u.shape, common, eeff)
                continue

            valid = np.isfinite(res.Vx) & np.isfinite(res.Vy)
            # Spatial differentiation is linear: grad(u/dt) = grad(u)/dt.
            # Reusing the pair gradients guarantees strain transport and strain
            # rate read the same measurement and avoids a second large fit.
            gradients_available = all(
                getattr(res, name, None) is not None
                for name in ("du_dx", "du_dy", "dv_dx", "dv_dy"))
            if gradients_available:
                dVx_dx = np.where(valid, np.asarray(res.du_dx) / dt, np.nan)
                dVx_dy = np.where(valid, np.asarray(res.du_dy) / dt, np.nan)
                dVy_dx = np.where(valid, np.asarray(res.dv_dx) / dt, np.nan)
                dVy_dy = np.where(valid, np.asarray(res.dv_dy) / dt, np.nan)
                Exy_rate = 0.5 * (dVx_dy + dVy_dx)
                rates = {
                    "dVx_dx": dVx_dx, "dVx_dy": dVx_dy,
                    "dVy_dx": dVy_dx, "dVy_dy": dVy_dy,
                    "Exx_rate": dVx_dx, "Eyy_rate": dVy_dy,
                    "Exy_rate": Exy_rate, "Gxy_rate": 2.0 * Exy_rate,
                    "Eeff_rate": von_mises_equivalent(
                        dVx_dx, dVy_dy, Exy_rate),
                }
            else:
                from .strain import compute_velocity_strains
                rates = compute_velocity_strains(
                    res.Vx, res.Vy, valid,
                    self.params.effective_strain_window(),
                    self.params.subset_spacing)

            # Exx_rate and Eyy_rate are exactly the corresponding diagonal
            # velocity gradients, so share those arrays too.
            res.dVx_dx = _result_f32(rates["dVx_dx"])
            res.dVx_dy = _result_f32(rates["dVx_dy"])
            res.dVy_dx = _result_f32(rates["dVy_dx"])
            res.dVy_dy = _result_f32(rates["dVy_dy"])
            res.Exx_rate = res.dVx_dx
            res.Exy_rate = _result_f32(rates["Exy_rate"])
            res.Gxy_rate = _result_f32(rates["Gxy_rate"])
            res.Eyy_rate = res.dVy_dy
            res.Eeff_rate = _result_f32(rates["Eeff_rate"])

    # ------------------------------------------------------------------
    # Marker-seeded trajectories
    # ------------------------------------------------------------------
    # Displacement fields are only populated at correlation grid points, so a
    # marker dropped at an arbitrary pixel needs local interpolation rather than
    # a direct lookup.

    def _sample_sparse(self, arr: np.ndarray, x: float, y: float,
                       search: Optional[int] = None) -> float:
        """Inverse-distance sample of a sparse (NaN-filled) field at a float position."""
        if arr is None:
            return float("nan")
        H, W = arr.shape
        if search is None:
            search = max(4, int(getattr(self.params, "subset_spacing", 3)) * 3)
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - search), min(W, xi + search + 1)
        y0, y1 = max(0, yi - search), min(H, yi + search + 1)
        if x1 <= x0 or y1 <= y0:
            return float("nan")
        if isinstance(arr, CompactField):
            # Indices are flat row-major and sorted, so each image row occupies
            # one contiguous run. Binary-searching the run for each row of the
            # window touches only the handful of subsets actually inside it.
            #
            # The previous version divided and modded EVERY index in the frame
            # -- around 100k subsets on a 1 MP field -- to keep the dozen that
            # fall in a 19x19 window, and it did that once per marker per frame.
            # Tracing five markers over a few hundred frames turned that into
            # hundreds of millions of operations and the window stopped
            # responding, which is what "streaklines crash the app" was.
            rows = np.arange(y0, y1, dtype=np.int64) * W
            lo = np.searchsorted(arr.indices, rows + x0, side="left")
            hi = np.searchsorted(arr.indices, rows + x1, side="left")
            counts = hi - lo
            total = int(counts.sum())
            if total == 0:
                return float("nan")
            # Ragged concatenation of the per-row runs without a Python loop:
            # two vectorised searchsorted calls and one arange, rather than one
            # scalar searchsorted per row. This is called twice per marker per
            # frame, so per-call overhead is what actually matters here.
            starts = np.repeat(lo, counts)
            within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            sel = starts + within
            vals = arr.values[sel]
            finite = np.isfinite(vals)
            if not finite.any():
                return float("nan")
            sel, vals = sel[finite], vals[finite]
            flat = arr.indices[sel].astype(np.int64, copy=False)
            ys, xs = flat // W, flat % W
            return self._interpolate_at(xs, ys, vals, x, y)
        win = arr[y0:y1, x0:x1]
        fin = np.isfinite(win)
        if not fin.any():
            return float("nan")
        ys, xs = np.nonzero(fin)
        return self._interpolate_at(xs + x0, ys + y0, win[fin], x, y)

    @staticmethod
    def _interpolate_at(xs, ys, vals, x: float, y: float) -> float:
        """Value of a scattered field at (x, y), by a local weighted plane fit.

        Displacement varies smoothly and close to linearly over a few subset
        spacings, so a plane through the neighbourhood is the right local model
        and it reproduces a linear field exactly.

        Inverse-distance weighting, used here previously, does not: on a regular
        grid it returns a distance-weighted blend that sits off the true value
        whenever the sample falls between centres. On a pure shear field that
        was a 1.6% error at the midpoint between rows. It matters more than the
        size suggests, because trajectory tracing sums the sampled displacement
        frame after frame -- the error is a bias, not noise, so it accumulates
        linearly rather than averaging out, and the traced path walks away from
        the material point it is supposed to follow.

        The design matrix is centred on the query point, so the constant term of
        the fit is the value there. Weighting by inverse distance keeps nearby
        centres dominant without breaking linear exactness -- for a truly linear
        field every weighting recovers the same plane.

        Falls back to inverse-distance weighting when there are too few points
        to determine a plane, or when they are degenerate (all in one row, say,
        at the edge of the analysed region).
        """
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        vals = np.asarray(vals, dtype=np.float64)

        d2 = (xs - x) ** 2 + (ys - y) ** 2
        if d2.size == 0:
            return float("nan")
        nearest = int(np.argmin(d2))
        if d2[nearest] < 1e-9:
            return float(vals[nearest])

        # A dozen points spans a couple of subset spacings in each direction:
        # enough to determine the plane, tight enough that the linear model
        # still holds across it.
        k = min(12, d2.size)
        idx = np.argpartition(d2, k - 1)[:k]
        xk, yk, vk, d2k = xs[idx], ys[idx], vals[idx], d2[idx]

        if k >= 3:
            # Solve the weighted normal equations directly. lstsq runs an SVD
            # per sample, which is four times the cost here and this is called
            # once per marker per field per frame -- enough to stall a scrub on
            # its own. The system is 3x3 and symmetric, so forming it from
            # reductions and solving once is both exact and cheap.
            dx, dy = xk - x, yk - y
            w = 1.0 / np.sqrt(d2k)
            wv = w * vk
            s00 = w.sum()
            s01 = (w * dx).sum()
            s02 = (w * dy).sum()
            s11 = (w * dx * dx).sum()
            s12 = (w * dx * dy).sum()
            s22 = (w * dy * dy).sum()
            m = np.array([[s00, s01, s02],
                          [s01, s11, s12],
                          [s02, s12, s22]])
            rhs = np.array([wv.sum(), (wv * dx).sum(), (wv * dy).sum()])
            det = float(np.linalg.det(m))
            # Reject a system the points cannot determine -- all on one row, or
            # collinear at the edge of the analysed region -- rather than
            # amplifying noise through a near-singular solve.
            scale = float(s00 * s11 * s22)
            if np.isfinite(det) and scale > 0.0 and abs(det) > 1e-9 * scale:
                try:
                    coef = np.linalg.solve(m, rhs)
                    if np.isfinite(coef[0]):
                        # The design is centred on the query point, so the
                        # constant term is the value there.
                        return float(coef[0])
                except np.linalg.LinAlgError:
                    pass

        w = 1.0 / d2k
        return float((vk * w).sum() / w.sum())

    def reference_from_current(self, x: float, y: float, frame_idx: int
                              ) -> Optional[tuple[tuple[float, float], float]]:
        """
        Map a click on the DISPLAYED (deformed) frame back to reference coordinates.

        Markers are stored in reference coordinates so that a marker follows the
        same material point for the whole sequence. Without this, a marker placed
        while scrubbed to frame 200 would be interpreted as a reference-frame
        position and would track the wrong material.

        Returns ((x_ref, y_ref), residual_px) or None.
        """
        if not self.results:
            return None
        idx = max(0, min(int(frame_idx), len(self.results) - 1))

        # Accumulate over the subset centres, never over the full image.
        #
        # This used to build three full-resolution arrays and then, for every
        # frame in the history, densify u and v twice each -- six megabyte-scale
        # allocations per frame. At a few hundred frames a single click took
        # seconds and the application stopped responding. Subset centres are a
        # fraction of a percent of the pixels, and they are all this needs.
        flat, u_cum, v_cum = self._cumulative_displacement(idx)
        if flat is None or flat.size == 0:
            return None

        W = int(self.results[0].u.shape[1])
        ys = (flat // W).astype(np.float64)
        xs = (flat % W).astype(np.float64)
        d2 = (xs + u_cum - x) ** 2 + (ys + v_cum - y) ** 2
        i = int(np.argmin(d2))
        return (float(xs[i]), float(ys[i])), float(np.sqrt(d2[i]))

    def _cumulative_displacement(self, idx: int):
        """Total displacement at each subset centre tracked through frame `idx`.

        Returns (flat_indices, u_cumulative, v_cumulative) over the points that
        stayed valid for the whole history, or (None, None, None) if none did.

        Works entirely on the packed values. Points that drop out are removed
        from the running set rather than masked in a dense array, so the work
        shrinks as the sequence goes on instead of staying at full frame size.
        """
        if not self.results:
            return None, None, None
        idx = max(0, min(int(idx), len(self.results) - 1))

        def _packed(field, shape):
            """(sorted flat indices, values) for a field, compact or dense."""
            if isinstance(field, CompactField):
                return field.indices.astype(np.int64, copy=False), field.values
            arr = np.asarray(field)
            finite = np.isfinite(arr)
            return np.flatnonzero(finite.reshape(-1)), arr.reshape(-1)[finite.reshape(-1)]

        first = self.results[0]
        shape = first.u.shape
        flat, u_vals = _packed(first.u, shape)
        _, v_vals = _packed(first.v, shape)
        if flat.size == 0:
            return None, None, None

        u_cum = np.zeros(flat.size, dtype=np.float64)
        v_cum = np.zeros(flat.size, dtype=np.float64)

        for k in range(0, idx + 1):
            res = self.results[k]
            f_u, u_k = _packed(res.u, shape)
            f_v, v_k = _packed(res.v, shape)

            # Keep only points this frame also measured.
            #
            # Almost always every frame carries the identical subset grid, and
            # then the intersection is the identity -- worth detecting, because
            # searchsorted over ~100k points on every frame of a long sequence
            # is most of the cost of placing a single marker. array_equal is one
            # cheap vectorised comparison against that.
            def _align(f_k):
                if f_k.size == flat.size and np.array_equal(f_k, flat):
                    return None                      # identity: index directly
                pos = np.searchsorted(f_k, flat)
                np.clip(pos, 0, max(0, f_k.size - 1), out=pos)
                return pos

            pos = _align(f_u)
            pos_v = pos if f_v is f_u else _align(f_v)

            if pos is None:
                keep = np.ones(flat.size, dtype=bool)
                du = u_k
            else:
                keep = (f_u.size > 0) & (f_u[pos] == flat)
                du = np.where(keep, u_k[pos], np.nan)

            if pos_v is None:
                dv = v_k
            else:
                keep = keep & ((f_v.size > 0) & (f_v[pos_v] == flat))
                dv = np.where(keep, v_k[pos_v], np.nan)

            keep = keep & np.isfinite(du) & np.isfinite(dv)
            if not keep.any():
                return None, None, None

            flat = flat[keep]
            u_cum = u_cum[keep] + du[keep]
            v_cum = v_cum[keep] + dv[keep]

        return flat, u_cum, v_cum

    def marker_positions(self, seeds, frame_idx: int) -> list[Optional[tuple[float, float]]]:
        """Where each reference-frame marker sits on the displayed frame."""
        if not self.results or not seeds:
            return [None] * len(seeds)
        idx = max(0, min(int(frame_idx), len(self.results) - 1))
        out = []
        for (sx, sy) in seeds:
            # Same traced path the streaklines use, so a marker and the end of
            # its own trail can never disagree, and neither pays to re-walk the
            # history the other already walked.
            path = self._traced_path(float(sx), float(sy), idx)
            lost = path["lost_at"]
            if lost is not None and lost <= idx:
                out.append(None)
                continue
            pts = path["points"]
            out.append(tuple(pts[idx + 1]) if idx + 1 < len(pts) else None)
        return out

    def get_trajectories_from_seeds(self, seeds, max_frame: int, trail: int = 0
                                   ) -> list[dict]:
        """
        Trace one trajectory per user-placed marker.

        seeds      : list of (x, y) in REFERENCE-frame coordinates
        max_frame  : trace up to and including this displayed frame
        trail      : 0 = full history, else keep only the last `trail` segments

        Each entry: {"points": [(x,y), ...], "lost_at": int|None, "seed": (x,y)}
        Strictly speaking these are pathlines (trajectories of individual
        material points). They coincide with streaklines only in steady flow --
        worth keeping in mind for segmented-chip conditions.
        """
        if not self.results or not seeds:
            return []
        last = min(int(max_frame), len(self.results) - 1)
        out: list[dict] = []
        for (sx, sy) in seeds:
            path = self._traced_path(float(sx), float(sy), last)
            pts = path["points"][:last + 2]          # seed + one point per frame
            lost_at = path["lost_at"]
            if lost_at is not None and lost_at > last:
                lost_at = None                        # not lost yet at this frame
            if trail and trail > 0 and len(pts) > trail + 1:
                pts = pts[-(trail + 1):]
            out.append({"points": list(pts), "lost_at": lost_at,
                        "seed": (float(sx), float(sy))})
        return out

    def _results_token(self) -> tuple:
        """Identity of the current result sequence, for cache validation.

        The epoch counter alone only catches the code paths inside this class
        that clear `results`. Callers also replace the list contents directly --
        loading a session, swapping in a re-run -- and a cached path that
        survives that is a path traced through data the viewer is no longer
        showing. Pairing the epoch with the length and the identity of the first
        result catches that too.
        """
        first = id(self.results[0]) if self.results else 0
        return (self._results_epoch, len(self.results), first)

    def _traced_path(self, sx: float, sy: float, last: int) -> dict:
        """Path of one material point, traced to at least frame `last`.

        Cached and extended in place. Scrubbing forward costs only the frames
        newly entered; scrubbing back costs nothing, because the shorter path is
        a prefix of the one already held.
        """
        cache = self._path_cache
        if cache is None:
            cache = self._path_cache = {}
        key = (round(sx, 3), round(sy, 3))
        entry = cache.get(key)
        if entry is None or entry["epoch"] != self._results_token():
            entry = {"epoch": self._results_token(),
                     "points": [(float(sx), float(sy))],
                     "traced_to": -1, "lost_at": None,
                     "u": 0.0, "v": 0.0}
            self._path_cache[key] = entry

        # Once a point is lost it stays lost; there is nothing further to trace.
        if entry["lost_at"] is not None:
            return entry

        for i in range(entry["traced_to"] + 1, last + 1):
            u = self._sample_sparse(self.results[i].u, sx, sy)
            v = self._sample_sparse(self.results[i].v, sx, sy)
            if not (np.isfinite(u) and np.isfinite(v)):
                entry["lost_at"] = i
                break
            entry["u"] += u
            entry["v"] += v
            entry["points"].append((sx + entry["u"], sy + entry["v"]))
            entry["traced_to"] = i

        # A cache that grows without bound across a long session is its own
        # problem; markers are few, so a small cap is ample. Re-insert this key
        # last so the entry just traced is never the one evicted.
        if len(self._path_cache) > 64:
            self._path_cache.pop(key, None)
            for stale in list(self._path_cache)[:-32]:
                self._path_cache.pop(stale, None)
            self._path_cache[key] = entry
        return entry

    # ------------------------------------------------------------------
    # Frame-pair analysis
    # ------------------------------------------------------------------
    # A "pair" is two frames (i, j) treated as one temporally smoothed
    # measurement interval. Displacement, velocity and strain rate describe that
    # interval. Accumulated strain remains the history state at endpoint j; a
    # sliding window must never reset or double-count that state.

    PAIR_FIELDS = ("u", "v", "u_inc", "v_inc", "mag_inc",
                   "Vx", "Vy", "Veff",
                   "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                   "Exx_rate", "Exy_rate", "Gxy_rate", "Eyy_rate", "Eeff_rate")

    def pair_interval(self, i: int, j: int) -> float:
        """Elapsed time between two frames, in seconds."""
        return abs(j - i) / max(self.fps, 1e-9)

    def _pair_step_gradients(
            self, result: "PairResult", strain_window: int,
            use_gpu: bool) -> dict[str, _CroppedTemporalField]:
        """Spatially fit one immediate interval once per window/backend.

        Sliding temporal windows reuse most constituent intervals. Without this
        cache, a span-4 sequence fitted nearly every interval four times. The
        finite-support crops keep the cache bounded and can be sampled without
        expanding full NaN images.
        """
        cache = getattr(self, "_temporal_gradient_cache", None)
        lock = getattr(self, "_temporal_gradient_lock", None)
        if cache is None or lock is None:
            cache = OrderedDict()
            lock = threading.RLock()
            self._temporal_gradient_cache = cache
            self._temporal_gradient_lock = lock
        key = (id(result), int(strain_window), bool(use_gpu))
        with lock:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                return hit

            from .strain import compute_velocity_strains
            valid = (np.asarray(result.valid, dtype=bool)
                     if result.valid is not None else
                     (np.isfinite(result.u) & np.isfinite(result.v)))
            fitted = compute_velocity_strains(
                np.asarray(result.u), np.asarray(result.v), valid,
                strain_window, self.params.subset_spacing, use_gpu=use_gpu)
            names = ("dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy")
            dense = [np.asarray(fitted[name]) for name in names]
            complete = np.logical_and.reduce(
                [np.isfinite(values) for values in dense])
            rows = complete.any(axis=1)
            cols = complete.any(axis=0)
            if rows.any() and cols.any():
                y0, y1 = int(rows.argmax()), int(len(rows) - rows[::-1].argmax()) + 1
                x0, x1 = int(cols.argmax()), int(len(cols) - cols[::-1].argmax()) + 1
            else:
                y0 = y1 = x0 = x1 = 0
            packed = {}
            for name, values in zip(names, dense):
                crop = values[y0:y1, x0:x1].astype(np.float32, copy=True)
                if crop.size:
                    crop[~complete[y0:y1, x0:x1]] = np.nan
                packed[name] = _CroppedTemporalField(
                    values.shape, y0, x0, crop)
            cache[key] = packed
            # Enough for several neighboring windows without turning a
            # full-sequence preprocessing run into another result-sized store.
            while len(cache) > 24:
                cache.popitem(last=False)
            return packed

    @staticmethod
    def _dense_temporal_gradient(field: _CroppedTemporalField) -> np.ndarray:
        """Expand one cached crop only for the path-transport step using it."""
        dense = np.full(field.shape, np.nan, dtype=np.float32)
        height, width = field.values.shape
        if height and width:
            dense[field.y0:field.y0 + height,
                  field.x0:field.x0 + width] = field.values
        return dense

    def _temporal_accumulated_strains(
            self, strain_window: int,
            use_gpu: bool,
            progress_cb: Optional[Callable[[float, str], None]] = None,
            cancel_flag: Optional[list] = None,
            min_frame: Optional[int] = None,
            max_frame: Optional[int] = None,
            ) -> tuple[dict[str, object], ...]:
        """Return analysis-history strain at every temporal endpoint.

        The selected temporal range owns its strain history. It starts from
        zero at that range's first displayed frame, consumes only the following
        measured increments, and remains cumulative across sliding outputs.
        """
        names = ("Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl")
        if cancel_flag is not None and cancel_flag[0]:
            raise RuntimeError("Temporal calculation cancelled.")
        limit = (len(self.results) if max_frame is None else
                 min(len(self.results), max(0, int(max_frame)) + 1))
        configured_start = int(np.clip(
            getattr(self, "strain_start_frame", 0), 0, len(self.results)))
        first = configured_start if min_frame is None else max(
            configured_start, min(len(self.results), max(0, int(min_frame))))
        selected_results = self.results[first:limit]
        original_window = self.params.effective_strain_window()
        if (min_frame is None and int(strain_window) == original_window and
                all(all(getattr(result, name, None) is not None
                        for name in names) for result in self.results[:limit])):
            return tuple({name: getattr(result, name) for name in names}
                         for result in self.results[:limit])

        cache = getattr(self, "_temporal_strain_cache", None)
        lock = getattr(self, "_temporal_strain_lock", None)
        if cache is None or lock is None:
            cache = OrderedDict()
            lock = threading.RLock()
            self._temporal_strain_cache = cache
            self._temporal_strain_lock = lock
        result_signature = (
            id(self.results), len(self.results),
            id(self.results[0]) if self.results else 0,
            id(self.results[-1]) if self.results else 0,
        )
        key = (result_signature, first, limit,
               int(strain_window), bool(use_gpu),
               int(getattr(self, "strain_start_frame", 0)),
               id(getattr(self, "_strain_origin_mask", None)))
        with lock:
            if cancel_flag is not None and cancel_flag[0]:
                raise RuntimeError("Temporal calculation cancelled.")
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                return hit

            if self._roi_mask is None:
                return tuple()
            from .strain_accum import StrainPathTracker

            origin = getattr(self, "_strain_origin_mask", None)
            if origin is None or not np.any(origin):
                origin = self._roi_mask
            tracker = StrainPathTracker(
                self._roi_mask.shape, origin, self._roi_mask,
                self.params.subset_radius, self.params.subset_spacing)
            persistent = (
                "Exx_inf", "Eyy_inf", "Exy_inf", "Eeff_inf",
                "Exx_gl", "Eyy_gl", "Exy_gl")
            swept = {name: np.full(self._roi_mask.shape, np.nan,
                                   dtype=np.float32)
                     for name in persistent}
            encountered = np.zeros(self._roi_mask.shape, dtype=bool)

            def deposit(snapshot) -> None:
                new = self._roi_mask & ~encountered
                for name in persistent:
                    new &= np.isfinite(np.asarray(snapshot[name]))
                if not new.any():
                    return
                for name in persistent:
                    swept[name][new] = np.asarray(snapshot[name])[new]
                encountered[new] = True

            def packed_snapshot() -> dict[str, CompactField]:
                values = {
                    "Exx_gl": swept["Exx_gl"],
                    "Eyy_gl": swept["Eyy_gl"],
                    "Exy_gl": swept["Exy_gl"],
                    "Eeff_gl": swept["Eeff_inf"],
                }
                complete = np.logical_and.reduce(
                    [np.isfinite(values[name]) for name in names])
                indices = np.flatnonzero(complete.reshape(-1)).astype(
                    np.uint32, copy=False)
                return {
                    name: CompactField.from_dense(values[name], indices=indices)
                    for name in names
                }

            blank = {name: CompactField.empty(self._roi_mask.shape)
                     for name in names}
            sequence = [blank for _ in range(first)]
            count = len(selected_results)
            for offset, result in enumerate(selected_results):
                if cancel_flag is not None and cancel_flag[0]:
                    raise RuntimeError("Temporal calculation cancelled.")
                if progress_cb:
                    progress_cb(
                        offset / max(1, count),
                        f"Strain history {offset + 1}/{count}")

                pair_valid = (result.valid if result.valid is not None else
                              (np.isfinite(result.u) & np.isfinite(result.v)))
                tracker.seed(pair_valid)
                deposit(tracker.snapshot())
                gradients = self._pair_step_gradients(
                    result, int(strain_window), use_gpu=use_gpu)
                if cancel_flag is not None and cancel_flag[0]:
                    raise RuntimeError("Temporal calculation cancelled.")
                tracker.advance(
                    np.asarray(result.u), np.asarray(result.v),
                    *(self._dense_temporal_gradient(gradients[name])
                      for name in ("dVx_dx", "dVx_dy",
                                   "dVy_dx", "dVy_dy")))
                deposit(tracker.snapshot())
                sequence.append(packed_snapshot())

            stored = tuple(sequence)
            if progress_cb:
                progress_cb(1.0, "Accumulated strain history ready")
            cache[key] = stored
            while len(cache) > 2:
                cache.popitem(last=False)
            return stored

    def pair_kinematics(
            self, i: int, j: int, strain_window: Optional[int] = None,
            use_gpu: bool = False,
            progress_cb: Optional[Callable[[float, str], None]] = None,
            cancel_flag: Optional[list] = None,
            include_strain: bool = True,
            include_rate: bool = True,
            ) -> "PairResult":
        """Material-path kinematics between displayed frames ``i`` and ``j``.

        Each adjacent displacement is sampled at the point's advected position,
        so the returned displacement maps material coordinates on frame ``i``
        to frame ``j``. Temporal strain rate is derived from that complete
        interval map. Accumulated Green--Lagrange strain is the analysis-history
        state at endpoint ``j``; it is never reset at ``i`` or averaged across
        overlapping windows.
        """
        n = len(self.results)
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"Frame pair ({i}, {j}) out of range 0..{n - 1}")
        if i == j:
            raise ValueError("A frame pair needs two different frames.")
        if j < i:
            i, j = j, i

        # Result k is the interval entering displayed frame k. Therefore the
        # motion from displayed frame i to j consumes intervals i+1..j.
        interval_results = self.results[i + 1:j + 1]
        shape = self.results[0].u.shape
        radius = max(0, int(self.params.subset_radius))
        spacing = max(1, int(self.params.subset_spacing))
        from .strain import compute_velocity_strains
        pair_window = self.params.effective_strain_window(window=strain_window)
        grid_x = np.arange(radius, shape[1] - radius, spacing, dtype=np.int32)
        grid_y = np.arange(radius, shape[0] - radius, spacing, dtype=np.int32)
        gx, gy = np.meshgrid(grid_x, grid_y)
        source_ok = (self._roi_mask[gy, gx] if self._roi_mask is not None
                     else np.ones(gx.shape, dtype=bool))
        x0 = gx[source_ok].astype(np.float64)
        y0 = gy[source_ok].astype(np.float64)
        x, y = x0.copy(), y0.copy()
        alive = np.ones(x.size, dtype=bool)

        def sample(field, qx_pixels, qy_pixels):
            """Strict bilinear sampling on the sparse regular DIC lattice."""
            out = np.full(qx_pixels.size, np.nan, dtype=np.float64)
            good = np.zeros(qx_pixels.size, dtype=bool)
            if field is None or not grid_x.size or not grid_y.size:
                return out, good
            qx = (qx_pixels - radius) / spacing
            qy = (qy_pixels - radius) / spacing
            inside = ((qx >= 0.0) & (qx <= grid_x.size - 1) &
                      (qy >= 0.0) & (qy <= grid_y.size - 1))
            rows = np.where(inside)[0]
            if not rows.size:
                return out, good
            xq, yq = qx[rows], qy[rows]
            ix0 = np.floor(xq).astype(np.intp)
            iy0 = np.floor(yq).astype(np.intp)
            ix1 = np.minimum(ix0 + 1, grid_x.size - 1)
            iy1 = np.minimum(iy0 + 1, grid_y.size - 1)
            tx, ty = xq - ix0, yq - iy0
            tx[ix0 == ix1] = 0.0
            ty[iy0 == iy1] = 0.0
            corner_y = np.column_stack((
                grid_y[iy0], grid_y[iy0], grid_y[iy1], grid_y[iy1]))
            corner_x = np.column_stack((
                grid_x[ix0], grid_x[ix1], grid_x[ix0], grid_x[ix1]))
            corner_flat = corner_y * field.shape[1] + corner_x
            if isinstance(field, CompactField):
                flat = corner_flat.reshape(-1).astype(np.uint32, copy=False)
                pos = np.searchsorted(field.indices, flat)
                exists = ((pos < field.indices.size) &
                          (field.indices[np.minimum(pos, max(0, field.indices.size - 1))]
                           == flat)) if field.indices.size else np.zeros(flat.size, bool)
                gathered = np.full(flat.size, np.nan, dtype=np.float64)
                if exists.any():
                    gathered[exists] = field.values[pos[exists]]
                samples = gathered.reshape(-1, 4)
            elif isinstance(field, _CroppedTemporalField):
                ly = corner_y - field.y0
                lx = corner_x - field.x0
                crop_h, crop_w = field.values.shape
                in_crop = ((ly >= 0) & (ly < crop_h) &
                           (lx >= 0) & (lx < crop_w))
                samples = np.full(corner_y.shape, np.nan, dtype=np.float64)
                if in_crop.any():
                    samples[in_crop] = field.values[ly[in_crop], lx[in_crop]]
            else:
                arr = np.asarray(field)
                samples = arr.reshape(-1)[corner_flat].astype(
                    np.float64, copy=False)
            weights = np.column_stack((
                (1.0 - tx) * (1.0 - ty), tx * (1.0 - ty),
                (1.0 - tx) * ty, tx * ty,
            ))
            needed = weights > 1e-12
            usable = np.all(~needed | np.isfinite(samples), axis=1)
            vals = np.sum(np.where(needed, samples, 0.0) * weights, axis=1)
            usable &= np.isfinite(vals)
            out[rows[usable]] = vals[usable]
            good[rows[usable]] = True
            return out, good

        # Compose displacement along each material path. A failed interpolation
        # ends only that path; it does not force unrelated subset centres out.
        for res in interval_results:
            if cancel_flag is not None and cancel_flag[0]:
                raise RuntimeError("Temporal calculation cancelled.")
            rows = np.where(alive)[0]
            if not rows.size:
                break
            uu, gu = sample(res.u, x[rows], y[rows])
            vv, gv = sample(res.v, x[rows], y[rows])
            keep = gu & gv
            alive[rows[~keep]] = False
            kept = rows[keep]
            x[kept] += uu[keep]
            y[kept] += vv[keep]

        du = np.full(shape, np.nan, dtype=np.float64)
        dv = np.full(shape, np.nan, dtype=np.float64)
        ok = np.zeros(shape, dtype=bool)
        if alive.any():
            sx = x0[alive].astype(np.intp)
            sy = y0[alive].astype(np.intp)
            du[sy, sx] = x[alive] - x0[alive]
            dv[sy, sx] = y[alive] - y0[alive]
            ok[sy, sx] = True
        with np.errstate(invalid="ignore"):
            mag = np.sqrt(du ** 2 + dv ** 2)

        dt = self.pair_interval(i, j)
        Vx, Vy = du / dt, dv / dt
        with np.errstate(invalid="ignore"):
            Veff = np.sqrt(Vx ** 2 + Vy ** 2)

        if include_rate:
            # Temporal smoothing is applied to velocity first. Refit every rate
            # component from that averaged velocity field, using the
            # pair-specific spatial window.
            from .strain import compute_velocity_strains
            rates = compute_velocity_strains(
                Vx, Vy, ok, pair_window, spacing, use_gpu=use_gpu)
            dVx_dx = rates["dVx_dx"]
            dVx_dy = rates["dVx_dy"]
            dVy_dx = rates["dVy_dx"]
            dVy_dy = rates["dVy_dy"]
            Exx_rate = rates["Exx_rate"]
            Exy_rate = rates["Exy_rate"]
            Eyy_rate = rates["Eyy_rate"]
            Eeff_rate = rates["Eeff_rate"]
            h11, h12 = dVx_dx * dt, dVx_dy * dt
            h21, h22 = dVy_dx * dt, dVy_dy * dt
        else:
            # Phase one of bulk smoothing exposes velocity only. Rate and
            # strain remain genuinely absent until all velocity averages exist.
            empty = CompactField.empty(shape)
            dVx_dx = dVx_dy = dVy_dx = dVy_dy = empty
            Exx_rate = Exy_rate = Eyy_rate = Eeff_rate = empty
            h11 = h12 = h21 = h22 = empty

        if include_strain:
            # Accumulated strain belongs to the complete analysis history
            # ending at j, not merely to this smoothing window i→j.
            strain_sequence = self._temporal_accumulated_strains(
                pair_window, use_gpu=use_gpu, progress_cb=progress_cb,
                cancel_flag=cancel_flag, min_frame=i + 1, max_frame=j)
            endpoint = strain_sequence[j]
            Exx_gl = endpoint["Exx_gl"]
            Eyy_gl = endpoint["Eyy_gl"]
            Exy_gl = endpoint["Exy_gl"]
            Eeff_gl = endpoint["Eeff_gl"]
        else:
            # Bulk preprocessing deliberately completes every averaged velocity
            # and derived rate before its one shared history pass. Empty compact
            # fields keep the intermediate pair disk format valid.
            Exx_gl = CompactField.empty(shape)
            Eyy_gl = CompactField.empty(shape)
            Exy_gl = CompactField.empty(shape)
            Eeff_gl = CompactField.empty(shape)

        # Pair timelines can contain hundreds of items. Match normal result
        # storage precision and share aliases instead of retaining duplicate
        # float64 arrays for every displayed quantity.
        du, dv, mag = map(_result_f32, (du, dv, mag))
        Vx, Vy, Veff = map(_result_f32, (Vx, Vy, Veff))
        h11, h12, h21, h22 = map(_result_f32, (h11, h12, h21, h22))
        dVx_dx, dVx_dy, dVy_dx, dVy_dy = map(
            _result_f32, (dVx_dx, dVx_dy, dVy_dx, dVy_dy))
        Exx_rate, Exy_rate, Eyy_rate, Eeff_rate = map(
            _result_f32, (Exx_rate, Exy_rate, Eyy_rate, Eeff_rate))
        Exx_gl, Exy_gl, Eyy_gl, Eeff_gl = map(
            _result_f32, (Exx_gl, Exy_gl, Eyy_gl, Eeff_gl))

        out = PairResult(
            image_path=f"pair {i + 1}→{j + 1}",
            u=du, v=dv,
            u_inc=du, v_inc=dv, mag_inc=mag,
            Exx=Exx_gl, Exy=Exy_gl, Eyy=Eyy_gl, Eeff=Eeff_gl,
            du_dx=h11, du_dy=h12, dv_dx=h21, dv_dy=h22,
            corr=None,
            Vx=Vx, Vy=Vy, Veff=Veff,
            dVx_dx=dVx_dx, dVx_dy=dVx_dy,
            dVy_dx=dVy_dx, dVy_dy=dVy_dy,
            Exx_rate=Exx_rate, Exy_rate=Exy_rate,
            Gxy_rate=_result_f32(2.0 * Exy_rate),
            Eyy_rate=Eyy_rate, Eeff_rate=Eeff_rate,
            valid=ok, elapsed=dt,
            Exx_gl=Exx_gl, Exy_gl=Exy_gl,
            Eyy_gl=Eyy_gl, Eeff_gl=Eeff_gl,
        )
        out.pair_start = i
        out.pair_end = j
        return out

    # Fields that are rates -- quantities per unit time. These are the only
    # ones a plain mean is valid on, because their value does not depend on how
    # long the interval that produced them happened to be.
    _PAIR_INTENSIVE = ("Vx", "Vy",
                       "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                       "Exx_rate", "Exy_rate", "Eyy_rate",
                       "corr")

    @staticmethod
    def _nan_mean(stack) -> tuple:
        """Per-pixel mean ignoring NaN, plus how many values fed each pixel.

        float64 accumulator regardless of the stored precision: averaging is
        the step meant to reduce noise, so it must not contribute any of its own.
        """
        arr = np.stack(stack).astype(np.float64, copy=False)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        counts = np.sum(np.isfinite(arr), axis=0)
        with np.errstate(invalid="ignore"):
            summed = np.nansum(arr, axis=0, dtype=np.float64)
            mean = np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)
        return mean, counts

    def average_pairs(self, pairs) -> "PairResult":
        """Combine several frame pairs into one representative measurement.

        Only the rate fields are averaged directly. Displacement is not
        averageable across pairs of unequal span: a pair covering three
        intervals accumulates roughly three times the displacement of a
        one-interval pair, so a plain mean of the two is a mean of
        incommensurable numbers that corresponds to no interval at all.

        Instead the average describes one representative interval, of the mean
        pair duration, and displacement is reconstructed from the averaged
        velocity over that duration. When every pair has the same span -- the
        usual case -- this is arithmetically identical to averaging the
        displacements, so nothing changes; when spans differ it is the
        difference between a defined quantity and a meaningless one.

        Magnitudes (|Δ|, effective velocity, equivalent strain rate) are
        recomputed from the averaged components rather than averaged
        themselves. Averaging magnitudes rectifies noise: every fluctuation
        contributes a positive amount regardless of sign, so the mean of
        magnitudes is biased high and never cancels. Averaging the signed
        components first lets it cancel, then the magnitude is taken once.

        Points are averaged over whichever pairs tracked them, so one dropout
        does not discard the rest; `pair_support` records how many pairs
        actually contributed to each point.
        """
        pairs = [tuple(p) for p in pairs]
        if not pairs:
            raise ValueError("Select at least one frame pair.")

        per_pair = [self.pair_kinematics(i, j) for i, j in pairs]
        shape = per_pair[0].u.shape
        nan = lambda: np.full(shape, np.nan)

        # 1. Average the rate fields.
        averaged = {}
        support = np.zeros(shape, dtype=np.int32)
        for name in self._PAIR_INTENSIVE:
            stack = [getattr(p, name) for p in per_pair]
            stack = [np.asarray(s) for s in stack if s is not None]
            if not stack:
                averaged[name] = None
                continue
            mean, counts = self._nan_mean(stack)
            averaged[name] = mean
            if name == "Vx":
                support = counts.astype(np.int32)

        # 2. Representative interval, and the displacement that spans it.
        T = float(np.mean([p.elapsed for p in per_pair]))
        Vx, Vy = averaged.get("Vx"), averaged.get("Vy")
        du = Vx * T if Vx is not None else nan()
        dv = Vy * T if Vy is not None else nan()

        # 3. Magnitudes from the averaged components, never from averaged
        #    magnitudes -- see the docstring.
        with np.errstate(invalid="ignore"):
            mag = np.sqrt(du ** 2 + dv ** 2)
            Veff = (np.sqrt(Vx ** 2 + Vy ** 2)
                    if Vx is not None and Vy is not None else nan())

        Exx_r = averaged.get("Exx_rate")
        Exy_r = averaged.get("Exy_rate")
        Eyy_r = averaged.get("Eyy_rate")
        if Exx_r is not None and Exy_r is not None and Eyy_r is not None:
            from .strain import von_mises_equivalent
            Eeff_r = von_mises_equivalent(Exx_r, Eyy_r, Exy_r)
            # Engineering shear is exactly twice tensor shear, by definition;
            # deriving it keeps that identity true of the averaged field.
            Gxy_r = 2.0 * Exy_r
        else:
            Eeff_r = Gxy_r = nan()

        label = ", ".join(f"{i + 1}→{j + 1}" for i, j in pairs)
        # A fresh result. The previous version returned per_pair[0] with its
        # arrays overwritten, so anything still holding that pair's result saw
        # it silently become the average.
        out = PairResult(
            image_path=(f"average of {len(pairs)} pairs [{label}]"
                        if len(pairs) > 1 else f"pair {label}"),
            u=du, v=dv,
            u_inc=du, v_inc=dv, mag_inc=mag,
            # Accumulated strain is not defined for an average of arbitrary
            # pairs; kept explicitly NaN rather than absent so anything reading
            # these gets "no value" instead of an attribute error.
            Exx=nan(), Exy=nan(), Eyy=nan(), Eeff=nan(),
            du_dx=nan(), du_dy=nan(), dv_dx=nan(), dv_dy=nan(),
            corr=averaged.get("corr"),
            Vx=Vx, Vy=Vy, Veff=Veff,
            dVx_dx=averaged.get("dVx_dx"), dVx_dy=averaged.get("dVx_dy"),
            dVy_dx=averaged.get("dVy_dx"), dVy_dy=averaged.get("dVy_dy"),
            Exx_rate=Exx_r, Exy_rate=Exy_r, Gxy_rate=Gxy_r,
            Eyy_rate=Eyy_r, Eeff_rate=Eeff_r,
            valid=support > 0,
            elapsed=T,
        )
        # Not a PairResult field: attached so the viewer can report how evenly
        # the pairs actually covered the specimen.
        out.pair_support = support
        out.pair_count = len(pairs)
        out.pair_spans_equal = len({abs(j - i) for i, j in pairs}) == 1
        return out

    def get_global_range(self, field: str,
                         coverage: float = 100.0) -> tuple[float, float]:
        """Colour limits spanning the whole sequence, for a fixed scale.

        `coverage` is the central share of the pooled values to span, matching
        the per-frame robust scaling. At 100 a single failed correlation
        anywhere in the sequence sets the limits for every frame, which is the
        common way a fixed scale ends up showing nothing.

        Values are pooled across frames rather than each frame's own
        percentiles being averaged: a percentile of percentiles is not a
        percentile, and would understate the true spread.
        """
        cov = float(np.clip(coverage, 1.0, 100.0))

        if cov >= 100.0:
            vmin, vmax = float('inf'), float('-inf')
            for res in self.results:
                arr = getattr(res, field, None)
                if arr is not None and arr.size:
                    valid = finite_values(arr)
                    if valid.size > 0:
                        vmin = min(vmin, float(valid.min()))
                        vmax = max(vmax, float(valid.max()))
            return (vmin, vmax) if vmin != float('inf') else (0.0, 1.0)

        # Subsample long sequences: pooling every finite value across a
        # thousand full-resolution frames is gigabytes for a number that a
        # sample estimates to well within its own display precision.
        stride = max(1, len(self.results) // 200)
        pooled = []
        for res in self.results[::stride]:
            arr = getattr(res, field, None)
            if arr is None or not arr.size:
                continue
            valid = finite_values(arr)
            if valid.size == 0:
                continue
            if valid.size > 50_000:
                step = valid.size // 50_000 + 1
                valid = valid[::step]
            pooled.append(valid)

        if not pooled:
            return (0.0, 1.0)
        limits = robust_limits(np.concatenate(pooled), cov)
        return limits if limits is not None else (0.0, 1.0)

    def export_csv(self, result_index, directory: str) -> None:
        """Write the selected frame's fields as CSV.

        Accepts a frame index or a PairResult directly, so a frame-pair average
        exports through exactly the same path as a single frame.
        """
        res = (self.results[result_index] if isinstance(result_index, (int, np.integer))
               else result_index)
        base = os.path.splitext(os.path.basename(res.image_path))[0]
        # Pair labels ("average of 3 pairs [1→2, ...]") are not filenames.
        base = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in base).strip("_")
        base = base or "export"
        fields = ("u", "v", "u_inc", "v_inc", "mag_inc",
                  "Exx_inf", "Eyy_inf", "Exy_inf", "Gxy_inf", "Eeff_inf",
                  "Exx_gl", "Eyy_gl", "Exy_gl", "Gxy_gl", "Eeff_gl",
                  "Vx", "Vy", "Veff",
                  "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                  "Exx_rate", "Exy_rate", "Gxy_rate", "Eyy_rate", "Eeff_rate", "corr")
        # CSV is a one-way export, so write the values as displayed and state the
        # unit in the header. (HDF5 is different: it must round-trip, so it keeps
        # raw pixels plus the calibration as metadata.) With no calibration set
        # this is byte-identical to the old behaviour.
        for name in fields:
            arr = getattr(res, name, None)
            if arr is None:
                continue
            # A field with nothing finite in it has no data to export. This is
            # how cumulative strain is skipped for a frame-pair average, where
            # it is deliberately undefined -- writing a grid of "nan" would
            # read as a failed measurement rather than an excluded one.
            if arr.size == 0 or finite_values(arr).size == 0:
                continue
            base_unit = _FIELD_BASE_UNIT.get(name, "")
            out, unit = self.calibration.convert(name, arr, base_unit)
            out = np.where(np.isfinite(out), out, np.nan)
            header = f"{name} [{unit}]" if unit else name
            if self.calibration.calibrated:
                header += f"  ({self.calibration.describe()})"
            np.savetxt(os.path.join(directory, f"{base}_{name}.csv"), out,
                       delimiter=",", header=header)

    def export_hdf5(
            self, path: str,
            progress_cb: Optional[Callable[[float], None]] = None,
            temporal_results=None, temporal_pairs=None,
            temporal_metadata: Optional[dict] = None) -> None:
        import h5py
        with h5py.File(path, "w") as f:
            # 1. Save Global Attributes
            f.attrs.update(dict(
                result_schema=5,
                storage_layout="compact_finite_subset_points",
                displacement_semantics="immediate_previous_frame",
                strain_semantics=(
                    "continuous_origin_pathline_transport_"
                    "with_first_arrival_frozen_coverage"),
                finite_strain_semantics="composed_incremental_deformation_gradient",
                equivalent_semantics="integral_of_nonnegative_equivalent_strain_rate",
                strain_start_frame=int(getattr(self, "strain_start_frame", 0)),
                reference_image=self.ref_path or "",
                subset_radius=self.params.subset_radius,
                subset_spacing=self.params.subset_spacing,
                strain_window=self.params.strain_window,
                fps=self.fps,
                # Datasets stay in the solver's native pixel units so a file
                # round-trips exactly; the calibration rides along as metadata
                # and is what the viewer applies on display.
                metres_per_pixel=(self.calibration.metres_per_pixel
                                  if self.calibration.calibrated else 0.0),
                display_unit=self.calibration.display_unit,
                length_units="pixels",
                analysis_backend=getattr(self, "last_backend", "unknown"),
                dynamic_roi=str(getattr(self.params, "dynamic_roi", "None")),
                # NaN is the HDF5-safe sentinel for automatic/Otsu selection.
                dynamic_roi_threshold=(
                    np.nan if getattr(self.params, "dynamic_roi_threshold", None) is None
                    else float(self.params.dynamic_roi_threshold)),
                dynamic_roi_min_area_frac=float(getattr(
                    self.params, "dynamic_roi_min_area_frac", 0.02)),
                dynamic_roi_fill_holes=bool(getattr(
                    self.params, "dynamic_roi_fill_holes", True)),
                dynamic_roi_override_semantics="pairwise_image_space",
            ))

            # 2. Save the ROI Mask (CRITICAL FIX)
            if self._roi_mask is not None:
                f.create_dataset(
                    "roi_mask",
                    data=self._roi_mask.astype(bool),
                    compression="gzip",
                    compression_opts=4
                )
            if getattr(self, "_strain_origin_mask", None) is not None:
                f.create_dataset(
                    "strain_origin_mask",
                    data=np.asarray(self._strain_origin_mask, dtype=bool),
                    compression="gzip", compression_opts=4)
            for name, mask in (
                    ("dynamic_include_mask", getattr(self, "dynamic_include_mask", None)),
                    ("dynamic_exclude_mask", getattr(self, "dynamic_exclude_mask", None))):
                if mask is not None:
                    f.create_dataset(
                        name, data=np.asarray(mask, dtype=bool),
                        compression="gzip", compression_opts=4)
            frame_overrides = getattr(self, "dynamic_frame_overrides", {})
            if frame_overrides:
                overrides_group = f.create_group("dynamic_frame_overrides")
                overrides_group.attrs["semantics"] = "exact_displayed_frame"
                for frame_index, entry in sorted(frame_overrides.items()):
                    group = overrides_group.create_group(
                        f"frame_{int(frame_index):06d}")
                    if entry.get("threshold") is not None:
                        group.attrs["threshold"] = float(entry["threshold"])
                    if "replace" in entry:
                        group.attrs["replace"] = bool(entry["replace"])
                    for channel in ("include", "exclude"):
                        mask = entry.get(channel)
                        if mask is not None and np.asarray(mask, dtype=bool).any():
                            group.create_dataset(
                                channel, data=np.asarray(mask, dtype=bool),
                                compression="gzip", compression_opts=4)
            future_overrides = getattr(self, "dynamic_future_overrides", {})
            if future_overrides:
                future_group = f.create_group("dynamic_future_overrides")
                future_group.attrs["semantics"] = "default_from_displayed_frame"
                for start_frame, entry in sorted(future_overrides.items()):
                    group = future_group.create_group(
                        f"from_{int(start_frame):06d}")
                    if bool(entry.get("reset", False)):
                        group.attrs["reset"] = True
                    if entry.get("threshold") is not None:
                        group.attrs["threshold"] = float(entry["threshold"])
                    if "replace" in entry:
                        group.attrs["replace"] = bool(entry["replace"])
                    for channel in ("include", "exclude"):
                        mask = entry.get(channel)
                        if mask is not None and np.asarray(mask, dtype=bool).any():
                            group.create_dataset(
                                channel, data=np.asarray(mask, dtype=bool),
                                compression="gzip", compression_opts=4)

            # 3. Save only independent, user-facing data. Velocity and
            # displacement magnitudes are deterministic views of u/v and dt;
            # gradients/correlation are solver workspaces, not result history.
            measurement_fields = ("u", "v")
            rate_fields = ("Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate")
            strain_fields = ("Exx_gl", "Exy_gl", "Eyy_gl", "Eeff_gl")

            def packed_common(res, names):
                packed = []
                for name in names:
                    field = getattr(res, name, None)
                    if field is None:
                        return np.zeros(0, np.uint32), []
                    if isinstance(field, CompactField):
                        idx, vals = field.indices, field.values
                    else:
                        dense = np.asarray(field)
                        idx = np.flatnonzero(np.isfinite(dense).reshape(-1)).astype(
                            np.uint32, copy=False)
                        vals = dense.reshape(-1)[idx].astype(np.float32, copy=False)
                    packed.append((idx, vals))
                common = packed[0][0]
                for idx, _ in packed[1:]:
                    common = np.intersect1d(common, idx, assume_unique=True)
                aligned = [vals[np.searchsorted(idx, common)]
                           for idx, vals in packed]
                return common, aligned

            def write_result_group(group, res) -> None:
                group.attrs["image_path"] = res.image_path
                group.attrs["elapsed_s"] = res.elapsed
                shape = getattr(res.u, "shape", None)
                if shape is None:
                    shape = self._roi_mask.shape
                group.attrs["field_shape"] = np.asarray(shape, dtype=np.int64)
                for prefix, names in (("valid", measurement_fields),
                                      ("rate", rate_fields),
                                      ("strain", strain_fields)):
                    indices, values = packed_common(res, names)
                    group.create_dataset(
                        f"{prefix}_indices", data=indices,
                        compression="gzip", compression_opts=4)
                    for name, data in zip(names, values):
                        group.create_dataset(
                            name, data=np.asarray(data, np.float32),
                            compression="gzip", compression_opts=4)

            temporal_count = (len(temporal_results)
                              if temporal_results is not None else 0)
            total_items = max(1, len(self.results) + temporal_count)

            for i, res in enumerate(self.results):
                if progress_cb:
                    progress_cb(i / total_items)
                g = f.create_group(f"frame_{i:04d}")
                write_result_group(g, res)

            if temporal_count:
                temporal = f.create_group("temporal_sequence")
                temporal.attrs["schema"] = 4
                temporal.attrs["complete"] = True
                temporal.attrs["count"] = temporal_count
                temporal.attrs["semantics"] = (
                    "velocity_average_then_rate_then_accumulated_strain")
                temporal.attrs["rate_semantics"] = (
                    "symmetric_gradient_of_composed_interval_mean_velocity")
                temporal.attrs["strain_semantics"] = (
                    "green_lagrange_history_of_temporally_averaged_frames")
                for name, value in dict(temporal_metadata or {}).items():
                    if value is not None:
                        temporal.attrs[str(name)] = value
                pairs = (list(temporal_pairs) if temporal_pairs is not None
                         else [(getattr(temporal_results[i], "pair_start", -1),
                                getattr(temporal_results[i], "pair_end", -1))
                               for i in range(temporal_count)])
                temporal.create_dataset(
                    "pairs", data=np.asarray(pairs, dtype=np.int64))
                for i in range(temporal_count):
                    if progress_cb:
                        progress_cb((len(self.results) + i) / total_items)
                    res = temporal_results[i]
                    group = temporal.create_group(f"pair_{i:06d}")
                    group.attrs["pair_start"] = int(pairs[i][0])
                    group.attrs["pair_end"] = int(pairs[i][1])
                    write_result_group(group, res)
            if progress_cb:
                progress_cb(1.0)

    def load_hdf5(
            self, path: str,
            progress_cb: Optional[Callable[[float, str], None]] = None
            ) -> None:
        import h5py

        def report(fraction: float, message: str) -> None:
            if progress_cb is not None:
                progress_cb(float(np.clip(fraction, 0.0, 1.0)), message)

        report(0.0, "Opening HDF5 session…")
        self._release_loaded_hdf5()
        self.results.clear()
        self._results_epoch += 1
        self.def_paths.clear()
        self.temporal_results = None
        self.temporal_pairs = []
        self.temporal_metadata = {}
        self._preview_frame_index = None
        self._preview_frame_image = None
        self.dynamic_frame_overrides = {}
        self.dynamic_future_overrides = {}

        f = h5py.File(path, "r")
        try:
            # 1. Restore Global Attributes
            report(0.04, "Reading session metadata…")
            result_schema = int(f.attrs.get("result_schema", 1))
            frame_keys = sorted(
                key for key in f.keys() if key.startswith("frame_"))
            logical_bytes = 0
            if frame_keys:
                first_frame = f[frame_keys[0]]
                bytes_per_frame = sum(
                    dataset.size * dataset.dtype.itemsize
                    for dataset in first_frame.values()
                    if isinstance(dataset, h5py.Dataset))
                logical_bytes = bytes_per_frame * len(frame_keys)
            lazy = (
                result_schema >= 3
                and logical_bytes >= HDF5_LAZY_THRESHOLD_BYTES)
            self.ref_path = f.attrs.get("reference_image", "")
            try:
                if self.ref_path and os.path.exists(self.ref_path):
                    self._ref_image = _load_image(self.ref_path)
            except Exception:
                pass

            self.params.subset_radius = int(f.attrs.get("subset_radius", self.params.subset_radius))
            self.params.subset_spacing = int(f.attrs.get("subset_spacing", self.params.subset_spacing))
            self.params.strain_window = int(f.attrs.get("strain_window", self.params.strain_window))
            self.fps = float(f.attrs.get("fps", 1.0))
            self.strain_start_frame = int(np.clip(
                int(f.attrs.get("strain_start_frame", 0)), 0, len(frame_keys)))

            def _attr_text(name: str, default: str) -> str:
                value = f.attrs.get(name, default)
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="replace")
                return str(value)

            self.last_backend = _attr_text("analysis_backend", "unknown")
            if "dynamic_roi" in f.attrs:
                self.params.dynamic_roi = _attr_text("dynamic_roi", "None")
                threshold = float(f.attrs.get("dynamic_roi_threshold", np.nan))
                self.params.dynamic_roi_threshold = (
                    None if not np.isfinite(threshold) else threshold)
                self.params.dynamic_roi_min_area_frac = float(f.attrs.get(
                    "dynamic_roi_min_area_frac",
                    getattr(self.params, "dynamic_roi_min_area_frac", 0.02)))
                self.params.dynamic_roi_fill_holes = bool(f.attrs.get(
                    "dynamic_roi_fill_holes",
                    getattr(self.params, "dynamic_roi_fill_holes", True)))

            mpp = float(f.attrs.get("metres_per_pixel", 0.0) or 0.0)
            self.calibration = Calibration(
                mpp if mpp > 0 else None,
                str(f.attrs.get("display_unit", "mm")))

            # 2. Restore the ROI Mask (CRITICAL FIX)
            report(0.08, "Restoring ROI and calibration…")
            if "roi_mask" in f:
                self._roi_mask = f["roi_mask"][:].astype(bool)
            else:
                self._roi_mask = None
            self._strain_origin_mask = (
                f["strain_origin_mask"][:].astype(bool)
                if "strain_origin_mask" in f else None)
            self.dynamic_include_mask = (
                f["dynamic_include_mask"][:].astype(bool)
                if "dynamic_include_mask" in f else None)
            self.dynamic_exclude_mask = (
                f["dynamic_exclude_mask"][:].astype(bool)
                if "dynamic_exclude_mask" in f else None)
            if "dynamic_frame_overrides" in f:
                overrides_group = f["dynamic_frame_overrides"]
                for key, group in overrides_group.items():
                    try:
                        frame_index = int(key.rsplit("_", 1)[-1])
                    except (TypeError, ValueError):
                        continue
                    entry: dict[str, object] = {}
                    if "threshold" in group.attrs:
                        entry["threshold"] = float(group.attrs["threshold"])
                    if "replace" in group.attrs:
                        entry["replace"] = bool(group.attrs["replace"])
                    for channel in ("include", "exclude"):
                        if channel in group:
                            entry[channel] = group[channel][:].astype(bool)
                    if entry:
                        self.dynamic_frame_overrides[frame_index] = entry
            if "dynamic_future_overrides" in f:
                future_group = f["dynamic_future_overrides"]
                for key, group in future_group.items():
                    try:
                        start_frame = int(key.rsplit("_", 1)[-1])
                    except (TypeError, ValueError):
                        continue
                    entry: dict[str, object] = {}
                    if bool(group.attrs.get("reset", False)):
                        entry["reset"] = True
                    if "threshold" in group.attrs:
                        entry["threshold"] = float(group.attrs["threshold"])
                    if "replace" in group.attrs:
                        entry["replace"] = bool(group.attrs["replace"])
                    for channel in ("include", "exclude"):
                        if channel in group:
                            entry[channel] = group[channel][:].astype(bool)
                    if entry:
                        self.dynamic_future_overrides[start_frame] = entry

            # 3. Restore Frame Data
            n_frames = len(frame_keys)
            for frame_index, k in enumerate(frame_keys):
                report(
                    0.10 + 0.82 * frame_index / max(1, n_frames),
                    f"Loading frame {frame_index + 1} of {n_frames}…")
                g = f[k]
                ipath = g.attrs.get("image_path", "")
                self.def_paths.append(ipath)

                if result_schema >= 5 and "valid_indices" in g:
                    shape = tuple(int(v) for v in g.attrs["field_shape"])

                    def compact_saved(name: str, prefix: str) -> CompactField:
                        indices = g[f"{prefix}_indices"][:].astype(
                            np.uint32, copy=False)
                        values = (g[name][:].astype(np.float32, copy=False)
                                  if name in g else np.zeros(0, np.float32))
                        return CompactField(shape, indices[:values.size], values)

                    u = compact_saved("u", "valid")
                    v = compact_saved("v", "valid")
                    valid_indices = g["valid_indices"][:].astype(
                        np.uint32, copy=False)
                    mag = CompactField(
                        shape, valid_indices,
                        np.hypot(u.values, v.values).astype(np.float32, copy=False))
                    scale = float(max(self.fps, 1e-9))
                    vx, vy = u.scaled(scale), v.scaled(scale)
                    veff = mag.scaled(scale)
                    exx_rate = compact_saved("Exx_rate", "rate")
                    exy_rate = compact_saved("Exy_rate", "rate")
                    eyy_rate = compact_saved("Eyy_rate", "rate")
                    eeff_rate = compact_saved("Eeff_rate", "rate")
                    res = PairResult(
                        image_path=ipath, u=u, v=v,
                        Exx=None, Exy=None, Eyy=None, Eeff=None,
                        du_dx=None, du_dy=None, dv_dx=None, dv_dy=None,
                        corr=None, u_inc=u, v_inc=v, mag_inc=mag,
                        Vx=vx, Vy=vy, Veff=veff,
                        dVx_dx=exx_rate, dVx_dy=None,
                        dVy_dx=None, dVy_dy=eyy_rate,
                        Exx_rate=exx_rate, Exy_rate=exy_rate,
                        Gxy_rate=None, Eyy_rate=eyy_rate,
                        Eeff_rate=eeff_rate,
                        valid=CompactMask(shape, valid_indices),
                        elapsed=float(g.attrs.get("elapsed_s", 0.0)),
                        Exx_gl=compact_saved("Exx_gl", "strain"),
                        Exy_gl=compact_saved("Exy_gl", "strain"),
                        Eyy_gl=compact_saved("Eyy_gl", "strain"),
                        Eeff_gl=compact_saved("Eeff_gl", "strain"),
                    )
                    self.results.append(res)
                    continue

                def read_field(name: str, default=None):
                    if name not in g:
                        return np.zeros(0) if default is None else default
                    return g[name] if lazy else g[name][:]

                res = PairResult(
                    image_path=ipath,
                    u=read_field("u"), v=read_field("v"),
                    Exx=read_field("Exx"), Exy=read_field("Exy"),
                    Eyy=read_field("Eyy"), Eeff=read_field("Eeff"),
                    du_dx=read_field("du_dx"), du_dy=read_field("du_dy"),
                    dv_dx=read_field("dv_dx"), dv_dy=read_field("dv_dy"),
                    corr=read_field("corr"),
                    valid=(read_field("valid") if "valid" in g else None),
                    elapsed=float(g.attrs.get("elapsed_s", 0.0))
                )
                extra_fields = ("u_inc", "v_inc", "mag_inc",
                                "Vx", "Vy", "Veff", "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                                "Exx_rate", "Exy_rate", "Gxy_rate", "Eyy_rate", "Eeff_rate",
                                "Exx_inf", "Eyy_inf", "Exy_inf", "Gxy_inf", "Eeff_inf",
                                "Exx_gl", "Eyy_gl", "Exy_gl", "Gxy_gl", "Eeff_gl")
                for rate in extra_fields:
                    if rate in g:
                        setattr(res, rate, (g[rate] if lazy else
                                           _result_f32(g[rate][:])))

                # Loaded files use the same compact in-memory representation as
                # a fresh run. Schema-3 u_inc/v_inc and the legacy strain names
                # are exact aliases by definition; reading their duplicate HDF5
                # datasets into separate arrays used nearly twice the necessary
                # memory when reopening a long result file.
                if not lazy:
                    for name in ("u", "v", "Exx", "Exy", "Eyy", "Eeff",
                                 "du_dx", "du_dy", "dv_dx", "dv_dy", "corr"):
                        setattr(res, name, _result_f32(getattr(res, name)))
                    if res.valid is not None:
                        res.valid = np.asarray(res.valid, dtype=bool)
                if result_schema >= 3:
                    res.u_inc = res.u
                    res.v_inc = res.v
                if res.Exx_inf is not None:
                    res.Exx = res.Exx_inf
                    res.Exy = res.Exy_inf
                    res.Eyy = res.Eyy_inf
                    res.Eeff = res.Eeff_inf
                if res.Exx_rate is not None:
                    res.dVx_dx = res.Exx_rate
                if res.Eyy_rate is not None:
                    res.dVy_dy = res.Eyy_rate
                self.results.append(res)

                # Backward-compatible interpretation for files that predate
                # explicit formulation names. Their legacy strain is exposed as
                # infinitesimal only; Green-Lagrange remains unavailable rather
                # than being silently fabricated.
                if res.Exx_inf is None:
                    res.Exx_inf = res.Exx
                    res.Eyy_inf = res.Eyy
                    res.Exy_inf = res.Exy
                    # A lazy HDF5 dataset cannot represent this derived legacy
                    # alias without allocating the entire field. Modern schema-3
                    # files save Gxy_inf explicitly; malformed/partial files
                    # simply leave this optional compatibility field unavailable.
                    res.Gxy_inf = None if lazy else 2.0 * res.Exy
                    res.Eeff_inf = res.Eeff

            if "temporal_sequence" in f:
                from .temporal import HDF5TemporalResultSequence
                temporal = f["temporal_sequence"]
                pairs = [tuple(int(v) for v in row)
                         for row in temporal["pairs"][:]]
                metadata = {}
                for name, value in temporal.attrs.items():
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    elif isinstance(value, np.generic):
                        value = value.item()
                    metadata[str(name)] = value
                self.temporal_pairs = pairs
                self.temporal_metadata = metadata
                self.temporal_results = HDF5TemporalResultSequence(path, pairs)

        except Exception:
            f.close()
            raise

        if lazy:
            self._hdf5_handle = f
            self.hdf5_lazy = True
        else:
            f.close()

        # Files written before schema 3 stored cumulative u/v. Convert their
        # display fields to frame increments once; schema-3 files already store
        # immediate displacement and only need direct aliases.
        report(0.94, "Finalising loaded fields…")
        if self.results and result_schema < 3:
            prev_u = prev_v = prev_valid = None
            for res in self.results:
                here = np.isfinite(res.u) & np.isfinite(res.v)
                if prev_u is None:
                    u_i = np.where(here, res.u, np.nan)
                    v_i = np.where(here, res.v, np.nan)
                else:
                    both = here & prev_valid
                    u_i = np.where(both, res.u - prev_u, np.nan)
                    v_i = np.where(both, res.v - prev_v, np.nan)
                prev_u, prev_v, prev_valid = res.u.copy(), res.v.copy(), here
                res.u, res.v = u_i, v_i
                res.valid = np.isfinite(u_i) & np.isfinite(v_i)
            self._compute_incremental_displacements()
            self._compute_velocities_and_rates()
        elif self.results and any(r.u_inc is None for r in self.results):
            self._compute_incremental_displacements()
        report(1.0, f"Loaded {len(self.results)} frames.")

    def _get_settings_path(self) -> str:
        import os
        # PYDIC_SETTINGS_PATH lets a test (or a second instance) point at its own
        # file. Relying on HOME/USERPROFILE for that is unreliable -- expanduser
        # consults several variables in a platform-specific order, so a test can
        # believe it is sandboxed and still write over the real settings.
        override = os.environ.get("PYDIC_SETTINGS_PATH")
        if override:
            return override
        return os.path.join(os.path.expanduser("~"), ".pydic_settings.json")

    def load_settings(self) -> None:
        import json, os
        path = self._get_settings_path()

        if os.path.exists(path):
            try:
                # utf-8-sig, not the platform default. A settings file carrying a
                # UTF-8 BOM (anything that has touched it from PowerShell, an
                # editor, or a sync tool) otherwise raised JSONDecodeError, the
                # except branch below quietly kept the defaults, and the next
                # save_settings() wrote those defaults straight over the user's
                # real values. A read failure must never be able to destroy the
                # file -- see the guard in the handler.
                with open(path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)

                # Settings written before the tolerance/cutoff semantics changed
                # must not be replayed: conv_tol is now in pixels of subset-edge
                # motion (1e-6 px is unreachable) and corr_cutoff=2.0 means
                # "accept everything" on the [0,4] ZNSSD scale. Drop those keys
                # and let the current defaults stand.
                SCHEMA = 2
                stale = int(data.get("schema_version", 1)) < SCHEMA
                DROP_ON_MIGRATE = {"conv_tol", "corr_cutoff"}
                migrated = []
                if "calibration" in data:
                    self.calibration = Calibration.from_dict(data.get("calibration"))
                self.prefer_gpu = bool(data.get("prefer_gpu", self.prefer_gpu))

                for k, v in data.items():
                    if k in ("schema_version", "calibration", "prefer_gpu"):
                        continue
                    if stale and k in DROP_ON_MIGRATE:
                        migrated.append(k)
                        self.settings_notices.append(
                            f"“{k}” was written under older semantics and has been "
                            f"reset to the current default "
                            f"({getattr(self.params, k, '?')}).")
                        continue
                    if hasattr(self.params, k):
                        setattr(self.params, k, v)
                # The schema gate above only fires once, so a value that is
                # unreachable under the CURRENT semantics can survive in a file
                # already stamped with the current schema. conv_tol is the one
                # that matters: it is a subset-edge motion in pixels, so
                # anything below ~1e-4 px can never be met and every subset
                # burns all max_iter iterations before giving up -- several
                # times slower for no gain in accuracy. Clamp on load.
                CONV_TOL_MIN, CONV_TOL_MAX = 1e-4, 1e-1
                ct = float(self.params.conv_tol)
                if not (CONV_TOL_MIN <= ct <= CONV_TOL_MAX):
                    clamped = min(max(ct, CONV_TOL_MIN), CONV_TOL_MAX)
                    print(f"[Settings] conv_tol={ct:g} px is outside the usable "
                          f"range [{CONV_TOL_MIN:g}, {CONV_TOL_MAX:g}]; using {clamped:g}.")
                    self.params.conv_tol = clamped
                    migrated.append("conv_tol")
                    self.settings_notices.append(
                        f"Convergence tolerance was {ct:g} px, which no subset can "
                        f"ever reach — every one ran the full iteration budget. "
                        f"Raised to {clamped:g} px.")

                if migrated:
                    print(f"[Settings] Migrated to schema {SCHEMA}; reset to new "
                          f"defaults: {', '.join(sorted(set(migrated)))}")
                    self.save_settings()

                # Load all specialized directories
                dirs = ["last_video_directory", "last_image_directory", "last_hdf5_directory"]
                for d in dirs:
                    if d in data and os.path.exists(data[d]):
                        setattr(self, d, data[d])

            except Exception as e:
                # Do NOT let a later save overwrite a file we could not read.
                # Failing to parse means the user's real settings are still in
                # there; silently replacing them with defaults turns a recoverable
                # read problem into permanent data loss.
                self._settings_readable = False
                print(f"[Warning] Failed to load settings: {e}")
                self.settings_notices.append(
                    f"Your settings file could not be read ({e}). Defaults are in "
                    f"use for this session and the file has been left untouched: "
                    f"{path}")
        else:
            self.save_settings()

    def save_settings(self) -> None:
        import json, os
        path = self._get_settings_path()

        # Refuse to write over a file we failed to parse -- the values in it are
        # the user's, and defaults would clobber them irrecoverably.
        if not getattr(self, "_settings_readable", True):
            print(f"[Settings] Not overwriting unreadable settings file: {path}")
            return

        try:
            data = {
                "schema_version": 2,
                "subset_radius": self.params.subset_radius,
                "subset_spacing": self.params.subset_spacing,
                "strain_window": self.params.strain_window,
                "max_iter": self.params.max_iter,
                "conv_tol": self.params.conv_tol,
                "corr_cutoff": self.params.corr_cutoff,
                "search_radius": self.params.search_radius,
                "rescue_radius": getattr(self.params, "rescue_radius", 12),
                "dynamic_roi": getattr(self.params, "dynamic_roi", "None"),
                "dynamic_roi_threshold": getattr(self.params, "dynamic_roi_threshold", None),
                "dynamic_roi_min_area_frac": getattr(self.params, "dynamic_roi_min_area_frac", 0.02),
                "dynamic_roi_fill_holes": getattr(self.params, "dynamic_roi_fill_holes", True),
                "calibration": self.calibration.to_dict(),
                "prefer_gpu": bool(getattr(self, "prefer_gpu", True)),
                "shape_order": getattr(self.params, "shape_order", 1),
                "mask_subsets_to_roi": getattr(self.params, "mask_subsets_to_roi", True),

                # Save all specialized directories
                "last_video_directory": getattr(self, "last_video_directory", os.path.expanduser("~")),
                "last_image_directory": getattr(self, "last_image_directory", os.path.expanduser("~")),
                "last_hdf5_directory": getattr(self, "last_hdf5_directory", os.path.expanduser("~")),
            }
            # Atomic replacement: a crash or forced close during json.dump must
            # not leave a half-written settings file that disconnects every UI
            # control from its cached value on the next launch.
            import tempfile
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".pydic_settings_", suffix=".tmp", dir=parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            print(f"[Error] Failed to save settings to {path}: {e}")

def _load_image(path: str) -> np.ndarray:
    if _HAVE_CV2:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read: {path}")
        mx = float(np.iinfo(img.dtype).max) if img.dtype.kind == "u" else 1.0
        # Source greyscale contains at most 16 useful bits. float32 represents
        # every normalised input level exactly enough for display and halves
        # the resident reference/preview image. Solvers promote to float64 at
        # their numerical boundary.
        return img.astype(np.float32) / np.float32(mx)
    elif _HAVE_PIL:
        return (np.asarray(PILImage.open(path).convert("L"), np.float32) /
                np.float32(255.0))
    else:
        raise ImportError("Install opencv-python or Pillow.")

class DynamicROI:
    """
    Texture-based dynamic ROI with a STATIONARY decision rule.

    The previous implementation recomputed both the normalisation and an Otsu
    threshold from every individual frame. Two consequences:

      * per-frame max normalisation meant one specular flash or the tool edge
        rescaled the whole metric, moving every pixel relative to the threshold;
      * per-frame Otsu moved the threshold itself as the scene evolved.

    Together they make the ROI boundary jitter frame to frame, which is what
    produces a ragged, frame-varying mask edge and inconsistent pair coverage.

    Here the scale and threshold are calibrated ONCE from the selected zero-strain frame and
    then held fixed, so "enough texture to correlate" means the same thing in
    frame 500 as in frame 1.
    """

    def __init__(self, method: str, keep_min_area_frac: float = 0.02,
                 threshold: Optional[float] = None,
                 include_mask: Optional[np.ndarray] = None,
                 exclude_mask: Optional[np.ndarray] = None,
                 roi_mask: Optional[np.ndarray] = None,
                 fill_holes: bool = True,
                 hysteresis: float = 0.03):
        self.method = method
        self.keep_min_area_frac = keep_min_area_frac
        # Fill regions that the texture metric rejected but which are completely
        # enclosed by kept material. A hole in the middle of valid material is
        # almost always a local dropout -- a glare spot, a patch where the
        # speckle is momentarily washed out -- not an absence of specimen, and
        # leaving it out fragments the field for no physical reason. Morphological
        # closing alone only ever fixed holes smaller than its kernel.
        self.fill_holes = bool(fill_holes)
        # The static ROI the operator drew. The dynamic mask REFINES that region
        # rather than competing with it: nothing outside it is ever kept, and --
        # just as importantly -- the threshold is calibrated only from pixels
        # inside it. Calibrating over the whole frame let background and tooling
        # dominate the statistics, so the threshold that came back described the
        # scene rather than the specimen.
        self.roi_mask = None if roi_mask is None else roi_mask.astype(bool)
        # Normalised texture threshold in [0, 1]. None means "pick it with Otsu
        # on the selected calibration frame", which stays the
        # default; the dynamic-ROI editor sets it explicitly when the user
        # drags the slider.
        self.threshold = threshold
        # A small Schmitt-trigger band stops borderline texture pixels from
        # alternating in/out on successive frames. It does not smooth geometry
        # or prevent a real scene change from moving the boundary.
        self.hysteresis = max(0.0, float(hysteresis))
        self._previous_mask: Optional[np.ndarray] = None
        # Hard user overrides, in REFERENCE-frame coordinates. include wins over
        # exclude wins over the texture metric, so a region the operator marked
        # is never silently reinterpreted frame to frame.
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask
        self.scale = None
        self.thresh = None
        self.auto_thresh = None
        self.enabled = method not in ("None", None) and _HAVE_CV2
        if method not in ("None", None) and not _HAVE_CV2:
            print("[Warning] cv2 not available for dynamic ROI. Ignoring.")

    def _metric(self, img: np.ndarray) -> Optional[np.ndarray]:
        # Dynamic ROI is a display/eligibility classifier, not the sub-pixel
        # solver. float32 halves its several full-frame blur/Sobel temporaries
        # and is amply precise for the final 8-bit threshold metric.
        work = np.asarray(img, dtype=np.float32)
        if self.method == "Contrast":
            mean = cv2.blur(work, (9, 9))
            var = cv2.blur(work ** 2, (9, 9)) - mean ** 2
            return np.sqrt(np.maximum(var, 0))
        if self.method == "Edge Detection":
            gxx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
            gyy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
            return np.sqrt(gxx ** 2 + gyy ** 2)
        if self.method == "Hybrid":
            mean = cv2.blur(work, (9, 9))
            var = cv2.blur(work ** 2, (9, 9)) - mean ** 2
            std = np.sqrt(np.maximum(var, 0))
            gxx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
            gyy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gxx ** 2 + gyy ** 2)
            # Put both terms on a comparable footing before adding them; the raw
            # sum was dominated by the unnormalised Sobel response.
            sd = std / max(std.std(), 1e-12)
            gd = grad / max(grad.std(), 1e-12)
            return sd + gd
        return None

    def calibrate(self, ref_image: np.ndarray) -> None:
        if not self.enabled:
            return
        m = self._metric(ref_image)
        if m is None:
            self.enabled = False
            return
        # Scale and Otsu threshold from ROI pixels only -- see roi_mask above.
        sel = m[self.roi_mask] if (self.roi_mask is not None and self.roi_mask.any()) else m
        self.scale = float(np.percentile(sel, 99.0)) or 1.0
        m8 = self._metric8(ref_image, m)
        sel8 = m8[self.roi_mask] if (self.roi_mask is not None and self.roi_mask.any()) else m8
        thr, _ = cv2.threshold(sel8.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.auto_thresh = float(thr)
        self.set_threshold_normalised(self.threshold)
        self._previous_mask = None

    def set_threshold_normalised(self, threshold: Optional[float]) -> None:
        """Select the threshold for the next frame without recalibrating scale."""
        self.threshold = threshold
        self.thresh = (self.auto_thresh if threshold is None
                       else float(np.clip(threshold, 0.0, 1.0)) * 255.0)

    def _metric8(self, img: np.ndarray, m: Optional[np.ndarray] = None):
        """Texture metric rescaled to uint8 against the calibrated scale."""
        if m is None:
            m = self._metric(img)
        if m is None:
            return None
        s = self.scale if self.scale else (float(np.percentile(m, 99.0)) or 1.0)
        m8 = np.clip(m / s * 255.0, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(m8, (5, 5), 0)

    def auto_threshold_normalised(self) -> Optional[float]:
        return None if self.auto_thresh is None else self.auto_thresh / 255.0

    def mask(self, cur_image: np.ndarray,
             reference_frame: bool = False) -> Optional[np.ndarray]:
        """Return the texture mask in the coordinate system of ``cur_image``.

        ``roi_mask`` and manual overrides are image-space regions previewed on
        the selected zero-strain image. During analysis, the unconstrained
        current-frame texture mask is sampled at each previous-frame subset
        centre plus that pair's displacement. Source-space overrides are then
        applied by :func:`_dynamic_measurement_mask`.
        """
        if not self.enabled:
            return None
        if self.thresh is None:
            self.calibrate(cur_image)
        m8 = self._metric8(cur_image)
        if m8 is None:
            return None
        if (not reference_frame and self._previous_mask is not None and
                self._previous_mask.shape == m8.shape and self.hysteresis > 0.0):
            band = self.hysteresis * 255.0
            # Previously kept pixels use the lower exit threshold; previously
            # rejected pixels use the upper entry threshold.
            mask = np.where(
                self._previous_mask,
                m8 >= max(0.0, self.thresh - band),
                m8 >= min(255.0, self.thresh + band),
            ).astype(np.uint8) * 255
        else:
            mask = (m8 >= self.thresh).astype(np.uint8) * 255

        # Overrides participate directly in the dense setup preview. During a
        # pair they are applied at source image-space centres by
        # _dynamic_measurement_mask.
        if reference_frame:
            if self.exclude_mask is not None:
                mask[self.exclude_mask] = 0
            if self.include_mask is not None:
                mask[self.include_mask] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Keep every sufficiently large connected region, not just the largest.
        # In machining the chip and the workpiece are separate bodies (and the
        # tool splits the field); keeping only the largest contour discards one
        # of them outright, and which one is "largest" can flip between frames.
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_lab <= 1:
            out = mask > 0
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            min_area = self.keep_min_area_frac * float(areas.max())
            keep = np.zeros(n_lab, bool)
            keep[1:] = areas >= min_area
            out = keep[labels]

        # The setup preview is clipped to the analysis ROI. Current-frame masks
        # stay in current image coordinates and are sampled after one pair's
        # displacement.
        if reference_frame and self.roi_mask is not None:
            out = out & self.roi_mask

        # Contour fill: anything fully surrounded by kept material becomes kept.
        # Unlike the morphological close above, this has no size limit -- a hole
        # is filled because it is enclosed, not because it is small.
        if self.fill_holes and out.any():
            from scipy.ndimage import binary_fill_holes
            out = binary_fill_holes(out)
            if reference_frame and self.roi_mask is not None:
                out = out & self.roi_mask

        # Re-apply the operator's decisions LAST. Setting them before morphology
        # was not enough to make them stick, and both were being partly undone:
        #
        #   * closing is a dilate-then-erode, so it grew neighbouring kept
        #     material back over the edge of an excluded region;
        #   * component pruning deleted a force-included patch whenever it was
        #     smaller than keep_min_area_frac of the largest region -- measured
        #     at 0 of 900 px surviving, i.e. include did nothing at all.
        #
        # Doing it here also means an explicit Exclude beats the hole fill, which
        # is the escape hatch when an enclosed gap is a genuine void rather than
        # a texture dropout. Precedence: include > exclude > fill > metric.
        if reference_frame:
            if self.exclude_mask is not None:
                out = out & ~self.exclude_mask
            if self.include_mask is not None:
                out = out | self.include_mask
                if self.roi_mask is not None:
                    out = out & self.roi_mask
        else:
            self._previous_mask = out.copy()
        return out


# The old _compute_dynamic_mask() wrapper is gone. It recalibrated scale and
# threshold on whichever frame it was handed, which is precisely the per-frame
# drift DynamicROI exists to avoid, and it had no way to receive the static ROI
# or the operator's include/exclude regions. Build the object through
# DICAnalysis.make_dynamic_roi() and calibrate it once on the selected frame.
