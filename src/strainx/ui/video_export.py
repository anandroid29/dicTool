"""
video_export.py — render result views to a video file or an image sequence.

Rendering runs entirely on numpy/OpenCV (see render.py) rather than by grabbing
the canvas widget, so it is safe on a worker thread, produces a fixed output
resolution regardless of the window size, and does not fight Qt's pixmap thread
affinity.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field as _dcfield
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from strainx.core.compact_field import finite_values

from strainx.core.units import Calibration
from . import render as R

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# Same palette as the on-screen markers (image_canvas.MARKER_PALETTE), as RGB.
# Duplicated as plain tuples so this module stays importable without Qt.
MARKER_RGB = [
    (0xff, 0x5c, 0x5c), (0xff, 0xd9, 0x3d), (0x4a, 0xde, 0x80),
    (0x38, 0xbd, 0xf8), (0xc0, 0x84, 0xfc), (0xfb, 0x92, 0x3c),
    (0x2d, 0xd4, 0xbf), (0xf4, 0x72, 0xb6), (0xa3, 0xe6, 0x35),
    (0x81, 0x8c, 0xf8),
]

# Container -> (extension, fourcc). H264/avc1 are omitted deliberately: on this
# machine cv2 reports the writer as opened but OpenH264 fails to initialise, so
# the result is unreliable. mp4v is the safe, widely-playable default; FFV1 is
# lossless for archival.
CODECS = {
    "MP4 (mp4v)":        (".mp4", "mp4v"),
    "AVI (XVID)":        (".avi", "XVID"),
    "AVI (MJPG)":        (".avi", "MJPG"),
    "AVI (FFV1 lossless)": (".avi", "FFV1"),
    "PNG image sequence": (".png", None),
}


@dataclass
class ExportSpec:
    rows: int = 1
    cols: int = 1
    panels: List[R.PanelSpec] = _dcfield(default_factory=list)
    cell_w: int = 640
    cell_h: int = 480
    fps: float = 25.0
    codec: str = "MP4 (mp4v)"
    gap: int = 4
    first: int = 0
    last: int = -1          # -1 = to the end
    trail: int = 0          # streakline trail length, 0 = full history

    @property
    def writes_sequence(self) -> bool:
        return CODECS.get(self.codec, (None, None))[1] is None


class ViewRenderer:
    """Turns (frame index, PanelSpec) into an RGB image.

    Holds no Qt state. Everything it needs comes from the DICAnalysis, so the
    exporter can run headless while the GUI carries on.
    """

    def __init__(self, analysis, markers: Optional[Sequence] = None,
                 trail: int = 0, results=None, pairs=None) -> None:
        self.analysis = analysis
        self.results = analysis.results if results is None else results
        self.pairs = None if pairs is None else list(pairs)
        self.markers = list(markers or [])
        self.trail = int(trail)
        self._img_cache: dict = {}
        self._result_cache: dict = {}
        self._traj_cache: dict = {}
        self._unit_cache: dict = {}
        self._global_cache: dict = {}

    def _source_index(self, idx: int) -> int:
        return int(self.pairs[idx][1]) if self.pairs is not None else int(idx)

    def _result(self, idx: int):
        if idx not in self._result_cache:
            if len(self._result_cache) >= 2:
                self._result_cache.clear()
            self._result_cache[idx] = self.results[idx]
        return self._result_cache[idx]

    # -- data access ------------------------------------------------------
    def _deformed(self, idx: int) -> Optional[np.ndarray]:
        source = self._source_index(idx)
        if source in self._img_cache:
            return self._img_cache[source]
        from strainx.core.analysis import _load_image
        try:
            if self.pairs is not None and source < len(self.analysis.def_paths):
                path = self.analysis.def_paths[source]
            else:
                path = self.analysis.results[source].image_path
            img = _load_image(path)
        except Exception:
            img = None
        # Only the two most recent frames are worth keeping; a full-sequence
        # cache is what turns a long export into an out-of-memory crash.
        if len(self._img_cache) > 2:
            self._img_cache.clear()
        self._img_cache[source] = img
        return img

    def field_array(self, idx: int, field: str) -> Tuple[Optional[np.ndarray], str]:
        """Field in display units, plus its unit label."""
        from strainx.ui.pages.results_page import FIELDS
        res = self._result(idx)
        arr = getattr(res, field, None)
        base = FIELDS.get(field, ("", ""))[1]
        if (arr is not None and bool(getattr(self.analysis, "hdf5_lazy", False))
                and not isinstance(arr, np.ndarray)):
            arr = np.asarray(arr)
        factor, unit = self._field_factor_and_unit(field, base, arr)
        if arr is None or factor == 1.0:
            return arr, unit
        return arr * factor, unit

    def _field_factor_and_unit(
            self, field: str, base: str, native_arr=None) -> Tuple[float, str]:
        """Use the same sequence-stable compact units as the results viewer."""
        cal: Calibration = self.analysis.calibration
        key = (field, cal.metres_per_pixel, cal.display_unit)
        cached = self._unit_cache.get(key)
        if cached is not None:
            return cached
        factor_unit = cal.factor_and_unit(field, base)
        if cal.calibrated and self.results:
            if bool(getattr(self.analysis, "hdf5_lazy", False)) and native_arr is not None:
                values = np.asarray(native_arr)
                finite = np.abs(values[np.isfinite(values)])
                magnitude = (float(np.percentile(finite, 99.0))
                             if finite.size else 0.0)
            else:
                lo, hi = self._native_global_range(field, 99.0)
                magnitude = max(abs(float(lo)), abs(float(hi)))
            factor_unit = cal.compact_factor_and_unit(field, magnitude, base)
        self._unit_cache[key] = factor_unit
        return factor_unit

    def global_range(self, field: str) -> Tuple[float, float]:
        lo, hi = self._native_global_range(field, 100.0)
        from strainx.ui.pages.results_page import FIELDS
        factor, _ = self._field_factor_and_unit(
            field, FIELDS.get(field, ("", ""))[1])
        return lo * factor, hi * factor

    def _native_global_range(self, field: str, coverage: float) -> Tuple[float, float]:
        key = (field, float(coverage))
        cached = self._global_cache.get(key)
        if cached is not None:
            return cached
        if self.results is self.analysis.results:
            value = self.analysis.get_global_range(field, coverage)
            self._global_cache[key] = value
            return value
        from strainx.core.stats import robust_limits
        pooled = []
        stride = max(1, len(self.results) // 200)
        for idx in range(0, len(self.results), stride):
            field_values = getattr(self._result(idx), field, None)
            if field_values is None:
                continue
            values = finite_values(field_values)
            if values.size > 50_000:
                values = values[::values.size // 50_000 + 1]
            if values.size:
                pooled.append(values)
        limits = robust_limits(np.concatenate(pooled), coverage) if pooled else None
        value = limits if limits is not None else (0.0, 1.0)
        self._global_cache[key] = value
        return value

    def _trajectories(self, idx: int):
        key = (idx, self.trail)
        if key not in self._traj_cache:
            self._traj_cache.clear()
            self._traj_cache[key] = self.analysis.get_trajectories_from_seeds(
                self.markers, self._source_index(idx), self.trail
            ) if self.markers else []
        return self._traj_cache[key]

    # -- rendering --------------------------------------------------------
    def render_panel(self, idx: int, spec: R.PanelSpec) -> np.ndarray:
        ref = self.analysis.reference_image
        deformed = self._deformed(idx)
        shape = (deformed if deformed is not None else ref).shape[:2]

        rgb, _alpha = R._background_rgb(spec, deformed, ref, shape)

        unit = ""
        vmin = vmax = None
        clip_low = clip_high = False
        if spec.content == "field":
            arr, unit = self.field_array(idx, spec.field)
            rng = spec.range_spec.resolve(
                arr, self.global_range(spec.field) if spec.range_spec.mode == "global" else None)
            if rng is not None and arr is not None:
                vmin, vmax = rng
                rgba = R.field_to_rgba(
                    arr, vmin, vmax, spec.cmap,
                    roi_mask=self.analysis.roi_mask,
                    spacing=getattr(self.analysis.params, "subset_spacing", 3),
                    mark_out_of_range=spec.mark_out_of_range)
                rgb = R.alpha_over(rgb, rgba)
                # An exported figure leaves the tool and gets read on its own,
                # so a trimmed scale has to declare itself on the colourbar.
                finite = finite_values(arr)
                clip_low = bool(np.any(finite < vmin))
                clip_high = bool(np.any(finite > vmax))

        if spec.wants_streaklines and self.markers:
            # Trajectory only. The marker circle used to be stamped on the
            # leading end of every streakline, which hid the very tip the
            # viewer is trying to follow.
            rgb = R.draw_streaklines(rgb, self._trajectories(idx), MARKER_RGB)

        if spec.content == "field" and spec.show_colorbar and vmin is not None:
            rgb = R.draw_colorbar(rgb, spec.cmap, vmin, vmax, unit,
                                  clipped_low=clip_low, clipped_high=clip_high)

        if spec.show_label:
            label = spec.label or self._default_label(spec)
            rgb = R.draw_label(rgb, label)
        return rgb

    def _default_label(self, spec: R.PanelSpec) -> str:
        if spec.content == "image":
            return "Raw frame"
        if spec.content == "streaklines":
            return "Streaklines"
        if spec.content == "field":
            from strainx.ui.pages.results_page import FIELDS
            return FIELDS.get(spec.field, (spec.field, ""))[0]
        return ""

    def render_frame(self, idx: int, spec: ExportSpec) -> np.ndarray:
        panels = [self.render_panel(idx, p) for p in spec.panels[:spec.rows * spec.cols]]
        if spec.rows == 1 and spec.cols == 1 and panels:
            return R.fit_into(panels[0], spec.cell_w, spec.cell_h)
        return R.compose_grid(panels, spec.rows, spec.cols, spec.cell_w, spec.cell_h, spec.gap)


def export_video(analysis, spec: ExportSpec, path: str,
                 markers: Optional[Sequence] = None,
                 progress_cb: Optional[Callable[[float, str], None]] = None,
                 cancel_flag: Optional[list] = None,
                 results=None, pairs=None) -> str:
    """Render frames [first, last] and write them out. Returns the output path."""
    if not _HAVE_CV2:
        raise RuntimeError("OpenCV is required for video export.")
    export_results = analysis.results if results is None else results
    n = len(export_results)
    if n == 0:
        raise RuntimeError("No results to export.")

    first = max(0, int(spec.first))
    last = n - 1 if spec.last is None or spec.last < 0 else min(int(spec.last), n - 1)
    if last < first:
        raise ValueError("Empty frame range.")

    renderer = ViewRenderer(
        analysis, markers, spec.trail, results=export_results, pairs=pairs)
    cancel_flag = cancel_flag if cancel_flag is not None else [False]

    probe = renderer.render_frame(first, spec)
    h, w = probe.shape[:2]

    writer = None
    seq_dir = None
    if spec.writes_sequence:
        seq_dir = os.path.splitext(path)[0]
        os.makedirs(seq_dir, exist_ok=True)
    else:
        ext, fourcc = CODECS[spec.codec]
        if not path.lower().endswith(ext):
            path = os.path.splitext(path)[0] + ext
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                                 float(spec.fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not open a video writer for {spec.codec}. "
                f"Try 'AVI (MJPG)', which needs no external codec.")

    total = last - first + 1
    try:
        for k, idx in enumerate(range(first, last + 1)):
            if cancel_flag[0]:
                break
            frame = probe if idx == first else renderer.render_frame(idx, spec)
            if frame.shape[:2] != (h, w):
                frame = R.fit_into(frame, w, h)
            if writer is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(os.path.join(seq_dir, f"frame_{idx:05d}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if progress_cb:
                progress_cb((k + 1) / total, f"Frame {idx + 1}/{last + 1}")
    finally:
        if writer is not None:
            writer.release()
    return seq_dir if seq_dir else path
