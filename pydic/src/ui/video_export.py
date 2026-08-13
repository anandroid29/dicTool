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

from src.core.units import Calibration
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
                 trail: int = 0) -> None:
        self.analysis = analysis
        self.markers = list(markers or [])
        self.trail = int(trail)
        self._img_cache: dict = {}
        self._traj_cache: dict = {}

    # -- data access ------------------------------------------------------
    def _deformed(self, idx: int) -> Optional[np.ndarray]:
        if idx in self._img_cache:
            return self._img_cache[idx]
        from src.core.analysis import _load_image
        try:
            img = _load_image(self.analysis.results[idx].image_path)
        except Exception:
            img = None
        # Only the two most recent frames are worth keeping; a full-sequence
        # cache is what turns a long export into an out-of-memory crash.
        if len(self._img_cache) > 2:
            self._img_cache.clear()
        self._img_cache[idx] = img
        return img

    def field_array(self, idx: int, field: str) -> Tuple[Optional[np.ndarray], str]:
        """Field in display units, plus its unit label."""
        from src.ui.pages.results_page import FIELDS
        res = self.analysis.results[idx]
        arr = getattr(res, field, None)
        base = FIELDS.get(field, ("", ""))[1]
        cal: Calibration = self.analysis.calibration
        return cal.convert(field, arr, base)

    def global_range(self, field: str) -> Tuple[float, float]:
        lo, hi = self.analysis.get_global_range(field)
        from src.ui.pages.results_page import FIELDS
        factor, _ = self.analysis.calibration.factor_and_unit(
            field, FIELDS.get(field, ("", ""))[1])
        return lo * factor, hi * factor

    def _trajectories(self, idx: int):
        key = (idx, self.trail)
        if key not in self._traj_cache:
            self._traj_cache.clear()
            self._traj_cache[key] = self.analysis.get_trajectories_from_seeds(
                self.markers, idx, self.trail) if self.markers else []
        return self._traj_cache[key]

    # -- rendering --------------------------------------------------------
    def render_panel(self, idx: int, spec: R.PanelSpec) -> np.ndarray:
        ref = self.analysis.reference_image
        deformed = self._deformed(idx)
        shape = (deformed if deformed is not None else ref).shape[:2]

        rgb, _alpha = R._background_rgb(spec, deformed, ref, shape)

        unit = ""
        vmin = vmax = None
        if spec.content == "field":
            arr, unit = self.field_array(idx, spec.field)
            rng = spec.range_spec.resolve(
                arr, self.global_range(spec.field) if spec.range_spec.mode == "global" else None)
            if rng is not None and arr is not None:
                vmin, vmax = rng
                rgba = R.field_to_rgba(
                    arr, vmin, vmax, spec.cmap,
                    roi_mask=self.analysis.roi_mask,
                    spacing=getattr(self.analysis.params, "subset_spacing", 3))
                rgb = R.alpha_over(rgb, rgba)

        if spec.wants_streaklines and self.markers:
            # Trajectory only. The marker circle used to be stamped on the
            # leading end of every streakline, which hid the very tip the
            # viewer is trying to follow.
            rgb = R.draw_streaklines(rgb, self._trajectories(idx), MARKER_RGB)

        if spec.content == "field" and spec.show_colorbar and vmin is not None:
            rgb = R.draw_colorbar(rgb, spec.cmap, vmin, vmax, unit)

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
            from src.ui.pages.results_page import FIELDS
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
                 cancel_flag: Optional[list] = None) -> str:
    """Render frames [first, last] and write them out. Returns the output path."""
    if not _HAVE_CV2:
        raise RuntimeError("OpenCV is required for video export.")
    n = len(analysis.results)
    if n == 0:
        raise RuntimeError("No results to export.")

    first = max(0, int(spec.first))
    last = n - 1 if spec.last is None or spec.last < 0 else min(int(spec.last), n - 1)
    if last < first:
        raise ValueError("Empty frame range.")

    renderer = ViewRenderer(analysis, markers, spec.trail)
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
