"""
render.py — pure numpy/OpenCV compositing of a result view.

Everything here works on arrays and returns arrays. Nothing touches a QWidget.

That is deliberate. ImageCanvas.paintEvent draws straight onto the widget with
`QPainter(self)` and scales by the live zoom/pan, so it can only ever produce
whatever the on-screen widget currently looks like at whatever size it happens
to be, on the GUI thread. Video export needs a fixed output resolution,
independent of the window, running on a worker thread, for hundreds of frames --
and QPixmap has thread affinity, so grabbing the widget from a worker is not an
option either.

So the field-to-RGBA step lives HERE and is called by both the on-screen overlay
and the exporter. That is the point: there is one implementation of "what does
this field look like", and the two paths cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dcfield
from typing import Optional, Sequence, Tuple

import numpy as np

# Colour limits are a display concern but the analysis layer needs the same
# definition for its sequence-wide range, so the primitive lives in core and
# both sides import it from there.
from src.core.stats import robust_limits

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

def get_cmap(name: str, n: int = 256):
    """matplotlib colormap, tolerant of the 3.9 API change."""
    import matplotlib
    try:
        return matplotlib.colormaps[name].resampled(n)
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        return cm.get_cmap(name, n)


# ---------------------------------------------------------------------------
# Colour range
# ---------------------------------------------------------------------------

@dataclass
class RangeSpec:
    """How the colour limits for a field are chosen.

    mode:
      "auto"   - limits from the frame being shown
      "global" - limits from the whole sequence, so frames stay comparable
      "manual" - the numbers the user typed

    `percentile` is the central coverage used by auto and global (100 = raw
    min/max). It does not apply to manual, where the user has stated the limits
    outright.

    Manual limits are in DISPLAY units (what the colourbar is labelled with),
    because that is what the user is reading when they type them. Auto/global
    are derived from the already-converted array, so all three end up on the
    same footing and the overlay, colourbar and statistics agree.
    """
    mode: str = "auto"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    symmetric: bool = False
    percentile: float = 100.0

    def resolve(self, arr: np.ndarray,
                global_range: Optional[Tuple[float, float]] = None
                ) -> Optional[Tuple[float, float]]:
        vmin = vmax = None
        if self.mode == "manual" and self.vmin is not None and self.vmax is not None:
            vmin, vmax = float(self.vmin), float(self.vmax)
        elif self.mode == "global" and global_range is not None:
            vmin, vmax = float(global_range[0]), float(global_range[1])
        else:
            if arr is None:
                return None
            limits = robust_limits(np.asarray(arr), self.percentile)
            if limits is None:
                return None
            vmin, vmax = limits

        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            return None
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if self.symmetric:
            lim = max(abs(vmin), abs(vmax))
            vmin, vmax = -lim, lim
        if vmin == vmax:
            # A perfectly uniform field. Widen by a relative amount: the old
            # fixed 1e-12 was below the resolution of a value of order 1e3, so
            # normalising against it amplified float noise into full-scale
            # colour banding across what should read as one flat value.
            pad = max(abs(vmin) * 1e-6, 1e-12)
            vmin, vmax = vmin - pad, vmax + pad
        return vmin, vmax


# ---------------------------------------------------------------------------
# Field -> RGBA overlay
# ---------------------------------------------------------------------------

# Colours for values outside the chosen range. Deliberately outside every
# scientific colormap's gamut so clipping is unmistakable rather than being
# mistaken for a genuine extreme.
OVER_RANGE_RGB  = (255, 0, 255)     # magenta -- above vmax
UNDER_RANGE_RGB = (0, 255, 255)     # cyan    -- below vmin


def field_to_rgba(arr: np.ndarray, vmin: float, vmax: float, cmap_name: str,
                  roi_mask: Optional[np.ndarray] = None,
                  spacing: int = 3, alpha: int = 195,
                  mark_out_of_range: bool = False) -> Optional[np.ndarray]:
    """Colour-mapped RGBA overlay for a sparse field, as uint8 (H, W, 4).

    The field is only populated on the correlation grid, so it is decimated to
    that grid, hole-filled by nearest neighbour, colour-mapped, and then resized
    back up. Doing it in that order is what avoids the blocky look: the
    interpolation happens on colours at grid resolution rather than on a
    mostly-NaN full-resolution array.

    The colormap is applied exactly as defined. Nothing is done to the colours
    afterwards, because the colourbar is drawn from the same colormap and any
    post-hoc adjustment here would break the correspondence between a colour on
    screen and the value the bar says it means.

    With `mark_out_of_range`, values beyond the limits are flagged in colours no
    colormap contains, so a scale that is hiding data says so.
    """
    if arr is None or not _HAVE_CV2:
        return None
    import matplotlib.colors as mc

    valid = np.isfinite(arr)
    if roi_mask is not None:
        valid &= np.asarray(roi_mask, dtype=bool)
    if not valid.any():
        return None

    ys, xs = np.where(valid)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())

    cropped = arr[ymin:ymax + 1, xmin:xmax + 1].copy()
    cropped_mask = valid[ymin:ymax + 1, xmin:xmax + 1]

    s = max(1, int(spacing))
    small = cropped[0::s, 0::s]
    small_mask = cropped_mask[0::s, 0::s]

    if not small_mask.any():
        return None

    norm = mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap_obj = get_cmap(cmap_name, 256)

    safe_small = np.where(small_mask, small, vmin)
    rgba_small = cmap_obj(norm(safe_small), bytes=True).astype(np.float32)

    if mark_out_of_range:
        # Compare before normalisation clips, so the test sees the real values.
        over = small_mask & (small > vmax)
        under = small_mask & (small < vmin)
        if over.any():
            rgba_small[over, :3] = OVER_RANGE_RGB
        if under.any():
            rgba_small[under, :3] = UNDER_RANGE_RGB

    # Resize premultiplied colour and coverage separately. Interpolating raw RGB
    # lets arbitrary colours stored at transparent/invalid pixels bleed into a
    # specimen edge; premultiplication makes invalid pixels contribute exactly
    # zero to both display and export.
    coverage_small = small_mask.astype(np.float32)
    premult_small = rgba_small[..., :3] * coverage_small[..., None]

    th, tw = ymax - ymin + 1, xmax - xmin + 1
    coverage = cv2.resize(
        coverage_small, (tw, th), interpolation=cv2.INTER_LINEAR)
    premult = cv2.resize(
        premult_small, (tw, th), interpolation=cv2.INTER_LINEAR)
    if premult.ndim == 2:
        premult = premult[..., None]
    rgb = np.zeros_like(premult, dtype=np.float32)
    np.divide(premult, coverage[..., None], out=rgb,
              where=coverage[..., None] > 1e-6)
    big = np.zeros((th, tw, 4), dtype=np.float32)
    big[..., :3] = rgb
    big[..., 3] = coverage * float(alpha)

    out = np.zeros((arr.shape[0], arr.shape[1], 4), np.float32)
    out[ymin:ymax + 1, xmin:xmax + 1] = big
    if roi_mask is not None:
        out[~roi_mask, 3] = 0.0
    out[~np.isfinite(out)] = 0.0
    return np.ascontiguousarray(out.astype(np.uint8))


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    """A [0,1] or [0,255] greyscale image as uint8 RGB."""
    a = np.asarray(img, np.float64)
    if a.size and np.nanmax(a) <= 1.5:
        a = a * 255.0
    a = np.clip(np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0),
                0, 255).astype(np.uint8)
    return np.dstack([a, a, a])


def alpha_over(base_rgb: np.ndarray, rgba: Optional[np.ndarray]) -> np.ndarray:
    """Standard source-over composite of an RGBA layer onto an RGB base."""
    if rgba is None:
        return base_rgb
    a = (rgba[..., 3:4].astype(np.float32) / 255.0)
    return np.clip(base_rgb.astype(np.float32) * (1.0 - a)
                   + rgba[..., :3].astype(np.float32) * a, 0, 255).astype(np.uint8)


def draw_streaklines(rgb: np.ndarray, trajectories: Sequence[dict],
                     colors: Sequence[Tuple[int, int, int]],
                     thickness: float = 1.8,
                     halo: bool = True) -> np.ndarray:
    """Draw trajectory paths in image coordinates onto an RGB array."""
    if not _HAVE_CV2 or not trajectories:
        return rgb
    out = rgb
    t = max(1, int(round(thickness)))
    for i, tr in enumerate(trajectories):
        pts = tr.get("points") if isinstance(tr, dict) else tr
        if not pts or len(pts) < 2:
            continue
        arr = np.asarray(pts, np.float64)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr) < 2:
            continue
        poly = np.round(arr).astype(np.int32).reshape(-1, 1, 2)
        if halo:
            cv2.polylines(out, [poly], False, (0, 0, 0),
                          max(1, int(round(t * 2.2))), cv2.LINE_AA)
        col = tuple(int(c) for c in colors[i % len(colors)]) if colors else (255, 80, 80)
        cv2.polylines(out, [poly], False, col, t, cv2.LINE_AA)
    return out


def draw_markers(rgb: np.ndarray, points, colors, radius: int = 6) -> np.ndarray:
    if not _HAVE_CV2 or not points:
        return rgb
    for i, p in enumerate(points):
        if p is None:
            continue
        x, y = p
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        c = tuple(int(v) for v in colors[i % len(colors)]) if colors else (255, 80, 80)
        cv2.circle(rgb, (int(round(x)), int(round(y))), radius, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(rgb, (int(round(x)), int(round(y))), radius, c, 2, cv2.LINE_AA)
        cv2.circle(rgb, (int(round(x)), int(round(y))), 2, c, -1, cv2.LINE_AA)
    return rgb


def draw_colorbar(rgb: np.ndarray, cmap_name: str, vmin: float, vmax: float,
                  unit: str = "", height: int = 12, margin: int = 12,
                  clipped_low: bool = False, clipped_high: bool = False
                  ) -> np.ndarray:
    """A horizontal colour ramp with end and midpoint labels, along the bottom.

    Drawn from the same colormap the field is coloured with, so a colour on the
    image and the same colour on the bar mean the same value.

    `clipped_low` / `clipped_high` add a cap in the out-of-range colour, so a
    figure taken out of the tool still says that the scale is hiding data
    rather than presenting a clipped extreme as the true one.
    """
    if not _HAVE_CV2:
        return rgb
    h, w = rgb.shape[:2]
    cap = 7 if (clipped_low or clipped_high) else 0
    bar_w = max(40, w - 2 * margin - 2 * cap)
    ramp = np.linspace(0.0, 1.0, bar_w, dtype=np.float32)[None, :]
    cols = (get_cmap(cmap_name, 256)(ramp)[..., :3] * 255).astype(np.uint8)
    bar = np.repeat(cols, height, axis=0)

    y0 = h - margin - height - 14
    if y0 < 0:
        return rgb
    x0 = margin + cap
    cv2.rectangle(rgb, (margin - 2, y0 - 2),
                  (margin + bar_w + 2 * cap + 2, y0 + height + 2), (25, 25, 25), -1)
    rgb[y0:y0 + height, x0:x0 + bar_w] = bar

    if clipped_low:
        rgb[y0:y0 + height, margin:x0] = UNDER_RANGE_RGB
    if clipped_high:
        rgb[y0:y0 + height, x0 + bar_w:x0 + bar_w + cap] = OVER_RANGE_RGB

    fs, th = 0.38, 1
    suffix = f" {unit}" if unit else ""
    mid = 0.5 * (vmin + vmax)
    lo, md, hi = f"{vmin:.4g}", f"{mid:.4g}", f"{vmax:.4g}{suffix}"
    ty = y0 + height + 12

    def _tw(s):
        return cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]

    placements = [(lo, x0), (hi, x0 + bar_w - _tw(hi))]
    # Only label the midpoint when it fits without colliding with the ends.
    mid_x = x0 + bar_w // 2 - _tw(md) // 2
    if mid_x > x0 + _tw(lo) + 10 and mid_x + _tw(md) < x0 + bar_w - _tw(hi) - 10:
        placements.insert(1, (md, mid_x))

    for text, tx in placements:
        cv2.putText(rgb, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(rgb, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), th, cv2.LINE_AA)
    return rgb


def draw_label(rgb: np.ndarray, text: str, margin: int = 10) -> np.ndarray:
    if not _HAVE_CV2 or not text:
        return rgb
    fs, th = 0.5, 1
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.rectangle(rgb, (margin - 5, margin - 4),
                  (margin + tw + 5, margin + tht + 6), (20, 20, 20), -1)
    cv2.putText(rgb, text, (margin, margin + tht),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), th, cv2.LINE_AA)
    return rgb


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

BACKGROUND_CHOICES = ["Deformed frame", "Reference frame", "Solid colour", "Transparent"]


@dataclass
class PanelSpec:
    """What one cell of the exported mosaic shows.

    content:
      "image"       - just the frame, no overlay
      "field"       - a result field over the chosen background
      "streaklines" - trajectories only, over the chosen background
      "empty"       - blank cell
    """
    content: str = "field"
    field: str = "Eeff_rate"
    cmap: str = "turbo"
    background: str = "Deformed frame"
    solid_colour: Tuple[int, int, int] = (0, 0, 0)
    show_colorbar: bool = True
    show_label: bool = True
    label: str = ""
    # Draw trajectories on top of whatever else this panel shows. Implied by
    # content == "streaklines"; opt-in for an image or field panel so you can
    # get paths over a strain map in one cell.
    overlay_streaklines: bool = False
    # Draw values outside the colour range in the out-of-range colours instead
    # of at the end colour, where they read as legitimate extremes.
    mark_out_of_range: bool = False
    range_spec: RangeSpec = _dcfield(default_factory=RangeSpec)

    @property
    def wants_streaklines(self) -> bool:
        return self.overlay_streaklines or self.content == "streaklines"


def _background_rgb(spec: PanelSpec, deformed: Optional[np.ndarray],
                    reference: Optional[np.ndarray],
                    shape: Tuple[int, int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Returns (rgb, alpha) — alpha is None unless the background is transparent."""
    h, w = shape
    if spec.background == "Reference frame" and reference is not None:
        return gray_to_rgb(reference), None
    if spec.background == "Deformed frame" and deformed is not None:
        return gray_to_rgb(deformed), None
    if spec.background == "Transparent":
        return (np.zeros((h, w, 3), np.uint8),
                np.zeros((h, w), np.uint8))
    c = np.asarray(spec.solid_colour, np.uint8)
    return np.tile(c.reshape(1, 1, 3), (h, w, 1)).copy(), None


def fit_into(img: np.ndarray, out_w: int, out_h: int,
             pad: Tuple[int, int, int] = (13, 17, 23)) -> np.ndarray:
    """Letterbox `img` into an (out_h, out_w) canvas, preserving aspect ratio."""
    if not _HAVE_CV2:
        return img
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.tile(np.asarray(pad, np.uint8).reshape(1, 1, 3), (out_h, out_w, 1))
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    ch = img.shape[2] if img.ndim == 3 else 1
    canvas = np.zeros((out_h, out_w, ch), img.dtype)
    if ch == 3:
        canvas[:] = np.asarray(pad, img.dtype)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def compose_grid(panels: Sequence[np.ndarray], rows: int, cols: int,
                 cell_w: int, cell_h: int, gap: int = 4,
                 bg: Tuple[int, int, int] = (13, 17, 23)) -> np.ndarray:
    """Tile rendered panels into a rows x cols mosaic."""
    out_h = rows * cell_h + (rows + 1) * gap
    out_w = cols * cell_w + (cols + 1) * gap
    canvas = np.tile(np.asarray(bg, np.uint8).reshape(1, 1, 3), (out_h, out_w, 1))
    for i, p in enumerate(panels):
        if i >= rows * cols or p is None:
            continue
        r, c = divmod(i, cols)
        y = gap + r * (cell_h + gap)
        x = gap + c * (cell_w + gap)
        canvas[y:y + cell_h, x:x + cell_w] = fit_into(p, cell_w, cell_h, bg)
    return canvas
