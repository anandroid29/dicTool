"""
image_canvas.py
---------------
Enhanced canvas with:
  - Memory-safe PyQt6/NumPy array bindings (resolves 0xC0000409 segmentation faults)
  - ROI edit mode: after committing a polygon or rectangle, switch to
    ROITool.NONE and click inside the shape to enter edit mode.
  - Snap-to-close for polygon (green ring + one click to finish)
  - Visible Seed Marker on Right-Click
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional, List, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap,
    QPolygonF, QPainterPath, QImage, QTransform, QFont,
)
from PyQt6.QtWidgets import QWidget, QSizePolicy


# ─────────────────────────────────────────────────────────────────────────────
# Tool enum & Constants
# ─────────────────────────────────────────────────────────────────────────────

class ROITool(Enum):
    NONE      = auto()
    POLYGON   = auto()
    POLYLINE  = auto()
    RECTANGLE = auto()
    CIRCLE    = auto()
    ERASE     = auto()


POLYGON_SNAP_RADIUS_PX: int = 12
VERTEX_HIT_PX:          int = 14
EDGE_HIT_PX:            int = 10
HANDLE_HIT_PX:          int = 10   
HANDLE_HALF:            int = 5    
MARKER_HIT_PX:          int = 13   


def _mask_border(mask: np.ndarray) -> np.ndarray:
    """Four-connected one-pixel border without importing SciPy in the UI."""
    src = np.asarray(mask, dtype=bool)
    interior = np.zeros_like(src)
    if src.shape[0] > 2 and src.shape[1] > 2:
        interior[1:-1, 1:-1] = (
            src[1:-1, 1:-1] & src[:-2, 1:-1] & src[2:, 1:-1] &
            src[1:-1, :-2] & src[1:-1, 2:]
        )
    return src & ~interior


def _indexed_pixmap(indices: np.ndarray, colors: list[QColor]) -> QPixmap:
    """Build a pixmap with a one-byte staging image instead of full RGBA."""
    data = np.ascontiguousarray(indices, dtype=np.uint8)
    height, width = data.shape
    image = QImage(
        data.data, width, height, data.strides[0],
        QImage.Format.Format_Indexed8,
    )
    image.setColorTable([color.rgba() for color in colors])
    # fromImage owns its converted pixels; ``data`` and ``image`` can be
    # released immediately instead of being retained beside the pixmap.
    return QPixmap.fromImage(image)

# Distinguishable, colour-blind-friendly-ish palette for trajectory markers.
MARKER_PALETTE = [
    "#ff5c5c", "#ffd93d", "#4ade80", "#38bdf8", "#c084fc",
    "#fb923c", "#2dd4bf", "#f472b6", "#a3e635", "#818cf8",
]


def marker_color(i: int) -> QColor:
    return QColor(MARKER_PALETTE[i % len(MARKER_PALETTE)])


TOOL_TOOLTIPS: dict = {
    ROITool.NONE:
        "Navigate / Edit — Pan: middle-drag · Zoom: scroll · Seed: right-click\n"
        "Click an existing ROI to edit it",
    ROITool.POLYGON:
        "Polygon ROI — Click to add vertices\n"
        "Hover near start to snap-close · click to finish\n"
        "Right-click removes last point · Enter/double-click to finish",
    ROITool.POLYLINE:
        "Line / curve — Click to add points or drag to sketch\n"
        "Enter or double-click finishes · right-click removes the last point",
    ROITool.RECTANGLE:
        "Rectangle ROI — Click and drag to draw · Release commits",
    ROITool.CIRCLE:
        "Circle ROI — Click centre, drag to radius · Release commits",
    ROITool.ERASE:
        "Erase ROI — Paint to remove mask · [ / ] to resize brush",
}

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)

def _dist_to_segment(px, py, ax, ay, bx, by) -> Tuple[float, float]:
    dx, dy = bx - ax, by - ay
    if dx == dy == 0: return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy)), t

_RECT_CURSORS = [
    Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeVerCursor, Qt.CursorShape.SizeBDiagCursor,
    Qt.CursorShape.SizeHorCursor, Qt.CursorShape.SizeHorCursor, Qt.CursorShape.SizeBDiagCursor,
    Qt.CursorShape.SizeVerCursor, Qt.CursorShape.SizeFDiagCursor
]

def _rect_handles(r: QRectF) -> List[QPointF]:
    x0, y0, x1, y1 = r.left(), r.top(), r.right(), r.bottom()
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    return [
        QPointF(x0, y0), QPointF(mx, y0), QPointF(x1, y0),
        QPointF(x0, my),                  QPointF(x1, my),
        QPointF(x0, y1), QPointF(mx, y1), QPointF(x1, y1),
    ]

def _apply_rect_handle_drag(r: QRectF, hi: int, dx_img: float, dy_img: float) -> QRectF:
    x0, y0, x1, y1 = r.left(), r.top(), r.right(), r.bottom()
    if hi in (0, 3, 5): x0 += dx_img
    if hi in (2, 4, 7): x1 += dx_img
    if hi in (0, 1, 2): y0 += dy_img
    if hi in (5, 6, 7): y1 += dy_img
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    return QRectF(x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _polyline_mask(points: List[QPointF], height: int, width: int) -> np.ndarray:
    """Rasterise an open, one-pixel-wide polyline without closing or filling it."""
    mask = np.zeros((height, width), dtype=bool)
    if len(points) < 2:
        return mask
    for start, end in zip(points[:-1], points[1:]):
        dx, dy = end.x() - start.x(), end.y() - start.y()
        count = max(2, int(math.ceil(max(abs(dx), abs(dy)))) + 1)
        xs = np.rint(np.linspace(start.x(), end.x(), count)).astype(np.intp)
        ys = np.rint(np.linspace(start.y(), end.y(), count)).astype(np.intp)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        mask[ys[inside], xs[inside]] = True
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# Edit State Containers
# ─────────────────────────────────────────────────────────────────────────────

class _PolyEdit:
    def __init__(self, pts: List[QPointF]):
        self.pts: List[QPointF] = list(pts)
        self.selected: Optional[int] = None
        self.dragging: bool = False

    def hit_vertex(self, wx, wy, zoom, pan_x, pan_y) -> Optional[int]:
        best_i, best_d = None, float("inf")
        for i, p in enumerate(self.pts):
            cx, cy = p.x() * zoom + pan_x, p.y() * zoom + pan_y
            d = _dist(wx, wy, cx, cy)
            if d < VERTEX_HIT_PX and d < best_d:
                best_d, best_i = d, i
        return best_i

    def hit_edge(self, wx, wy, zoom, pan_x, pan_y):
        n = len(self.pts)
        if n < 2: return None, None
        num_edges = n if n >= 3 else n - 1
        best_i, best_d, best_t = None, float("inf"), 0.0
        for i in range(num_edges):
            a, b = self.pts[i], self.pts[(i + 1) % n]
            ax, ay = a.x() * zoom + pan_x, a.y() * zoom + pan_y
            bx, by = b.x() * zoom + pan_x, b.y() * zoom + pan_y
            d, t = _dist_to_segment(wx, wy, ax, ay, bx, by)
            if d < EDGE_HIT_PX and d < best_d:
                best_d, best_i, best_t = d, i, t
        return best_i, best_t

class _RectEdit:
    def __init__(self, rect: QRectF):
        self.rect = QRectF(rect)
        self.handle: Optional[int] = None   
        self.moving: bool = False            

    def hit_handle(self, wx, wy, zoom, pan_x, pan_y) -> Optional[int]:
        handles = _rect_handles(self.rect)
        for i, h in enumerate(handles):
            if _dist(wx, wy, h.x() * zoom + pan_x, h.y() * zoom + pan_y) < HANDLE_HIT_PX:
                return i
        return None

    def hit_interior(self, img_pt: QPointF) -> bool:
        return self.rect.contains(img_pt)

# ─────────────────────────────────────────────────────────────────────────────
# Main Widget
# ─────────────────────────────────────────────────────────────────────────────

class ImageCanvas(QWidget):
    roi_changed  = pyqtSignal(object)
    shape_drawing_changed = pyqtSignal(bool)
    seed_placed  = pyqtSignal(int, int)
    cursor_moved = pyqtSignal(int, int, float)
    markers_changed  = pyqtSignal(object)   # list[(x, y)] in image coords
    marker_selected  = pyqtSignal(int)      # index, or -1
    marker_requested = pyqtSignal(float, float)  # raw click, page maps to reference

    _ROI_FILL_COLOR   = QColor(47, 129, 247,  55)
    _ROI_BORDER_COLOR = QColor(47, 129, 247, 200)
    _POLY_VERT_COLOR  = QColor(255, 200,  50, 230)
    _ERASE_COLOR      = QColor(248,  81,  73, 120)
    _SNAP_RING_COLOR  = QColor( 80, 220, 120, 220)
    _EDIT_VERT_COLOR  = QColor(  0, 229, 255, 240)   
    _EDIT_SEL_COLOR   = QColor(255, 107,  53, 240)   
    _HANDLE_COLOR     = QColor(  0, 229, 255, 220)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 200)
        self.setMouseTracking(True)

        # ── MEMORY SAFETY: Instance-bound NumPy & QImage variables ──
        self._image_arr: Optional[np.ndarray] = None
        self._image_u8: Optional[np.ndarray] = None
        self._image_qimg: Optional[QImage] = None
        self._image_px: Optional[QPixmap] = None

        self._result_arr: Optional[np.ndarray] = None
        self._result_rgba: Optional[np.ndarray] = None
        self._result_qimg: Optional[QImage] = None
        self._result_px: Optional[QPixmap] = None

        self._roi_mask: Optional[np.ndarray] = None
        self._roi_rgba: Optional[np.ndarray] = None
        self._roi_qimg: Optional[QImage] = None
        self._roi_px: Optional[QPixmap] = None
        self._roi_fill_color = QColor(self._ROI_FILL_COLOR)
        self._roi_border_color = QColor(self._ROI_BORDER_COLOR)
        # Optional blue underlay and hard drawing constraint. The ROI page uses
        # both while the amber strain-origin line is being edited.
        self._context_mask: Optional[np.ndarray] = None
        self._context_rgba: Optional[np.ndarray] = None
        self._context_qimg: Optional[QImage] = None
        self._context_px: Optional[QPixmap] = None
        self._constraint_mask: Optional[np.ndarray] = None

        # VISUAL SEED & SUBSET PREVIEW
        self._seed_xy: Optional[Tuple[int, int]] = None
        self._subset_radius: Optional[int] = None
        self.seed_enabled: bool = True
        self._streaklines: list[list[tuple[float, float]]] | None = None
        self._streak_path: Optional[QPainterPath] = None
        self._streak_paths: list[tuple[QPainterPath, QColor, bool]] = []
        self.streakline_thickness: float = 1.8

        # ── Trajectory markers ──
        # Stored positions are whatever the page hands us (reference-frame
        # coords); _marker_draw_pts is where to render them on the CURRENT frame,
        # so a marker visually sticks to its material point while scrubbing.
        self._markers: List[QPointF] = []
        self._marker_draw_pts: List[Optional[QPointF]] = []
        self._marker_mode: bool = False
        self._marker_sel: int = -1
        self._marker_drag: bool = False
        self.show_marker_labels: bool = True
        # ─────────────────────────────────────────────────────────────

        self._committed_poly: Optional[List[QPointF]] = None
        self._committed_rect: Optional[QRectF] = None

        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._dragging: bool = False
        self._drag_start: QPoint = QPoint()

        self._tool: ROITool = ROITool.NONE
        self._poly_pts: List[QPointF] = []
        self._poly_snapped: bool = False
        self._rect_start: Optional[QPointF] = None
        self._rect_cur: Optional[QPointF] = None
        self._circ_centre: Optional[QPointF] = None
        self._circ_radius: float = 0.0
        self._mouse_img: Optional[QPointF] = None
        self._mouse_widget: Optional[QPointF] = None
        self._erase_pts: List[QPointF] = []
        self._erase_radius: int = 20

        self._poly_edit: Optional[_PolyEdit] = None
        self._rect_edit: Optional[_RectEdit] = None

        # Use fast pixmap scaling during continuous pointer interaction, then
        # restore smooth rendering once the pointer settles. Scaling a large
        # scientific frame smoothly for every mouse event is needlessly slow.
        self._fast_paint = False
        self._smooth_paint_timer = QTimer(self)
        self._smooth_paint_timer.setSingleShot(True)
        self._smooth_paint_timer.setInterval(90)
        self._smooth_paint_timer.timeout.connect(self._restore_smooth_paint)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ─────────────────────────────────────────────────────────────────────
    # Safely Handled Pixmap Generation
    # ─────────────────────────────────────────────────────────────────────

    def _begin_fast_paint(self) -> None:
        self._fast_paint = True
        self._smooth_paint_timer.start()

    def _restore_smooth_paint(self) -> None:
        if self._fast_paint:
            self._fast_paint = False
            self.update()

    def set_image(self, arr: np.ndarray, keep_view: bool = False,
                  display_scale: float = 1.0) -> None:
        self._image_arr = arr
        image_u8 = np.ascontiguousarray(
            np.clip(np.asarray(arr) * (255.0 * float(display_scale)),
                    0, 255).astype(np.uint8))
        H, W = image_u8.shape
        image_qimg = QImage(
            image_u8.data, W, H, image_u8.strides[0],
            QImage.Format.Format_Grayscale8)
        self._image_px = QPixmap.fromImage(image_qimg)
        # QPixmap owns a converted copy. Retaining the NumPy/QImage staging
        # pair doubled the display allocation for every hidden wizard page.
        self._image_u8 = None
        self._image_qimg = None

        if not keep_view:
            self.clear_result_overlay()
            self.clear_roi()
            self.set_context_mask(None)
            self.set_draw_constraint(None)
            self._fit_to_window()

        self._rebuild_roi_pixmap()

        self.update()
    def set_result_overlay_rgba(self, rgba) -> None:
        if rgba is None:
            self.clear_result_overlay(); return

        result_rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        H, W = result_rgba.shape[:2]
        result_qimg = QImage(
            result_rgba.data, W, H, result_rgba.strides[0],
            QImage.Format.Format_RGBA8888)
        self._result_px = QPixmap.fromImage(result_qimg)
        self._result_rgba = None
        self._result_qimg = None
        self.update()

    def set_result_overlay_indexed(
            self, indices: np.ndarray, colors: list[QColor]) -> None:
        """Set a categorical overlay using one byte per staging pixel."""
        if indices is None:
            self.clear_result_overlay()
            return
        self._result_px = _indexed_pixmap(indices, colors)
        self._result_rgba = None
        self._result_qimg = None
        self.update()

    def _rebuild_roi_pixmap(self) -> None:
        if (self._image_arr is None or
                (self._roi_mask is None and self._context_mask is None)):
            self._roi_px = None
            self._context_px = None
            self._roi_rgba = None
            self._roi_qimg = None
            return

        H, W = self._image_arr.shape
        indices = np.zeros((H, W), dtype=np.uint8)
        colors = [
            QColor(0, 0, 0, 0),
            QColor(47, 129, 247, 28),
            QColor(47, 129, 247, 205),
            QColor(self._roi_fill_color),
            QColor(self._roi_border_color),
        ]
        if self._context_mask is not None:
            indices[self._context_mask] = 1
            indices[_mask_border(self._context_mask)] = 2
        if self._roi_mask is not None:
            indices[self._roi_mask] = 3
            indices[_mask_border(self._roi_mask)] = 4

        # Context and active ROI share one categorical pixmap. The previous two
        # independent RGBA layers cost eight bytes per source pixel in Qt alone.
        self._roi_px = _indexed_pixmap(indices, colors)
        self._context_px = self._roi_px if self._context_mask is not None else None
        self._roi_rgba = None
        self._roi_qimg = None
        self._context_rgba = None
        self._context_qimg = None

    def _rebuild_context_pixmap(self) -> None:
        # Context is composited into the same indexed pixmap as the active ROI.
        self._rebuild_roi_pixmap()

    def clear_result_overlay(self) -> None:
        self._result_arr = None; self._result_rgba = None; self._result_qimg = None; self._result_px = None
        self.update()

    def clear_roi(self) -> None:
        self._roi_mask = None;
        self._roi_rgba = None;
        self._roi_qimg = None;
        self._poly_pts = [];
        self.shape_drawing_changed.emit(False)
        self._rect_start = None
        self._committed_poly = None;
        self._committed_rect = None
        self._poly_edit = None;
        self._rect_edit = None
        self._seed_xy = None
        self._rebuild_roi_pixmap()
        self.update()

    def release_display_buffers(self) -> None:
        """Drop recreatable full-resolution canvas state on a hidden page."""
        self._image_arr = None
        self._image_u8 = None
        self._image_qimg = None
        self._image_px = None
        self._result_arr = None
        self._result_rgba = None
        self._result_qimg = None
        self._result_px = None
        self._roi_rgba = None
        self._roi_qimg = None
        self._roi_px = None
        self._context_rgba = None
        self._context_qimg = None
        self._context_px = None
        self._roi_mask = None
        self._context_mask = None
        self._constraint_mask = None
        self.update()

    # ─────────────────────────────────────────────────────────────────────
    # Public API Overrides
    # ─────────────────────────────────────────────────────────────────────
    
    def set_roi_mask(self, mask: np.ndarray) -> None:
        self._roi_mask = mask.astype(bool)
        # A programmatically supplied mask may belong to another editing
        # channel (analysis ROI versus strain origin). Do not leave handles from
        # the previously displayed geometry attached to it.
        self._poly_pts = []
        self.shape_drawing_changed.emit(False)
        self._committed_poly = None
        self._committed_rect = None
        self._poly_edit = None
        self._rect_edit = None
        self._rebuild_roi_pixmap(); self.update()

    def set_context_mask(self, mask: Optional[np.ndarray]) -> None:
        """Show a non-editable blue ROI beneath the active mask."""
        if (mask is not None and self._image_arr is not None and
                np.asarray(mask).shape != self._image_arr.shape):
            raise ValueError("Context mask shape does not match the image.")
        # Context and constraints are read-only views of model masks. Copying
        # both added two more full-resolution arrays on the strain ROI screen.
        self._context_mask = (None if mask is None else
                              np.asarray(mask, dtype=bool))
        self._rebuild_context_pixmap()
        self.update()

    def set_draw_constraint(self, mask: Optional[np.ndarray]) -> None:
        """Clip every newly committed active-mask pixel to ``mask``."""
        if (mask is not None and self._image_arr is not None and
                np.asarray(mask).shape != self._image_arr.shape):
            raise ValueError("Drawing constraint shape does not match the image.")
        self._constraint_mask = (None if mask is None else
                                 np.asarray(mask, dtype=bool))

    def set_roi_role(self, origin: bool) -> None:
        """Use blue for analysis ROI and amber for the strain-origin line."""
        if origin:
            self._roi_fill_color = QColor(245, 158, 11, 55)
            self._roi_border_color = QColor(245, 158, 11, 220)
        else:
            self._roi_fill_color = QColor(self._ROI_FILL_COLOR)
            self._roi_border_color = QColor(self._ROI_BORDER_COLOR)
        self._rebuild_roi_pixmap()
        self.update()

    def set_roi_colors(self, fill: QColor, border: QColor) -> None:
        self._roi_fill_color = QColor(fill)
        self._roi_border_color = QColor(border)
        self._rebuild_roi_pixmap()
        self.update()

    def set_tool(self, tool: ROITool) -> None:
        self._commit_poly_edit(); self._commit_rect_edit()
        self._tool = tool; self._poly_pts = []; self._poly_snapped = False
        self.shape_drawing_changed.emit(False)
        self._rect_start = None; self._circ_centre = None
        self._poly_edit = None; self._rect_edit = None
        cursor = Qt.CursorShape.ArrowCursor if tool == ROITool.NONE else Qt.CursorShape.CrossCursor
        self.setCursor(cursor); self.update()

    def finish_active_shape(self) -> bool:
        """Commit the currently drawn open/closed point shape, if complete."""
        if self._tool == ROITool.POLYLINE and len(self._poly_pts) >= 2:
            self._commit_polyline()
            return True
        if self._tool == ROITool.POLYGON and len(self._poly_pts) >= 3:
            self._commit_polygon()
            return True
        return False
        
    def set_seed_xy(self, xy: Optional[Tuple[int, int]]) -> None:
        self._seed_xy = xy
        self.update()

    @property
    def roi_mask(self) -> Optional[np.ndarray]: return self._roi_mask

    def zoom_fit(self): self.fit_image()
    def set_roi_tool(self, t): self.set_tool(t)

    def fit_image(self) -> None:
        self._fit_to_window(); self.update()

    # ─────────────────────────────────────────────────────────────────────
    # Mouse events
    # ─────────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        pos, wx, wy = event.position(), event.position().x(), event.position().y()

        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = True;
            self._drag_start = pos.toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor);
            return

        # ── Marker mode takes precedence over all ROI interaction ──
        if self._marker_mode and self._image_arr is not None:
            hit = self._hit_marker(wx, wy)
            if event.button() == Qt.MouseButton.LeftButton:
                if hit is not None:
                    self.select_marker(hit)
                    self._marker_drag = True
                else:
                    ip = self._widget_to_image(pos)
                    if ip is not None:
                        H, W = self._image_arr.shape
                        if 0 <= ip.x() < W and 0 <= ip.y() < H:
                            # The page decides how to interpret the click (it may
                            # need to map a deformed-frame position back to
                            # reference coords), then calls add_marker().
                            self.marker_requested.emit(ip.x(), ip.y())
                return
            if event.button() == Qt.MouseButton.RightButton:
                if hit is not None:
                    self.remove_marker(hit)
                return

        if self._poly_edit is not None:
            pe = self._poly_edit
            if event.button() == Qt.MouseButton.LeftButton:
                vi = pe.hit_vertex(wx, wy, self._zoom, self._pan_x, self._pan_y)
                if vi is not None:
                    pe.selected = vi;
                    pe.dragging = True;
                    self.update();
                    return
                ei, t = pe.hit_edge(wx, wy, self._zoom, self._pan_x, self._pan_y)
                if ei is not None:
                    p1, p2 = pe.pts[ei], pe.pts[(ei + 1) % len(pe.pts)]
                    pe.pts.insert(ei + 1, QPointF(p1.x() + t * (p2.x() - p1.x()), p1.y() + t * (p2.y() - p1.y())))
                    pe.selected = ei + 1;
                    pe.dragging = True;
                    self.update();
                    return
                self._commit_poly_edit();
                self.update()
            elif event.button() == Qt.MouseButton.RightButton:
                vi = pe.hit_vertex(wx, wy, self._zoom, self._pan_x, self._pan_y)
                if vi is not None and len(pe.pts) > 3:
                    pe.pts.pop(vi)
                    if pe.selected == vi:
                        pe.selected = None
                    elif pe.selected is not None and pe.selected > vi:
                        pe.selected -= 1
                    self.update()
            return

        if self._rect_edit is not None:
            re = self._rect_edit
            if event.button() == Qt.MouseButton.LeftButton:
                hi = re.hit_handle(wx, wy, self._zoom, self._pan_x, self._pan_y)
                if hi is not None:
                    re.handle = hi;
                    self.update();
                    return
                img_pt = self._widget_to_image(pos)
                if img_pt and re.hit_interior(img_pt):
                    re.moving = True;
                    self._drag_start = pos.toPoint();
                    self.update();
                    return
                self._commit_rect_edit();
                self.update()
            return

        if event.button() == Qt.MouseButton.RightButton:
            if self._tool in (ROITool.POLYGON, ROITool.POLYLINE) and self._poly_pts:
                self._poly_pts.pop();
                self._poly_snapped = False;
                self.shape_drawing_changed.emit(bool(self._poly_pts))
                self.update();
                return
            img_pt = self._widget_to_image(pos)
            if self.seed_enabled and img_pt and self._image_arr is not None:
                x, y = int(round(img_pt.x())), int(round(img_pt.y()))
                H, W = self._image_arr.shape
                if 0 <= x < W and 0 <= y < H:
                    # ── ONLY place seed if an ROI exists and the point is inside it ──
                    if self._roi_mask is not None and self._roi_mask[y, x]:
                        self._seed_xy = (x, y)
                        self.seed_placed.emit(x, y)
                        self.update()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(pos)
            if img_pt is None: return

            if self._tool == ROITool.NONE:
                self._try_enter_edit(pos, img_pt)
            elif self._tool == ROITool.POLYGON:
                if self._poly_snapped and len(self._poly_pts) >= 3:
                    self._commit_polygon();
                    return
                self._poly_pts.append(img_pt);
                self.shape_drawing_changed.emit(True)
                self.update()
            elif self._tool == ROITool.POLYLINE:
                self._poly_pts.append(img_pt)
                self.shape_drawing_changed.emit(True)
                self.update()
            elif self._tool == ROITool.RECTANGLE:
                self._rect_start = img_pt;
                self._rect_cur = img_pt
            elif self._tool == ROITool.CIRCLE:
                self._circ_centre = img_pt;
                self._circ_radius = 0.0
            elif self._tool == ROITool.ERASE:
                self._erase_pts = [img_pt]

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool == ROITool.POLYGON and len(self._poly_pts) >= 3:
                self._commit_polygon()
            elif self._tool == ROITool.POLYLINE and len(self._poly_pts) >= 2:
                self._commit_polyline()

    def mouseMoveEvent(self, event) -> None:
        # Nearest-neighbour scaling is useful while actively dragging a large
        # image, but applying it to ordinary hover motion makes the canvas blur
        # and then sharpen again after every pointer event.  Keep hover previews
        # and zooming smooth; use the fast path only during a button drag.
        if event.buttons() != Qt.MouseButton.NoButton:
            self._begin_fast_paint()
        if (self._marker_mode and self._marker_drag and 0 <= self._marker_sel < len(self._markers)):
            ip = self._widget_to_image(event.position())
            if ip is not None and self._image_arr is not None:
                H, W = self._image_arr.shape
                self._markers[self._marker_sel] = QPointF(
                    min(max(ip.x(), 0.0), W - 1.0), min(max(ip.y(), 0.0), H - 1.0))
                self._marker_draw_pts = []
                self.markers_changed.emit(self.markers())
                self.update()
            return
        if self._marker_mode:
            self._mouse_widget = event.position()
            self._mouse_img = self._widget_to_image(event.position())
            hit = self._hit_marker(event.position().x(), event.position().y())
            self.setCursor(Qt.CursorShape.OpenHandCursor if hit is not None
                           else Qt.CursorShape.CrossCursor)
            self.update()
            return
        pos, wx, wy = event.position(), event.position().x(), event.position().y()
        img_pt = self._widget_to_image(pos)
        self._mouse_img, self._mouse_widget = img_pt, pos

        if self._dragging:
            delta = pos.toPoint() - self._drag_start
            self._pan_x += delta.x(); self._pan_y += delta.y()
            self._drag_start = pos.toPoint(); self.update(); return

        if self._poly_edit is not None:
            pe = self._poly_edit
            if pe.dragging and pe.selected is not None and img_pt is not None:
                if event.buttons() & Qt.MouseButton.LeftButton:
                    pe.pts[pe.selected] = img_pt; self.update(); return
            vi = pe.hit_vertex(wx, wy, self._zoom, self._pan_x, self._pan_y)
            if vi is not None: self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                ei, _ = pe.hit_edge(wx, wy, self._zoom, self._pan_x, self._pan_y)
                self.setCursor(Qt.CursorShape.CrossCursor if ei is not None else Qt.CursorShape.ArrowCursor)
            self.update(); return

        if self._rect_edit is not None:
            re = self._rect_edit
            if event.buttons() & Qt.MouseButton.LeftButton:
                if re.handle is not None and img_pt is not None:
                    prev_img = self._widget_to_image(QPointF(self._drag_start))
                    if prev_img:
                        re.rect = _apply_rect_handle_drag(re.rect, re.handle, img_pt.x() - prev_img.x(), img_pt.y() - prev_img.y())
                    self._drag_start = pos.toPoint(); self.update(); return
                if re.moving and img_pt is not None:
                    prev_img = self._widget_to_image(QPointF(self._drag_start))
                    if prev_img:
                        re.rect.translate(img_pt.x() - prev_img.x(), img_pt.y() - prev_img.y())
                    self._drag_start = pos.toPoint(); self.update(); return
            hi = re.hit_handle(wx, wy, self._zoom, self._pan_x, self._pan_y)
            if hi is not None: self.setCursor(_RECT_CURSORS[hi])
            elif img_pt and re.hit_interior(img_pt): self.setCursor(Qt.CursorShape.SizeAllCursor)
            else: self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update(); return

        if self._tool == ROITool.POLYGON and len(self._poly_pts) >= 3:
            self._poly_snapped = self._check_snap(pos)
        else: self._poly_snapped = False

        if event.buttons() & Qt.MouseButton.LeftButton and img_pt is not None:
            if self._tool in (ROITool.POLYGON, ROITool.POLYLINE) and self._poly_pts:
                last_pt = self._poly_pts[-1]
                if math.hypot(img_pt.x() - last_pt.x(), img_pt.y() - last_pt.y()) * self._zoom > 10:
                    self._poly_pts.append(img_pt)
            elif self._tool == ROITool.RECTANGLE: self._rect_cur = img_pt
            elif self._tool == ROITool.CIRCLE and self._circ_centre:
                self._circ_radius = math.hypot(img_pt.x() - self._circ_centre.x(), img_pt.y() - self._circ_centre.y())
            elif self._tool == ROITool.ERASE:
                self._erase_pts.append(img_pt); self._apply_erase()

        self.update()

        if img_pt is not None:
            px, py = int(img_pt.x()), int(img_pt.y())
            val = float("nan")
            if self._result_arr is not None:
                H, W = self._result_arr.shape[:2]
                if 0 <= py < H and 0 <= px < W: val = float(self._result_arr[py, px])
            self.cursor_moved.emit(px, py, val)

    def mouseReleaseEvent(self, event) -> None:
        if self._marker_drag:
            self._marker_drag = False
            self.markers_changed.emit(self.markers())
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.CrossCursor if self._tool != ROITool.NONE else Qt.CursorShape.ArrowCursor)

        if event.button() == Qt.MouseButton.LeftButton:
            if self._poly_edit is not None: self._poly_edit.dragging = False; return
            if self._rect_edit is not None: self._rect_edit.handle = None; self._rect_edit.moving = False; return
            if self._tool == ROITool.RECTANGLE and self._rect_start: self._commit_rectangle()
            elif self._tool == ROITool.CIRCLE and self._circ_centre:
                if self._circ_radius > 2: self._commit_circle()
            elif self._tool == ROITool.ERASE: self._erase_pts = []

    def wheelEvent(self, event) -> None:
        delta  = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        pos    = event.position()
        self._pan_x = pos.x() + (self._pan_x - pos.x()) * factor
        self._pan_y = pos.y() + (self._pan_y - pos.y()) * factor
        self._zoom  = max(0.1, min(self._zoom * factor, 40.0))
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if self._marker_mode:
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                if self._marker_sel >= 0:
                    self.remove_marker(self._marker_sel)
                return
            if key == Qt.Key.Key_Escape:
                self.select_marker(-1)
                return
        if key == Qt.Key.Key_Escape:
            if self._poly_edit: self._commit_poly_edit()
            elif self._rect_edit: self._commit_rect_edit()
            elif self._tool in (ROITool.POLYGON, ROITool.POLYLINE):
                self._poly_pts = []; self._poly_snapped = False
                self.shape_drawing_changed.emit(False)
            self.update()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._poly_edit: self._commit_poly_edit()
            elif self._rect_edit: self._commit_rect_edit()
            elif self._tool == ROITool.POLYGON and len(self._poly_pts) >= 3: self._commit_polygon()
            elif self._tool == ROITool.POLYLINE and len(self._poly_pts) >= 2: self._commit_polyline()
        elif key == Qt.Key.Key_BracketLeft:
            self._erase_radius = max(4, self._erase_radius - 4); self.update()
        elif key == Qt.Key.Key_BracketRight:
            self._erase_radius = min(200, self._erase_radius + 4); self.update()

    # ─────────────────────────────────────────────────────────────────────
    # Painting
    # ─────────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform, not self._fast_paint)
        painter.fillRect(self.rect(), QColor("#1a1c1e"))

        if self._image_px is None:
            painter.setPen(QColor("#6b7378"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a reference image to begin")
            return

        painter.save()
        painter.setTransform(self._get_transform())
        painter.drawPixmap(0, 0, self._image_px)
        if self._result_px is not None: painter.drawPixmap(0, 0, self._result_px)
        # The context mask is already composited into _roi_px.
        if self._roi_px is not None: painter.drawPixmap(0, 0, self._roi_px)
        self._paint_roi_preview(painter)

        # ─── DRAW VISUAL SEED & SUBSET CIRCLE ───
        if self._seed_xy is not None:
            sx, sy = self._seed_xy

            # If radius is set (Params Page), use it. Otherwise (ROI page), use a small dot.
            sr = float(self._subset_radius) if getattr(self, '_subset_radius', None) is not None else (4.0 / self._zoom)

            # Red theme for the circle
            outline_color = QColor(255, 60, 60, 220)
            fill_color = QColor(255, 60, 60, 45)

            painter.setPen(QPen(outline_color, 1.5 / self._zoom, Qt.PenStyle.SolidLine))
            painter.setBrush(QBrush(fill_color))

            painter.drawEllipse(QPointF(sx, sy), sr, sr)

        # ─── DRAW TRAJECTORIES (one colour per marker) ───
        if self._streak_paths:
            thick = max(0.6, getattr(self, 'streakline_thickness', 1.8)) / self._zoom
            for path, col, lost in self._streak_paths:
                # Halo first so paths stay legible over a bright field overlay.
                halo = QColor(0, 0, 0, 150)
                painter.setPen(QPen(halo, thick * 2.2, Qt.PenStyle.SolidLine,
                                    Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
                pen = QPen(col, thick, Qt.PenStyle.DashLine if lost else Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.drawPath(path)

        # ─── DRAW MARKERS ───
        # The head circle sits right on the streakline's leading end and buries
        # it, so it is suppressed while trajectories are on screen. Marker mode
        # keeps it: there it is the grab handle, not decoration.
        if self._markers and not (self._streak_paths and not self._marker_mode):
            r_out = 6.5 / self._zoom
            r_in = 2.4 / self._zoom
            for i in range(len(self._markers)):
                p = self._marker_render_pt(i)
                if p is None:
                    continue
                col = marker_color(i)
                sel = (i == self._marker_sel)
                lost = (i < len(self._marker_draw_pts) and self._marker_draw_pts[i] is None)

                painter.setPen(QPen(QColor(0, 0, 0, 190), 3.0 / self._zoom))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(p, r_out, r_out)

                painter.setPen(QPen(col, (2.4 if sel else 1.6) / self._zoom,
                                    Qt.PenStyle.DotLine if lost else Qt.PenStyle.SolidLine))
                painter.drawEllipse(p, r_out, r_out)

                fill = QColor(col)
                fill.setAlpha(90 if lost else 255)
                painter.setBrush(QBrush(fill))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(p, r_in, r_in)

                if sel:
                    ring = QColor(255, 255, 255, 210)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(ring, 1.2 / self._zoom, Qt.PenStyle.DashLine))
                    painter.drawEllipse(p, r_out * 1.7, r_out * 1.7)
        # ────────────────────────

        painter.restore()

        if self._poly_edit is not None: self._paint_poly_edit(painter)
        if self._rect_edit is not None: self._paint_rect_edit(painter)

        # Marker numbers are drawn unscaled so they stay readable at any zoom.
        if self._markers and self.show_marker_labels:
            f = QFont(); f.setPointSize(8); f.setBold(True)
            painter.setFont(f)
            for i in range(len(self._markers)):
                p = self._marker_render_pt(i)
                if p is None:
                    continue
                sx = p.x() * self._zoom + self._pan_x + 9
                sy = p.y() * self._zoom + self._pan_y - 8
                label = str(i + 1)
                painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
                painter.drawText(QPointF(sx + 0.7, sy + 0.7), label)
                painter.setPen(marker_color(i))
                painter.drawText(QPointF(sx, sy), label)

        if self._marker_mode:
            hint = ("MARKER MODE — click to place · drag to move · "
                    "right-click or Del to remove")
            f2 = QFont(); f2.setPointSize(8); f2.setBold(True)
            painter.setFont(f2)
            tw = painter.fontMetrics().horizontalAdvance(hint) + 16
            bar = QRectF(8, 8, tw, 22)
            painter.setBrush(QBrush(QColor(11, 22, 38, 225)))
            painter.setPen(QPen(QColor(59, 130, 246, 200), 1))
            painter.drawRoundedRect(bar, 4, 4)
            painter.setPen(QColor("#9dc4ff"))
            painter.drawText(bar, Qt.AlignmentFlag.AlignCenter, hint)

        if self._mouse_img and self._image_arr is not None:
            x, y = self._mouse_img.x(), self._mouse_img.y()
            H, W = self._image_arr.shape
            if 0 <= x < W and 0 <= y < H:
                val = self._image_arr[int(y), int(x)]
                extra = ""
                if self._tool in (ROITool.POLYGON, ROITool.POLYLINE) and self._poly_pts:
                    extra = f"  pts={len(self._poly_pts)}"
                    if self._tool == ROITool.POLYGON and self._poly_snapped:
                        extra += "  ● click to close"
                    else:
                        extra += "  RMB=undo · ↵/dbl=finish"
                elif self._poly_edit is not None:
                    extra = "  EDIT POLY — drag vertex · click edge=insert · RMB=delete · ↵=done"
                elif self._rect_edit is not None:
                    extra = "  EDIT RECT — drag handle · drag interior=move · ↵=done"
                txt = f"  x={int(x)}  y={int(y)}  I={val:.3f}  zoom={self._zoom:.2f}×{extra}"
                painter.setPen(QColor("#a2a8ad"))
                painter.drawText(4, self.height() - 6, txt)

    def _paint_roi_preview(self, painter: QPainter) -> None:
        if self._tool in (ROITool.POLYGON, ROITool.POLYLINE) and self._poly_pts:
            pen = QPen(self._roi_border_color, 1.5 / self._zoom)
            painter.setPen(pen); painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(self._poly_pts) - 1):
                painter.drawLine(self._poly_pts[i], self._poly_pts[i + 1])
            if self._mouse_img:
                closed = self._tool == ROITool.POLYGON and self._poly_snapped
                tgt  = self._poly_pts[0] if closed else self._mouse_img
                c    = self._SNAP_RING_COLOR if closed else self._roi_border_color
                painter.setPen(QPen(c, 1.5 / self._zoom, Qt.PenStyle.DashLine))
                painter.drawLine(self._poly_pts[-1], tgt)
            r = 3.5 / self._zoom
            painter.setPen(QPen(self._POLY_VERT_COLOR, 1.5 / self._zoom))
            painter.setBrush(QBrush(self._POLY_VERT_COLOR))
            for pt in self._poly_pts: painter.drawEllipse(pt, r, r)
            if self._tool == ROITool.POLYGON and self._poly_snapped:
                snap_r = POLYGON_SNAP_RADIUS_PX / self._zoom
                painter.setPen(QPen(self._SNAP_RING_COLOR, 2.0 / self._zoom))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(self._poly_pts[0], snap_r, snap_r)

        elif self._tool == ROITool.RECTANGLE and self._rect_start and self._rect_cur:
            x1, y1 = self._rect_start.x(), self._rect_start.y()
            x2, y2 = self._rect_cur.x(),   self._rect_cur.y()
            painter.setPen(QPen(self._roi_border_color, 1.5 / self._zoom))
            painter.setBrush(QBrush(self._roi_fill_color))
            painter.drawRect(QRectF(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)))

        elif self._tool == ROITool.CIRCLE and self._circ_centre and self._circ_radius > 0:
            painter.setPen(QPen(self._roi_border_color, 1.5 / self._zoom))
            painter.setBrush(QBrush(self._roi_fill_color))
            painter.drawEllipse(self._circ_centre, self._circ_radius, self._circ_radius)

        elif self._tool == ROITool.ERASE and self._mouse_img:
            r = self._erase_image_radius()
            painter.setPen(QPen(self._ERASE_COLOR, 1.5 / self._zoom))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self._mouse_img, r, r)

    def _paint_poly_edit(self, painter: QPainter) -> None:
        pe, n = self._poly_edit, len(self._poly_edit.pts)
        cpts = [QPointF(p.x() * self._zoom + self._pan_x, p.y() * self._zoom + self._pan_y) for p in pe.pts]
        if n >= 3:
            path = QPainterPath(); path.addPolygon(QPolygonF(cpts + [cpts[0]]))
            painter.fillPath(path, QBrush(QColor(0, 229, 255, 30)))
        edge_pen = QPen(QColor(0, 229, 255, 160), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(edge_pen)
        for i in range(n if n >= 3 else n - 1):
            a, b = cpts[i], cpts[(i + 1) % n]
            painter.drawLine(a, b)
            painter.setPen(QPen(QColor(0, 229, 255, 180), 1)); painter.setBrush(QBrush(QColor("#1a1c1e")))
            painter.drawEllipse(QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2), 4.0, 4.0)
            painter.setPen(edge_pen)
        font = QFont("Courier", 7, QFont.Weight.Bold)
        for i, cp in enumerate(cpts):
            is_sel = (i == pe.selected)
            col = self._EDIT_SEL_COLOR if is_sel else self._EDIT_VERT_COLOR
            r = 9.0 if is_sel else 7.0
            painter.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 55), 6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(cp, r + 4, r + 4)
            painter.setPen(QPen(col, 2)); painter.setBrush(QBrush(QColor("#1a1c1e")))
            painter.drawEllipse(cp, r, r)
            painter.setPen(QPen(col)); painter.setFont(font)
            painter.drawText(QRect(int(cp.x()) - 8, int(cp.y()) - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, str(i + 1))

    def _paint_rect_edit(self, painter: QPainter) -> None:
        re = self._rect_edit
        wx0, wy0 = re.rect.left() * self._zoom + self._pan_x, re.rect.top() * self._zoom + self._pan_y
        wx1, wy1 = re.rect.right() * self._zoom + self._pan_x, re.rect.bottom() * self._zoom + self._pan_y
        painter.setBrush(QBrush(QColor(0, 229, 255, 25)))
        painter.setPen(QPen(QColor(0, 229, 255, 180), 1.5, Qt.PenStyle.DashLine))
        painter.drawRect(QRectF(wx0, wy0, wx1 - wx0, wy1 - wy0))
        for h in _rect_handles(re.rect):
            cx, cy = h.x() * self._zoom + self._pan_x, h.y() * self._zoom + self._pan_y
            painter.setPen(QPen(self._HANDLE_COLOR, 1.5)); painter.setBrush(QBrush(QColor("#1a1c1e")))
            painter.drawRect(QRectF(cx - float(HANDLE_HALF), cy - float(HANDLE_HALF), float(HANDLE_HALF) * 2, float(HANDLE_HALF) * 2))

    # ─────────────────────────────────────────────────────────────────────
    # Commit Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _try_enter_edit(self, widget_pos: QPointF, img_pt: QPointF) -> None:
        if self._committed_poly and len(self._committed_poly) >= 3:
            if _point_in_polygon(img_pt, self._committed_poly):
                self._poly_edit = _PolyEdit(self._committed_poly)
                self._rect_edit = None; self._committed_rect = None
                self.setCursor(Qt.CursorShape.SizeAllCursor); self.update(); return
        if self._committed_rect is not None:
            if _RectEdit(self._committed_rect).hit_handle(widget_pos.x(), widget_pos.y(), self._zoom, self._pan_x, self._pan_y) is not None or self._committed_rect.contains(img_pt):
                self._rect_edit = _RectEdit(self._committed_rect)
                self._poly_edit = None; self._committed_poly = None
                self.setCursor(Qt.CursorShape.SizeAllCursor); self.update()

    def _commit_poly_edit(self) -> None:
        if self._poly_edit is None: return
        pe = self._poly_edit
        if self._image_arr is not None and len(pe.pts) >= 3:
            H, W = self._image_arr.shape
            self._committed_poly = list(pe.pts)
            self._roi_mask = _polygon_mask(pe.pts, H, W)
            self._rebuild_roi_pixmap()
            self.roi_changed.emit(self._roi_mask.copy())
        self._poly_edit = None; self.setCursor(Qt.CursorShape.ArrowCursor); self.update()

    def _commit_rect_edit(self) -> None:
        if self._rect_edit is None: return
        re = self._rect_edit
        if self._image_arr is not None:
            H, W = self._image_arr.shape
            x1, y1 = max(0, int(re.rect.left())), max(0, int(re.rect.top()))
            x2, y2 = min(W, int(re.rect.right())), min(H, int(re.rect.bottom()))
            mask = np.zeros((H, W), dtype=bool); mask[y1:y2, x1:x2] = True
            self._committed_rect = QRectF(re.rect)
            self._roi_mask = mask; self._rebuild_roi_pixmap()
            self.roi_changed.emit(self._roi_mask.copy())
        self._rect_edit = None; self.setCursor(Qt.CursorShape.ArrowCursor); self.update()

    def _commit_polygon(self) -> None:
        if self._image_arr is None or len(self._poly_pts) < 3: return
        H, W = self._image_arr.shape
        self._committed_poly = list(self._poly_pts)
        self._committed_rect = None
        self._merge_mask(_polygon_mask(self._poly_pts, H, W))
        self._poly_pts = []; self._poly_snapped = False; self.update()
        self.shape_drawing_changed.emit(False)

    def _commit_polyline(self) -> None:
        if self._image_arr is None or len(self._poly_pts) < 2:
            return
        H, W = self._image_arr.shape
        self._committed_poly = None
        self._committed_rect = None
        self._merge_mask(_polyline_mask(self._poly_pts, H, W))
        self._poly_pts = []
        self._poly_snapped = False
        self.shape_drawing_changed.emit(False)
        self.update()

    def _commit_rectangle(self) -> None:
        if self._image_arr is None or not self._rect_start or not self._rect_cur: return
        H, W = self._image_arr.shape
        x1, y1 = max(0, int(min(self._rect_start.x(), self._rect_cur.x()))), max(0, int(min(self._rect_start.y(), self._rect_cur.y())))
        x2, y2 = min(W, int(max(self._rect_start.x(), self._rect_cur.x()))), min(H, int(max(self._rect_start.y(), self._rect_cur.y())))
        self._committed_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        self._committed_poly = None
        mask = np.zeros((H, W), dtype=bool); mask[y1:y2, x1:x2] = True
        self._merge_mask(mask)
        self._rect_start = self._rect_cur = None; self.update()

    def _commit_circle(self) -> None:
        if self._image_arr is None or not self._circ_centre: return
        H, W = self._image_arr.shape
        yg, xg = np.ogrid[:H, :W]
        self._merge_mask((xg - self._circ_centre.x()) ** 2 + (yg - self._circ_centre.y()) ** 2 <= self._circ_radius ** 2)
        self._circ_centre = None; self._circ_radius = 0.0; self.update()

    def _apply_erase(self) -> None:
        if self._image_arr is None or not self._erase_pts or self._roi_mask is None: return
        H, W = self._image_arr.shape
        r = self._erase_image_radius()
        for pt in self._erase_pts[-2:]:
            cx, cy = pt.x(), pt.y()
            y1, y2 = max(0, math.floor(cy-r)), min(H, math.ceil(cy+r)+1)
            x1, x2 = max(0, math.floor(cx-r)), min(W, math.ceil(cx+r)+1)
            yg, xg = np.ogrid[y1:y2, x1:x2]
            self._roi_mask[y1:y2, x1:x2][(xg-cx)**2+(yg-cy)**2 <= r**2] = False
        self._rebuild_roi_pixmap(); self.roi_changed.emit(self._roi_mask.copy())

    def _erase_image_radius(self) -> float:
        """Return the fixed screen-space eraser radius in image pixels."""
        return self._erase_radius / max(self._zoom, 1e-9)

    def _merge_mask(self, new_mask: np.ndarray) -> None:
        if self._constraint_mask is not None:
            if self._constraint_mask.shape != new_mask.shape:
                raise ValueError("Drawing constraint shape does not match the image.")
            new_mask = np.asarray(new_mask, dtype=bool) & self._constraint_mask
        self._roi_mask = new_mask if self._roi_mask is None else (self._roi_mask | new_mask)
        self._rebuild_roi_pixmap(); self.roi_changed.emit(self._roi_mask.copy())

    def _check_snap(self, widget_pos: QPointF) -> bool:
        if not self._poly_pts: return False
        return _dist(widget_pos.x(), widget_pos.y(), self._poly_pts[0].x() * self._zoom + self._pan_x, self._poly_pts[0].y() * self._zoom + self._pan_y) < POLYGON_SNAP_RADIUS_PX

    def _get_transform(self) -> QTransform:
        t = QTransform(); t.translate(self._pan_x, self._pan_y); t.scale(self._zoom, self._zoom); return t

    def _widget_to_image(self, pos: QPointF) -> Optional[QPointF]:
        if self._image_arr is None: return None
        return QPointF((pos.x() - self._pan_x) / self._zoom, (pos.y() - self._pan_y) / self._zoom)

    def _fit_to_window(self) -> None:
        if self._image_px is None: return
        iw, ih = self._image_px.width(), self._image_px.height()
        ww, wh = max(1, self.width()), max(1, self.height())
        self._zoom = min(ww / iw, wh / ih) * 0.92
        self._pan_x = (ww - iw * self._zoom) / 2.0
        self._pan_y = (wh - ih * self._zoom) / 2.0

    def resizeEvent(self, event) -> None:
        if self._image_px is not None: self._fit_to_window()
        super().resizeEvent(event)

    # ─────────────────────────────────────────────────────────────────────
    # Trajectory markers
    # ─────────────────────────────────────────────────────────────────────

    def set_marker_mode(self, on: bool) -> None:
        self._marker_mode = bool(on)
        if not on:
            self._marker_drag = False
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        self.update()

    @property
    def markers(self) -> list[tuple[float, float]]:
        return [(p.x(), p.y()) for p in self._markers]

    def set_markers(self, pts) -> None:
        self._markers = [QPointF(float(x), float(y)) for x, y in (pts or [])]
        if self._marker_sel >= len(self._markers):
            self._marker_sel = -1
        self._marker_draw_pts = []
        self.update()

    def set_marker_draw_positions(self, pts) -> None:
        """Advected on-screen positions for the current frame (None = lost)."""
        self._marker_draw_pts = [
            None if p is None else QPointF(float(p[0]), float(p[1])) for p in (pts or [])
        ]
        self.update()

    def add_marker(self, x: float, y: float) -> int:
        self._markers.append(QPointF(float(x), float(y)))
        self._marker_sel = len(self._markers) - 1
        self._marker_draw_pts = []
        self.markers_changed.emit(self.markers())
        self.marker_selected.emit(self._marker_sel)
        self.update()
        return self._marker_sel

    def remove_marker(self, i: int) -> None:
        if 0 <= i < len(self._markers):
            self._markers.pop(i)
            if i < len(self._marker_draw_pts):
                self._marker_draw_pts.pop(i)
            self._marker_sel = min(self._marker_sel, len(self._markers) - 1)
            self.markers_changed.emit(self.markers())
            self.marker_selected.emit(self._marker_sel)
            self.update()

    def clear_markers(self) -> None:
        self._markers = []
        self._marker_draw_pts = []
        self._marker_sel = -1
        self._streak_paths = []
        self._streak_path = None
        self.markers_changed.emit([])
        self.marker_selected.emit(-1)
        self.update()

    def select_marker(self, i: int) -> None:
        self._marker_sel = i if 0 <= i < len(self._markers) else -1
        self.marker_selected.emit(self._marker_sel)
        self.update()

    @property
    def selected_marker(self) -> int:
        return self._marker_sel

    def _marker_render_pt(self, i: int) -> Optional[QPointF]:
        if i < len(self._marker_draw_pts):
            return self._marker_draw_pts[i]
        return self._markers[i] if i < len(self._markers) else None

    def _hit_marker(self, wx: float, wy: float) -> Optional[int]:
        """Topmost marker under the widget-space point, or None."""
        best, best_d = None, MARKER_HIT_PX
        for i in range(len(self._markers) - 1, -1, -1):
            p = self._marker_render_pt(i)
            if p is None:
                continue
            sx = p.x() * self._zoom + self._pan_x
            sy = p.y() * self._zoom + self._pan_y
            d = _dist(wx, wy, sx, sy)
            if d <= best_d:
                best, best_d = i, d
        return best

    # ─────────────────────────────────────────────────────────────────────

    def set_streaklines(self, lines, colors=None, lost_flags=None) -> None:
        """
        lines      : list of point-lists in image coords
        colors     : optional per-line QColor / colour string
        lost_flags : optional per-line bool -- track was lost before the frame
        """
        self._streaklines = lines
        self._streak_path = None
        self._streak_paths = []
        if not lines:
            self.update()
            return
        for i, pts in enumerate(lines):
            if not pts or len(pts) < 2:
                continue
            path = QPainterPath()
            path.moveTo(QPointF(pts[0][0], pts[0][1]))
            for (x, y) in pts[1:]:
                path.lineTo(QPointF(x, y))
            if colors is not None and i < len(colors) and colors[i] is not None:
                c = QColor(colors[i])
            else:
                c = marker_color(i)
            lost = bool(lost_flags[i]) if (lost_flags is not None and i < len(lost_flags)) else False
            self._streak_paths.append((path, c, lost))
        self.update()

        self.update()

    def set_subset_radius(self, radius: Optional[int]) -> None:
        self._subset_radius = radius
        self.update()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry Low-Level Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _point_in_polygon(pt: QPointF, poly: List[QPointF]) -> bool:
    x, y, n, inside, j = pt.x(), pt.y(), len(poly), False, len(poly) - 1
    for i in range(n):
        xi, yi = poly[i].x(), poly[i].y()
        xj, yj = poly[j].x(), poly[j].y()
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi): inside = not inside
        j = i
    return inside

def _polygon_mask(pts: List[QPointF], H: int, W: int) -> np.ndarray:
    from PIL import Image as PILImage, ImageDraw
    img = PILImage.new("L", (W, H), 0); draw = ImageDraw.Draw(img)
    draw.polygon([(p.x(), p.y()) for p in pts], fill=255)
    return np.array(img) > 0
