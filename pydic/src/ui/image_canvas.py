"""
image_canvas.py
---------------
Enhanced canvas with:
  - Memory-safe PyQt6/NumPy array bindings (resolves 0xC0000409 segmentation faults)
  - Proper irregular-polygon icon via make_polygon_tool_icon()
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
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, pyqtSignal, QSize
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap,
    QPolygonF, QPainterPath, QImage, QCursor,
    QTransform, QIcon, QFont,
)
from PyQt6.QtWidgets import QWidget, QSizePolicy


# ─────────────────────────────────────────────────────────────────────────────
# Tool enum & Constants
# ─────────────────────────────────────────────────────────────────────────────

class ROITool(Enum):
    NONE      = auto()
    POLYGON   = auto()
    RECTANGLE = auto()
    CIRCLE    = auto()
    ERASE     = auto()

POLYGON_SNAP_RADIUS_PX: int = 12   
VERTEX_HIT_PX:          int = 14   
EDGE_HIT_PX:            int = 10   
HANDLE_HIT_PX:          int = 10   
HANDLE_HALF:            int = 5    

TOOL_TOOLTIPS: dict = {
    ROITool.NONE:
        "Navigate / Edit — Pan: middle-drag · Zoom: scroll · Seed: right-click\n"
        "Click an existing ROI to edit it",
    ROITool.POLYGON:
        "Polygon ROI — Click to add vertices\n"
        "Hover near start to snap-close · click to finish\n"
        "Right-click removes last point · Enter/double-click to finish",
    ROITool.RECTANGLE:
        "Rectangle ROI — Click and drag to draw · Release commits",
    ROITool.CIRCLE:
        "Circle ROI — Click centre, drag to radius · Release commits",
    ROITool.ERASE:
        "Erase ROI — Paint to remove mask · [ / ] to resize brush",
}

# ─────────────────────────────────────────────────────────────────────────────
# Icon & Geometry Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_polygon_tool_icon(size: int = 24, color: QColor = None) -> QIcon:
    if color is None: color = QColor("#2f81f7")
    px = QPixmap(size, size); px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size - 2  
    raw = [(0.48, 0.03), (0.90, 0.22), (0.97, 0.64), (0.68, 0.97), (0.19, 0.88), (0.04, 0.46)]
    pts = [QPointF(x * s + 1, y * s + 1) for x, y in raw]
    fill = QColor(color); fill.setAlpha(38)
    p.setBrush(QBrush(fill)); p.setPen(QPen(color, max(1.1, size / 16.0)))
    p.drawPolygon(QPolygonF(pts))
    dc = QColor(color); dc.setAlpha(230)
    p.setBrush(QBrush(dc)); p.setPen(Qt.PenStyle.NoPen)
    dot_r = max(1.3, size / 13.0)
    for pt in pts: p.drawEllipse(pt, dot_r, dot_r)
    p.end()
    return QIcon(px)

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
    seed_placed  = pyqtSignal(int, int)
    cursor_moved = pyqtSignal(int, int, float)

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

        # VISUAL SEED & SUBSET PREVIEW
        self._seed_xy: Optional[Tuple[int, int]] = None
        self._subset_radius: Optional[int] = None
        self.seed_enabled: bool = True
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

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ─────────────────────────────────────────────────────────────────────
    # Safely Handled Pixmap Generation
    # ─────────────────────────────────────────────────────────────────────

    def set_image(self, arr: np.ndarray) -> None:
        self._image_arr = arr
        self._image_u8 = np.ascontiguousarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
        H, W = self._image_u8.shape
        self._image_qimg = QImage(self._image_u8.data, W, H, W, QImage.Format.Format_Grayscale8)
        self._image_px = QPixmap.fromImage(self._image_qimg)
        
        self.clear_result_overlay()
        self.clear_roi()
        self._fit_to_window()
        self.update()

    def set_result_overlay_rgba(self, rgba) -> None:
        if rgba is None:
            self.clear_result_overlay(); return
            
        self._result_rgba = np.ascontiguousarray(rgba)
        H, W = self._result_rgba.shape[:2]
        self._result_qimg = QImage(self._result_rgba.data, W, H, W * 4, QImage.Format.Format_RGBA8888)
        self._result_px = QPixmap.fromImage(self._result_qimg)
        self.update()

    def set_result_overlay(self, field, mask, colormap="RdYlBu_r", vmin=None, vmax=None, alpha=0.75) -> None:
        if self._image_arr is None: return
        self._result_arr = field
        import matplotlib, matplotlib.cm as mcm
        cmap = mcm.get_cmap(colormap)
        valid = mask & ~np.isnan(field)

        if valid.any():
            valid_data = field[valid]
            # Robust scaling: ignore top/bottom 2% of extreme outliers
            if vmin is None: vmin = float(np.nanpercentile(valid_data, 2))
            if vmax is None: vmax = float(np.nanpercentile(valid_data, 98))
        else:
            if vmin is None: vmin = 0.0
            if vmax is None: vmax = 1.0

        if vmax == vmin: vmax = vmin + 1e-12
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        H, W = field.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        colors = cmap(norm(np.where(valid, field, 0.0)))
        rgba[..., :3] = (colors[..., :3] * 255).astype(np.uint8)
        rgba[..., 3] = np.where(valid, int(alpha * 255), 0).astype(np.uint8)
        self.set_result_overlay_rgba(rgba)

    def _rebuild_roi_pixmap(self) -> None:
        if self._roi_mask is None or self._image_arr is None:
            self._roi_px = None; self._roi_rgba = None; self._roi_qimg = None; return
        H, W = self._roi_mask.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[self._roi_mask, 0] = 47; rgba[self._roi_mask, 1] = 129
        rgba[self._roi_mask, 2] = 247; rgba[self._roi_mask, 3] = 50
        from scipy.ndimage import binary_erosion
        border = self._roi_mask & ~binary_erosion(self._roi_mask)
        rgba[border, 0] = 47; rgba[border, 1] = 129
        rgba[border, 2] = 247; rgba[border, 3] = 180
        
        self._roi_rgba = np.ascontiguousarray(rgba)
        self._roi_qimg = QImage(self._roi_rgba.data, W, H, W * 4, QImage.Format.Format_RGBA8888)
        self._roi_px = QPixmap.fromImage(self._roi_qimg)

    def clear_result_overlay(self) -> None:
        self._result_arr = None; self._result_rgba = None; self._result_qimg = None; self._result_px = None
        self.update()

    def clear_roi(self) -> None:
        self._roi_mask = None;
        self._roi_rgba = None;
        self._roi_qimg = None;
        self._roi_px = None
        self._poly_pts = [];
        self._rect_start = None
        self._committed_poly = None;
        self._committed_rect = None
        self._poly_edit = None;
        self._rect_edit = None
        self._seed_xy = None
        self.update()

    # ─────────────────────────────────────────────────────────────────────
    # Public API Overrides
    # ─────────────────────────────────────────────────────────────────────
    
    def set_roi_mask(self, mask: np.ndarray) -> None:
        self._roi_mask = mask.astype(bool)
        self._rebuild_roi_pixmap(); self.update()

    def set_tool(self, tool: ROITool) -> None:
        self._commit_poly_edit(); self._commit_rect_edit()
        self._tool = tool; self._poly_pts = []; self._poly_snapped = False
        self._rect_start = None; self._circ_centre = None
        self._poly_edit = None; self._rect_edit = None
        cursor = Qt.CursorShape.ArrowCursor if tool == ROITool.NONE else Qt.CursorShape.CrossCursor
        self.setCursor(cursor); self.update()
        
    def set_seed_xy(self, xy: Optional[Tuple[int, int]]) -> None:
        self._seed_xy = xy
        self.update()

    @property
    def roi_mask(self) -> Optional[np.ndarray]: return self._roi_mask

    def set_base_image(self, arr): self.set_image(arr)
    def zoom_fit(self): self.fit_image()
    def set_roi_tool(self, t): self.set_tool(t)

    def fit_image(self) -> None:
        self._fit_to_window(); self.update()

    def set_zoom(self, factor: float) -> None:
        if self._image_px is None: return
        self._zoom = max(0.1, min(factor, 40.0))
        ww, wh = max(1, self.width()), max(1, self.height())
        iw, ih = self._image_px.width(), self._image_px.height()
        self._pan_x = (ww - iw * self._zoom) / 2.0
        self._pan_y = (wh - ih * self._zoom) / 2.0
        self.update()

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
            if self._tool == ROITool.POLYGON and self._poly_pts:
                self._poly_pts.pop();
                self._poly_snapped = False;
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

    def mouseMoveEvent(self, event) -> None:
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
            if self._tool == ROITool.POLYGON and self._poly_pts:
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
        if key == Qt.Key.Key_Escape:
            if self._poly_edit: self._commit_poly_edit()
            elif self._rect_edit: self._commit_rect_edit()
            elif self._tool == ROITool.POLYGON: self._poly_pts = []; self._poly_snapped = False
            self.update()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._poly_edit: self._commit_poly_edit()
            elif self._rect_edit: self._commit_rect_edit()
            elif self._tool == ROITool.POLYGON and len(self._poly_pts) >= 3: self._commit_polygon()
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
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#0d1117"))

        if self._image_px is None:
            painter.setPen(QColor("#484f58"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a reference image to begin")
            return

        painter.save()
        painter.setTransform(self._get_transform())
        painter.drawPixmap(0, 0, self._image_px)
        if self._result_px is not None: painter.drawPixmap(0, 0, self._result_px)
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
        # ────────────────────────────────────────

        painter.restore()

        if self._poly_edit is not None: self._paint_poly_edit(painter)
        if self._rect_edit is not None: self._paint_rect_edit(painter)

        if self._mouse_img and self._image_arr is not None:
            x, y = self._mouse_img.x(), self._mouse_img.y()
            H, W = self._image_arr.shape
            if 0 <= x < W and 0 <= y < H:
                val = self._image_arr[int(y), int(x)]
                extra = ""
                if self._tool == ROITool.POLYGON and self._poly_pts:
                    extra = f"  pts={len(self._poly_pts)}"
                    extra += ("  ● click to close" if self._poly_snapped else "  RMB=undo · ↵/dbl=finish")
                elif self._poly_edit is not None:
                    extra = "  EDIT POLY — drag vertex · click edge=insert · RMB=delete · ↵=done"
                elif self._rect_edit is not None:
                    extra = "  EDIT RECT — drag handle · drag interior=move · ↵=done"
                txt = f"  x={int(x)}  y={int(y)}  I={val:.3f}  zoom={self._zoom:.2f}×{extra}"
                painter.setPen(QColor("#8b949e"))
                painter.drawText(4, self.height() - 6, txt)

    def _paint_roi_preview(self, painter: QPainter) -> None:
        if self._tool == ROITool.POLYGON and self._poly_pts:
            pen = QPen(self._ROI_BORDER_COLOR, 1.5 / self._zoom)
            painter.setPen(pen); painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(self._poly_pts) - 1):
                painter.drawLine(self._poly_pts[i], self._poly_pts[i + 1])
            if self._mouse_img:
                tgt  = self._poly_pts[0] if self._poly_snapped else self._mouse_img
                c    = self._SNAP_RING_COLOR if self._poly_snapped else self._ROI_BORDER_COLOR
                painter.setPen(QPen(c, 1.5 / self._zoom, Qt.PenStyle.DashLine))
                painter.drawLine(self._poly_pts[-1], tgt)
            r = 3.5 / self._zoom
            painter.setPen(QPen(self._POLY_VERT_COLOR, 1.5 / self._zoom))
            painter.setBrush(QBrush(self._POLY_VERT_COLOR))
            for pt in self._poly_pts: painter.drawEllipse(pt, r, r)
            if self._poly_snapped:
                snap_r = POLYGON_SNAP_RADIUS_PX / self._zoom
                painter.setPen(QPen(self._SNAP_RING_COLOR, 2.0 / self._zoom))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(self._poly_pts[0], snap_r, snap_r)

        elif self._tool == ROITool.RECTANGLE and self._rect_start and self._rect_cur:
            x1, y1 = self._rect_start.x(), self._rect_start.y()
            x2, y2 = self._rect_cur.x(),   self._rect_cur.y()
            painter.setPen(QPen(self._ROI_BORDER_COLOR, 1.5 / self._zoom))
            painter.setBrush(QBrush(self._ROI_FILL_COLOR))
            painter.drawRect(QRectF(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)))

        elif self._tool == ROITool.CIRCLE and self._circ_centre and self._circ_radius > 0:
            painter.setPen(QPen(self._ROI_BORDER_COLOR, 1.5 / self._zoom))
            painter.setBrush(QBrush(self._ROI_FILL_COLOR))
            painter.drawEllipse(self._circ_centre, self._circ_radius, self._circ_radius)

        elif self._tool == ROITool.ERASE and self._mouse_img:
            r = self._erase_radius / self._zoom
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
            painter.setPen(QPen(QColor(0, 229, 255, 180), 1)); painter.setBrush(QBrush(QColor("#161B22")))
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
            painter.setPen(QPen(col, 2)); painter.setBrush(QBrush(QColor("#0d1117")))
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
            painter.setPen(QPen(self._HANDLE_COLOR, 1.5)); painter.setBrush(QBrush(QColor("#0d1117")))
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
        H, W = self._image_arr.shape; r = self._erase_radius
        for pt in self._erase_pts[-2:]:
            cx, cy = int(pt.x()), int(pt.y())
            y1, y2 = max(0, cy-r), min(H, cy+r+1)
            x1, x2 = max(0, cx-r), min(W, cx+r+1)
            yg, xg = np.ogrid[y1:y2, x1:x2]
            self._roi_mask[y1:y2, x1:x2][(xg-cx)**2+(yg-cy)**2 <= r**2] = False
        self._rebuild_roi_pixmap(); self.roi_changed.emit(self._roi_mask.copy())

    def _merge_mask(self, new_mask: np.ndarray) -> None:
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
