"""
image_canvas.py
---------------
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional, List

import numpy as np
from PyQt6.QtCore import (Qt, QPoint, QPointF, QRect, QRectF,
                           pyqtSignal, QSize)
from PyQt6.QtGui import (QPainter, QPen, QBrush, QColor, QPixmap,
                          QPolygonF, QPainterPath, QImage, QCursor,
                          QTransform, QIcon)
from PyQt6.QtWidgets import QWidget, QSizePolicy


class ROITool(Enum):
    NONE      = auto()
    POLYGON   = auto()
    RECTANGLE = auto()
    CIRCLE    = auto()
    ERASE     = auto()


class ImageCanvas(QWidget):
    """
    A widget that displays a greyscale image and lets the user draw a Region
    Of Interest.  Results can be overlaid as a semi-transparent colormap.

    Signals
    -------
    roi_changed(np.ndarray)
        Emitted when the ROI mask is updated (bool array, same shape as image).
    seed_placed(int, int)
        Emitted when the user right-clicks to set a seed point.
    """

    roi_changed  = pyqtSignal(object)    # np.ndarray bool mask
    seed_placed  = pyqtSignal(int, int)  # x, y
    cursor_moved = pyqtSignal(int, int, float)  # px, py, field_value

    # Colours
    _ROI_FILL_COLOR   = QColor(47, 129, 247, 55)   # #2f81f7 at 22% alpha
    _ROI_BORDER_COLOR = QColor(47, 129, 247, 200)
    _POLY_VERT_COLOR  = QColor(255, 200, 50, 230)
    _ERASE_COLOR      = QColor(248, 81, 73, 120)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 200)
        self.setMouseTracking(True)

        self._image_arr:   Optional[np.ndarray] = None 
        self._image_px:    Optional[QPixmap]    = None   
        self._result_arr:  Optional[np.ndarray] = None  
        self._result_px:   Optional[QPixmap]    = None  
        self._roi_mask:    Optional[np.ndarray] = None   
        self._roi_px:      Optional[QPixmap]    = None   

        # Viewport transform
        self._zoom:    float  = 1.0
        self._pan_x:   float  = 0.0
        self._pan_y:   float  = 0.0
        self._dragging: bool  = False
        self._drag_start: QPoint = QPoint()

        self._tool:        ROITool    = ROITool.NONE
        self._poly_pts:    List[QPointF] = []  
        self._rect_start:  Optional[QPointF] = None
        self._rect_cur:    Optional[QPointF] = None
        self._circ_centre: Optional[QPointF] = None
        self._circ_radius: float = 0.0
        self._mouse_img:   Optional[QPointF] = None
        self._erase_pts:   List[QPointF] = []
        self._erase_radius: int = 20

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


    def set_image(self, arr: np.ndarray) -> None:
        """Load a float64 [0,1] greyscale array as the display image."""
        self._image_arr  = arr
        self._result_arr = None
        self._result_px  = None
        self._roi_mask   = None
        self._roi_px     = None
        self._image_px   = _array_to_pixmap_gray(arr)
        self._fit_to_window()
        self.update()

    def set_result_overlay(
        self,
        field: np.ndarray,
        mask: np.ndarray,
        colormap: str = "RdYlBu_r",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha: float = 0.75,
    ) -> None:
        """
        Paint a scalar field over the image as a semi-transparent colormap.

        Parameters
        ----------
        field : (H, W) float array (NaN = transparent)
        mask  : (H, W) bool array  (False = transparent)
        """
        if self._image_arr is None:
            return
        self._result_arr = field
        self._result_px  = _field_to_pixmap(field, mask, colormap, vmin, vmax, alpha)
        self.update()

    def clear_result_overlay(self) -> None:
        self._result_arr = None
        self._result_px  = None
        self.update()

    def set_roi_mask(self, mask: np.ndarray) -> None:
        """Set and display an existing ROI mask."""
        self._roi_mask = mask.astype(bool)
        self._rebuild_roi_pixmap()
        self.update()

    def clear_roi(self) -> None:
        self._roi_mask  = None
        self._roi_px    = None
        self._poly_pts  = []
        self._rect_start = None
        self.update()

    def set_tool(self, tool: ROITool) -> None:
        self._tool = tool
        self._poly_pts = []
        self._rect_start = None
        self._circ_centre = None
        if tool == ROITool.NONE:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif tool == ROITool.ERASE:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    @property
    def roi_mask(self) -> Optional[np.ndarray]:
        return self._roi_mask


    def mousePressEvent(self, event) -> None:
        pos = event.position()

        if event.button() == Qt.MouseButton.MiddleButton:
            # Pan start
            self._dragging  = True
            self._drag_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.RightButton:
            img_pt = self._widget_to_image(pos)
            if img_pt and self._image_arr is not None:
                x = int(round(img_pt.x()))
                y = int(round(img_pt.y()))
                H, W = self._image_arr.shape
                if 0 <= x < W and 0 <= y < H:
                    self.seed_placed.emit(x, y)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            img_pt = self._widget_to_image(pos)
            if img_pt is None:
                return

            if self._tool == ROITool.POLYGON:
                self._poly_pts.append(img_pt)
                self.update()

            elif self._tool == ROITool.RECTANGLE:
                self._rect_start = img_pt
                self._rect_cur   = img_pt

            elif self._tool == ROITool.CIRCLE:
                self._circ_centre = img_pt
                self._circ_radius = 0.0

            elif self._tool == ROITool.ERASE:
                self._erase_pts = [img_pt]

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool == ROITool.POLYGON and len(self._poly_pts) >= 3:
                self._commit_polygon()
                return

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        img_pt = self._widget_to_image(pos)
        self._mouse_img = img_pt

        if self._dragging:
            delta = pos.toPoint() - self._drag_start
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._drag_start = pos.toPoint()
            self.update()
            return

        if event.buttons() & Qt.MouseButton.LeftButton:
            if img_pt is None:
                return

            if self._tool == ROITool.RECTANGLE:
                self._rect_cur = img_pt
                self.update()

            elif self._tool == ROITool.CIRCLE:
                if self._circ_centre:
                    dx = img_pt.x() - self._circ_centre.x()
                    dy = img_pt.y() - self._circ_centre.y()
                    self._circ_radius = math.sqrt(dx*dx + dy*dy)
                    self.update()

            elif self._tool == ROITool.ERASE:
                self._erase_pts.append(img_pt)
                self._apply_erase()
                self.update()

        self.update()  # repaint cursor position

        # Emit cursor position + field value for status bar
        if img_pt is not None:
            px, py = int(img_pt.x()), int(img_pt.y())
            val = float("nan")
            if self._result_arr is not None:
                H, W = self._result_arr.shape[:2]
                if 0 <= py < H and 0 <= px < W:
                    val = float(self._result_arr[py, px])
            self.cursor_moved.emit(px, py, val)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.CrossCursor
                           if self._tool != ROITool.NONE
                           else Qt.CursorShape.ArrowCursor)

        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool == ROITool.RECTANGLE and self._rect_start:
                self._commit_rectangle()

            elif self._tool == ROITool.CIRCLE and self._circ_centre:
                if self._circ_radius > 2:
                    self._commit_circle()

            elif self._tool == ROITool.ERASE:
                self._erase_pts = []

    def wheelEvent(self, event) -> None:
        delta   = event.angleDelta().y()
        factor  = 1.15 if delta > 0 else 1.0 / 1.15
        pos     = event.position()

        # Zoom around cursor position
        self._pan_x = pos.x() + (self._pan_x - pos.x()) * factor
        self._pan_y = pos.y() + (self._pan_y - pos.y()) * factor
        self._zoom  = max(0.1, min(self._zoom * factor, 40.0))
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self._tool == ROITool.POLYGON:
                self._poly_pts = []
                self.update()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._tool == ROITool.POLYGON and len(self._poly_pts) >= 3:
                self._commit_polygon()


    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor("#0d1117"))

        if self._image_px is None:
            # Placeholder text
            painter.setPen(QColor("#484f58"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Load a reference image to begin")
            return

        # Apply viewport transform
        painter.save()
        t = self._get_transform()
        painter.setTransform(t)

        # Base image
        painter.drawPixmap(0, 0, self._image_px)

        # Result overlay
        if self._result_px is not None:
            painter.drawPixmap(0, 0, self._result_px)

        # ROI mask overlay
        if self._roi_px is not None:
            painter.drawPixmap(0, 0, self._roi_px)

        # Active ROI drawing preview
        self._paint_roi_preview(painter)

        painter.restore()

        # HUD: coordinates
        if self._mouse_img and self._image_arr is not None:
            x = self._mouse_img.x()
            y = self._mouse_img.y()
            H, W = self._image_arr.shape
            if 0 <= x < W and 0 <= y < H:
                val = self._image_arr[int(y), int(x)]
                txt = f"  x={int(x)}  y={int(y)}  I={val:.3f}  zoom={self._zoom:.2f}×"
                painter.setPen(QColor("#8b949e"))
                painter.drawText(4, self.height() - 6, txt)

    def _paint_roi_preview(self, painter: QPainter) -> None:
        """Paint the in-progress ROI shape."""
        if self._tool == ROITool.POLYGON and self._poly_pts:
            pen = QPen(self._ROI_BORDER_COLOR, 1.5 / self._zoom)
            painter.setPen(pen)
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))

            if len(self._poly_pts) > 1:
                for i in range(len(self._poly_pts) - 1):
                    painter.drawLine(self._poly_pts[i], self._poly_pts[i + 1])

            # Dashed line from last point to cursor
            if self._mouse_img and len(self._poly_pts) >= 1:
                pen2 = QPen(self._ROI_BORDER_COLOR, 1.0 / self._zoom,
                            Qt.PenStyle.DashLine)
                painter.setPen(pen2)
                painter.drawLine(self._poly_pts[-1], self._mouse_img)

            # Vertices
            r = 3.5 / self._zoom
            painter.setPen(QPen(self._POLY_VERT_COLOR, 1.5 / self._zoom))
            painter.setBrush(QBrush(self._POLY_VERT_COLOR))
            for pt in self._poly_pts:
                painter.drawEllipse(pt, r, r)

        elif self._tool == ROITool.RECTANGLE and self._rect_start and self._rect_cur:
            x1, y1 = self._rect_start.x(), self._rect_start.y()
            x2, y2 = self._rect_cur.x(), self._rect_cur.y()
            rect = QRectF(min(x1, x2), min(y1, y2),
                          abs(x2 - x1), abs(y2 - y1))
            painter.setPen(QPen(self._ROI_BORDER_COLOR, 1.5 / self._zoom))
            painter.setBrush(QBrush(self._ROI_FILL_COLOR))
            painter.drawRect(rect)

        elif self._tool == ROITool.CIRCLE and self._circ_centre and self._circ_radius > 0:
            painter.setPen(QPen(self._ROI_BORDER_COLOR, 1.5 / self._zoom))
            painter.setBrush(QBrush(self._ROI_FILL_COLOR))
            r = self._circ_radius
            painter.drawEllipse(self._circ_centre, r, r)

        elif self._tool == ROITool.ERASE and self._mouse_img:
            r = self._erase_radius / self._zoom
            painter.setPen(QPen(self._ERASE_COLOR, 1.5 / self._zoom))
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawEllipse(self._mouse_img, r, r)

    # ------------------------------------------------------------------
    # ROI commit helpers
    # ------------------------------------------------------------------

    def _commit_polygon(self) -> None:
        if self._image_arr is None or len(self._poly_pts) < 3:
            return
        H, W = self._image_arr.shape
        mask = _polygon_mask(self._poly_pts, H, W)
        self._merge_mask(mask)
        self._poly_pts = []
        self.update()

    def _commit_rectangle(self) -> None:
        if self._image_arr is None or not self._rect_start or not self._rect_cur:
            return
        H, W = self._image_arr.shape
        x1 = int(min(self._rect_start.x(), self._rect_cur.x()))
        y1 = int(min(self._rect_start.y(), self._rect_cur.y()))
        x2 = int(max(self._rect_start.x(), self._rect_cur.x()))
        y2 = int(max(self._rect_start.y(), self._rect_cur.y()))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        mask = np.zeros((H, W), dtype=bool)
        mask[y1:y2, x1:x2] = True
        self._merge_mask(mask)
        self._rect_start = self._rect_cur = None
        self.update()

    def _commit_circle(self) -> None:
        if self._image_arr is None or not self._circ_centre:
            return
        H, W = self._image_arr.shape
        cx, cy = self._circ_centre.x(), self._circ_centre.y()
        r = self._circ_radius
        yg, xg = np.ogrid[:H, :W]
        mask = (xg - cx) ** 2 + (yg - cy) ** 2 <= r ** 2
        self._merge_mask(mask)
        self._circ_centre = None
        self._circ_radius = 0.0
        self.update()

    def _apply_erase(self) -> None:
        if self._image_arr is None or not self._erase_pts:
            return
        if self._roi_mask is None:
            return
        H, W = self._image_arr.shape
        r = self._erase_radius
        for pt in self._erase_pts[-2:]:  # only process newest points
            cx, cy = int(pt.x()), int(pt.y())
            y1, y2 = max(0, cy - r), min(H, cy + r + 1)
            x1, x2 = max(0, cx - r), min(W, cx + r + 1)
            yg, xg = np.ogrid[y1:y2, x1:x2]
            circle = (xg - cx) ** 2 + (yg - cy) ** 2 <= r ** 2
            self._roi_mask[y1:y2, x1:x2][circle] = False
        self._rebuild_roi_pixmap()
        self.roi_changed.emit(self._roi_mask.copy())

    def _merge_mask(self, new_mask: np.ndarray) -> None:
        if self._roi_mask is None:
            self._roi_mask = new_mask
        else:
            self._roi_mask = self._roi_mask | new_mask
        self._rebuild_roi_pixmap()
        self.roi_changed.emit(self._roi_mask.copy())

    def _rebuild_roi_pixmap(self) -> None:
        if self._roi_mask is None or self._image_arr is None:
            self._roi_px = None
            return
        H, W = self._roi_mask.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[self._roi_mask, 0] = 47   # R
        rgba[self._roi_mask, 1] = 129  # G
        rgba[self._roi_mask, 2] = 247  # B
        rgba[self._roi_mask, 3] = 50   # A (20%)
        # Border: erode mask slightly for outline
        from scipy.ndimage import binary_erosion
        border = self._roi_mask & ~binary_erosion(self._roi_mask)
        rgba[border, 0] = 47
        rgba[border, 1] = 129
        rgba[border, 2] = 247
        rgba[border, 3] = 180
        self._roi_px = _rgba_array_to_pixmap(rgba)

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _get_transform(self) -> QTransform:
        t = QTransform()
        t.translate(self._pan_x, self._pan_y)
        t.scale(self._zoom, self._zoom)
        return t

    def _widget_to_image(self, pos: QPointF) -> Optional[QPointF]:
        """Convert widget coordinates to image pixel coordinates."""
        if self._image_arr is None:
            return None
        ix = (pos.x() - self._pan_x) / self._zoom
        iy = (pos.y() - self._pan_y) / self._zoom
        return QPointF(ix, iy)

    def _fit_to_window(self) -> None:
        if self._image_px is None:
            return
        iw = self._image_px.width()
        ih = self._image_px.height()
        ww = max(1, self.width())
        wh = max(1, self.height())
        scale = min(ww / iw, wh / ih) * 0.92
        self._zoom  = scale
        self._pan_x = (ww - iw * scale) / 2.0
        self._pan_y = (wh - ih * scale) / 2.0

    def resizeEvent(self, event) -> None:
        if self._image_px is not None:
            self._fit_to_window()
        super().resizeEvent(event)

    def fit_image(self) -> None:
        """Reset zoom so the full image fits the window."""
        self._fit_to_window()
        self.update()

    def set_base_image(self, arr: np.ndarray) -> None:
        """Alias for set_image — load a float64 [0,1] greyscale array."""
        self.set_image(arr)

    def zoom_fit(self) -> None:
        """Alias for fit_image."""
        self.fit_image()

    def set_zoom(self, factor: float) -> None:
        """Set an absolute zoom level, keeping image centred."""
        if self._image_px is None:
            return
        self._zoom = max(0.1, min(factor, 40.0))
        ww, wh = max(1, self.width()), max(1, self.height())
        iw, ih = self._image_px.width(), self._image_px.height()
        self._pan_x = (ww - iw * self._zoom) / 2.0
        self._pan_y = (wh - ih * self._zoom) / 2.0
        self.update()

    def set_roi_tool(self, tool: ROITool) -> None:
        """Alias for set_tool."""
        self.set_tool(tool)

    def set_result_overlay_rgba(self, rgba: np.ndarray) -> None:
        """
        Set result overlay from a pre-rendered H×W×4 uint8 RGBA array
        (faster path when the caller has already applied a colourmap).
        """
        if rgba is None:
            self.clear_result_overlay()
            return
        self._result_px = _rgba_array_to_pixmap(rgba)
        self.update()

def _array_to_pixmap_gray(arr: np.ndarray) -> QPixmap:
    """Convert float64 [0,1] greyscale array to QPixmap."""
    u8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
    H, W = u8.shape
    img = QImage(u8.data, W, H, W, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(img.copy())


def _field_to_pixmap(
    field: np.ndarray,
    mask: np.ndarray,
    colormap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    alpha: float,
) -> QPixmap:
    """Render a scalar field to a QPixmap using a matplotlib colormap."""
    import matplotlib
    import matplotlib.cm as mcm
    cmap = mcm.get_cmap(colormap)

    valid = mask & ~np.isnan(field)
    if vmin is None:
        vmin = float(np.nanmin(field[valid])) if valid.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(field[valid])) if valid.any() else 1.0
    if vmax == vmin:
        vmax = vmin + 1e-12

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    H, W = field.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    fv = np.where(valid, field, 0.0)
    colors = cmap(norm(fv))   # (H, W, 4) float [0,1]
    rgba[..., :3] = (colors[..., :3] * 255).astype(np.uint8)
    rgba[..., 3]  = np.where(valid, int(alpha * 255), 0).astype(np.uint8)
    return _rgba_array_to_pixmap(rgba)


def _rgba_array_to_pixmap(rgba: np.ndarray) -> QPixmap:
    H, W = rgba.shape[:2]
    contiguous = np.ascontiguousarray(rgba)
    img = QImage(contiguous.data, W, H, W * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(img.copy())


def _polygon_mask(pts: List[QPointF], H: int, W: int) -> np.ndarray:
    """Rasterize a polygon (given as image-space QPointF list) to a bool mask."""
    from PIL import Image as PILImage, ImageDraw
    img = PILImage.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    poly = [(p.x(), p.y()) for p in pts]
    draw.polygon(poly, fill=255)
    return np.array(img) > 0
