"""Fast, interactive 3D parameter plot painted entirely by Qt."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from strainx.ui import render
from strainx.ui.result_controls import DEFAULT_CMAP


class ParametricPlot3D(QWidget):
    """A small rectilinear surface that never uses Matplotlib or OpenGL."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._points: list[tuple[float, float, float]] = []
        cmap = render.get_cmap(DEFAULT_CMAP, 64)
        self._colors = tuple(tuple(int(channel * 255)
                                   for channel in cmap(i / 63)[:3])
                             for i in range(64))
        self._color_limits = (0.0, 1.0)
        self._mark_out_of_range = False
        self._title = self._x_label = self._y_label = self._z_label = ""
        self._yaw = math.radians(38)
        self._elevation = math.radians(26)
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._drag_position = None
        self._drag_button = None

    def set_data(self, points, title: str, x_label: str,
                 y_label: str, z_label: str, *, colors=None,
                 limits=None, mark_out_of_range: bool = False) -> None:
        grouped = {}
        for x, y, z in points:
            if all(math.isfinite(float(v)) for v in (x, y, z)):
                grouped.setdefault((float(x), float(y)), []).append(float(z))
        self._points = [
            (x, y, sum(values) / len(values))
            for (x, y), values in sorted(grouped.items())]
        if colors is not None:
            self._colors = tuple(colors)
        values = [point[2] for point in self._points]
        self._color_limits = (tuple(limits) if limits is not None else
                              (min(values), max(values)) if values else
                              (0.0, 1.0))
        self._mark_out_of_range = mark_out_of_range
        self._title, self._x_label = title, x_label
        self._y_label, self._z_label = y_label, z_label
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MouseButton.LeftButton,
                              Qt.MouseButton.RightButton,
                              Qt.MouseButton.MiddleButton):
            self._drag_position = event.position()
            self._drag_button = event.button()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_position is None:
            return super().mouseMoveEvent(event)
        delta = event.position() - self._drag_position
        self._drag_position = event.position()
        if self._drag_button == Qt.MouseButton.LeftButton:
            self._yaw += math.radians(delta.x() * 0.55)
            self._elevation = max(math.radians(-80), min(
                math.radians(80),
                self._elevation + math.radians(delta.y() * 0.45)))
        else:
            self._pan += delta
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_position is not None:
            self._drag_position = None
            self._drag_button = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        self._zoom = max(0.45, min(3.5, self._zoom * (1.12 ** steps)))
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        self._yaw = math.radians(38)
        self._elevation = math.radians(26)
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()
        event.accept()

    def _colour(self, value: float, alpha: int = 255) -> QColor:
        return QColor(*render.sample_colorbar(
            value, *self._color_limits, self._colors,
            self._mark_out_of_range), alpha)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#24282c"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        painter.setPen(QColor("#e6e8ea"))
        painter.drawText(self.rect().adjusted(14, 8, -14, -8),
                         Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                         self._title)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#8f9ba3"))
        painter.drawText(self.rect().adjusted(14, 8, -14, -8),
                         Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
                         "Drag: rotate  ·  Wheel: zoom  ·  Right-drag: pan")
        if not self._points:
            painter.setPen(QColor("#a2a8ad"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No finite values for this selection")
            return

        xs = sorted({point[0] for point in self._points})
        ys = sorted({point[1] for point in self._points})
        zs = [point[2] for point in self._points]
        minimum = (xs[0], ys[0], min(zs))
        maximum = (xs[-1], ys[-1], max(zs))
        spans = tuple(max(hi - lo, 1e-12) for lo, hi in zip(minimum, maximum))
        width, height = max(1, self.width()), max(1, self.height())
        base_y = max(80, height - 65) / 1.4
        scale_y = base_y * self._zoom
        scale_x = min(max(100, width - 150) / 2.5,
                      base_y * 2.0) * self._zoom
        center = QPointF(width * 0.5 + self._pan.x(),
                         height * 0.52 + self._pan.y())
        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        ce, se = math.cos(self._elevation), math.sin(self._elevation)

        def project(x: float, y: float, z: float):
            xx, yy, zz = x - 0.5, y - 0.5, z - 0.5
            across = cy * xx - sy * yy
            away = sy * xx + cy * yy
            up = se * away - ce * zz
            depth = ce * away + se * zz
            return QPointF(center.x() + across * scale_x,
                           center.y() + up * scale_y), depth

        def norm(point):
            return tuple((value - lo) / span for value, lo, span in
                         zip(point, minimum, spans))

        corners = [(x, y, z) for x in (0.0, 1.0)
                   for y in (0.0, 1.0) for z in (0.0, 1.0)]
        painter.setPen(QPen(QColor("#49525a"), 1))
        for first in corners:
            for axis in range(3):
                if first[axis] != 0.0:
                    continue
                second = list(first)
                second[axis] = 1.0
                painter.drawLine(project(*first)[0], project(*second)[0])

        grid = {(x, y): z for x, y, z in self._points}
        triangles = []
        for ix in range(len(xs) - 1):
            for iy in range(len(ys) - 1):
                cells = [(xs[ix], ys[iy]), (xs[ix + 1], ys[iy]),
                         (xs[ix], ys[iy + 1]), (xs[ix + 1], ys[iy + 1])]
                if not all(cell in grid for cell in cells):
                    continue
                for indices in ((0, 1, 3), (0, 3, 2)):
                    vertices = [norm((*cells[index], grid[cells[index]]))
                                for index in indices]
                    positions = [project(*vertex) for vertex in vertices]
                    triangles.append((
                        sum(item[1] for item in positions) / 3,
                        QPolygonF([item[0] for item in positions]),
                        sum(grid[cells[index]] for index in indices) / 3))
        for _depth, polygon, value in sorted(triangles, key=lambda item: item[0]):
            painter.setPen(QPen(QColor("#96b4c2"), 0.6))
            painter.setBrush(QBrush(self._colour(value, 115)))
            painter.drawPolygon(polygon)

        dots = [(project(*norm(point)), point[2])
                for point in self._points]
        painter.setPen(QPen(QColor("#edf4f7"), 0.8))
        for (position, _depth), value in sorted(dots, key=lambda item: item[0][1]):
            painter.setBrush(QBrush(self._colour(value)))
            painter.drawEllipse(position, 3.6, 3.6)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#cbd3d8"), 1.7))
        origin = project(0, 0, 0)[0]
        for endpoint in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            painter.drawLine(origin, project(*endpoint)[0])

        painter.setPen(QColor("#cbd3d8"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        captions = ((f"X: {self._x_label}", (1, 0, 0)),
                    (f"Y: {self._y_label}", (0, 1, 0)),
                    (f"Z: {self._z_label}", (0, 0, 1)))
        metrics = painter.fontMetrics()
        for index, (text, point) in enumerate(captions):
            end = project(*point)[0]
            text = metrics.elidedText(
                text, Qt.TextElideMode.ElideRight, max(40, width - 16))
            text_width = metrics.horizontalAdvance(text)
            if index == 2:
                x = end.x() - text_width / 2
                baseline = end.y() - 8
            else:
                x = end.x() + (8 if end.x() >= origin.x() else
                               -text_width - 8)
                baseline = end.y() + (metrics.height() + 3
                                      if end.y() >= origin.y() else -6)
            x = max(8, min(width - text_width - 8, x))
            baseline = max(metrics.height() + 6,
                           min(height - metrics.height() - 22, baseline))
            painter.drawText(QPointF(x, baseline), text)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#a2a8ad"))
        labels = (
            f"X: {minimum[0]:g}–{maximum[0]:g}",
            f"Y: {minimum[1]:g}–{maximum[1]:g}",
            f"Z: {minimum[2]:.4g}–{maximum[2]:.4g}",
        )
        for index, label in enumerate(labels):
            painter.drawText(
                self.rect().adjusted(12 + index * width // 3, 0,
                                     -(width - (index + 1) * width // 3), -6),
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                label)
