"""
bspline.py — Biquintic (order-5) B-spline image interpolation.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import spline_filter, map_coordinates

BSPLINE_ORDER: int = 5          # quintic — matches Ncorr


class BSplineInterpolator:
    """Precomputed biquintic B-spline interpolator for a 2-D greyscale image."""

    def __init__(self, image: np.ndarray) -> None:
        if image.ndim != 2:
            raise ValueError("Expected 2-D greyscale image.")
        img = image.astype(np.float64, copy=False)
        self.coefficients: np.ndarray = spline_filter(
            img, order=BSPLINE_ORDER, mode="mirror", output=np.float64
        )
        self.shape: tuple[int, int] = img.shape

    def eval(self, x, y) -> np.ndarray:
        """Interpolate at sub-pixel (column x, row y) coordinates."""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        out = map_coordinates(
            self.coefficients, [y.ravel(), x.ravel()],
            order=BSPLINE_ORDER, mode="mirror", prefilter=False,
        )
        return out.reshape(x.shape)

    def gradient(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        xr, yr = x.ravel(), y.ravel()
        h = 1e-4

        def _eval(dx, dy):
            return map_coordinates(
                self.coefficients, [yr + dy, xr + dx],
                order=BSPLINE_ORDER, mode="mirror", prefilter=False,
            )

        df_dx = (_eval(h, 0) - _eval(-h, 0)) / (2 * h)
        df_dy = (_eval(0, h) - _eval(0, -h)) / (2 * h)
        return df_dx.reshape(x.shape), df_dy.reshape(y.shape)


def image_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = image.astype(np.float64, copy=False)
    coeff = spline_filter(img, order=BSPLINE_ORDER, mode="mirror", output=np.float64)
    H, W = img.shape

    yr, xr = np.mgrid[0:H, 0:W]
    yr = yr.ravel().astype(np.float64)
    xr = xr.ravel().astype(np.float64)
    h = 1e-4

    def _mc(dx, dy):
        return map_coordinates(
            coeff, [yr + dy, xr + dx],
            order=BSPLINE_ORDER, mode="mirror", prefilter=False,
        )

    grad_x = ((_mc(h, 0) - _mc(-h, 0)) / (2 * h)).reshape(H, W)
    grad_y = ((_mc(0, h) - _mc(0, -h)) / (2 * h)).reshape(H, W)
    return grad_x, grad_y


def circular_subset(radius: int) -> tuple[np.ndarray, np.ndarray]:
    r = int(radius)
    y_g, x_g = np.mgrid[-r:r + 1, -r:r + 1]
    inside = x_g ** 2 + y_g ** 2 <= r ** 2
    return x_g[inside], y_g[inside]
