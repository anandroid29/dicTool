"""
icgn.py
-------
Inverse Compositional Gauss-Newton (IC-GN) optimizer for DIC.
Optimized for speed by eliminating matrix allocations in the inner loop.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from .bspline import BSplineInterpolator

class SubsetData:
    __slots__ = (
        "center_x", "center_y", "dx", "dy",
        "f_norm", "sigma_f", "sd", "H", "L_fac", "valid"
    )

    def __init__(
        self, center_x: int, center_y: int,
        dx: np.ndarray, dy: np.ndarray,
        f_norm: np.ndarray, sigma_f: float,
        sd: np.ndarray, H: np.ndarray, L_fac
    ) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.dx = dx
        self.dy = dy
        self.f_norm = f_norm
        self.sigma_f = sigma_f
        self.sd = sd  # steepest descent -> Jacobian
        self.H = H  # Hessian matrix
        self.L_fac = L_fac  # Cholesky factorization of Hessian
        self.valid = (L_fac is not None) and (sigma_f > 1e-12)


def precompute_subset(
    ref_image: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray,
    center_x: int, center_y: int, dx: np.ndarray, dy: np.ndarray,
) -> SubsetData:
    H_im, W_im = ref_image.shape
    xs = center_x + dx
    ys = center_y + dy
    valid_px = (xs >= 0) & (xs < W_im) & (ys >= 0) & (ys < H_im)
    xs, ys, dx_, dy_ = xs[valid_px], ys[valid_px], dx[valid_px], dy[valid_px]

    n_px = len(xs)
    if n_px < 6:
        return SubsetData(center_x, center_y, dx_, dy_, np.zeros(n_px), 0.0,
                          np.zeros((n_px, 6)), np.zeros((6, 6)), None)

    f = ref_image[ys, xs]
    f_c = f - f.mean()
    sigma_f = float(np.sqrt((f_c ** 2).sum()))

    if sigma_f < 1e-12:
        return SubsetData(center_x, center_y, dx_, dy_, np.zeros(n_px), sigma_f,
                          np.zeros((n_px, 6)), np.zeros((6, 6)), None)

    f_norm = f_c / sigma_f
    gx, gy = grad_x[ys, xs], grad_y[ys, xs]
    dx_f, dy_f = dx_.astype(np.float64), dy_.astype(np.float64)

    SD = np.column_stack([gx, gy, gx * dx_f, gx * dy_f, gy * dx_f, gy * dy_f])
    sd = SD / sigma_f
    H_mat = sd.T @ sd

    try:
        L_fac = cho_factor(H_mat, lower=True)
    except LinAlgError:
        L_fac = None

    return SubsetData(
        center_x, center_y, dx_, dy_,
        f_norm, sigma_f, sd, H_mat, L_fac,
    )


def run_icgn(
    cur_interp: BSplineInterpolator, subset: SubsetData,
    p_init: np.ndarray, max_iter: int = 50, conv_tol: float = 1e-4,
) -> tuple[np.ndarray, float, bool]:

    if not subset.valid:
        return p_init.copy(), 2.0, False

    p = p_init.astype(np.float64).copy()
    cx, cy = float(subset.center_x), float(subset.center_y)
    dx, dy = subset.dx.astype(np.float64), subset.dy.astype(np.float64)
    f_norm, sd, L_fac = subset.f_norm, subset.sd, subset.L_fac

    converged = False
    CLS = 2.0

    for _it in range(max_iter):
        x_cur = cx + dx + p[0] + p[2] * dx + p[3] * dy
        y_cur = cy + dy + p[1] + p[4] * dx + p[5] * dy

        g = cur_interp.eval(x_cur, y_cur)
        g_c = g - g.mean()
        sigma_g = float(np.sqrt((g_c ** 2).sum()))

        if sigma_g < 1e-12:
            break

        residual = (g_c / sigma_g) - f_norm
        CLS = float((residual ** 2).sum())
        b = sd.T @ residual

        try:
            delta_p = cho_solve(L_fac, b)
        except Exception:
            break

        # Analytical compositional update (Zero intermediate array allocations)
        a1, b1, c1 = 1.0 + p[2], p[3], p[0]
        d1, e1, f1 = p[4], 1.0 + p[5], p[1]

        a2, b2, c2 = 1.0 + delta_p[2], delta_p[3], delta_p[0]
        d2, e2, f2 = delta_p[4], 1.0 + delta_p[5], delta_p[1]

        det2 = a2 * e2 - b2 * d2
        if abs(det2) < 1e-12: break
        inv_det = 1.0 / det2

        i_a, i_b = e2 * inv_det, -b2 * inv_det
        i_c = (b2 * f2 - c2 * e2) * inv_det
        i_d, i_e = -d2 * inv_det, a2 * inv_det
        i_f = (c2 * d2 - a2 * f2) * inv_det

        p[0] = a1 * i_c + b1 * i_f + c1
        p[1] = d1 * i_c + e1 * i_f + f1
        p[2] = (a1 * i_a + b1 * i_d) - 1.0
        p[3] = a1 * i_b + b1 * i_e
        p[4] = d1 * i_a + e1 * i_d
        p[5] = (d1 * i_b + e1 * i_e) - 1.0

        if np.sqrt(delta_p[0]**2 + delta_p[1]**2 + delta_p[2]**2 +
                   delta_p[3]**2 + delta_p[4]**2 + delta_p[5]**2) < conv_tol:
            converged = True
            break

    return p, CLS, converged