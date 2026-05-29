"""
icgn.py
-------
Inverse Compositional Gauss-Newton (IC-GN) optimizer for DIC.

Implements the algorithm described in Blaber et al. (2015) §"Non-Linear
Optimization Scheme" and Appendix A1, following Baker & Matthews (2004).

The deformation vector is:
    p = [u, v, du/dx, du/dy, dv/dx, dv/dy]ᵀ   (6 parameters)

The ZNSSD (zero-normalized sum of squared differences) criterion C_LS is
minimized using iterative compositional warp updates.  The Hessian is
precomputed once per subset, giving the IC-GN method its speed advantage.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky, LinAlgError, solve

from .bspline import BSplineInterpolator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class SubsetData:
    """Precomputed quantities for one reference subset (used by IC-GN)."""

    __slots__ = (
        "center_x", "center_y",
        "dx", "dy",            # int arrays: relative pixel coords in subset
        "f_norm",              # normalised reference intensities
        "sigma_f",             # standard deviation of f
        "sd",                  # steepest-descent images (n_px × 6), normalised
        "H",                   # 6×6 Hessian
        "L",                   # Cholesky factor of H (or None if failed)
        "valid",               # bool: was Cholesky feasible?
    )

    def __init__(
        self,
        center_x: int, center_y: int,
        dx: np.ndarray, dy: np.ndarray,
        f_norm: np.ndarray, sigma_f: float,
        sd: np.ndarray, H: np.ndarray, L,
    ) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.dx = dx
        self.dy = dy
        self.f_norm = f_norm
        self.sigma_f = sigma_f
        self.sd = sd
        self.H = H
        self.L = L
        self.valid = (L is not None) and (sigma_f > 1e-12)


# ---------------------------------------------------------------------------
# Subset precomputation
# ---------------------------------------------------------------------------

def precompute_subset(
    ref_image: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    center_x: int,
    center_y: int,
    dx: np.ndarray,
    dy: np.ndarray,
) -> SubsetData:
    """
    Precompute all constant quantities for one reference subset.

    Parameters
    ----------
    ref_image : (H, W) float64
        Reference image (integer-pixel values — B-spline coefficients not
        needed here because the reference is sampled at integer locations).
    grad_x, grad_y : (H, W) float64
        Pre-computed ∂I/∂x and ∂I/∂y for the entire reference image.
    center_x, center_y : int
        Subset centre in pixel coordinates.
    dx, dy : 1-D int arrays
        Relative coordinates of subset pixels (from circular_subset()).

    Returns
    -------
    SubsetData
    """
    H_im, W_im = ref_image.shape

    # Absolute integer coordinates of subset pixels
    xs = center_x + dx
    ys = center_y + dy

    # Boundary check — keep only pixels within image
    valid_px = (xs >= 0) & (xs < W_im) & (ys >= 0) & (ys < H_im)
    xs = xs[valid_px]
    ys = ys[valid_px]
    dx_ = dx[valid_px]
    dy_ = dy[valid_px]

    n_px = len(xs)
    if n_px < 6:
        # Not enough pixels to define 6 parameters
        return SubsetData(center_x, center_y, dx_, dy_,
                          np.zeros(n_px), 0.0,
                          np.zeros((n_px, 6)), np.zeros((6, 6)), None)

    # Reference intensities
    f = ref_image[ys, xs]          # (n_px,)
    f_m = f.mean()
    f_c = f - f_m
    sigma_f = float(np.sqrt((f_c ** 2).sum()))

    if sigma_f < 1e-12:
        return SubsetData(center_x, center_y, dx_, dy_,
                          np.zeros(n_px), sigma_f,
                          np.zeros((n_px, 6)), np.zeros((6, 6)), None)

    f_norm = f_c / sigma_f

    # Gradients at subset pixels
    gx = grad_x[ys, xs]   # ∂f/∂x at each subset pixel
    gy = grad_y[ys, xs]   # ∂f/∂y

    # Steepest-descent images  (n_px × 6)
    # SD_k = [∂f/∂x, ∂f/∂y, ∂f/∂x·Δx, ∂f/∂x·Δy, ∂f/∂y·Δx, ∂f/∂y·Δy]
    dx_f = dx_.astype(np.float64)
    dy_f = dy_.astype(np.float64)
    SD = np.column_stack([gx, gy, gx * dx_f, gx * dy_f, gy * dx_f, gy * dy_f])

    # Normalised steepest-descent: sd = SD / sigma_f
    sd = SD / sigma_f   # (n_px × 6)

    # Hessian H = sdᵀ sd  (6 × 6)
    H_mat = sd.T @ sd

    # Cholesky factorization (fails if H is not positive-definite)
    try:
        L = cholesky(H_mat)
    except LinAlgError:
        L = None

    return SubsetData(
        center_x, center_y, dx_, dy_,
        f_norm, sigma_f, sd, H_mat, L,
    )


# ---------------------------------------------------------------------------
# IC-GN iteration
# ---------------------------------------------------------------------------

def run_icgn(
    cur_interp: BSplineInterpolator,
    subset: SubsetData,
    p_init: np.ndarray,
    max_iter: int = 50,
    conv_tol: float = 1e-4,
) -> tuple[np.ndarray, float, bool]:
    """
    Run the IC-GN optimizer for one subset.

    Parameters
    ----------
    cur_interp : BSplineInterpolator
        Precomputed B-spline interpolator for the current (deformed) image.
    subset : SubsetData
        Precomputed reference-subset quantities.
    p_init : (6,) float array
        Initial deformation vector [u, v, du/dx, du/dy, dv/dx, dv/dy].
    max_iter : int
        Maximum number of IC-GN iterations.
    conv_tol : float
        Convergence threshold on ‖Δp‖.

    Returns
    -------
    p_opt : (6,) float array
        Optimised deformation vector.
    CLS : float
        Final ZNSSD value (0 = perfect match, 2 = worst).
    converged : bool
    """
    if not subset.valid:
        return p_init.copy(), 2.0, False

    p = p_init.astype(np.float64).copy()
    cx = float(subset.center_x)
    cy = float(subset.center_y)
    dx = subset.dx.astype(np.float64)
    dy = subset.dy.astype(np.float64)
    f_norm = subset.f_norm
    sd = subset.sd
    L = subset.L

    converged = False
    CLS = 2.0

    for _it in range(max_iter):
        # ---- Warp current image at p -------------------------------------
        x_cur = cx + dx + p[0] + p[2] * dx + p[3] * dy
        y_cur = cy + dy + p[1] + p[4] * dx + p[5] * dy

        g = cur_interp.eval(x_cur, y_cur)

        g_m = g.mean()
        g_c = g - g_m
        sigma_g = float(np.sqrt((g_c ** 2).sum()))

        if sigma_g < 1e-12:
            break

        g_norm = g_c / sigma_g

        # ---- Residual and cost -------------------------------------------
        residual = g_norm - f_norm   # (n_px,)
        CLS = float((residual ** 2).sum())

        # ---- Gradient (right-hand side) ----------------------------------
        # b = sdᵀ (g_norm − f_norm) / sigma_f  — but sd already has /sigma_f
        # From eq.21: ∇C_LS(0) = 2/sigma_f * sdᵀ * residual
        # We want Δp from H·Δp = b where b makes the gradient zero.
        # b = sd.T @ residual   (H already incorporates 1/sigma_f²)
        b = sd.T @ residual    # (6,)

        # ---- Solve H·Δp = b via Cholesky --------------------------------
        try:
            delta_p = solve(L.T, solve(L, b))
        except Exception:
            break

        # ---- Compositional warp update ----------------------------------
        # M_new = M_old · M(Δp)⁻¹
        M_old = _p_to_matrix(p)
        M_dp  = _p_to_matrix(delta_p)
        try:
            M_new = M_old @ np.linalg.inv(M_dp)
        except np.linalg.LinAlgError:
            break
        p = _matrix_to_p(M_new)

        # ---- Convergence check ------------------------------------------
        # Ncorr-identical convergence: simple Euclidean norm of full Δp vector.
        # Ncorr cutoff_diffnorm = 1e-6 (unweighted, matches Baker & Matthews 2004).
        diff_norm = np.sqrt(
            delta_p[0]**2 + delta_p[1]**2 +
            delta_p[2]**2 + delta_p[3]**2 +
            delta_p[4]**2 + delta_p[5]**2
        )
        if diff_norm < conv_tol:
            converged = True
            break

    return p, CLS, converged


# ---------------------------------------------------------------------------
# Warp matrix helpers
# ---------------------------------------------------------------------------

def _p_to_matrix(p: np.ndarray) -> np.ndarray:
    """Convert 6-parameter deformation vector to 3×3 warp matrix M."""
    return np.array([
        [1.0 + p[2],       p[3],  p[0]],
        [      p[4], 1.0 + p[5],  p[1]],
        [      0.0,         0.0,   1.0],
    ])


def _matrix_to_p(M: np.ndarray) -> np.ndarray:
    """Extract 6-parameter deformation vector from 3×3 warp matrix M."""
    return np.array([
        M[0, 2],         # u
        M[1, 2],         # v
        M[0, 0] - 1.0,   # du/dx
        M[0, 1],         # du/dy
        M[1, 0],         # dv/dx
        M[1, 1] - 1.0,   # dv/dy
    ])
