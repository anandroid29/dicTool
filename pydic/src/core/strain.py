"""
strain.py
---------
Green-Lagrangian strain computation via least-squares plane fit.
Optimized using mathematically separable 1D filters.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import correlate1d

def compute_velocity_strains(
    Vx: np.ndarray,
    Vy: np.ndarray,
    valid_mask: np.ndarray,
    strain_window: int,
) -> dict[str, np.ndarray]:
    """Compute strain rates directly from the spatial gradients of velocity using separable filters."""
    r = int(strain_window)

    # 1D Separable Kernels
    k_ones  = np.ones(2 * r + 1, dtype=np.float64)
    k_ramp  = np.arange(-r, r + 1, dtype=np.float64)
    k_ramp2 = k_ramp ** 2

    valid = valid_mask & ~np.isnan(Vx) & ~np.isnan(Vy)
    u_z   = np.where(valid, Vx, 0.0)
    v_z   = np.where(valid, Vy, 0.0)
    cnt   = valid.astype(np.float64)

    def sep_corr(arr: np.ndarray, ky: np.ndarray, kx: np.ndarray) -> np.ndarray:
        """Applies a 2D correlation by separating it into two O(N) 1D correlations."""
        temp = correlate1d(arr, ky, axis=0, mode='constant', cval=0.0)
        return correlate1d(temp, kx, axis=1, mode='constant', cval=0.0)

    # Replace O(N^2) 2D correlations with O(2N) separable passes
    N      = sep_corr(cnt, k_ones, k_ones)
    sum_x  = sep_corr(cnt, k_ones, k_ramp)
    sum_y  = sep_corr(cnt, k_ramp, k_ones)
    sum_x2 = sep_corr(cnt, k_ones, k_ramp2)
    sum_y2 = sep_corr(cnt, k_ramp2, k_ones)
    sum_xy = sep_corr(cnt, k_ramp, k_ramp)

    sum_u  = sep_corr(u_z, k_ones, k_ones)
    sum_v  = sep_corr(v_z, k_ones, k_ones)
    sum_ux = sep_corr(u_z, k_ones, k_ramp)
    sum_uy = sep_corr(u_z, k_ramp, k_ones)
    sum_vx = sep_corr(v_z, k_ones, k_ramp)
    sum_vy = sep_corr(v_z, k_ramp, k_ones)

    safe_N = np.maximum(N, 1.0)

    Sxx = sum_x2 - sum_x**2 / safe_N
    Syy = sum_y2 - sum_y**2 / safe_N
    Sxy = sum_xy - sum_x * sum_y / safe_N

    Sux = sum_ux - sum_u * sum_x / safe_N
    Suy = sum_uy - sum_u * sum_y / safe_N
    Svx = sum_vx - sum_v * sum_x / safe_N
    Svy = sum_vy - sum_v * sum_y / safe_N

    det     = Sxx * Syy - Sxy**2
    min_pts = 6
    enough  = (N >= min_pts) & (det > 1e-12)
    safe_d  = np.where(det > 1e-12, det, 1.0)

    dVx_dx = np.where(enough, (Sux * Syy - Suy * Sxy) / safe_d, np.nan)
    dVx_dy = np.where(enough, (Suy * Sxx - Sux * Sxy) / safe_d, np.nan)
    dVy_dx = np.where(enough, (Svx * Syy - Svy * Sxy) / safe_d, np.nan)
    dVy_dy = np.where(enough, (Svy * Sxx - Svx * Sxy) / safe_d, np.nan)

    dVx_dx[~valid_mask] = np.nan
    dVx_dy[~valid_mask] = np.nan
    dVy_dx[~valid_mask] = np.nan
    dVy_dy[~valid_mask] = np.nan

    # Rate of Deformation Tensor D = 0.5 * (L + L^T)
    Exx_rate  = dVx_dx
    Eyy_rate  = dVy_dy
    Exy_rate  = 0.5 * (dVx_dy + dVy_dx)

    with np.errstate(invalid='ignore'):
        Eeff_rate = np.sqrt(np.maximum(
            (2.0/3.0)*(Exx_rate**2 + Eyy_rate**2 + 2.0*Exy_rate**2 - Exx_rate*Eyy_rate), 0.0))

    return dict(Exx_rate=Exx_rate, Exy_rate=Exy_rate, Eyy_rate=Eyy_rate, Eeff_rate=Eeff_rate,
                dVx_dx=dVx_dx, dVx_dy=dVx_dy, dVy_dx=dVy_dx, dVy_dy=dVy_dy)
