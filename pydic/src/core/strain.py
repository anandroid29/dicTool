"""
strain.py
---------
Green-Lagrangian strain computation via least-squares plane fit.
Optimized using mathematically separable 1D filters.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, correlate1d, label


def connected_support_labels(valid: np.ndarray, grid_spacing: int = 1):
    """Label material regions without treating sparse grid gaps as physical cuts.

    DIC values only exist at subset centres. Dilating by half a grid step joins
    neighbouring centres, while a missing row/column of centres remains a gap.
    Gradient and recovery neighbourhoods can then be restricted to one material
    component instead of bridging a cut or invalid background.
    """
    support = np.asarray(valid, dtype=bool)
    grow = max(0, int(grid_spacing) // 2)
    if grow:
        support = binary_dilation(
            support, structure=np.ones((3, 3), dtype=bool), iterations=grow)
    return label(support, structure=np.ones((3, 3), dtype=np.uint8))

def compute_velocity_strains(
    Vx: np.ndarray,
    Vy: np.ndarray,
    valid_mask: np.ndarray,
    strain_window: int,
    grid_spacing: int = 1,
) -> dict[str, np.ndarray]:
    """Fit velocity gradients using finite points from one material region.

    Invalid values never enter a fit, and a neighbourhood never borrows points
    from a disconnected component on the other side of a cut or dropout.
    """
    r = int(strain_window)

    # 1D Separable Kernels
    k_ones  = np.ones(2 * r + 1, dtype=np.float64)
    k_ramp  = np.arange(-r, r + 1, dtype=np.float64)
    k_ramp2 = k_ramp ** 2

    valid = (np.asarray(valid_mask, dtype=bool) &
             np.isfinite(Vx) & np.isfinite(Vy))

    def sep_corr(arr: np.ndarray, ky: np.ndarray, kx: np.ndarray) -> np.ndarray:
        """Applies a 2D correlation by separating it into two O(N) 1D correlations."""
        temp = correlate1d(arr, ky, axis=0, mode='constant', cval=0.0)
        return correlate1d(temp, kx, axis=1, mode='constant', cval=0.0)

    gradients = [np.full(Vx.shape, np.nan, dtype=np.float64) for _ in range(4)]
    labels, n_components = connected_support_labels(valid, grid_spacing)

    # Usually one or two components. Each pass remains separable O(N), while
    # the component restriction prevents a least-squares plane crossing a cut.
    for component_id in range(1, n_components + 1):
        component = valid & (labels == component_id)
        if np.count_nonzero(component) < 6:
            continue
        cnt = component.astype(np.float64)
        u_z = np.where(component, Vx, 0.0).astype(np.float64, copy=False)
        v_z = np.where(component, Vy, 0.0).astype(np.float64, copy=False)

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

        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            det = Sxx * Syy - Sxy**2
            enough = component & (N >= 6) & np.isfinite(det) & (det > 1e-12)
            safe_d = np.where(enough, det, 1.0)
            fitted = (
                (Sux * Syy - Suy * Sxy) / safe_d,
                (Suy * Sxx - Sux * Sxy) / safe_d,
                (Svx * Syy - Svy * Sxy) / safe_d,
                (Svy * Sxx - Svx * Sxy) / safe_d,
            )
        for output, values in zip(gradients, fitted):
            keep = enough & np.isfinite(values)
            output[keep] = values[keep]

    dVx_dx, dVx_dy, dVy_dx, dVy_dy = gradients

    # Rate of Deformation Tensor D = 0.5 * (L + L^T)
    Exx_rate  = dVx_dx
    Eyy_rate  = dVy_dy
    Exy_rate  = 0.5 * (dVx_dy + dVy_dx)
    Gxy_rate  = 2.0 * Exy_rate

    with np.errstate(invalid='ignore'):
        Eeff_rate = np.sqrt(np.maximum(
            (2.0/3.0)*(Exx_rate**2 + Eyy_rate**2 + 2.0*Exy_rate**2 - Exx_rate*Eyy_rate), 0.0))

    result = dict(Exx_rate=Exx_rate, Exy_rate=Exy_rate, Gxy_rate=Gxy_rate,
                  Eyy_rate=Eyy_rate, Eeff_rate=Eeff_rate,
                  dVx_dx=dVx_dx, dVx_dy=dVx_dy,
                  dVy_dx=dVy_dx, dVy_dy=dVy_dy)
    for values in result.values():
        values[~np.isfinite(values)] = np.nan
    return result
