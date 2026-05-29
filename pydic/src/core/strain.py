"""
strain.py
---------
Green-Lagrangian strain computation via least-squares plane fit.

Matches Ncorr (Blaber et al. 2015, eqs. 13-18) exactly.

Critical bug fixed in this version
------------------------------------
scipy.ndimage.convolve performs TRUE convolution which FLIPS the kernel
before applying it.  For an antisymmetric kernel such as

    x_kern[i,j] = j - r   (Δx offsets)

the flipped kernel equals -x_kern, so

    convolve(u, x_kern)  =  -Σ u·Δx  ← WRONG (sign-flipped gradient)

Using scipy.ndimage.correlate (cross-correlation, no kernel flip) gives

    correlate(u, x_kern)  =  +Σ u·Δx  ← CORRECT

Symmetric kernels (Δx², Δy², ΔxΔy, 1) are unaffected.
This sign error caused Exx and Eyy to have the opposite sign to Ncorr.

Full centred OLS formula is also implemented here (including the Sxy
cross-term) so that boundary pixels with asymmetric windows are handled
correctly.
"""

from __future__ import annotations
import numpy as np
# NOTE: correlate, NOT convolve — see module docstring
from scipy.ndimage import correlate


def compute_strains(
    u: np.ndarray,
    v: np.ndarray,
    valid_mask: np.ndarray,
    strain_window: int,
) -> dict[str, np.ndarray]:
    """
    Compute Green-Lagrangian strains from displacement fields.

    Parameters
    ----------
    u, v         : (H, W) float64 — displacement fields (NaN where invalid)
    valid_mask   : (H, W) bool    — True where u, v are valid
    strain_window: int             — plane-fit window half-width (pixels)

    Returns
    -------
    dict with 'Exx', 'Exy', 'Eyy', 'Eeff', 'du_dx', 'du_dy', 'dv_dx', 'dv_dy'
    """
    r = int(strain_window)

    # ── Coordinate kernels ─────────────────────────────────────────────────
    # y_kern[i,j] = (i - r) = Δy   (row offset, positive downward)
    # x_kern[i,j] = (j - r) = Δx   (col offset, positive rightward)
    y_kern, x_kern = np.mgrid[-r:r+1, -r:r+1]
    x_kern  = x_kern.astype(np.float64)
    y_kern  = y_kern.astype(np.float64)
    x2_kern = x_kern ** 2
    y2_kern = y_kern ** 2
    xy_kern = x_kern * y_kern    # ΔxΔy  (symmetric → correlate = convolve)
    ones_k  = np.ones_like(x_kern)

    valid = valid_mask & ~np.isnan(u) & ~np.isnan(v)
    u_z   = np.where(valid, u, 0.0)
    v_z   = np.where(valid, v, 0.0)
    cnt   = valid.astype(np.float64)

    kw = dict(mode='constant', cval=0.0)

    # ── Sums via cross-correlation (no kernel flip) ────────────────────────
    N      = correlate(cnt,  ones_k,  **kw)   # number of valid pts
    sum_x  = correlate(cnt,  x_kern,  **kw)   # Σ Δx
    sum_y  = correlate(cnt,  y_kern,  **kw)   # Σ Δy
    sum_x2 = correlate(cnt,  x2_kern, **kw)   # Σ Δx²
    sum_y2 = correlate(cnt,  y2_kern, **kw)   # Σ Δy²
    sum_xy = correlate(cnt,  xy_kern, **kw)   # Σ ΔxΔy

    sum_u  = correlate(u_z,  ones_k,  **kw)   # Σ u
    sum_v  = correlate(v_z,  ones_k,  **kw)   # Σ v
    sum_ux = correlate(u_z,  x_kern,  **kw)   # Σ u·Δx
    sum_uy = correlate(u_z,  y_kern,  **kw)   # Σ u·Δy
    sum_vx = correlate(v_z,  x_kern,  **kw)   # Σ v·Δx
    sum_vy = correlate(v_z,  y_kern,  **kw)   # Σ v·Δy

    # ── Full centred OLS (2×2 system after eliminating constant term) ──────
    #
    # Fitting u(Δx,Δy) = a₀ + (du/dx)·Δx + (du/dy)·Δy over the window.
    # Eliminating a₀ via the normal equations gives the 2×2 system:
    #
    #   [Sxx  Sxy] [dudx]   [Sux]
    #   [Sxy  Syy] [dudy] = [Suy]
    #
    # where Sxx = Σ Δx² − (Σ Δx)²/N, Sxy = Σ ΔxΔy − (Σ Δx)(Σ Δy)/N, etc.
    # For symmetric/full windows Sxy = Σ Δx = Σ Δy = 0, system decouples.
    # For truncated windows (ROI boundary), Sxy ≠ 0 and must be included.
    safe_N = np.maximum(N, 1.0)

    Sxx = sum_x2 - sum_x**2 / safe_N
    Syy = sum_y2 - sum_y**2 / safe_N
    Sxy = sum_xy - sum_x * sum_y / safe_N

    Sux = sum_ux - sum_u * sum_x / safe_N
    Suy = sum_uy - sum_u * sum_y / safe_N
    Svx = sum_vx - sum_v * sum_x / safe_N
    Svy = sum_vy - sum_v * sum_y / safe_N

    det    = Sxx * Syy - Sxy**2
    min_pts = 6
    enough  = (N >= min_pts) & (det > 1e-12)
    safe_d  = np.where(det > 1e-12, det, 1.0)

    du_dx = np.where(enough, (Sux * Syy - Suy * Sxy) / safe_d, np.nan)
    du_dy = np.where(enough, (Suy * Sxx - Sux * Sxy) / safe_d, np.nan)
    dv_dx = np.where(enough, (Svx * Syy - Svy * Sxy) / safe_d, np.nan)
    dv_dy = np.where(enough, (Svy * Sxx - Svx * Sxy) / safe_d, np.nan)

    # Mask outside ROI
    du_dx[~valid_mask] = np.nan
    du_dy[~valid_mask] = np.nan
    dv_dx[~valid_mask] = np.nan
    dv_dy[~valid_mask] = np.nan

    # ── Green-Lagrangian strains (Blaber et al. eqs. 13-15) ──────────────
    Exx  = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
    Eyy  = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
    Exy  = 0.5 * (du_dy + dv_dx + du_dx*du_dy + dv_dx*dv_dy)

    with np.errstate(invalid='ignore'):
        Eeff = np.sqrt(np.maximum(
            (2.0/3.0)*(Exx**2 + Eyy**2 + 2.0*Exy**2 - Exx*Eyy), 0.0))

    return dict(Exx=Exx, Exy=Exy, Eyy=Eyy, Eeff=Eeff,
                du_dx=du_dx, du_dy=du_dy, dv_dx=dv_dx, dv_dy=dv_dy)
