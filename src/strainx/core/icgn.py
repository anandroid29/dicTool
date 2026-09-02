"""
icgn.py
-------
Inverse Compositional Gauss-Newton (IC-GN) optimizer for DIC.
Optimized for speed by eliminating matrix allocations in the inner loop.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from .bspline import BSplineInterpolator

# A subset must retain at least this fraction of its nominal pixels (after
# image-bound and ROI clipping) to be considered trustworthy.
MIN_SUPPORT_FRACTION = 0.30



# ─────────────────────────────────────────────────────────────────────────────
# Second-order (quadratic) shape functions
# ─────────────────────────────────────────────────────────────────────────────
# Parameter layout keeps the first six entries identical to the first-order
# vector, so everything downstream (p[0]=u, p[1]=v, p[2:6]=displacement
# gradients) is unchanged:
#
#   p = [u, v, ux, uy, vx, vy, uxx, uxy, uyy, vxx, vxy, vyy]
#
# Quadratic warps are not closed under composition, so unlike the affine case
# there is no exact matrix inverse. The standard treatment (Gao et al. 2015)
# represents the warp as a 6x6 operator on the monomial vector
# [1, dx, dy, dx^2, dx*dy, dy^2] and truncates products above degree two;
# composition and inversion then reduce to matrix operations exactly as in the
# 3x3 affine case.

def _warp_matrix_2(p: np.ndarray) -> np.ndarray:
    u, v, ux, uy, vx, vy, uxx, uxy, uyy, vxx, vxy, vyy = p
    S1, S2 = 1.0 + ux, 1.0 + vy
    W = np.zeros((6, 6), dtype=np.float64)
    W[0] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    W[1] = (u, S1, uy, 0.5 * uxx, uxy, 0.5 * uyy)
    W[2] = (v, vx, S2, 0.5 * vxx, vxy, 0.5 * vyy)
    W[3] = (u * u, 2.0 * u * S1, 2.0 * u * uy,
            S1 * S1 + u * uxx, 2.0 * (S1 * uy + u * uxy), uy * uy + u * uyy)
    W[4] = (u * v, v * S1 + u * vx, v * uy + u * S2,
            S1 * vx + 0.5 * (u * vxx + v * uxx),
            S1 * S2 + uy * vx + u * vxy + v * uxy,
            uy * S2 + 0.5 * (u * vyy + v * uyy))
    W[5] = (v * v, 2.0 * v * vx, 2.0 * v * S2,
            vx * vx + v * vxx, 2.0 * (vx * S2 + v * vxy), S2 * S2 + v * vyy)
    return W


def _params_from_warp_2(W: np.ndarray) -> np.ndarray:
    return np.array([
        W[1, 0], W[2, 0],
        W[1, 1] - 1.0, W[1, 2], W[2, 1], W[2, 2] - 1.0,
        2.0 * W[1, 3], W[1, 4], 2.0 * W[1, 5],
        2.0 * W[2, 3], W[2, 4], 2.0 * W[2, 5],
    ], dtype=np.float64)


def warp_points(p: np.ndarray, dx: np.ndarray, dy: np.ndarray):
    """Offsets of the warped subset points for a 6- or 12-parameter vector."""
    if len(p) >= 12:
        return (p[0] + (1.0 + p[2]) * dx + p[3] * dy
                + 0.5 * p[6] * dx * dx + p[7] * dx * dy + 0.5 * p[8] * dy * dy,
                p[1] + p[4] * dx + (1.0 + p[5]) * dy
                + 0.5 * p[9] * dx * dx + p[10] * dx * dy + 0.5 * p[11] * dy * dy)
    return (p[0] + (1.0 + p[2]) * dx + p[3] * dy,
            p[1] + p[4] * dx + (1.0 + p[5]) * dy)


def compose_inverse_2(p: np.ndarray, dp: np.ndarray) -> Optional[np.ndarray]:
    """p <- p o dp^-1 for second-order warps."""
    try:
        Wi = np.linalg.inv(_warp_matrix_2(dp))
    except np.linalg.LinAlgError:
        return None
    out = _params_from_warp_2(_warp_matrix_2(p) @ Wi)
    return out if np.all(np.isfinite(out)) else None


# A subset containing a large near-uniform region (occlusion silhouette,
# saturated glare, tool shadow) can correlate deceptively well against ANY
# other large uniform region -- a big flat patch matches a big flat patch
# regardless of position, because the ZNSSD cost is dominated by the many
# near-identical pixels and barely sees the few that carry real texture.
# This is what turns a subset straddling an occlusion boundary into a
# plausible-looking wrong answer instead of a clean, honest failure.
#
# These MUST be expressed relative to the image's intensity range, not as
# absolute 8-bit numbers. Images enter this module normalised to [0, 1] (see
# analysis._load_image), so the previous absolute cutoffs of 12 / 250 marked
# EVERY pixel of EVERY subset as "near-black" -- occluded_frac was 1.0
# everywhere, every subset was rejected before correlation, and the whole CPU
# solver silently returned an empty field instead of raising.
OCCLUSION_FRACTION_LOW = 12.0 / 255.0    # near-black: silhouette / shadow
OCCLUSION_FRACTION_HIGH = 250.0 / 255.0  # near-saturated: glare / overexposure
MAX_OCCLUDED_FRACTION = 0.12             # reject subset if more of it is occluded

# A small but ONE-SIDED occluded patch biases the affine fit (it pulls the
# gradient-weighted centroid toward the textured side) well before the total
# fraction crosses MAX_OCCLUDED_FRACTION. Checking the occluded pixels'
# centroid offset from the subset center catches an advancing boundary a few
# frames earlier than a bare area fraction would.
MAX_OCCLUDED_ASYMMETRY = 0.35      # centroid offset / subset radius

# Default assumes 8-bit-valued data so that direct callers passing 0-255 arrays
# keep working; run_rg_dic infers the real scale and passes it explicitly.
DEFAULT_INTENSITY_SCALE = 255.0


def infer_intensity_scale(*images: np.ndarray) -> float:
    """Container range of the image data, snapped to the usual quantisations.

    Snapping (rather than using the raw max) keeps the occlusion cutoffs
    meaning the same physical thing on a dark frame as on a bright one.
    """
    m = 0.0
    for img in images:
        if img is None or img.size == 0:
            continue
        mx = float(np.nanmax(img))
        if np.isfinite(mx):
            m = max(m, mx)
    if m <= 1.5:
        return 1.0
    if m <= 255.0:
        return 255.0
    if m <= 65535.0:
        return 65535.0
    return m


def _occluded(vals: np.ndarray, scale: float) -> np.ndarray:
    return ((vals <= OCCLUSION_FRACTION_LOW * scale) |
            (vals >= OCCLUSION_FRACTION_HIGH * scale))


def _occlusion_asymmetry(vals: np.ndarray, dxf: np.ndarray, dyf: np.ndarray,
                         r_eff: float, scale: float) -> float:
    occ = _occluded(vals, scale)
    n = int(occ.sum())
    if n == 0:
        return 0.0
    cx, cy = float(dxf[occ].mean()), float(dyf[occ].mean())
    return float(np.hypot(cx, cy) / max(r_eff, 1.0)) * (n / len(vals)) * 4.0


class SubsetData:
    __slots__ = (
        "center_x", "center_y", "dx", "dy",
        "f_norm", "sigma_f", "sd", "H", "L_fac", "valid", "r_eff", "order", "cond",
        "occluded_frac",
    )

    def __init__(
        self, center_x: int, center_y: int,
        dx: np.ndarray, dy: np.ndarray,
        f_norm: np.ndarray, sigma_f: float,
        sd: np.ndarray, H: np.ndarray, L_fac, order: int = 1, cond: float = np.inf,
        occluded_frac: float = 0.0,
    ) -> None:
        self.order = order
        self.cond = cond
        self.occluded_frac = occluded_frac
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
        # Characteristic subset radius, used to put the affine terms of the
        # convergence test into pixel units.
        self.r_eff = float(np.sqrt((dx.astype(np.float64) ** 2 +
                                    dy.astype(np.float64) ** 2).max())) if dx.size else 1.0
        if self.r_eff < 1.0:
            self.r_eff = 1.0


# Second order trades bias for variance: it removes the systematic error caused
# by curvature inside the subset, but the six extra parameters amplify noise in
# the displacement estimate. Whether that trade pays off depends entirely on the
# pattern, so decide per subset rather than globally.
#
# For a ZNSSD-normalised subset the parameter covariance goes as sigma_n^2 * H^-1,
# so (H^-1)[0,0] is proportional to the variance of u. Comparing that quantity
# between the 12- and 6-parameter fits gives the noise penalty directly, which is
# a far better test than the condition number alone (which barely moves on weak,
# anisotropic patterns where second order is in fact ruinous).
# NOTE on automatic order selection: neither the Hessian condition number nor
# the variance-inflation ratio (H2^-1)[0,0]/(H1^-1)[0,0] discriminates usefully.
# The inflation ratio sits at ~4.0 for dense speckle and for weak anisotropic
# texture alike -- it is a structural consequence of adding six parameters, not a
# measure of pattern quality. So the guard below only rejects genuinely
# degenerate subsets; the bias/variance judgement is left to the user, informed
# by shape_order_report() which works on the real images.
MAX_COND_2ND = 1.0e9


def precompute_subset(
    ref_image: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray,
    center_x: int, center_y: int, dx: np.ndarray, dy: np.ndarray,
    roi_mask: np.ndarray | None = None, order: int = 1,
    intensity_scale: float = DEFAULT_INTENSITY_SCALE,
) -> SubsetData:
    H_im, W_im = ref_image.shape
    xs = center_x + dx
    ys = center_y + dy
    n_nominal = len(dx)

    valid_px = (xs >= 0) & (xs < W_im) & (ys >= 0) & (ys < H_im)
    xs, ys, dx_, dy_ = xs[valid_px], ys[valid_px], dx[valid_px], dy[valid_px]

    # Restrict the subset to pixels inside the ROI. Without this, subsets whose
    # centre is near the ROI border pull in out-of-ROI content (background,
    # grips, a different material) and fail correlation for no good reason,
    # eroding a band of width ~subset_radius inward from every ROI edge.
    if roi_mask is not None and len(xs):
        in_roi = roi_mask[ys, xs]
        xs, ys, dx_, dy_ = xs[in_roi], ys[in_roi], dx_[in_roi], dy_[in_roi]

    n_px = len(xs)
    if n_px < 6 or n_px < MIN_SUPPORT_FRACTION * n_nominal:
        return SubsetData(center_x, center_y, dx_, dy_, np.zeros(n_px), 0.0,
                          np.zeros((n_px, 6)), np.zeros((6, 6)), None)

    f = ref_image[ys, xs]

    # Occlusion check on the REFERENCE subset. A subset that already straddles
    # a silhouette/glare boundary in the reference frame is the case that
    # produces a confidently-wrong match, so it is rejected before any
    # correlation is attempted rather than caught after the fact.
    occ_frac = float(np.mean(_occluded(f, intensity_scale)))
    r_nom = float(np.sqrt((dx_.astype(np.float64) ** 2 +
                          dy_.astype(np.float64) ** 2).max())) if n_px else 1.0
    occ_asym = _occlusion_asymmetry(f, dx_.astype(np.float64), dy_.astype(np.float64),
                                    r_nom, intensity_scale)
    if occ_frac > MAX_OCCLUDED_FRACTION or occ_asym > MAX_OCCLUDED_ASYMMETRY:
        return SubsetData(center_x, center_y, dx_, dy_, np.zeros(n_px), 0.0,
                          np.zeros((n_px, 6)), np.zeros((6, 6)), None,
                          occluded_frac=occ_frac)

    f_c = f - f.mean()
    sigma_f = float(np.sqrt((f_c ** 2).sum()))

    if sigma_f < 1e-12:
        return SubsetData(center_x, center_y, dx_, dy_, np.zeros(n_px), sigma_f,
                          np.zeros((n_px, 6)), np.zeros((6, 6)), None,
                          occluded_frac=occ_frac)

    f_norm = f_c / sigma_f
    gx, gy = grad_x[ys, xs], grad_y[ys, xs]
    dx_f, dy_f = dx_.astype(np.float64), dy_.astype(np.float64)

    cols = [gx, gy, gx * dx_f, gx * dy_f, gy * dx_f, gy * dy_f]
    if order >= 2:
        xx, xy, yy = dx_f * dx_f, dx_f * dy_f, dy_f * dy_f
        cols += [0.5 * gx * xx, gx * xy, 0.5 * gx * yy,
                 0.5 * gy * xx, gy * xy, 0.5 * gy * yy]
    SD = np.column_stack(cols)
    # Mean-correction projection for ZNSSD (Pan et al. 2009):
    # sd[i,k] = (1/sigma_f) * (SD[i,k] - f_norm[i] * sum_j f_norm[j]*SD[j,k])
    correction = f_norm @ SD  # shape (6,)
    sd = (SD - np.outer(f_norm, correction)) / sigma_f
    H_mat = sd.T @ sd

    try:
        cond = float(np.linalg.cond(H_mat))
    except Exception:
        cond = np.inf

    # Retry at first order rather than returning a noise-dominated quadratic fit.
    if order >= 2:
        if (not np.isfinite(cond)) or cond > MAX_COND_2ND:
            return precompute_subset(ref_image, grad_x, grad_y, center_x,
                                     center_y, dx, dy, roi_mask, order=1,
                                     intensity_scale=intensity_scale)

    try:
        L_fac = cho_factor(H_mat, lower=True)
    except LinAlgError:
        L_fac = None

    return SubsetData(
        center_x, center_y, dx_, dy_,
        f_norm, sigma_f, sd, H_mat, L_fac, order, cond, occ_frac,
    )


def run_icgn(
    cur_interp: BSplineInterpolator, subset: SubsetData,
    p_init: np.ndarray, max_iter: int = 50, conv_tol: float = 1e-4,
    intensity_scale: float = DEFAULT_INTENSITY_SCALE,
) -> tuple[np.ndarray, float, bool]:

    if not subset.valid:
        return p_init.copy(), np.inf, False

    order = getattr(subset, "order", 1)
    n_par = 12 if order >= 2 else 6
    p = np.zeros(n_par, dtype=np.float64)
    p[:min(n_par, len(p_init))] = np.asarray(p_init, dtype=np.float64)[:n_par]
    cx, cy = float(subset.center_x), float(subset.center_y)
    dx, dy = subset.dx.astype(np.float64), subset.dy.astype(np.float64)
    f_norm, sd, L_fac = subset.f_norm, subset.sd, subset.L_fac
    r_eff = subset.r_eff

    H_im, W_im = cur_interp.shape

    converged = False
    # Track the best iterate seen. IC-GN can wander after a near-singular step;
    # returning the last iterate rather than the best one is how a subset that
    # was fine at iteration 3 gets reported as a failure at iteration 50.
    best_p = p.copy()
    best_CLS = np.inf

    for _it in range(max_iter):
        ox, oy = warp_points(p, dx, dy)
        x_cur, y_cur = cx + ox, cy + oy

        # The interpolator mirrors out-of-bounds coordinates, which silently
        # fabricates intensity data. Reject instead of correlating against it.
        if (x_cur.min() < -0.5 or x_cur.max() > W_im - 0.5 or
                y_cur.min() < -0.5 or y_cur.max() > H_im - 0.5):
            break

        g = cur_interp.eval(x_cur, y_cur)

        # Current-frame occlusion check. The reference subset can be clean
        # while the DEFORMED subset (where the warp currently points) has
        # moved onto an occlusion boundary -- e.g. the tool advances into
        # material the subset used to see clearly. Catching this only via the
        # final ZNSSD is too late: a big uniform occluded patch can still
        # produce a low-looking cost by matching against noise structure in
        # the small remaining textured area, especially early in the sweep
        # when the occluded fraction is still small enough to be masked by
        # otherwise-good correlation. Check directly instead of inferring it
        # from the fit quality.
        g_occ_frac = float(np.mean(_occluded(g, intensity_scale)))
        g_occ_asym = _occlusion_asymmetry(g, dx, dy, r_eff, intensity_scale)
        if g_occ_frac > MAX_OCCLUDED_FRACTION or g_occ_asym > MAX_OCCLUDED_ASYMMETRY:
            break

        g_c = g - g.mean()
        sigma_g = float(np.sqrt((g_c ** 2).sum()))

        if sigma_g < 1e-12:
            break

        residual = (g_c / sigma_g) - f_norm
        CLS = float((residual ** 2).sum())

        if CLS < best_CLS:
            best_CLS = CLS
            best_p = p.copy()

        b = sd.T @ residual

        try:
            delta_p = cho_solve(L_fac, b)
        except Exception:
            break

        if not np.all(np.isfinite(delta_p)):
            break

        if order >= 2:
            p_new = compose_inverse_2(p, delta_p)
            if p_new is None:
                break
            p = p_new
        else:
            # Analytical compositional update (zero intermediate allocations)
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

        # Convergence test in consistent units. delta_p[0:2] are pixels while
        # delta_p[2:6] are dimensionless; scaling the affine terms by the subset
        # radius converts them to the pixel motion they cause at the subset edge,
        # which is what "converged" should actually mean.
        step_sq = (delta_p[0] ** 2 + delta_p[1] ** 2 +
                   (r_eff ** 2) * (delta_p[2] ** 2 + delta_p[3] ** 2 +
                                   delta_p[4] ** 2 + delta_p[5] ** 2))
        if order >= 2:
            # Quadratic terms move the subset edge by ~r^2 * coefficient.
            step_sq += (r_eff ** 4) * float((delta_p[6:12] ** 2).sum())
        step_px = np.sqrt(step_sq)
        if step_px < conv_tol:
            converged = True
            break

    # Report the correlation of the parameters actually being returned.
    ox, oy = warp_points(best_p, dx, dy)
    x_cur, y_cur = cx + ox, cy + oy
    if (x_cur.min() < -0.5 or x_cur.max() > W_im - 0.5 or
            y_cur.min() < -0.5 or y_cur.max() > H_im - 0.5):
        return best_p, np.inf, False
    g = cur_interp.eval(x_cur, y_cur)
    g_occ_frac = float(np.mean(_occluded(g, intensity_scale)))
    g_occ_asym = _occlusion_asymmetry(g, dx, dy, r_eff, intensity_scale)
    if g_occ_frac > MAX_OCCLUDED_FRACTION or g_occ_asym > MAX_OCCLUDED_ASYMMETRY:
        return best_p, np.inf, False
    g_c = g - g.mean()
    sigma_g = float(np.sqrt((g_c ** 2).sum()))
    if sigma_g < 1e-12:
        return best_p, np.inf, False
    final_CLS = float((((g_c / sigma_g) - f_norm) ** 2).sum())

    return best_p, final_CLS, converged
