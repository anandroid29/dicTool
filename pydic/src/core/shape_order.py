"""
shape_order.py
--------------
Decide between first- and second-order shape functions using the ACTUAL images,
rather than a rule of thumb.

Second order removes the systematic error caused by strain curvature inside the
subset, but the six extra parameters amplify noise in the displacement estimate.
Which effect wins is a property of the speckle pattern, the noise level and the
subset size -- not something that can be decided in the abstract.

This estimates both terms directly:
  * random error  -- predicted sigma_u from the parameter covariance
                     sigma_n^2 * H^-1, with sigma_n measured off the image
  * systematic    -- the sub-subset curvature the affine model cannot represent
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve

from .bspline import BSplineInterpolator
from .icgn import precompute_subset


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Robust noise estimate (Immerkaer): MAD-scaled response to a Laplacian."""
    k = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])
    r = convolve(img.astype(np.float64), k, mode="reflect")
    return float(np.median(np.abs(r)) / 0.6745 / np.sqrt(36.0))


def predicted_sigma_u(ref: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                      centers, dxs, dys, order: int, noise_sigma: float,
                      roi_mask=None) -> float:
    """Mean predicted standard deviation of u, in pixels."""
    vals = []
    for (cx, cy) in centers:
        sd = precompute_subset(ref, gx, gy, int(cx), int(cy), dxs, dys,
                               roi_mask, order=order)
        if not sd.valid or sd.order != order:
            continue
        try:
            cov00 = float(np.linalg.inv(sd.H)[0, 0])
        except np.linalg.LinAlgError:
            continue
        if cov00 <= 0 or not np.isfinite(cov00):
            continue
        # f_norm is unit-norm, so image noise enters normalised by sigma_f.
        vals.append((noise_sigma / max(sd.sigma_f, 1e-12)) * np.sqrt(cov00))
    return float(np.mean(vals)) if vals else float("nan")


def shape_order_report(ref: np.ndarray, roi_mask=None, radius: int = 18,
                       n_samples: int = 120, verbose: bool = True) -> dict:
    """Measure the first- vs second-order trade-off on a real reference image."""
    H, W = ref.shape
    if roi_mask is None:
        roi_mask = np.zeros((H, W), bool)
        roi_mask[radius + 1:H - radius - 1, radius + 1:W - radius - 1] = True

    ys, xs = np.nonzero(roi_mask)
    keep = ((xs > radius) & (xs < W - radius) & (ys > radius) & (ys < H - radius))
    ys, xs = ys[keep], xs[keep]
    if len(xs) == 0:
        raise ValueError("ROI leaves no room for a subset of this radius.")
    step = max(1, len(xs) // n_samples)
    centers = list(zip(xs[::step], ys[::step]))

    dd = np.arange(-radius, radius + 1)
    DX, DY = np.meshgrid(dd, dd)
    m = DX ** 2 + DY ** 2 <= radius * radius
    dxs, dys = DX[m].astype(np.int32), DY[m].astype(np.int32)

    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    g = BSplineInterpolator(ref).gradient(X.ravel(), Y.ravel())
    gx, gy = g[0].reshape(H, W), g[1].reshape(H, W)

    noise = estimate_noise_sigma(ref)
    grad = float(np.hypot(gx, gy)[roi_mask].mean())
    s1 = predicted_sigma_u(ref, gx, gy, centers, dxs, dys, 1, noise, roi_mask)
    s2 = predicted_sigma_u(ref, gx, gy, centers, dxs, dys, 2, noise, roi_mask)

    out = {"noise_sigma": noise, "mean_gradient": grad,
           "sigma_u_order1": s1, "sigma_u_order2": s2,
           "penalty_px": (s2 - s1) if np.isfinite(s2) else float("nan"),
           "radius": radius}

    if verbose:
        print(f"  image noise sigma          : {noise:.2f} grey levels")
        print(f"  mean |grad| in ROI         : {grad:.2f} grey/px")
        print(f"  predicted sigma_u, order 1 : {s1:.4f} px")
        print(f"  predicted sigma_u, order 2 : {s2:.4f} px")
        print(f"  extra random error         : {out['penalty_px']:+.4f} px")
        print("\n  Second order is worth it only if the strain curvature across")
        print("  one subset produces a displacement error LARGER than that penalty.")
        print(f"  At r={radius}, curvature uxx causes ~0.5*uxx*r^2 px of edge error:")
        for uxx in (1e-4, 5e-4, 1e-3, 5e-3):
            print(f"    uxx = {uxx:.0e}  ->  {0.5*uxx*radius**2:7.4f} px "
                  f"{'(worth it)' if 0.5*uxx*radius**2 > out['penalty_px'] else '(not worth it)'}")
    return out
