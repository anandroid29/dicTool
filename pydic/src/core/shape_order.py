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
from scipy.linalg import cho_solve

from .bspline import BSplineInterpolator
from .icgn import infer_intensity_scale, precompute_subset


def estimate_noise_sigma(img: np.ndarray, roi_mask=None) -> float:
    """Estimate image noise with Immerkaer's 3x3 Laplacian method.

    The published estimator uses the *mean* absolute filter response.  A median
    response (used previously) becomes exactly zero on quantised images whenever
    more than half of the local 3x3 patches are flat, even though the textured
    part of the ROI still contains measurable noise.
    """
    k = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])
    r = convolve(img.astype(np.float64), k, mode="reflect")
    response = np.abs(r[1:-1, 1:-1])
    if roi_mask is not None:
        mask = np.asarray(roi_mask, dtype=bool)
        if mask.shape != img.shape:
            raise ValueError("ROI mask shape does not match the reference image.")
        # A filter response is usable only when its complete 3x3 footprint is
        # inside the ROI; otherwise the ROI boundary is mistaken for noise.
        support = convolve(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                           mode="constant", cval=0)[1:-1, 1:-1] == 9
        response = response[support]
    if response.size == 0:
        raise ValueError("ROI leaves no complete 3x3 patches for noise estimation.")
    return float(np.sqrt(np.pi / 2.0) * response.mean() / 6.0)


def predicted_sigma_u(ref: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                      centers, dxs, dys, order: int, noise_sigma: float,
                      roi_mask=None, intensity_scale: float | None = None,
                      return_count: bool = False):
    """Mean predicted standard deviation of u, in pixels.

    ``precompute_subset`` needs the image's actual intensity scale for its
    saturation/occlusion checks.  Reference images loaded by the application
    are normalised to [0, 1]; relying on its legacy 8-bit default therefore
    rejected every sampled subset and made this diagnostic return NaN.
    """
    if intensity_scale is None:
        intensity_scale = infer_intensity_scale(ref)
    vals = []
    for (cx, cy) in centers:
        sd = precompute_subset(ref, gx, gy, int(cx), int(cy), dxs, dys,
                               roi_mask, order=order,
                               intensity_scale=intensity_scale)
        if not sd.valid or sd.order != order:
            continue
        try:
            # Solve only the covariance column we need, using the same
            # Cholesky factor already validated by the optimizer.  Forming a
            # full inverse is both less stable and needlessly expensive.
            e0 = np.zeros(sd.H.shape[0], dtype=np.float64)
            e0[0] = 1.0
            cov00 = float(cho_solve(sd.L_fac, e0)[0])
        except Exception:
            continue
        if cov00 <= 0 or not np.isfinite(cov00):
            continue
        # f_norm is unit-norm, so image noise enters normalised by sigma_f.
        vals.append((noise_sigma / max(sd.sigma_f, 1e-12)) * np.sqrt(cov00))
    value = float(np.mean(vals)) if vals else float("nan")
    return (value, len(vals)) if return_count else value


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

    intensity_scale = infer_intensity_scale(ref)
    noise_native = estimate_noise_sigma(ref, roi_mask)
    grad_native = float(np.hypot(gx, gy)[roi_mask].mean())
    s1, n1 = predicted_sigma_u(
        ref, gx, gy, centers, dxs, dys, 1, noise_native, roi_mask,
        intensity_scale=intensity_scale, return_count=True)
    s2, n2 = predicted_sigma_u(
        ref, gx, gy, centers, dxs, dys, 2, noise_native, roi_mask,
        intensity_scale=intensity_scale, return_count=True)

    if n1 == 0 or n2 == 0:
        raise ValueError(
            "The selected ROI has too little usable texture for this estimate "
            f"({n1} first-order and {n2} second-order valid samples out of "
            f"{len(centers)}). Try a larger textured ROI or a smaller subset."
        )

    # The application stores images normalised to [0, 1].  Convert display
    # values back to familiar 8-bit grey levels; the displacement uncertainty
    # above remains scale-invariant because noise and image gradients use the
    # same native scale.
    to_grey_levels = 255.0 / intensity_scale
    noise = noise_native * to_grey_levels
    grad = grad_native * to_grey_levels

    out = {"noise_sigma": noise, "mean_gradient": grad,
           "sigma_u_order1": s1, "sigma_u_order2": s2,
           "penalty_px": s2 - s1,
           "radius": radius, "sample_count": len(centers),
           "valid_samples_order1": n1, "valid_samples_order2": n2}

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
