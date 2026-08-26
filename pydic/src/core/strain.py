"""
strain.py
---------
Green-Lagrangian strain computation via least-squares plane fit.
Optimized using mathematically separable 1D filters.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, correlate1d, label


def von_mises_equivalent(exx, eyy, exy):
    """von Mises equivalent of a symmetric 2-D strain (or strain-rate) tensor.

    Plastic incompressibility fixes the out-of-plane term, e_zz = -(e_xx+e_yy),
    and the equivalent measure is sqrt(2/3 * e_ij e_ij) over the restored 3-D
    tensor. Restoring e_zz is the part that is easy to drop, and dropping it is
    not a small error: it leaves equibiaxial strain reading 59 % low, because
    equibiaxial deformation is carried almost entirely by the thickness change
    the 2-D tensor cannot see.

    Reduces to the textbook results:
      uniaxial (e_yy = -e_xx/2) -> e_xx
      pure shear                -> 2*e_xy/sqrt(3)
      equibiaxial (e_xx = e_yy) -> 2*e_xx

    This is the single definition of "equivalent" in the codebase. Accumulated
    strain and strain rate both route through it so the two cannot disagree --
    they previously used different expressions, and the rate was the wrong one.
    """
    ezz = -(exx + eyy)
    contraction = exx ** 2 + eyy ** 2 + ezz ** 2 + 2.0 * exy ** 2
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.maximum((2.0 / 3.0) * contraction, 0.0))


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
    use_gpu: bool = False,
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

    valid_full = (np.asarray(valid_mask, dtype=bool) &
                  np.isfinite(Vx) & np.isfinite(Vy))

    # DIC fields are sparse and the ROI is often a small fraction of a full-HD
    # image. Running every separable correlation across the whole sensor was the
    # reason a frame-pair sequence appeared to hang. Crop to the finite support,
    # retaining enough invalid padding that constant-zero boundary conditions
    # produce exactly the same fit at every valid subset centre.
    yy, xx = np.nonzero(valid_full)
    full_shape = Vx.shape
    if not yy.size:
        names = ("Exx_rate", "Exy_rate", "Gxy_rate", "Eyy_rate", "Eeff_rate",
                 "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy")
        return {name: np.full(full_shape, np.nan, dtype=np.float64)
                for name in names}
    grow = max(0, int(grid_spacing) // 2)
    pad = r + grow + 1
    y0, y1 = max(0, int(yy.min()) - pad), min(full_shape[0], int(yy.max()) + pad + 1)
    x0, x1 = max(0, int(xx.min()) - pad), min(full_shape[1], int(xx.max()) + pad + 1)
    crop = np.s_[y0:y1, x0:x1]
    Vx = np.asarray(Vx)[crop]
    Vy = np.asarray(Vy)[crop]
    valid = valid_full[crop]

    # GPU startup and host/device transfer cost more than the filters on a
    # small crop. Only use CuPy for a genuinely large finite-support box; any
    # missing package, driver, or cupyx feature falls back to the identical CPU
    # calculation. Pair playback passes the user's existing GPU preference.
    gpu_enabled = bool(use_gpu and Vx.size >= 512 * 512)
    cp = cp_correlate1d = None
    gpu_arrays: dict[int, object] = {}
    gpu_kernels: dict[bytes, object] = {}
    if gpu_enabled:
        try:
            import cupy as cp  # type: ignore[no-redef]
            from cupyx.scipy.ndimage import correlate1d as cp_correlate1d  # type: ignore[no-redef]
            # This is the first operation that authoritatively validates the
            # CUDA runtime/device instead of merely finding the Python module.
            cp.cuda.runtime.getDeviceCount()
        except Exception:
            gpu_enabled = False
            cp = cp_correlate1d = None

    def sep_corr(arr: np.ndarray, ky: np.ndarray, kx: np.ndarray) -> np.ndarray:
        """Apply a separable 2-D correlation on the selected backend."""
        nonlocal gpu_enabled
        if gpu_enabled:
            try:
                key = id(arr)
                gpu_arr = gpu_arrays.get(key)
                if gpu_arr is None:
                    gpu_arr = cp.asarray(arr, dtype=cp.float64)
                    gpu_arrays[key] = gpu_arr

                def gpu_kernel(kernel):
                    kkey = np.asarray(kernel, dtype=np.float64).tobytes()
                    value = gpu_kernels.get(kkey)
                    if value is None:
                        value = cp.asarray(kernel, dtype=cp.float64)
                        gpu_kernels[kkey] = value
                    return value

                temp = cp_correlate1d(
                    gpu_arr, gpu_kernel(ky), axis=0,
                    mode='constant', cval=0.0)
                out = cp_correlate1d(
                    temp, gpu_kernel(kx), axis=1,
                    mode='constant', cval=0.0)
                return cp.asnumpy(out)
            except Exception as exc:
                # A low-memory device or incompatible cupyx build should make
                # this one calculation slower, never make pair strain fail.
                print(f"[Pair strain] GPU filters unavailable ({exc}); using CPU.")
                gpu_enabled = False
                gpu_arrays.clear()
        temp = correlate1d(arr, ky, axis=0, mode='constant', cval=0.0)
        return correlate1d(temp, kx, axis=1, mode='constant', cval=0.0)

    gradients = [np.full(Vx.shape, np.nan, dtype=np.float64) for _ in range(4)]
    labels, n_components = connected_support_labels(valid, grid_spacing)

    # Usually one or two components. Each pass remains separable O(N), while
    # the component restriction prevents a least-squares plane crossing a cut.
    for component_id in range(1, n_components + 1):
        gpu_arrays.clear()
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

    Eeff_rate = von_mises_equivalent(Exx_rate, Eyy_rate, Exy_rate)

    result = dict(Exx_rate=Exx_rate, Exy_rate=Exy_rate, Gxy_rate=Gxy_rate,
                  Eyy_rate=Eyy_rate, Eeff_rate=Eeff_rate,
                  dVx_dx=dVx_dx, dVx_dy=dVx_dy,
                  dVy_dx=dVy_dx, dVy_dy=dVy_dy)
    for values in result.values():
        values[~np.isfinite(values)] = np.nan

    # Restore the public full-image shape. Several result names intentionally
    # alias the same derivative array; preserve those aliases while expanding.
    expanded: dict[str, np.ndarray] = {}
    by_id: dict[int, np.ndarray] = {}
    for name, values in result.items():
        shared = by_id.get(id(values))
        if shared is None:
            shared = np.full(full_shape, np.nan, dtype=np.float64)
            shared[crop] = values
            by_id[id(values)] = shared
        expanded[name] = shared
    return expanded
