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


def iter_velocity_strains_multi_window(
    Vx: np.ndarray,
    Vy: np.ndarray,
    valid_mask: np.ndarray,
    strain_windows,
    grid_spacing: int = 1,
    use_gpu: bool = False,
):
    """Yield exact plane-fit gradients for many windows from shared integrals.

    The single-window implementation performs twelve separable image scans for
    every radius. A parameter sweep needs the same sufficient statistics at up
    to fifteen radii, so integral images reduce each additional radius to box
    lookups while preserving the normal-equation fit exactly.
    """
    Vx, Vy = np.asarray(Vx), np.asarray(Vy)
    valid = (np.asarray(valid_mask, dtype=bool) &
             np.isfinite(Vx) & np.isfinite(Vy))
    windows = tuple(sorted({max(0, int(value)) for value in strain_windows}))
    if not windows:
        return
    if use_gpu:
        for radius in windows:
            yield radius, compute_velocity_strains(
                Vx, Vy, valid, radius, grid_spacing, use_gpu=True)
        return
    shape = Vx.shape

    # Correlation values occupy one regular subset lattice inside a full-image
    # array.  Regressing on that lattice is mathematically identical because we
    # retain physical pixel coordinates, and avoids scanning the invalid pixels
    # between subset centres (25x fewer cells for grid spacing 5).
    spacing = max(1, int(grid_spacing))
    rows, columns = np.nonzero(valid)
    if rows.size:
        row_offset = int(rows[0] % spacing)
        column_offset = int(columns[0] % spacing)
        aligned = (np.all(rows % spacing == row_offset)
                   and np.all(columns % spacing == column_offset))
    else:
        row_offset = column_offset = 0
        aligned = True
    if not aligned:
        spacing = 1
        row_offset = column_offset = 0

    row_slice = slice(row_offset, None, spacing)
    column_slice = slice(column_offset, None, spacing)
    lattice_slice = (row_slice, column_slice)
    Vx_lattice = Vx[lattice_slice]
    Vy_lattice = Vy[lattice_slice]
    valid_lattice = valid[lattice_slice]
    lattice_shape = valid_lattice.shape
    labels, count = connected_support_labels(valid_lattice, 1)
    yy = (row_offset + np.arange(lattice_shape[0], dtype=np.float64)
          * spacing)[:, None]
    xx = (column_offset + np.arange(lattice_shape[1], dtype=np.float64)
          * spacing)[None, :]
    lattice_radii = {window: window // spacing for window in windows}
    pad = max(lattice_radii.values())

    def integral(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, pad, mode="constant")
        summed = padded.cumsum(axis=0, dtype=np.float64).cumsum(
            axis=1, dtype=np.float64)
        return np.pad(summed, ((1, 0), (1, 0)), mode="constant")

    def box(summed: np.ndarray, radius: int) -> np.ndarray:
        top, bottom = pad - radius, pad + radius + 1
        left, right = top, bottom
        h, w = lattice_shape
        return (summed[bottom:bottom + h, right:right + w]
                - summed[top:top + h, right:right + w]
                - summed[bottom:bottom + h, left:left + w]
                + summed[top:top + h, left:left + w])

    components = []
    for component_id in range(1, count + 1):
        component = valid_lattice & (labels == component_id)
        if np.count_nonzero(component) < 6:
            continue
        mask = component.astype(np.float64)
        u = np.where(component, Vx_lattice, 0.0).astype(
            np.float64, copy=False)
        v = np.where(component, Vy_lattice, 0.0).astype(
            np.float64, copy=False)
        bases = (
            mask, mask * xx, mask * yy, mask * xx * xx,
            mask * yy * yy, mask * xx * yy,
            u, v, u * xx, u * yy, v * xx, v * yy,
        )
        components.append((component, tuple(integral(values)
                                             for values in bases)))

    cached = {}
    for radius in windows:
        lattice_radius = lattice_radii[radius]
        if lattice_radius in cached:
            yield radius, cached[lattice_radius]
            continue
        gradients = [np.full(lattice_shape, np.nan, dtype=np.float64)
                     for _ in range(4)]
        for component, summed in components:
            (N, abs_x, abs_y, abs_x2, abs_y2, abs_xy,
             sum_u, sum_v, abs_ux, abs_uy, abs_vx, abs_vy) = (
                box(values, lattice_radius) for values in summed)
            sum_x = abs_x - xx * N
            sum_y = abs_y - yy * N
            sum_x2 = abs_x2 - 2.0 * xx * abs_x + xx * xx * N
            sum_y2 = abs_y2 - 2.0 * yy * abs_y + yy * yy * N
            sum_xy = (abs_xy - xx * abs_y - yy * abs_x
                      + xx * yy * N)
            sum_ux = abs_ux - xx * sum_u
            sum_uy = abs_uy - yy * sum_u
            sum_vx = abs_vx - xx * sum_v
            sum_vy = abs_vy - yy * sum_v
            safe_n = np.maximum(N, 1.0)
            Sxx = sum_x2 - sum_x ** 2 / safe_n
            Syy = sum_y2 - sum_y ** 2 / safe_n
            Sxy = sum_xy - sum_x * sum_y / safe_n
            Sux = sum_ux - sum_u * sum_x / safe_n
            Suy = sum_uy - sum_u * sum_y / safe_n
            Svx = sum_vx - sum_v * sum_x / safe_n
            Svy = sum_vy - sum_v * sum_y / safe_n
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                det = Sxx * Syy - Sxy ** 2
                enough = (component & (N >= 6) & np.isfinite(det)
                          & (det > 1e-12))
                safe_det = np.where(enough, det, 1.0)
                fitted = (
                    (Sux * Syy - Suy * Sxy) / safe_det,
                    (Suy * Sxx - Sux * Sxy) / safe_det,
                    (Svx * Syy - Svy * Sxy) / safe_det,
                    (Svy * Sxx - Svx * Sxy) / safe_det,
                )
            for output, values in zip(gradients, fitted):
                keep = enough & np.isfinite(values)
                output[keep] = values[keep]
        expanded = []
        for values in gradients:
            full = np.full(shape, np.nan, dtype=np.float64)
            full[lattice_slice] = values
            expanded.append(full)
        dVx_dx, dVx_dy, dVy_dx, dVy_dy = expanded
        Exx_rate, Eyy_rate = dVx_dx, dVy_dy
        Exy_rate = 0.5 * (dVx_dy + dVy_dx)
        result = {
            "dVx_dx": dVx_dx, "dVx_dy": dVx_dy,
            "dVy_dx": dVy_dx, "dVy_dy": dVy_dy,
            "Eeff_rate": von_mises_equivalent(
                Exx_rate, Eyy_rate, Exy_rate),
        }
        cached[lattice_radius] = result
        yield radius, result

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

    gpu_enabled = bool(use_gpu)
    native_plane_fit = None
    if gpu_enabled:
        try:
            from .cuda_native import native_cuda_available, native_plane_fit as _native_plane_fit
            gpu_enabled = native_cuda_available()
            native_plane_fit = _native_plane_fit if gpu_enabled else None
        except Exception:
            gpu_enabled = False

    def sep_corr(arr: np.ndarray, ky: np.ndarray, kx: np.ndarray) -> np.ndarray:
        """Apply the unchanged separable SciPy CPU fallback."""
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
        if gpu_enabled and native_plane_fit is not None:
            try:
                fitted = native_plane_fit(Vx, Vy, component, r)
                for output, values in zip(gradients, fitted):
                    keep = component & np.isfinite(values)
                    output[keep] = values[keep]
                continue
            except Exception as exc:
                print(f"[Pair strain] Native CUDA plane fit unavailable ({exc}); using CPU.")
                gpu_enabled = False
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
