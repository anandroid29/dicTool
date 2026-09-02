"""Material-path strain transport for immediate-frame DIC.

Correlation is performed once for each adjacent image pair on a fresh spatial
grid. This module consumes those pairwise fields; it never runs correlation of
its own. Material states are continuously injected through an operator-drawn
origin region, advected by the measured pair displacement, and accumulated
along their paths.

Signed infinitesimal components integrate the symmetric displacement gradient.
Finite Green--Lagrange components come from composition of the incremental
deformation gradients. Equivalent accumulated strain is the non-negative path
integral of equivalent strain rate and therefore never decreases.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .strain import von_mises_equivalent


_STATE_NAMES = (
    "Exx_inf", "Eyy_inf", "Exy_inf", "Eeff_inf",
    "Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl",
)


class StrainPathTracker:
    """Continuously seed and transport accumulated strain through an ROI.

    Fields supplied to :meth:`advance` are sparse full-image arrays populated
    on the regular DIC subset grid. Particle state is kept at floating-point
    material positions, while :meth:`snapshot` remeshes it onto that same grid
    for rendering and HDF5 export.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        origin_mask: np.ndarray,
        domain_mask: np.ndarray,
        subset_radius: int,
        grid_spacing: int,
    ) -> None:
        self.shape = tuple(int(v) for v in shape)
        self.radius = max(0, int(subset_radius))
        self.spacing = max(1, int(grid_spacing))
        self.domain_mask = np.asarray(domain_mask, dtype=bool)
        self.origin_mask = np.asarray(origin_mask, dtype=bool) & self.domain_mask
        if self.domain_mask.shape != self.shape:
            raise ValueError("Strain domain mask shape does not match the images.")
        if self.origin_mask.shape != self.shape:
            raise ValueError("Strain-origin mask shape does not match the images.")

        h, w = self.shape
        self.grid_x = np.arange(self.radius, w - self.radius, self.spacing,
                                dtype=np.int32)
        self.grid_y = np.arange(self.radius, h - self.radius, self.spacing,
                                dtype=np.int32)
        if self.grid_x.size and self.grid_y.size:
            gx, gy = np.meshgrid(self.grid_x, self.grid_y)
            candidates = self.domain_mask[gy, gx]
            source = np.zeros(gx.shape, dtype=bool)
            # A wall can be thinner than the correlation spacing, or lie at an
            # image/ROI boundary where no complete subset centre can physically
            # sit. Snap each disconnected origin component to its nearest
            # measurable lattice layer instead of silently producing no strain.
            from scipy.ndimage import distance_transform_edt, label
            labels, count = label(self.origin_mask)
            for component_id in range(1, count + 1):
                component = labels == component_id
                distances = distance_transform_edt(~component)[gy, gx]
                available = distances[candidates]
                if available.size:
                    nearest = float(np.min(available))
                    source |= candidates & (
                        distances <= nearest + 0.51 * self.spacing)
            self._source_x = gx[source].astype(np.float64)
            self._source_y = gy[source].astype(np.float64)
            self._source_flat = (
                np.searchsorted(self.grid_y, gy[source]) * self.grid_x.size
                + np.searchsorted(self.grid_x, gx[source]))
        else:
            self._source_x = np.empty(0, dtype=np.float64)
            self._source_y = np.empty(0, dtype=np.float64)
            self._source_flat = np.empty(0, dtype=np.int64)

        self.x = np.empty(0, dtype=np.float64)
        self.y = np.empty(0, dtype=np.float64)
        self.Exx_inf = np.empty(0, dtype=np.float64)
        self.Eyy_inf = np.empty(0, dtype=np.float64)
        self.Exy_inf = np.empty(0, dtype=np.float64)
        self.equivalent = np.empty(0, dtype=np.float64)
        self.F11 = np.empty(0, dtype=np.float64)
        self.F12 = np.empty(0, dtype=np.float64)
        self.F21 = np.empty(0, dtype=np.float64)
        self.F22 = np.empty(0, dtype=np.float64)
        self.age = np.empty(0, dtype=np.int32)

    @property
    def count(self) -> int:
        return int(self.x.size)

    def _append_zeros(self, x: np.ndarray, y: np.ndarray) -> None:
        n = int(x.size)
        if not n:
            return
        self.x = np.concatenate((self.x, x.astype(np.float64, copy=False)))
        self.y = np.concatenate((self.y, y.astype(np.float64, copy=False)))
        zeros = np.zeros(n, dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        for name in ("Exx_inf", "Eyy_inf", "Exy_inf", "equivalent",
                     "F12", "F21"):
            setattr(self, name, np.concatenate((getattr(self, name), zeros)))
        self.F11 = np.concatenate((self.F11, ones))
        self.F22 = np.concatenate((self.F22, ones))
        self.age = np.concatenate((self.age, np.zeros(n, dtype=np.int32)))

    def _particle_grid_cells(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.count or not self.grid_x.size or not self.grid_y.size:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, np.empty(0, dtype=bool)
        ix = np.rint((self.x - self.radius) / self.spacing).astype(np.int64)
        iy = np.rint((self.y - self.radius) / self.spacing).astype(np.int64)
        inside = ((ix >= 0) & (ix < self.grid_x.size) &
                  (iy >= 0) & (iy < self.grid_y.size))
        return ix, iy, inside

    def seed(self, pair_valid: Optional[np.ndarray] = None) -> int:
        """Inject unoccupied subset centres from the spatial origin region."""
        if not self._source_x.size:
            return 0
        eligible = np.ones(self._source_x.size, dtype=bool)
        if pair_valid is not None:
            valid = np.asarray(pair_valid, dtype=bool)
            eligible &= valid[self._source_y.astype(np.intp),
                              self._source_x.astype(np.intp)]

        # Replenish the inlet without stacking a new state on material that has
        # not yet moved away from its source cell.
        if self.count:
            ix, iy, inside = self._particle_grid_cells()
            pidx = np.where(inside)[0]
            if pidx.size:
                px = self.radius + ix[pidx] * self.spacing
                py = self.radius + iy[pidx] * self.spacing
                close = ((self.x[pidx] - px) ** 2 + (self.y[pidx] - py) ** 2
                         <= (0.60 * self.spacing) ** 2)
                pidx = pidx[close]
                if pidx.size:
                    flats = iy[pidx] * self.grid_x.size + ix[pidx]
                    eligible &= ~np.isin(self._source_flat, np.unique(flats))

        x = self._source_x[eligible]
        y = self._source_y[eligible]
        self._append_zeros(x, y)
        return int(x.size)

    def _sample(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Strict bilinear sampling of a sparse regular-grid result field."""
        values = np.full(self.count, np.nan, dtype=np.float64)
        if (not self.count or field is None or not self.grid_x.size
                or not self.grid_y.size):
            return values, np.zeros(self.count, dtype=bool)
        arr = np.asarray(field)
        qx = (self.x - self.radius) / self.spacing
        qy = (self.y - self.radius) / self.spacing
        in_grid = ((qx >= 0.0) & (qx <= self.grid_x.size - 1) &
                   (qy >= 0.0) & (qy <= self.grid_y.size - 1))
        rows = np.where(in_grid)[0]
        if not rows.size:
            return values, in_grid

        xq, yq = qx[rows], qy[rows]
        ix0 = np.floor(xq).astype(np.intp)
        iy0 = np.floor(yq).astype(np.intp)
        ix1 = np.minimum(ix0 + 1, self.grid_x.size - 1)
        iy1 = np.minimum(iy0 + 1, self.grid_y.size - 1)
        tx, ty = xq - ix0, yq - iy0
        tx[ix0 == ix1] = 0.0
        ty[iy0 == iy1] = 0.0

        xs0, xs1 = self.grid_x[ix0], self.grid_x[ix1]
        ys0, ys1 = self.grid_y[iy0], self.grid_y[iy1]
        samples = np.column_stack((
            arr[ys0, xs0], arr[ys0, xs1],
            arr[ys1, xs0], arr[ys1, xs1],
        )).astype(np.float64, copy=False)
        weights = np.column_stack((
            (1.0 - tx) * (1.0 - ty), tx * (1.0 - ty),
            (1.0 - tx) * ty, tx * ty,
        ))
        needed = weights > 1e-12
        good = np.all(~needed | np.isfinite(samples), axis=1)
        interpolated = np.sum(np.where(needed, samples, 0.0) * weights, axis=1)
        good &= np.isfinite(interpolated)
        values[rows[good]] = interpolated[good]
        valid = np.zeros(self.count, dtype=bool)
        valid[rows[good]] = True
        return values, valid

    def _retain(self, keep: np.ndarray) -> None:
        for name in ("x", "y", "Exx_inf", "Eyy_inf", "Exy_inf",
                     "equivalent", "F11", "F12", "F21", "F22", "age"):
            setattr(self, name, getattr(self, name)[keep])

    def advance(
        self,
        u: np.ndarray,
        v: np.ndarray,
        du_dx: np.ndarray,
        du_dy: np.ndarray,
        dv_dx: np.ndarray,
        dv_dy: np.ndarray,
    ) -> int:
        """Transport all live states through one measured frame interval."""
        if not self.count:
            return 0
        sampled = [self._sample(a) for a in
                   (u, v, du_dx, du_dy, dv_dx, dv_dy)]
        vals = [item[0] for item in sampled]
        good = np.logical_and.reduce([item[1] for item in sampled])
        if not good.any():
            self._retain(good)
            return 0

        uu, vv, h11, h12, h21, h22 = vals
        new_x = self.x + uu
        new_y = self.y + vv
        h, w = self.shape
        in_frame = (np.isfinite(new_x) & np.isfinite(new_y) &
                    (new_x >= 0.0) & (new_x <= w - 1.0) &
                    (new_y >= 0.0) & (new_y <= h - 1.0))
        # Material points may leave the initial domain_mask (ROI) but still be on-screen
        # and valid if the solver tracked them. Kill them only if they fall off the image.
        good &= in_frame
        if not good.any():
            self._retain(good)
            return 0

        deq = von_mises_equivalent(h11, h22, 0.5 * (h12 + h21))
        good &= np.isfinite(deq)
        if not good.any():
            self._retain(good)
            return 0

        # Signed small-strain components integrate D*dt. H is the displacement
        # gradient for this frame interval, so dt is already included.
        self.Exx_inf[good] += h11[good]
        self.Eyy_inf[good] += h22[good]
        self.Exy_inf[good] += 0.5 * (h12[good] + h21[good])
        self.equivalent[good] += deq[good]

        # Updated-Lagrangian finite strain: F_total(n+1) = F_inc * F_total(n).
        f11, f12 = self.F11.copy(), self.F12.copy()
        f21, f22 = self.F21.copy(), self.F22.copy()
        a11, a12 = 1.0 + h11, h12
        a21, a22 = h21, 1.0 + h22
        self.F11[good] = a11[good] * f11[good] + a12[good] * f21[good]
        self.F12[good] = a11[good] * f12[good] + a12[good] * f22[good]
        self.F21[good] = a21[good] * f11[good] + a22[good] * f21[good]
        self.F22[good] = a21[good] * f12[good] + a22[good] * f22[good]
        good &= (np.isfinite(self.F11) & np.isfinite(self.F12) &
                 np.isfinite(self.F21) & np.isfinite(self.F22))

        self.x[good], self.y[good] = new_x[good], new_y[good]
        self.age[good] += 1
        self._retain(good)
        return self.count

    def snapshot(self) -> dict[str, np.ndarray]:
        """Remesh live material state to the current regular subset grid."""
        out = {name: np.full(self.shape, np.nan, dtype=np.float32)
               for name in _STATE_NAMES if name != "Eeff_gl"}
        out["Eeff_gl"] = out["Eeff_inf"]
        if not self.count:
            return out
        ix, iy, inside = self._particle_grid_cells()
        rows = np.where(inside)[0]
        if not rows.size:
            return out

        # Work only with particles that can be remeshed. A live point may still
        # be inside the image while sitting in the subset-radius margin; mixing
        # the full ix/iy arrays with this filtered row list made snapshot()
        # broadcast incompatible shapes as soon as that happened.
        ix_inside = ix[rows]
        iy_inside = iy[rows]
        flat = iy_inside * self.grid_x.size + ix_inside
        distance = (
            (self.x[rows] - (self.radius + ix_inside * self.spacing)) ** 2
            + (self.y[rows] - (self.radius + iy_inside * self.spacing)) ** 2)
        order = np.lexsort((-self.age[rows], distance, flat))
        ordered_flat = flat[order]
        first = np.empty(order.size, dtype=bool)
        first[0] = True
        first[1:] = ordered_flat[1:] != ordered_flat[:-1]
        chosen = rows[order[first]]
        cx = (self.radius + np.rint((self.x[chosen] - self.radius) /
              self.spacing).astype(np.int64) * self.spacing)
        cy = (self.radius + np.rint((self.y[chosen] - self.radius) /
              self.spacing).astype(np.int64) * self.spacing)

        exx_gl = 0.5 * (self.F11 ** 2 + self.F21 ** 2 - 1.0)
        eyy_gl = 0.5 * (self.F12 ** 2 + self.F22 ** 2 - 1.0)
        exy_gl = 0.5 * (self.F11 * self.F12 + self.F21 * self.F22)
        values = {
            "Exx_inf": self.Exx_inf,
            "Eyy_inf": self.Eyy_inf,
            "Exy_inf": self.Exy_inf,
            "Eeff_inf": self.equivalent,
            "Exx_gl": exx_gl,
            "Eyy_gl": eyy_gl,
            "Exy_gl": exy_gl,
            # This is the non-negative path integral requested by the UI, not
            # von_mises_equivalent(E_total), which can decrease on unloading.
            "Eeff_gl": self.equivalent,
        }
        finite_xy = ((cx >= 0) & (cx < self.shape[1]) &
                     (cy >= 0) & (cy < self.shape[0]))
        cx, cy, chosen = cx[finite_xy], cy[finite_xy], chosen[finite_xy]
        for name, state in values.items():
            vals = state[chosen]
            finite = np.isfinite(vals)
            out[name][cy[finite], cx[finite]] = vals[finite]
        # Both formulations use the same requested accumulated equivalent-rate
        # path integral. Share storage instead of retaining a duplicate full
        # image for every frame.
        out["Eeff_gl"] = out["Eeff_inf"]
        return out
