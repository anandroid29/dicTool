"""Backend-neutral accumulation for immediate-frame DIC.

Correlation always measures the current frame against the immediately previous
frame.  Displacement remains an interval quantity; only strain is accumulated.
The accumulator also keeps a private position history used solely to find the
same material in the next frame.  That history must never leak into the public
``u``/``v`` result fields.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter
from .strain import (compute_velocity_strains, connected_support_labels,
                     von_mises_equivalent)

# Minimum number of currently-live neighbours required before a re-appearing
# point's accumulated total may be re-baselined from them.
_MIN_NEIGHBOUR_SUPPORT = 3.0
# Box half-widths tried, smallest first, when looking for that support.
_REPAIR_RADII = (3, 6, 12, 24)


def _neighbour_estimate(field: np.ndarray, live: np.ndarray,
                        eligible: np.ndarray,
                        grid_spacing: int = 1,
                        radii=_REPAIR_RADII) -> np.ndarray:
    """Estimate `field` where it is not live, from the mean of nearby live values.

    Returns NaN wherever no radius in `radii` gathered enough support, so the
    caller can tell "repaired" from "genuinely unrecoverable".
    """
    est = np.full(field.shape, np.nan, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    live = np.asarray(live, dtype=bool) & eligible & np.isfinite(field)
    need = eligible & ~live
    if not need.any():
        return est
    labels, n_components = connected_support_labels(eligible, grid_spacing)
    for component_id in range(1, n_components + 1):
        component = eligible & (labels == component_id)
        source = live & component
        component_need = need & component
        if not source.any() or not component_need.any():
            continue
        fz = np.where(source, field, 0.0).astype(np.float64)
        cz = source.astype(np.float64)
        for r in radii:
            k = 2 * int(r) + 1
            area = float(k * k)
            s = uniform_filter(fz, size=k, mode="constant", cval=0.0) * area
            n = uniform_filter(cz, size=k, mode="constant", cval=0.0) * area
            ok = component_need & (n >= _MIN_NEIGHBOUR_SUPPORT)
            if ok.any():
                values = s[ok] / n[ok]
                finite = np.isfinite(values)
                indices = np.where(ok)
                est[indices[0][finite], indices[1][finite]] = values[finite]
                component_need[indices[0][finite], indices[1][finite]] = False
            if not component_need.any():
                break
    return est


def _principal_2x2(Axx, Ayy, Axy):
    """Eigenvalues of a symmetric 2x2 field, larger first."""
    tr = Axx + Ayy
    disc = np.sqrt(np.maximum((Axx - Ayy) ** 2 + 4.0 * Axy ** 2, 0.0))
    return 0.5 * (tr + disc), 0.5 * (tr - disc)


def _equivalent_from_principal(e1, e2):
    """von Mises equivalent strain from the in-plane principal values.

    Shear vanishes in the principal frame, so this is the shared definition
    evaluated there. Routing through it keeps accumulated strain and strain
    rate on one formula.
    """
    return von_mises_equivalent(e1, e2, 0.0)


def incremental_strains(du_dx, du_dy, dv_dx, dv_dy):
    """Return infinitesimal and Green-Lagrange strain for one frame interval.

    ``Exy`` is tensor shear.  Engineering shear is always exactly ``2*Exy`` and
    is returned under a separate, explicit name.  Equivalent strain is a
    non-negative magnitude; the accumulator sums that magnitude frame by frame.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        # Infinitesimal strain: symmetric part of the displacement gradient.
        inf_xx = du_dx
        inf_yy = dv_dy
        inf_xy = 0.5 * (du_dy + dv_dx)

        # Incremental Green-Lagrange strain: E = 0.5 * (F.T F - I).
        F11 = 1.0 + du_dx
        F12 = du_dy
        F21 = dv_dx
        F22 = 1.0 + dv_dy
        gl_xx = 0.5 * (F11 * F11 + F21 * F21 - 1.0)
        gl_yy = 0.5 * (F12 * F12 + F22 * F22 - 1.0)
        gl_xy = 0.5 * (F11 * F12 + F21 * F22)

        inf_1, inf_2 = _principal_2x2(inf_xx, inf_yy, inf_xy)
        gl_1, gl_2 = _principal_2x2(gl_xx, gl_yy, gl_xy)
        inf_eff = _equivalent_from_principal(inf_1, inf_2)
        gl_eff = _equivalent_from_principal(gl_1, gl_2)

    result = dict(
        Exx_inf=inf_xx, Eyy_inf=inf_yy, Exy_inf=inf_xy,
        Gxy_inf=2.0 * inf_xy, Eeff_inf=inf_eff,
        Exx_gl=gl_xx, Eyy_gl=gl_yy, Exy_gl=gl_xy,
        Gxy_gl=2.0 * gl_xy, Eeff_gl=gl_eff,
    )
    for values in result.values():
        values[~np.isfinite(values)] = np.nan
    return result


class StrainAccumulator:
    """Accumulate both requested strain measures, but not displacement.

    The private ``u``/``v`` arrays are position hints for temporal tracking.
    Public result displacement is supplied by the current frame's solver.  A
    dropout is recoverable: retained history is hidden while invalid and is
    resumed if the point is measured again.
    """

    def __init__(self, shape, strain_window: int, grid_spacing: int = 1):
        self.shape = shape
        self.strain_window = int(strain_window)
        self.grid_spacing = max(1, int(grid_spacing))
        self.u = np.full(shape, np.nan)
        self.v = np.full(shape, np.nan)
        self.Exx_inf = np.full(shape, np.nan)
        self.Eyy_inf = np.full(shape, np.nan)
        self.Exy_inf = np.full(shape, np.nan)
        self.Eeff_inf = np.full(shape, np.nan)
        self.Exx_gl = np.full(shape, np.nan)
        self.Eyy_gl = np.full(shape, np.nan)
        self.Exy_gl = np.full(shape, np.nan)
        self.Eeff_gl = np.full(shape, np.nan)
        self.n_frames = np.zeros(shape, np.int32)
        self.started = np.zeros(shape, bool)    # has ever been successfully tracked
        # Carries a usable running total RIGHT NOW. Recomputed every frame, so a
        # point can leave this set and come back into it.
        self.tracked = np.zeros(shape, bool)
        self.strain_tracked = np.zeros(shape, bool)
        # Sticky, informational: this point's total no longer traces back to
        # frame 0 unbroken (it was re-baselined or started late).
        self.rebased = np.zeros(shape, bool)
        self._first = True

    # Kept as a read-only alias: "not trustworthy for this frame".
    @property
    def broken(self):
        return self.started & ~self.tracked

    def mark_lost(self, lost: np.ndarray) -> None:
        """Drop points out of the live set for this frame (e.g. dynamic-ROI loss).

        Their accumulated totals are retained, so if they come back the repair
        path can use them; they are simply not reported as valid right now.
        """
        self.tracked &= ~lost

    def add_frame(self, inc_u, inc_v, inc_du_dx, inc_du_dy, inc_dv_dx, inc_dv_dy):
        ok = np.isfinite(inc_u) & np.isfinite(inc_v)

        # Derive gradients from the same immediate displacement fields on both
        # backends. Solver-affine terms are deliberately not used for strain:
        # they differ between CPU/GPU implementations and bypass strain_window.
        grad = compute_velocity_strains(
            inc_u, inc_v, ok, self.strain_window, self.grid_spacing)
        inc_du_dx = grad["dVx_dx"]
        inc_du_dy = grad["dVx_dy"]
        inc_dv_dx = grad["dVy_dx"]
        inc_dv_dy = grad["dVy_dy"]
        self.last_gradients = {
            "du_dx": inc_du_dx, "du_dy": inc_du_dy,
            "dv_dx": inc_dv_dx, "dv_dy": inc_dv_dy,
        }
        strain_ok = (ok & np.isfinite(inc_du_dx) & np.isfinite(inc_du_dy) &
                     np.isfinite(inc_dv_dx) & np.isfinite(inc_dv_dy))

        increments = incremental_strains(
            inc_du_dx, inc_du_dy, inc_dv_dx, inc_dv_dy)

        if self._first:
            self.u[ok] = inc_u[ok]
            self.v[ok] = inc_v[ok]
            for name in ("Exx_inf", "Eyy_inf", "Exy_inf", "Eeff_inf",
                         "Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"):
                arr = getattr(self, name)
                arr[strain_ok] = 0.0
            self.started |= ok
            self._first = False
        else:
            # Totals that were trustworthy going into this frame.
            live = self.tracked & np.isfinite(self.u) & np.isfinite(self.v)

            cont = ok & live

            # Measured now but without an intact total: either returning from a
            # gap or seen for the first time. Both are handled the same way --
            # take the neighbours' accumulated total as this point's total
            # through the PREVIOUS frame, then add this frame's own increment.
            #
            # The estimate MUST be read before `cont` is advanced below. Reading
            # it afterwards samples neighbours that already include this frame's
            # motion, and adding inc on top double-counts one increment.
            need = ok & ~cont
            est_u = est_v = None
            if need.any():
                est_u = _neighbour_estimate(
                    self.u, live, eligible=ok, grid_spacing=self.grid_spacing)
                est_v = _neighbour_estimate(
                    self.v, live, eligible=ok, grid_spacing=self.grid_spacing)

            # Intact history: accumulate normally.
            self.u[cont] += inc_u[cont]
            self.v[cont] += inc_v[cont]

            if need.any():
                rep = need & np.isfinite(est_u) & np.isfinite(est_v)
                self.u[rep] = est_u[rep] + inc_u[rep]
                self.v[rep] = est_v[rep] + inc_v[rep]

                # No live neighbours anywhere near: nothing to borrow. Start this
                # point's own history here, which is honest but means its
                # displacement is relative to now, not to frame 0.
                orphan = need & ~rep
                self.u[orphan] = inc_u[orphan]
                self.v[orphan] = inc_v[orphan]

                self.started |= need
                self.rebased |= need

                # Newly appearing points have no strain history. Returning
                # points retain their own pre-dropout history. This avoids both
                # permanent death and fabrication of a local strain peak from a
                # neighbour's value.
                new = need & ~np.isfinite(self.Exx_inf)
                for name in ("Exx_inf", "Eyy_inf", "Exy_inf", "Eeff_inf",
                             "Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"):
                    getattr(self, name)[new] = 0.0

        # A point is trustworthy for THIS frame exactly when it was measured for
        # this frame. Freezing a stale total and still reporting it as valid was
        # the other half of the problem: downstream velocity differencing then
        # saw a perfectly ordinary number that had quietly stopped updating.
        self.tracked = ok & self.started & np.isfinite(self.u) & np.isfinite(self.v)

        # Accumulate tensor components and non-negative equivalent magnitudes
        # independently for each requested formulation.
        self.strain_tracked = self.tracked & strain_ok
        good = self.strain_tracked
        first_strain = good & ~np.isfinite(self.Exx_inf)
        if first_strain.any():
            for name in ("Exx_inf", "Eyy_inf", "Exy_inf", "Eeff_inf",
                         "Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"):
                getattr(self, name)[first_strain] = 0.0
        for name in ("Exx_inf", "Eyy_inf", "Exy_inf", "Eeff_inf",
                     "Exx_gl", "Eyy_gl", "Exy_gl", "Eeff_gl"):
            total = getattr(self, name)
            inc = increments[name]
            acc = good & np.isfinite(total) & np.isfinite(inc)
            total[acc] += inc[acc]
        self.n_frames[good] += 1

    @property
    def valid(self):
        """Points carrying a live, trustworthy accumulated total this frame."""
        return self.tracked & np.isfinite(self.u) & np.isfinite(self.v)

    def position_hint(self):
        """Best-known (x, y) offsets for EVERY started point, dropouts included.

        Deliberately not masked by `valid`: a point that missed this frame is
        still far more likely to be near its last known position than at its
        frame-0 position, and handing the solver NaN here makes a temporary
        dropout permanent.
        """
        return self.u, self.v

    def results(self):
        v = self.valid
        strain_v = v & self.strain_tracked
        out = {
            "u_total": self.u,
            "v_total": self.v,
            "Exx_inf": np.where(strain_v, self.Exx_inf, np.nan),
            "Eyy_inf": np.where(strain_v, self.Eyy_inf, np.nan),
            "Exy_inf": np.where(strain_v, self.Exy_inf, np.nan),
            "Gxy_inf": np.where(strain_v, 2.0 * self.Exy_inf, np.nan),
            "Eeff_inf": np.where(strain_v, self.Eeff_inf, np.nan),
            "Exx_gl": np.where(strain_v, self.Exx_gl, np.nan),
            "Eyy_gl": np.where(strain_v, self.Eyy_gl, np.nan),
            "Exy_gl": np.where(strain_v, self.Exy_gl, np.nan),
            "Gxy_gl": np.where(strain_v, 2.0 * self.Exy_gl, np.nan),
            "Eeff_gl": np.where(strain_v, self.Eeff_gl, np.nan),
        }
        for name, arr in getattr(self, "last_gradients", {}).items():
            out[name] = np.where(strain_v, arr, np.nan)
        # Legacy field aliases remain readable for old integrations. They now
        # mean accumulated infinitesimal strain and are not shown under these
        # ambiguous names in the UI.
        out["Exx"] = out["Exx_inf"]
        out["Eyy"] = out["Eyy_inf"]
        out["Exy"] = out["Exy_inf"]
        out["Eeff"] = out["Eeff_inf"]
        out["valid"] = v
        out["strain_valid"] = strain_v
        out["broken"] = self.broken
        out["rebased"] = self.rebased
        return out
