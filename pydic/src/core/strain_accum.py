"""
strain_accum.py
---------------
Accumulated finite strain for incremental (updated-Lagrangian) DIC.

Two distinct quantities are provided, because at machining strain levels they
are NOT interchangeable:

1. Total strain from the accumulated displacement field. This is a state
   measure -- it describes the current shape relative to the undeformed
   reference and forgets how the material got there.

2. Path-integrated equivalent strain. This sums the equivalent strain
   increment frame by frame. For non-proportional deformation -- which is
   exactly what the primary shear zone is, since material rotates strongly as
   it turns into the chip -- this is the quantity that corresponds to
   accumulated plastic work, and it is strictly larger than (1).

For simple shear of amount gamma, (2) converges to the textbook machining
result gamma/sqrt(3), while (1) falls increasingly short as gamma grows.
Reporting (1) as "accumulated strain" understates shear-zone strain badly.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter
from .strain import compute_velocity_strains

# Minimum number of currently-live neighbours required before a re-appearing
# point's accumulated total may be re-baselined from them.
_MIN_NEIGHBOUR_SUPPORT = 3.0
# Box half-widths tried, smallest first, when looking for that support.
_REPAIR_RADII = (3, 6, 12, 24)


def _neighbour_estimate(field: np.ndarray, live: np.ndarray,
                        radii=_REPAIR_RADII) -> np.ndarray:
    """Estimate `field` where it is not live, from the mean of nearby live values.

    Returns NaN wherever no radius in `radii` gathered enough support, so the
    caller can tell "repaired" from "genuinely unrecoverable".
    """
    est = np.full(field.shape, np.nan, dtype=np.float64)
    need = ~live
    if not need.any():
        return est
    fz = np.where(live & np.isfinite(field), field, 0.0).astype(np.float64)
    cz = (live & np.isfinite(field)).astype(np.float64)
    for r in radii:
        k = 2 * int(r) + 1
        area = float(k * k)
        s = uniform_filter(fz, size=k, mode="constant", cval=0.0) * area
        n = uniform_filter(cz, size=k, mode="constant", cval=0.0) * area
        ok = need & (n >= _MIN_NEIGHBOUR_SUPPORT)
        if ok.any():
            est[ok] = s[ok] / n[ok]
            need = need & ~ok
        if not need.any():
            break
    return est


def _principal_2x2(Axx, Ayy, Axy):
    """Eigenvalues of a symmetric 2x2 field, larger first."""
    tr = Axx + Ayy
    disc = np.sqrt(np.maximum((Axx - Ayy) ** 2 + 4.0 * Axy ** 2, 0.0))
    return 0.5 * (tr + disc), 0.5 * (tr - disc)


def _equivalent_from_principal(e1, e2):
    """von Mises equivalent strain, plane strain + plastic incompressibility."""
    e3 = -(e1 + e2)
    return np.sqrt((2.0 / 3.0) * (e1 ** 2 + e2 ** 2 + e3 ** 2))


def deformation_gradient(u_total, v_total, valid, strain_window):
    """F = I + grad_X(u_total), by least-squares plane fit on the reference grid."""
    g = compute_velocity_strains(u_total, v_total, valid, strain_window)
    F11 = 1.0 + g["dVx_dx"]
    F12 = g["dVx_dy"]
    F21 = g["dVy_dx"]
    F22 = 1.0 + g["dVy_dy"]
    return F11, F12, F21, F22


def total_strain(u_total, v_total, valid, strain_window):
    """Finite strain measures from the accumulated displacement field."""
    F11, F12, F21, F22 = deformation_gradient(u_total, v_total, valid, strain_window)

    # Right Cauchy-Green C = F^T F
    Cxx = F11 ** 2 + F21 ** 2
    Cyy = F12 ** 2 + F22 ** 2
    Cxy = F11 * F12 + F21 * F22

    # Green-Lagrange E = (C - I)/2
    Exx = 0.5 * (Cxx - 1.0)
    Eyy = 0.5 * (Cyy - 1.0)
    Exy = 0.5 * Cxy

    # Hencky (true/logarithmic) strain from the principal stretches. This is the
    # appropriate large-strain measure; Green-Lagrange overstates badly above
    # ~20% strain because it is quadratic in the stretch.
    l1, l2 = _principal_2x2(Cxx, Cyy, Cxy)
    with np.errstate(invalid="ignore", divide="ignore"):
        h1 = 0.5 * np.log(np.maximum(l1, 1e-12))
        h2 = 0.5 * np.log(np.maximum(l2, 1e-12))
        eq_log = _equivalent_from_principal(h1, h2)
        eq_gl = _equivalent_from_principal(*_principal_2x2(Exx, Eyy, Exy))
        detF = F11 * F22 - F12 * F21

    return dict(
        F11=F11, F12=F12, F21=F21, F22=F22, detF=detF,
        Exx=Exx, Eyy=Eyy, Exy=Exy, Eeff_GL=eq_gl,
        e1_log=h1, e2_log=h2, Eeff_log=eq_log,
    )


class StrainAccumulator:
    """
    Accumulates displacement and path-integrated equivalent strain across frames.

    Feed it one frame at a time. It keeps its own validity bookkeeping so that a
    subset which drops out for a single frame does NOT have its accumulated
    history silently replaced by that frame's increment.

    Dropouts are RECOVERABLE. A point that goes missing (occlusion, a frame the
    solver could not match, a dynamic-ROI flicker) and later comes back is
    re-baselined from the accumulated totals of its still-live neighbours, then
    resumes accumulating normally. The displacement field is spatially smooth,
    so the neighbours carry the information the gap destroyed.

    The previous version instead latched a permanent `broken` flag on any such
    point -- and since the flag was only ever OR-ed in, never cleared, a single
    bad frame removed that point from the output for the rest of the sequence.
    Over a long sequence the valid field only ever shrank, which is the "once a
    pixel dies it stays dead" behaviour. It also fed NaN totals back into the
    solver's position hint, so a dropped point was then searched for at its
    frame-0 location and could not re-acquire even in principle.

    Re-baselining borrows the local mean displacement, so a repaired point does
    not preserve its own strain concentration across the gap. `rebased` marks
    those points so that can be seen rather than guessed at.
    """

    def __init__(self, shape, strain_window: int):
        self.shape = shape
        self.strain_window = int(strain_window)
        self.u = np.full(shape, np.nan)
        self.v = np.full(shape, np.nan)
        self.eq_path = np.full(shape, np.nan)   # path-integrated equivalent strain
        self.n_frames = np.zeros(shape, np.int32)
        self.started = np.zeros(shape, bool)    # has ever been successfully tracked
        # Carries a usable running total RIGHT NOW. Recomputed every frame, so a
        # point can leave this set and come back into it.
        self.tracked = np.zeros(shape, bool)
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

        if self._first:
            self.u[ok] = inc_u[ok]
            self.v[ok] = inc_v[ok]
            self.eq_path[ok] = 0.0
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
            est_u = est_v = est_p = None
            if need.any():
                est_u = _neighbour_estimate(self.u, live)
                est_v = _neighbour_estimate(self.v, live)
                est_p = _neighbour_estimate(self.eq_path, live)

            # Intact history: accumulate normally.
            self.u[cont] += inc_u[cont]
            self.v[cont] += inc_v[cont]

            if need.any():
                rep = need & np.isfinite(est_u) & np.isfinite(est_v)
                self.u[rep] = est_u[rep] + inc_u[rep]
                self.v[rep] = est_v[rep] + inc_v[rep]
                self.eq_path[rep] = np.where(np.isfinite(est_p[rep]), est_p[rep], 0.0)

                # No live neighbours anywhere near: nothing to borrow. Start this
                # point's own history here, which is honest but means its
                # displacement is relative to now, not to frame 0.
                orphan = need & ~rep
                self.u[orphan] = inc_u[orphan]
                self.v[orphan] = inc_v[orphan]
                self.eq_path[orphan] = 0.0

                self.started |= need
                self.rebased |= need

        # A point is trustworthy for THIS frame exactly when it was measured for
        # this frame. Freezing a stale total and still reporting it as valid was
        # the other half of the problem: downstream velocity differencing then
        # saw a perfectly ordinary number that had quietly stopped updating.
        self.tracked = ok & self.started & np.isfinite(self.u) & np.isfinite(self.v)

        # --- path-integrated equivalent strain increment ---
        # For small per-frame increments the incremental Hencky strain is well
        # approximated by the symmetric part of the incremental displacement
        # gradient. Rotation drops out, which is precisely why this works through
        # the shear zone where the total-strain measure fails.
        with np.errstate(invalid="ignore"):
            dxx = inc_du_dx
            dyy = inc_dv_dy
            dxy = 0.5 * (inc_du_dy + inc_dv_dx)
            d1, d2 = _principal_2x2(dxx, dyy, dxy)
            deq = _equivalent_from_principal(d1, d2)

        good = self.tracked & np.isfinite(deq)
        acc = good & np.isfinite(self.eq_path)
        self.eq_path[acc] += deq[acc]
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
        out = total_strain(self.u, self.v, v, self.strain_window)
        out["u_total"] = self.u
        out["v_total"] = self.v
        out["Eeff_path"] = np.where(v, self.eq_path, np.nan)
        out["valid"] = v
        out["broken"] = self.broken
        out["rebased"] = self.rebased
        for k in ("Exx", "Eyy", "Exy", "Eeff_GL", "Eeff_log",
                  "e1_log", "e2_log", "detF", "F11", "F12", "F21", "F22"):
            out[k] = np.where(v, out[k], np.nan)
        return out
