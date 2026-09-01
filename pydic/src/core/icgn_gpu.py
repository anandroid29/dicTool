
# ── NOT YET IMPLEMENTED on GPU ──
# icgn.py (CPU path) now rejects subsets where a large or one-sided fraction of
# pixels are near-black/near-saturated (occlusion silhouette, glare), which was
# the root cause of tracked points wandering before being lost near a tool/chip
# boundary. This module has no equivalent check yet -- CuPy isn't available in
# this environment so it could not be validated here. Port
# OCCLUSION_INTENSITY_LOW/HIGH, MAX_OCCLUDED_FRACTION, MAX_OCCLUDED_ASYMMETRY,
# and _occlusion_asymmetry() from icgn.py before relying on GPU tracking near
# occlusion boundaries; until then, expect the same wandering there.
# src/core/icgn_gpu.py
"""
icgn_gpu.py
-----------
Wavefront GPU-accelerated IC-GN solver using CuPy.
Fixed: Re-anchored the safety jump check to track IC-GN drift instead of inter-frame delta.
Double precision float64 used to eliminate noise. Warm Start queue implemented up to 40,000 subsets.
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage
import sys

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

# Cubic B-spline prefiltering runs once per frame over the WHOLE image. Doing it
# with scipy meant a full-resolution CPU pass plus a host->device copy on every
# frame, on the critical path of an otherwise GPU-resident pipeline. CuPy has
# the same filter; fall back to scipy only if this CuPy build lacks it.
try:
    from cupyx.scipy.ndimage import spline_filter as _cp_spline_filter
    _HAS_CP_SPLINE = _HAS_CUPY
except ImportError:
    _HAS_CP_SPLINE = False

DEBUG_BATCHES = False


class GPUWavefrontDIC:
    def __init__(self, params):
        if not _HAS_CUPY:
            raise RuntimeError("CuPy is required for GPU execution.")
        self.params = params
        self._initialized = False
        self._warned_cpu_spline = False

        self.state = None
        self.p_global = None
        self.cls_global = None
        self.ref_value = None
        self.ref_grad_x = None
        self.ref_grad_y = None
        self._cur_image = None
        self._cur_gpu = None
        self._cur_coeff = None

    def _spline_coeff(self, img: np.ndarray) -> "cp.ndarray":
        """Cubic B-spline coefficients of `img`, computed on the GPU when possible."""
        if _HAS_CP_SPLINE:
            try:
                gpu_img = cp.asarray(img, dtype=cp.float64)
                return _cp_spline_filter(gpu_img, order=3, output=cp.float64, mode='mirror')
            except Exception as e:
                if not self._warned_cpu_spline:
                    print(f"[GPU] CuPy spline_filter unavailable ({e}); "
                          f"falling back to the scipy CPU path.")
                    self._warned_cpu_spline = True
        return cp.asarray(
            scipy.ndimage.spline_filter(np.asarray(img, dtype=np.float64),
                                        order=3, mode='mirror'),
            dtype=cp.float64)

    def precompute_reference(self, ref_image: np.ndarray, roi_mask: np.ndarray):
        self.H, self.W = ref_image.shape
        self.r = self.params.subset_radius
        self.s = self.params.subset_spacing

        ys = np.arange(self.r, self.H - self.r, self.s)
        xs = np.arange(self.r, self.W - self.r, self.s)
        self.grid_shape = (len(ys), len(xs))

        gx, gy = np.meshgrid(xs, ys)

        self.gx_flat = gx.ravel()
        self.gy_flat = gy.ravel()
        self.valid_mask = roi_mask[self.gy_flat, self.gx_flat]
        self.N_total = len(self.gx_flat)

        self.ref_coeff = self._spline_coeff(ref_image)
        self._invalidate_reference_maps()

        dy_sub, dx_sub = np.mgrid[-self.r:self.r+1, -self.r:self.r+1]
        mask_sub = (dx_sub**2 + dy_sub**2) <= self.r**2
        self.dx_sub = cp.asarray(dx_sub[mask_sub], dtype=cp.float64)
        self.dy_sub = cp.asarray(dy_sub[mask_sub], dtype=cp.float64)
        self.N_px = len(self.dx_sub)

        self.gx_gpu = cp.asarray(self.gx_flat, dtype=cp.float64)
        self.gy_gpu = cp.asarray(self.gy_flat, dtype=cp.float64)

        self._initialized = True

    def _invalidate_reference_maps(self) -> None:
        self.ref_value = self.ref_grad_x = self.ref_grad_y = None

    def _prepare_current(self, image: np.ndarray):
        """Upload and prefilter a frame once, including targeted recovery."""
        if self._cur_image is image and self._cur_coeff is not None:
            return self._cur_gpu, self._cur_coeff
        self._cur_image = image
        self._cur_gpu = cp.asarray(np.asarray(image, dtype=np.float64))
        if _HAS_CP_SPLINE:
            try:
                self._cur_coeff = _cp_spline_filter(
                    self._cur_gpu, order=3, output=cp.float64, mode='mirror')
            except Exception:
                self._cur_coeff = self._spline_coeff(image)
        else:
            self._cur_coeff = self._spline_coeff(image)
        return self._cur_gpu, self._cur_coeff

    def _prepare_reference_maps(self) -> None:
        """Evaluate the fixed reference lattice once for all wavefront batches."""
        if self.ref_value is not None:
            return
        self.ref_value = cp.empty((self.H, self.W), dtype=cp.float64)
        self.ref_grad_x = cp.empty_like(self.ref_value)
        self.ref_grad_y = cp.empty_like(self.ref_value)
        # Row chunks avoid constructing a multi-hundred-MB coordinate stack.
        h = 1e-3
        rows_per_chunk = 256
        x = cp.arange(self.W, dtype=cp.float64)[None, :]
        for y0 in range(0, self.H, rows_per_chunk):
            y1 = min(self.H, y0 + rows_per_chunk)
            yy = cp.arange(y0, y1, dtype=cp.float64)[:, None]
            xx = cp.broadcast_to(x, (y1 - y0, self.W))
            yy = cp.broadcast_to(yy, xx.shape)
            base = cp.stack((yy.ravel(), xx.ravel()))
            xp = cp.stack((yy.ravel(), (xx + h).ravel()))
            xm = cp.stack((yy.ravel(), (xx - h).ravel()))
            yp = cp.stack(((yy + h).ravel(), xx.ravel()))
            ym = cp.stack(((yy - h).ravel(), xx.ravel()))
            shape = (y1 - y0, self.W)
            self.ref_value[y0:y1] = map_coordinates(
                self.ref_coeff, base, order=3, mode='mirror',
                prefilter=False).reshape(shape)
            self.ref_grad_x[y0:y1] = (map_coordinates(
                self.ref_coeff, xp, order=3, mode='mirror', prefilter=False) -
                map_coordinates(self.ref_coeff, xm, order=3, mode='mirror',
                                prefilter=False)).reshape(shape) / (2 * h)
            self.ref_grad_y[y0:y1] = (map_coordinates(
                self.ref_coeff, yp, order=3, mode='mirror', prefilter=False) -
                map_coordinates(self.ref_coeff, ym, order=3, mode='mirror',
                                prefilter=False)).reshape(shape) / (2 * h)

    def update_reference_image(self, new_ref_image: np.ndarray):
        """Update reference B-spline coefficients for Updated Lagrangian tracking.
        Grid geometry and subset offsets are unchanged — only the reference
        intensity data is replaced with the new (previous) frame."""
        if self._cur_image is new_ref_image and self._cur_coeff is not None:
            self.ref_coeff = self._cur_coeff
        else:
            self.ref_coeff = self._spline_coeff(new_ref_image)
        self._invalidate_reference_maps()
        self._cur_image = self._cur_gpu = self._cur_coeff = None

    @staticmethod
    def release_temporary_memory() -> None:
        """Return unused per-frame workspaces to the CUDA driver.

        CuPy's default allocator is a caching pool.  That is normally good for
        repeated equal-size operations, but wavefront and rescue batches vary
        with tracking survival, leaving differently sized multi-hundred-MB
        blocks cached across frames.  ``free_all_blocks`` never frees live
        arrays (reference coefficients, grid state, warm starts); it only drops
        blocks whose owning temporaries have already gone out of scope.
        """
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def _out_of_bounds(self, xs: "cp.ndarray", ys: "cp.ndarray") -> "cp.ndarray":
        """Per-subset flag: does any pixel of this subset fall outside the image?

        Half-pixel margins match the CPU solver's test in icgn.run_icgn.
        """
        return ((xs.min(axis=1) < -0.5) | (xs.max(axis=1) > self.W - 0.5) |
                (ys.min(axis=1) < -0.5) | (ys.max(axis=1) > self.H - 0.5))

    def solve_frame(self, cur_image: np.ndarray, seed_idx: int = -1,
                    seed_p: np.ndarray = None,
                    warm_start: bool = False, *, recovery_seeds=None):
        if not self._initialized:
            raise RuntimeError("Must precompute reference before solving frames.")

        cur_gpu, cur_coeff = self._prepare_current(cur_image)
        self._prepare_reference_maps()

        if recovery_seeds is not None:
            # Preserve reliable warm-start solutions and open only failed ROI
            # regions for propagation from targeted NCC seeds.
            pending = (self.state != 2) & cp.asarray(self.valid_mask)
            self.state[pending] = 0
            self.state[~cp.asarray(self.valid_mask)] = -1
            self.retry[pending] = 0
            for idx, params in recovery_seeds:
                if bool(pending[int(idx)]):
                    self.state[int(idx)] = 1
                    self.p_global[int(idx)] = cp.asarray(params, dtype=cp.float64)
        elif not warm_start:
            self.state = cp.zeros(self.N_total, dtype=cp.int8)
            self.state[~cp.asarray(self.valid_mask)] = -1
            self.p_global = cp.zeros((self.N_total, 6), dtype=cp.float64)
            self.cls_global = cp.full(self.N_total, cp.nan, dtype=cp.float64)
            # Retry budget. A subset that fails from one parent may still
            # succeed from a better one; without this, the first failure is
            # permanent and the wavefront cannot propagate past it, so one bad
            # subset removes everything downstream of it.
            self.retry = cp.zeros(self.N_total, dtype=cp.int8)

            self.state[seed_idx] = 1
            self.p_global[seed_idx] = cp.asarray(seed_p, dtype=cp.float64)
        else:
            self.state[self.state == 2] = 1
            # Re-open failed points inside the analysis ROI so a reliable
            # neighbour can repair them on this fresh previous->current pair.
            # The grid is Eulerian/image-space: material-path transport is a
            # separate strain operation, so warm starting must never grow the
            # pairwise solve outside the operator's analysis ROI.
            self.state[self.state == -1] = 0
            self.state[~cp.asarray(self.valid_mask)] = -1
            if getattr(self, "retry", None) is None or len(self.retry) != self.N_total:
                self.retry = cp.zeros(self.N_total, dtype=cp.int8)
            self.retry[:] = 0

        Ny, Nx = self.grid_shape

        def get_neighbors(indices):
            y = indices // Nx
            x = indices % Nx
            neighbors = []
            if (y > 0).any(): neighbors.append(indices[y > 0] - Nx)
            if (y < Ny - 1).any(): neighbors.append(indices[y < Ny - 1] + Nx)
            if (x > 0).any(): neighbors.append(indices[x > 0] - 1)
            if (x < Nx - 1).any(): neighbors.append(indices[x < Nx - 1] + 1)

            if not neighbors: return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32)
            all_n = cp.concatenate(neighbors)
            parents = cp.concatenate([indices[y > 0], indices[y < Ny - 1], indices[x > 0], indices[x < Nx - 1]])
            return all_n, parents

        MAX_RETRY = 3
        batch_count = 0

        while True:
            batch_count += 1
            active_mask = self.state == 1
            active_indices = cp.where(active_mask)[0]

            MAX_BATCH_SIZE = 40000
            if len(active_indices) > MAX_BATCH_SIZE:
                active_indices = active_indices[:MAX_BATCH_SIZE]

            N_act = len(active_indices)
            if N_act == 0: break

            p_act = self.p_global[active_indices]

            gx_act = self.gx_gpu[active_indices]
            gy_act = self.gy_gpu[active_indices]

            # update_reference_image() makes the immediately previous frame
            # the reference. Its subset centres are therefore the fixed
            # image-space grid, not frame-0 centres plus a cumulative offset.
            # p_global still supplies the previous pair as a numerical warm
            # start, but never changes which points this pair is allowed to
            # measure.
            ref_xs_act = gx_act[:, None] + self.dx_sub[None, :]
            ref_ys_act = gy_act[:, None] + self.dy_sub[None, :]

            # Every map_coordinates call here uses mode='mirror', which does not
            # fail on out-of-range coordinates -- it REFLECTS the image and
            # returns fabricated intensities. A subset tracking material that
            # has left the frame therefore keeps correlating, against a mirror
            # image of the interior, and can score well enough to be accepted.
            # Its affine terms are then meaningless, and because Eeff is a
            # monotonically increasing path integral those bogus increments are
            # summed in permanently -- which is why strain skyrockets exactly
            # where material exits the frame. The CPU solver has always
            # rejected this (run_icgn's bounds test); the GPU never did.
            ref_oob = self._out_of_bounds(ref_xs_act, ref_ys_act)

            ref_xi = ref_xs_act.astype(cp.int32)
            ref_yi = ref_ys_act.astype(cp.int32)
            f = self.ref_value[ref_yi, ref_xi]
            f_mean = f.mean(axis=1, keepdims=True)
            f_c = f - f_mean
            sigma_f_act = cp.maximum(cp.sqrt((f_c**2).sum(axis=1, keepdims=True)), 1e-12)
            f_norm_act = f_c / sigma_f_act

            fx = self.ref_grad_x[ref_yi, ref_xi]
            fy = self.ref_grad_y[ref_yi, ref_xi]

            # --- Bug 1 Fix: Correct ZNSSD steepest descent with mean-correction ---
            # Raw Jacobian dF/dp (before normalization)
            dfdp = cp.empty((N_act, self.N_px, 6), dtype=cp.float64)
            dfdp[:, :, 0] = fx; dfdp[:, :, 1] = fy
            dfdp[:, :, 2] = fx * self.dx_sub; dfdp[:, :, 3] = fx * self.dy_sub
            dfdp[:, :, 4] = fy * self.dx_sub; dfdp[:, :, 5] = fy * self.dy_sub

            # Mean-correction projection (Pan et al. 2009):
            # SD[i,k] = (1/σ_f) * (dfdp[i,k] - f_norm[i] * Σ_j f_norm[j]*dfdp[j,k])
            correction = cp.einsum('np,npk->nk', f_norm_act, dfdp)  # (N_act, 6)
            SD_act = (dfdp - f_norm_act[:, :, None] * correction[:, None, :]) / sigma_f_act[:, :, None]

            H_mat = cp.matmul(SD_act.transpose(0, 2, 1), SD_act)
            H_mat += cp.eye(6, dtype=cp.float64).reshape(1, 6, 6) * 1e-6
            H_inv_act = cp.linalg.inv(H_mat)

            # Integer-shift rescue search.
            #
            # This previously ran unconditionally over a 25x25 shift grid for
            # every subset in every batch. Two problems: it costs 625 full
            # subset resamples per batch (the dominant runtime term), and it
            # actively destroys good propagated guesses by snapping them onto a
            # spurious local ZNSSD minimum -- speckle is quasi-periodic, so a
            # +/-12 px window around a correct guess reliably contains false
            # minima. It now only rescues subsets whose propagated guess is
            # already poor, and leaves good guesses alone.
            def _znssd_grid(rows, base_u, base_v, dxs, dys, mask_oob=True):
                """Best integer shift per row over the offsets (dxs, dys).

                Every (row, shift) pair is evaluated in batched GPU work rather
                than one kernel launch per shift, which is what made a wide
                search unaffordable before. Returns (score, du, dv) per row.
                """
                XS = ref_xs_act if rows is None else ref_xs_act[rows]
                YS = ref_ys_act if rows is None else ref_ys_act[rows]
                FN = f_norm_act if rows is None else f_norm_act[rows]
                n = XS.shape[0]
                best_s = cp.full(n, cp.inf, dtype=cp.float64)
                best_du = cp.zeros(n, dtype=cp.float64)
                best_dv = cp.zeros(n, dtype=cp.float64)
                if n == 0 or len(dxs) == 0:
                    return best_s, best_du, best_dv

                # Keep each batch's temporaries near a fixed element budget so a
                # large rescue set cannot blow up device memory.
                budget = 8_000_000
                k_max = max(1, int(budget // max(1, n * self.N_px)))
                for s0 in range(0, len(dxs), k_max):
                    dxc = cp.asarray(dxs[s0:s0 + k_max], dtype=cp.float64)
                    dyc = cp.asarray(dys[s0:s0 + k_max], dtype=cp.float64)
                    k = int(dxc.size)
                    xt = XS[None] + (base_u[None, :] + dxc[:, None])[:, :, None]
                    yt = YS[None] + (base_v[None, :] + dyc[:, None])[:, :, None]
                    c = cp.stack([yt.ravel(), xt.ravel()], axis=0)
                    gt = map_coordinates(cur_gpu, c, order=1, mode='mirror',
                                         prefilter=False).reshape(k, n, self.N_px)
                    gtc = gt - gt.mean(axis=2, keepdims=True)
                    gtn = gtc / cp.maximum(
                        cp.sqrt((gtc ** 2).sum(axis=2, keepdims=True)), 1e-12)
                    sc = ((gtn - FN[None]) ** 2).sum(axis=2)          # (k, n)
                    if mask_oob:
                        # A candidate shift that pushes the subset off the image
                        # is scored as unusable, so the rescue cannot snap onto
                        # mirrored data.
                        oob = ((xt.min(axis=2) < -0.5) | (xt.max(axis=2) > self.W - 0.5) |
                               (yt.min(axis=2) < -0.5) | (yt.max(axis=2) > self.H - 0.5))
                        sc = cp.where(oob, cp.inf, sc)

                    jb = cp.argmin(sc, axis=0)                        # (n,)
                    sb = cp.take_along_axis(sc, jb[None, :], axis=0)[0]
                    imp = sb < best_s
                    best_s = cp.where(imp, sb, best_s)
                    best_du = cp.where(imp, dxc[jb], best_du)
                    best_dv = cp.where(imp, dyc[jb], best_dv)
                return best_s, best_du, best_dv

            best_u, best_v = p_act[:, 0].copy(), p_act[:, 1].copy()
            # Scored WITHOUT the out-of-bounds substitution on purpose. This
            # value only decides "is the current guess poor enough to search
            # from". Scoring an already-off-frame guess as infinitely bad sends
            # it into the rescue, which then finds some in-bounds speckle match
            # and hands back a confident wrong answer -- measured at 49 such
            # subsets on the off-frame regression, versus 0 when they are left
            # alone to be rejected by the bounds test at acceptance time.
            best_score, _, _ = _znssd_grid(None, p_act[:, 0], p_act[:, 1],
                                           np.zeros(1), np.zeros(1), mask_oob=False)

            # Absolute ZNSSD quality bar for 'this guess is poor, go search'.
            # Must NOT be tied to corr_cutoff: that is an acceptance
            # threshold, and a permissive one disables the rescue entirely.
            SEARCH_TRIGGER = 0.25
            need = best_score > SEARCH_TRIGGER
            if bool(need.any()):
                rows = cp.where(need)[0]

                # Rescue radius is its own parameter, NOT a function of
                # params.search_radius.
                #
                # search_radius sizes the NCC template search that seeds a
                # frame; reusing it here reads as though it also widens this
                # per-subset sweep, and it did feed a min(12, search_radius//4)
                # expression that silently ignored anything above 48. Widening
                # it is not the fix, though: measured on a synthetic pure
                # translation, taking this sweep out to +/-50 made subsets lock
                # onto false ZNSSD minima -- displacement error rose from 0.000
                # to 236 px and spurious gradients appeared where the truth is
                # exactly zero. Speckle is quasi-periodic, so a wide window
                # reliably contains minima as deep as the true one, and nothing
                # local can tell them apart.
                #
                # So the radius stays deliberately small and is now stated
                # outright. Frame-scale motion is the NCC fallback's job, not
                # this sweep's.
                R = int(max(1, getattr(self.params, "rescue_radius", 12)))

                base_u = p_act[rows, 0].copy()
                base_v = p_act[rows, 1].copy()

                g = np.arange(-R, R + 1, dtype=np.float64)
                gxg, gyg = np.meshgrid(g, g)
                s1, du1, dv1 = _znssd_grid(rows, base_u, base_v,
                                           gxg.ravel(), gyg.ravel())

                imp = s1 < best_score[rows]
                best_score[rows] = cp.where(imp, s1, best_score[rows])
                best_u[rows] = cp.where(imp, base_u + du1, base_u)
                best_v[rows] = cp.where(imp, base_v + dv1, base_v)

            # --- Bug 3 Fix: Reset gradients when integer search shifts significantly ---
            # Only during wavefront propagation (not warm-start). In warm-start,
            # the previous frame's gradients are a good starting point even if the
            # displacement shifted by several pixels (rigid-body chip motion).
            # During wavefront, a large shift means the parent is across a discontinuity.
            shift_mag = cp.sqrt((best_u - p_act[:, 0])**2 + (best_v - p_act[:, 1])**2)
            p_act[:, 0] = best_u; p_act[:, 1] = best_v
            if not warm_start:
                reset_mask = shift_mag > 4.0
                p_act[reset_mask, 2:6] = 0.0

            p_icgn_start = p_act.copy()

            converged = cp.zeros(N_act, dtype=bool)
            failed = cp.zeros(N_act, dtype=bool)

            # Track the best iterate seen, exactly as the CPU solver does.
            # Without this, a subset that reached an excellent correlation at
            # iteration 3 and then took one oversized step was written off
            # entirely: `failed` fed straight into the accept test, so its good
            # iterate was discarded. Measured against the CPU path on real
            # frames, that rejected subsets the CPU solved at ZNSSD ~0.13 --
            # and when it caught the wavefront seed, the whole frame returned
            # nothing and the driver fell back to a full NCC re-run.
            best_p = p_act.copy()
            best_cls = cp.full(N_act, cp.inf, dtype=cp.float64)

            for it in range(self.params.max_iter):
                mask_proc = ~(converged | failed)
                if not mask_proc.any(): break

                p_curr = p_act[mask_proc]
                xs_curr = ref_xs_act[mask_proc]
                ys_curr = ref_ys_act[mask_proc]

                x_prime = xs_curr + p_curr[:, 0:1] + p_curr[:, 2:3]*self.dx_sub + p_curr[:, 3:4]*self.dy_sub
                y_prime = ys_curr + p_curr[:, 1:2] + p_curr[:, 4:5]*self.dx_sub + p_curr[:, 5:6]*self.dy_sub

                coords_def = cp.stack([y_prime.ravel(), x_prime.ravel()], axis=0)
                g = map_coordinates(cur_coeff, coords_def, order=3, mode='mirror', prefilter=False).reshape(len(p_curr), self.N_px)

                g_mean = g.mean(axis=1, keepdims=True)
                g_c = g - g_mean
                sigma_g = cp.maximum(cp.sqrt((g_c**2).sum(axis=1, keepdims=True)), 1e-12)
                g_norm = g_c / sigma_g

                residual = g_norm - f_norm_act[mask_proc]

                # Score the CURRENT parameters before stepping away from them.
                # An iterate that has warped the subset off the image is scored
                # as infinitely bad so it can never win the best-iterate test --
                # its correlation is against mirrored, fabricated data.
                cls_iter = (residual ** 2).sum(axis=1)
                cls_iter = cp.where(self._out_of_bounds(x_prime, y_prime), cp.inf, cls_iter)
                proc_rows = cp.where(mask_proc)[0]
                imp = cls_iter < best_cls[proc_rows]
                if bool(imp.any()):
                    imp_rows = proc_rows[imp]
                    best_cls[imp_rows] = cls_iter[imp]
                    best_p[imp_rows] = p_curr[imp]

                b = cp.einsum('npi,np->ni', SD_act[mask_proc], residual)
                dp = cp.einsum('nij,nj->ni', H_inv_act[mask_proc], b)

                # --- Bug 2 Fix: Separate displacement and gradient divergence checks ---
                # Displacement (pixels) and strain gradients (dimensionless) have
                # different scales; mixing them in a single norm is meaningless.
                dp_norm = cp.linalg.norm(dp, axis=1)
                disp_norm = cp.sqrt(dp[:, 0]**2 + dp[:, 1]**2)
                grad_norm = cp.sqrt(dp[:, 2]**2 + dp[:, 3]**2 + dp[:, 4]**2 + dp[:, 5]**2)
                diverged = (disp_norm > 5.0) | (grad_norm > 8.0) | cp.isnan(dp_norm)

                if diverged.any():
                    failed_global = failed.copy()
                    failed_global[cp.where(mask_proc)[0][diverged]] = True
                    failed = failed_global
                    dp = dp[~diverged]
                    p_curr = p_curr[~diverged]
                    if len(dp) == 0: continue

                dp0, dp1, dp2, dp3, dp4, dp5 = dp[:,0], dp[:,1], dp[:,2], dp[:,3], dp[:,4], dp[:,5]
                a2, b2, c2 = 1.0 + dp2, dp3, dp0
                d2, e2, f2 = dp4, 1.0 + dp5, dp1

                det2 = a2 * e2 - b2 * d2
                sing = cp.abs(det2) < 1e-12
                if sing.any(): det2[sing] = 1.0

                inv_det = 1.0 / det2
                i_a, i_b = e2 * inv_det, -b2 * inv_det
                i_c = (b2 * f2 - c2 * e2) * inv_det
                i_d, i_e = -d2 * inv_det, a2 * inv_det
                i_f = (c2 * d2 - a2 * f2) * inv_det

                p0, p1, p2, p3, p4, p5 = p_curr[:,0], p_curr[:,1], p_curr[:,2], p_curr[:,3], p_curr[:,4], p_curr[:,5]
                a1, b1, c1 = 1.0 + p2, p3, p0
                d1, e1, f1 = p4, 1.0 + p5, p1

                p_new = cp.empty_like(p_curr)
                p_new[:,0] = a1 * i_c + b1 * i_f + c1
                p_new[:,1] = d1 * i_c + e1 * i_f + f1
                p_new[:,2] = (a1 * i_a + b1 * i_d) - 1.0
                p_new[:,3] = a1 * i_b + b1 * i_e
                p_new[:,4] = d1 * i_a + e1 * i_d
                p_new[:,5] = (d1 * i_b + e1 * i_e) - 1.0

                valid_mask_idx = cp.where(mask_proc)[0][~diverged]
                p_act[valid_mask_idx] = p_new

                conv_global = converged.copy()
                conv_global[valid_mask_idx] = dp_norm[~diverged] < self.params.conv_tol
                converged = conv_global

            # Score the final parameters too -- the last accepted update is
            # never scored inside the loop -- then keep whichever of {last,
            # best-seen} correlates better.
            x_prime = ref_xs_act + p_act[:, 0:1] + p_act[:, 2:3]*self.dx_sub + p_act[:, 3:4]*self.dy_sub
            y_prime = ref_ys_act + p_act[:, 1:2] + p_act[:, 4:5]*self.dx_sub + p_act[:, 5:6]*self.dy_sub

            coords_def = cp.stack([y_prime.ravel(), x_prime.ravel()], axis=0)
            g = map_coordinates(cur_coeff, coords_def, order=3, mode='mirror', prefilter=False).reshape(N_act, self.N_px)
            g_c = g - g.mean(axis=1, keepdims=True)
            g_norm = g_c / cp.maximum(cp.sqrt((g_c**2).sum(axis=1, keepdims=True)), 1e-12)
            final_cls = ((g_norm - f_norm_act) ** 2).sum(axis=1)
            final_cls = cp.where(cp.isfinite(final_cls), final_cls, cp.inf)

            final_cls = cp.where(self._out_of_bounds(x_prime, y_prime), cp.inf, final_cls)

            better = final_cls < best_cls
            cls_act = cp.where(better, final_cls, best_cls)
            p_act = cp.where(better[:, None], p_act, best_p)

            # Re-test the parameters actually being returned: the winner of the
            # best/final comparison may differ from either candidate tested above.
            xw = ref_xs_act + p_act[:, 0:1] + p_act[:, 2:3]*self.dx_sub + p_act[:, 3:4]*self.dy_sub
            yw = ref_ys_act + p_act[:, 1:2] + p_act[:, 4:5]*self.dx_sub + p_act[:, 5:6]*self.dy_sub
            oob = ref_oob | self._out_of_bounds(xw, yw)

            # Relaxed the jump threshold to reduce erosion (was subset_spacing + 1, 5.0)
            cutoff_disp = float(max(self.params.subset_spacing * 1.5, 10.0))
            jump_x = cp.abs(p_act[:, 0] - p_icgn_start[:, 0])
            jump_y = cp.abs(p_act[:, 1] - p_icgn_start[:, 1])

            mask_failed_cls = cls_act >= self.params.corr_cutoff
            mask_failed_jump = (jump_x >= cutoff_disp) | (jump_y >= cutoff_disp)

            # `failed` only means "stopped iterating early" now. Whether the
            # subset is usable is decided by the correlation of the parameters
            # actually being returned, not by how the iteration ended.
            accepted = ~mask_failed_cls & ~mask_failed_jump & ~oob

            if DEBUG_BATCHES:
                print(f"\n[DEBUG] --- BATCH {batch_count} | Mode: {'Warm-Start' if warm_start else 'Wavefront'} ---")
                print(f"[DEBUG] Processing {N_act} subsets. Accepted {int(accepted.sum().get())}. "
                      f"Rejections: {int(mask_failed_jump.sum().get())} Jump, "
                      f"{int(mask_failed_cls.sum().get())} ZNSSD, "
                      f"{int(oob.sum().get())} off-frame "
                      f"({int(failed.sum().get())} stopped iterating early).")
                sys.stdout.flush()

            # Only commit parameters that passed. Rejected points currently
            # always get re-seeded from a parent (or never read again) before
            # p_global is used, so this is not load-bearing today -- but writing
            # diverged iterates into the array that seeds the next frame's warm
            # start is a trap waiting for the next change to the state machine.
            success_idx = active_indices[accepted]
            fail_idx = active_indices[~accepted]

            self.p_global[success_idx] = p_act[accepted]
            self.cls_global[success_idx] = cls_act[accepted]
            self.cls_global[fail_idx] = cp.nan

            self.state[success_idx] = 2
            # Return failures to the pending pool while they still have retry
            # budget, so a later, more reliable neighbour can re-seed them.
            self.retry[fail_idx] += 1
            retryable = fail_idx[self.retry[fail_idx] < MAX_RETRY]
            exhausted = fail_idx[self.retry[fail_idx] >= MAX_RETRY]
            self.state[retryable] = 0
            self.state[exhausted] = -1

            if len(success_idx) > 0:
                n_idx, p_idx = get_neighbors(success_idx)
                valid_n = self.state[n_idx] == 0
                n_sel, p_sel = n_idx[valid_n], p_idx[valid_n]
                if len(n_sel) == 0:
                    continue
                # cp.unique(..., return_index=True) returns whichever parent
                # happened to come first in array order. Reliability-guided DIC
                # requires the *lowest-ZNSSD* parent, otherwise this degrades
                # into a plain breadth-first flood fill and poor estimates get
                # to seed their neighbours.
                pcls = cp.nan_to_num(self.cls_global[p_sel], nan=cp.inf, posinf=1e30)
                order = cp.lexsort(cp.stack([pcls, n_sel.astype(cp.float64)]))
                n_sorted, p_sorted = n_sel[order], p_sel[order]
                first = cp.empty(len(n_sorted), dtype=bool)
                first[0] = True
                first[1:] = n_sorted[1:] != n_sorted[:-1]
                unique_n, selected_parents = n_sorted[first], p_sorted[first]

                self.state[unique_n] = 1
                p_par = self.p_global[selected_parents]

                dx = self.gx_gpu[unique_n] - self.gx_gpu[selected_parents]
                dy = self.gy_gpu[unique_n] - self.gy_gpu[selected_parents]

                p_child = p_par.copy()
                p_child[:, 0] += p_par[:, 2] * dx + p_par[:, 3] * dy
                p_child[:, 1] += p_par[:, 4] * dx + p_par[:, 5] * dy
                self.p_global[unique_n] = p_child

        p_cpu = self.p_global.get()
        cls_cpu = self.cls_global.get()
        state_cpu = self.state.get()

        u_f, v_f = np.full((self.H, self.W), np.nan), np.full((self.H, self.W), np.nan)
        du_dx, du_dy = np.full((self.H, self.W), np.nan), np.full((self.H, self.W), np.nan)
        dv_dx, dv_dy = np.full((self.H, self.W), np.nan), np.full((self.H, self.W), np.nan)
        corr_f = np.full((self.H, self.W), np.nan)

        solved = state_cpu == 2
        gx_s, gy_s = self.gx_flat[solved], self.gy_flat[solved]

        u_f[gy_s, gx_s] = p_cpu[solved, 0]; v_f[gy_s, gx_s] = p_cpu[solved, 1]
        du_dx[gy_s, gx_s] = p_cpu[solved, 2]; du_dy[gy_s, gx_s] = p_cpu[solved, 3]
        dv_dx[gy_s, gx_s] = p_cpu[solved, 4]; dv_dy[gy_s, gx_s] = p_cpu[solved, 5]
        corr_f[gy_s, gx_s] = cls_cpu[solved]

        return u_f, v_f, du_dx, du_dy, dv_dx, dv_dy, corr_f
