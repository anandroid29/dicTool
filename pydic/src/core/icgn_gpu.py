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

class GPUWavefrontDIC:
    def __init__(self, params):
        if not _HAS_CUPY:
            raise RuntimeError("CuPy is required for GPU execution.")
        self.params = params
        self._initialized = False

        self.state = None
        self.p_global = None
        self.cls_global = None

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

        ref_f64 = ref_image.astype(np.float64)
        ref_coeff_cpu = scipy.ndimage.spline_filter(ref_f64, order=3, mode='mirror')
        self.ref_coeff = cp.asarray(ref_coeff_cpu, dtype=cp.float64)

        dy_sub, dx_sub = np.mgrid[-self.r:self.r+1, -self.r:self.r+1]
        mask_sub = (dx_sub**2 + dy_sub**2) <= self.r**2
        self.dx_sub = cp.asarray(dx_sub[mask_sub], dtype=cp.float64)
        self.dy_sub = cp.asarray(dy_sub[mask_sub], dtype=cp.float64)
        self.N_px = len(self.dx_sub)

        self.gx_gpu = cp.asarray(self.gx_flat, dtype=cp.float64)
        self.gy_gpu = cp.asarray(self.gy_flat, dtype=cp.float64)

        self._initialized = True

    def solve_frame(self, cur_image: np.ndarray, seed_idx: int = -1, seed_p: np.ndarray = None, warm_start: bool = False):
        if not self._initialized:
            raise RuntimeError("Must precompute reference before solving frames.")

        cur_f64 = cur_image.astype(np.float64)
        cur_coeff_cpu = scipy.ndimage.spline_filter(cur_f64, order=3, mode='mirror')
        cur_coeff = cp.asarray(cur_coeff_cpu, dtype=cp.float64)
        cur_gpu = cp.asarray(cur_f64, dtype=cp.float64)

        if not warm_start:
            self.state = cp.zeros(self.N_total, dtype=cp.int8)
            self.state[~cp.asarray(self.valid_mask)] = -1
            self.p_global = cp.zeros((self.N_total, 6), dtype=cp.float64)
            self.cls_global = cp.full(self.N_total, cp.nan, dtype=cp.float64)

            self.state[seed_idx] = 1
            self.p_global[seed_idx] = cp.asarray(seed_p, dtype=cp.float64)
        else:
            self.state[self.state == 2] = 1
            valid_gpu = cp.asarray(self.valid_mask)
            self.state[(self.state == -1) & valid_gpu] = 0

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

            ref_xs_act = gx_act[:, None] + self.dx_sub[None, :]
            ref_ys_act = gy_act[:, None] + self.dy_sub[None, :]

            coords_ref = cp.stack([ref_ys_act.ravel(), ref_xs_act.ravel()], axis=0)

            f = map_coordinates(self.ref_coeff, coords_ref, order=3, mode='mirror', prefilter=False).reshape(N_act, self.N_px)
            f_mean = f.mean(axis=1, keepdims=True)
            f_c = f - f_mean
            sigma_f_act = cp.maximum(cp.sqrt((f_c**2).sum(axis=1, keepdims=True)), 1e-12)
            f_norm_act = f_c / sigma_f_act

            h = 1e-3
            cx_p = cp.stack([ref_ys_act.ravel(), ref_xs_act.ravel() + h], axis=0)
            cx_m = cp.stack([ref_ys_act.ravel(), ref_xs_act.ravel() - h], axis=0)
            fx = (map_coordinates(self.ref_coeff, cx_p, order=3, mode='mirror', prefilter=False) -
                  map_coordinates(self.ref_coeff, cx_m, order=3, mode='mirror', prefilter=False)) / (2*h)
            fx = fx.reshape(N_act, self.N_px)

            cy_p = cp.stack([ref_ys_act.ravel() + h, ref_xs_act.ravel()], axis=0)
            cy_m = cp.stack([ref_ys_act.ravel() - h, ref_xs_act.ravel()], axis=0)
            fy = (map_coordinates(self.ref_coeff, cy_p, order=3, mode='mirror', prefilter=False) -
                  map_coordinates(self.ref_coeff, cy_m, order=3, mode='mirror', prefilter=False)) / (2*h)
            fy = fy.reshape(N_act, self.N_px)

            SD_act = cp.empty((N_act, self.N_px, 6), dtype=cp.float64)
            SD_act[:, :, 0] = fx; SD_act[:, :, 1] = fy
            SD_act[:, :, 2] = fx * self.dx_sub; SD_act[:, :, 3] = fx * self.dy_sub
            SD_act[:, :, 4] = fy * self.dx_sub; SD_act[:, :, 5] = fy * self.dy_sub
            SD_act /= sigma_f_act[:, :, None]

            H_mat = cp.matmul(SD_act.transpose(0, 2, 1), SD_act)
            H_mat += cp.eye(6, dtype=cp.float64).reshape(1, 6, 6) * 1e-6
            H_inv_act = cp.linalg.inv(H_mat)

            sr = 3
            shifts = cp.arange(-sr, sr + 1)
            sy_grid, sx_grid = cp.meshgrid(shifts, shifts)
            sy_flat, sx_flat = sy_grid.ravel(), sx_grid.ravel()

            best_u, best_v = p_act[:, 0].copy(), p_act[:, 1].copy()
            best_score = cp.full(N_act, cp.inf, dtype=cp.float64)

            for sx_shift, sy_shift in zip(sx_flat, sy_flat):
                x_test = ref_xs_act + p_act[:, 0:1] + sx_shift
                y_test = ref_ys_act + p_act[:, 1:2] + sy_shift
                coords_test = cp.stack([y_test.ravel(), x_test.ravel()], axis=0)

                g_test = map_coordinates(cur_gpu, coords_test, order=1, mode='mirror', prefilter=False).reshape(N_act, self.N_px)
                g_test_c = g_test - g_test.mean(axis=1, keepdims=True)
                g_test_norm = g_test_c / cp.maximum(cp.sqrt((g_test_c**2).sum(axis=1, keepdims=True)), 1e-12)

                score = ((g_test_norm - f_norm_act)**2).sum(axis=1)
                improve = score < best_score
                best_score[improve] = score[improve]
                best_u[improve] = p_act[improve, 0] + sx_shift
                best_v[improve] = p_act[improve, 1] + sy_shift

            p_act[:, 0] = best_u; p_act[:, 1] = best_v

            p_icgn_start = p_act.copy()

            converged = cp.zeros(N_act, dtype=bool)
            failed = cp.zeros(N_act, dtype=bool)

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
                b = cp.einsum('npi,np->ni', SD_act[mask_proc], residual)
                dp = cp.einsum('nij,nj->ni', H_inv_act[mask_proc], b)

                dp_norm = cp.linalg.norm(dp, axis=1)
                diverged = (dp_norm > 3.0) | cp.isnan(dp_norm)

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

            final_eval_mask = ~failed
            cls_act = cp.full(N_act, cp.inf, dtype=cp.float64)

            if final_eval_mask.any():
                p_final = p_act[final_eval_mask]
                xs_final = ref_xs_act[final_eval_mask]
                ys_final = ref_ys_act[final_eval_mask]

                x_prime = xs_final + p_final[:, 0:1] + p_final[:, 2:3]*self.dx_sub + p_final[:, 3:4]*self.dy_sub
                y_prime = ys_final + p_final[:, 1:2] + p_final[:, 4:5]*self.dx_sub + p_final[:, 5:6]*self.dy_sub

                coords_def = cp.stack([y_prime.ravel(), x_prime.ravel()], axis=0)
                g = map_coordinates(cur_coeff, coords_def, order=3, mode='mirror', prefilter=False).reshape(len(p_final), self.N_px)
                g_c = g - g.mean(axis=1, keepdims=True)
                g_norm = g_c / cp.maximum(cp.sqrt((g_c**2).sum(axis=1, keepdims=True)), 1e-12)
                res = g_norm - f_norm_act[final_eval_mask]
                cls_act[final_eval_mask] = (res**2).sum(axis=1)

            cutoff_disp = float(max(self.params.subset_spacing + 1, 5.0))
            jump_x = cp.abs(p_act[:, 0] - p_icgn_start[:, 0])
            jump_y = cp.abs(p_act[:, 1] - p_icgn_start[:, 1])

            mask_failed_cls = cls_act >= self.params.corr_cutoff
            mask_failed_jump = (jump_x >= cutoff_disp) | (jump_y >= cutoff_disp)

            accepted = ~failed & ~mask_failed_cls & ~mask_failed_jump

            if batch_count <= 2:
                print(f"\n[DEBUG] --- BATCH {batch_count} | Mode: {'Warm-Start' if warm_start else 'Wavefront'} ---")
                print(f"[DEBUG] Processing {N_act} subsets. Rejections: {failed.sum().get()} IC-GN, {(final_eval_mask & mask_failed_jump).sum().get()} Jump, {(final_eval_mask & mask_failed_cls).sum().get()} ZNSSD.")
                sys.stdout.flush()

            self.p_global[active_indices] = p_act
            self.cls_global[active_indices] = cls_act

            success_idx = active_indices[accepted]
            fail_idx = active_indices[~accepted]

            self.state[success_idx] = 2
            self.state[fail_idx] = -1

            if len(success_idx) > 0:
                n_idx, p_idx = get_neighbors(success_idx)
                valid_n = self.state[n_idx] == 0
                unique_n, unique_indices = cp.unique(n_idx[valid_n], return_index=True)
                selected_parents = p_idx[valid_n][unique_indices]

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