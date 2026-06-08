"""
rg_dic.py — Parallel domain-decomposed RG-DIC matching Ncorr's multithreaded scheme.
"""
from __future__ import annotations
import heapq, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from .bspline import BSplineInterpolator, circular_subset, image_gradient
from .ncc import ncc_initial_guess
from .icgn import precompute_subset, run_icgn


@dataclass
class DICParams:
    subset_radius:  int   = 21
    subset_spacing: int   = 3
    strain_window:  int   = 15
    max_iter:       int   = 50
    conv_tol:       float = 1e-6
    corr_cutoff:    float = 2.0
    search_radius:  int   = 50


@dataclass
class DICResult:
    u: np.ndarray; v: np.ndarray
    du_dx: np.ndarray; du_dy: np.ndarray
    dv_dx: np.ndarray; dv_dy: np.ndarray
    corr: np.ndarray; analyzed: np.ndarray
    grid_x: np.ndarray; grid_y: np.ndarray


def run_rg_dic(
    ref_image: np.ndarray,
    cur_image: np.ndarray,
    roi_mask:  np.ndarray,
    params:    DICParams,
    seed_xy:   Optional[tuple] = None,
    progress_cb: Optional[Callable] = None,
    cancel_flag: Optional[list] = None,
    guess_u: float = 0.0,
    guess_v: float = 0.0,
) -> DICResult:
    if cancel_flag is None:
        cancel_flag = [False]

    H, W = ref_image.shape
    grid_x, grid_y = _build_grid(roi_mask, params.subset_radius, params.subset_spacing)
    n_total = len(grid_x)
    if n_total == 0:
        raise ValueError("No valid subset centres. Reduce subset_radius or subset_spacing.")

    _report(progress_cb, 0.0, "Precomputing B-spline coefficients…")
    ref_f64       = ref_image.astype(np.float64)
    cur_image_raw = cur_image.astype(np.float64)
    cur_interp    = BSplineInterpolator(cur_image_raw)
    grad_x, grad_y = image_gradient(ref_f64)
    dx_sub, dy_sub = circular_subset(params.subset_radius)
    if seed_xy is None:
        ys_roi, xs_roi = np.where(roi_mask)
        seed_xy = (int(xs_roi.mean()), int(ys_roi.mean()))

    n_workers = os.cpu_count() or 4
    if n_total < 200 or n_workers < 2:
        n_workers = 1

    order  = np.lexsort((grid_x, grid_y))
    gx_s, gy_s = grid_x[order], grid_y[order]
    splits = np.array_split(np.arange(n_total), n_workers)
    domains = [(gx_s[s], gy_s[s]) for s in splits if len(s) > 0]
    n_workers = len(domains)

    domain_seeds = []
    for i, (dxg, dyg) in enumerate(domains):
        if i == 0:
            best = int(np.argmin((dxg - seed_xy[0])**2 + (dyg - seed_xy[1])**2))
        else:
            cx, cy = dxg.mean(), dyg.mean()
            best = int(np.argmin((dxg - cx)**2 + (dyg - cy)**2))
        domain_seeds.append((int(dxg[best]), int(dyg[best])))

    shape = (H, W)

    import threading
    shared_state = {"done": 0, "total": n_total}
    progress_lock = threading.Lock()

    def global_cb(frac, msg):
        if progress_cb:
            try:
                progress_cb(0.05 + 0.95 * frac, msg)
            except Exception:
                pass

    args_list = [
        (ref_f64, cur_image_raw, cur_interp, grad_x, grad_y,
         dxg, dyg, dx_sub, dy_sub,
         params, seed, shape, cancel_flag, global_cb, shared_state, progress_lock, guess_u, guess_v)
        for i, ((dxg, dyg), seed) in enumerate(zip(domains, domain_seeds))
    ]

    results = [None] * n_workers
    if n_workers == 1:
        results[0] = _run_domain(*args_list[0])
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            fmap = {pool.submit(_run_domain, *a): i for i, a in enumerate(args_list)}
            for fut in as_completed(fmap):
                results[fmap[fut]] = fut.result()

    u_f = np.full(shape, np.nan); v_f = np.full(shape, np.nan)
    du_dx_f = np.full(shape, np.nan); du_dy_f = np.full(shape, np.nan)
    dv_dx_f = np.full(shape, np.nan); dv_dy_f = np.full(shape, np.nan)
    corr_f  = np.full(shape, np.nan); ana = np.zeros(shape, dtype=bool)
    for r in results:
        if r is None: continue
        m = r["analyzed"]
        u_f[m]=r["u"][m]; v_f[m]=r["v"][m]
        du_dx_f[m]=r["du_dx"][m]; du_dy_f[m]=r["du_dy"][m]
        dv_dx_f[m]=r["dv_dx"][m]; dv_dy_f[m]=r["dv_dy"][m]
        corr_f[m]=r["corr"][m]; ana[m]=True

    _report(progress_cb, 1.0, "Done.")
    return DICResult(
        u=u_f, v=v_f, du_dx=du_dx_f, du_dy=du_dy_f,
        dv_dx=dv_dx_f, dv_dy=dv_dy_f,
        corr=corr_f, analyzed=ana,
        grid_x=grid_x, grid_y=grid_y,
    )


def _run_domain(ref_f64, cur_image_raw, cur_interp, grad_x, grad_y,
                domain_gx, domain_gy, dx_sub, dy_sub,
                params, seed_xy, shape, cancel_flag, progress_cb,
                shared_state, lock, guess_u, guess_v):

    step = params.subset_spacing
    cutoff_disp = float(step + 1)

    grid_map = {(int(domain_gx[i]), int(domain_gy[i])): i
                for i in range(len(domain_gx))}

    u_f = np.full(shape, np.nan); v_f = np.full(shape, np.nan)
    du_dx = np.full(shape, np.nan); du_dy = np.full(shape, np.nan)
    dv_dx = np.full(shape, np.nan); dv_dy = np.full(shape, np.nan)
    corr_f = np.full(shape, np.nan); ana = np.zeros(shape, dtype=bool)

    sx, sy = _snap(seed_xy[0], seed_xy[1], domain_gx, domain_gy)
    seed_idx = grid_map.get((sx, sy), 0)

    u0, v0, _ = ncc_initial_guess(ref_f64, cur_image_raw, sx, sy,
                                  params.subset_radius, params.search_radius,
                                  guess_u, guess_v)

    p0 = np.array([u0, v0, 0., 0., 0., 0.])
    sd = precompute_subset(ref_f64, grad_x, grad_y, sx, sy, dx_sub, dy_sub)
    p0, cls0, _ = run_icgn(cur_interp, sd, p0, params.max_iter, params.conv_tol)
    _store(u_f, v_f, du_dx, du_dy, dv_dx, dv_dy, corr_f, ana, sx, sy, p0, cls0)

    done = np.zeros(len(domain_gx), dtype=bool)
    done[seed_idx] = True

    with lock:
        shared_state["done"] += 1

    heap = [(cls0, seed_idx, p0.copy())]

    local_steps = 0
    report_interval = max(1, shared_state["total"] // 100)

    while heap and not cancel_flag[0]:
        cls_p, pidx, p_par = heapq.heappop(heap)
        px, py = int(domain_gx[pidx]), int(domain_gy[pidx])

        for nx, ny in [(px + step, py), (px - step, py), (px, py + step), (px, py - step)]:
            nbidx = grid_map.get((nx, ny))
            if nbidx is None or done[nbidx]:
                continue

            ddx, ddy = float(nx - px), float(ny - py)
            u_i = p_par[0] + p_par[2] * ddx + p_par[3] * ddy
            v_i = p_par[1] + p_par[4] * ddx + p_par[5] * ddy
            p_i = np.array([u_i, v_i, p_par[2], p_par[3], p_par[4], p_par[5]])

            sd_nb = precompute_subset(ref_f64, grad_x, grad_y, nx, ny, dx_sub, dy_sub)
            p_opt, cls_opt, _ = run_icgn(cur_interp, sd_nb, p_i,
                                         params.max_iter, params.conv_tol)
            done[nbidx] = True
            local_steps += 1

            if (cls_opt < params.corr_cutoff and
                    abs(p_opt[0] - u_i) < cutoff_disp and
                    abs(p_opt[1] - v_i) < cutoff_disp):
                _store(u_f, v_f, du_dx, du_dy, dv_dx, dv_dy,
                       corr_f, ana, nx, ny, p_opt, cls_opt)
                heapq.heappush(heap, (cls_opt, nbidx, p_opt.copy()))

            if local_steps >= report_interval:
                with lock:
                    shared_state["done"] += local_steps
                    curr_done = shared_state["done"]
                local_steps = 0
                _report(progress_cb, curr_done / shared_state["total"],
                        f"Analysing … {curr_done}/{shared_state['total']}")

    if local_steps > 0:
        with lock:
            shared_state["done"] += local_steps
            curr_done = shared_state["done"]
        _report(progress_cb, curr_done / shared_state["total"],
                f"Analysing … {curr_done}/{shared_state['total']}")

    return dict(u=u_f, v=v_f, du_dx=du_dx, du_dy=du_dy,
                dv_dx=dv_dx, dv_dy=dv_dy, corr=corr_f, analyzed=ana)

def _build_grid(roi_mask, radius, spacing):
    H, W = roi_mask.shape
    ys = np.arange(radius, H-radius, spacing, dtype=int)
    xs = np.arange(radius, W-radius, spacing, dtype=int)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    m = roi_mask[gy, gx]
    return gx[m], gy[m]


def _snap(x, y, gx, gy):
    i = int(np.argmin((gx-x)**2+(gy-y)**2))
    return int(gx[i]), int(gy[i])


def _store(u_f,v_f,du_dx,du_dy,dv_dx,dv_dy,corr_f,ana,cx,cy,p,cls):
    u_f[cy,cx]=p[0]; v_f[cy,cx]=p[1]
    du_dx[cy,cx]=p[2]; du_dy[cy,cx]=p[3]
    dv_dx[cy,cx]=p[4]; dv_dy[cy,cx]=p[5]
    corr_f[cy,cx]=cls; ana[cy,cx]=True


def _report(cb, frac, msg):
    if cb:
        try: cb(frac, msg)
        except Exception: pass