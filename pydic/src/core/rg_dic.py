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
from .icgn import precompute_subset, run_icgn, infer_intensity_scale


@dataclass
class DICParams:
    subset_radius:  int   = 21
    subset_spacing: int   = 3
    strain_window:  int   = 15
    max_iter:       int   = 50
    # In pixels of subset-edge motion per iteration (see icgn.run_icgn).
    conv_tol:       float = 1e-3
    # ZNSSD of two unit-normalised subsets lies in [0, 4] and equals
    # 2*(1 - ZNCC). A cutoff of 2.0 therefore accepts ZNCC >= 0, i.e. it
    # accepts everything. 0.3 corresponds to ZNCC >= 0.85.
    corr_cutoff:    float = 0.30
    # Half-width of the NCC template search used to SEED a frame.
    search_radius:  int   = 50
    # Half-width of the GPU solver's per-subset integer-shift rescue sweep.
    # Deliberately separate from search_radius and deliberately small: a wide
    # sweep over quasi-periodic speckle locks onto false ZNSSD minima. Raising
    # this trades robustness for reach -- see icgn_gpu.py for the measurement.
    rescue_radius:  int   = 12
    dynamic_roi:    str   = "Hybrid"
    # Texture threshold for the dynamic ROI, normalised to [0, 1]. None keeps
    # the automatic (Otsu on the reference frame) choice.
    dynamic_roi_threshold: Optional[float] = None
    # Drop connected regions smaller than this fraction of the largest one.
    dynamic_roi_min_area_frac: float = 0.02
    # Keep regions the texture metric rejected when they are fully enclosed by
    # kept material -- a hole inside valid specimen is a local dropout, not a
    # gap in the material.
    dynamic_roi_fill_holes: bool = True
    # Restrict subset pixels to the ROI. Correct when the ROI outlines a
    # material boundary (specimen edge, hole, grip). Turn off when the ROI is
    # just a crop of a larger uniform speckle field, where the surrounding
    # pixels are valid data and masking only throws away support.
    mask_subsets_to_roi: bool = True
    # 1 = affine shape function (6 params), 2 = quadratic (12 params).
    # Second order captures curvature inside the subset and reduces systematic
    # error where the strain gradient is high, at the cost of substantially
    # higher noise sensitivity -- it needs a well-textured, reasonably large
    # subset to be worth using.
    shape_order: int = 1

    # strain_window is a half-width in PIXELS, but the least-squares plane fit
    # behind every strain field only ever sees correlation GRID points, which
    # sit subset_spacing apart. The number of points actually feeding the fit is
    # therefore (2*strain_window/subset_spacing + 1)^2, and strain.py needs at
    # least 6 of them. Small window + large spacing silently produced an
    # all-NaN strain field with no error anywhere -- the strain views just came
    # up blank. Clamp instead, and say so.
    MIN_STRAIN_PTS_PER_AXIS = 5

    def effective_strain_window(self, warn: bool = True) -> int:
        need = int(np.ceil((self.MIN_STRAIN_PTS_PER_AXIS - 1) / 2.0)) * max(1, int(self.subset_spacing))
        sw = int(self.strain_window)
        if sw >= need:
            return sw
        if warn and not getattr(self, "_strain_window_warned", False):
            print(f"[Params] strain_window={sw} px spans only "
                  f"{2 * sw // max(1, self.subset_spacing) + 1} grid points at "
                  f"subset_spacing={self.subset_spacing}; too few for a strain fit. "
                  f"Using {need} px instead.")
            object.__setattr__(self, "_strain_window_warned", True)
        return need


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
    use_gpu = False
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
    # Occlusion cutoffs in icgn.py are fractions of the intensity range, so the
    # solver has to be told what range this data actually uses.
    intensity_scale = infer_intensity_scale(ref_f64, cur_image_raw)
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
                progress_cb(frac, msg)
            except Exception:
                pass

    args_list = [
        (ref_f64, cur_image_raw, cur_interp, grad_x, grad_y,
         dxg, dyg, dx_sub, dy_sub,
         params, seed, shape, cancel_flag, global_cb, shared_state, progress_lock,
         guess_u, guess_v, roi_mask, intensity_scale)
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
                shared_state, lock, guess_u, guess_v, roi_mask=None,
                intensity_scale=255.0):

    step = params.subset_spacing
    cutoff_disp = float(step + 1)
    n_pts = len(domain_gx)

    grid_map = {(int(domain_gx[i]), int(domain_gy[i])): i for i in range(n_pts)}

    u_f = np.full(shape, np.nan); v_f = np.full(shape, np.nan)
    du_dx = np.full(shape, np.nan); du_dy = np.full(shape, np.nan)
    dv_dx = np.full(shape, np.nan); dv_dy = np.full(shape, np.nan)
    corr_f = np.full(shape, np.nan); ana = np.zeros(shape, dtype=bool)

    solved   = np.zeros(n_pts, dtype=bool)
    attempts = np.zeros(n_pts, dtype=np.int8)
    best_cls = np.full(n_pts, np.inf)

    # precompute_subset is pure w.r.t. the reference image, so cache it. With
    # retries enabled a point can be visited several times and this keeps the
    # extra Cholesky factorisations from showing up in the runtime.
    sd_cache: dict[int, object] = {}

    def get_sd(idx, x, y):
        sd = sd_cache.get(idx)
        if sd is None:
            sd = precompute_subset(ref_f64, grad_x, grad_y, x, y, dx_sub, dy_sub,
                                   roi_mask if params.mask_subsets_to_roi else None,
                                   order=int(getattr(params, 'shape_order', 1)),
                                   intensity_scale=intensity_scale)
            sd_cache[idx] = sd
        return sd

    def try_point(idx, p_guess, disp_limit):
        """Run IC-GN at grid point idx. Returns (accepted, p, cls)."""
        x, y = int(domain_gx[idx]), int(domain_gy[idx])
        sd_nb = get_sd(idx, x, y)
        if not sd_nb.valid:
            return False, None, np.inf
        p_opt, cls_opt, conv = run_icgn(cur_interp, sd_nb, p_guess,
                                        params.max_iter, params.conv_tol,
                                        intensity_scale=intensity_scale)
        good = (cls_opt < params.corr_cutoff and
                abs(p_opt[0] - p_guess[0]) < disp_limit and
                abs(p_opt[1] - p_guess[1]) < disp_limit)
        return bool(good), p_opt, cls_opt

    def commit(idx, p_opt, cls_opt):
        x, y = int(domain_gx[idx]), int(domain_gy[idx])
        _store(u_f, v_f, du_dx, du_dy, dv_dx, dv_dy, corr_f, ana, x, y, p_opt, cls_opt)
        solved[idx] = True
        best_cls[idx] = cls_opt

    # ---------------- seed ----------------
    # A domain whose seed is bad produces nothing at all, so try several
    # candidates rather than trusting the first one unconditionally.
    sx, sy = _snap(seed_xy[0], seed_xy[1], domain_gx, domain_gy)
    seed_idx = grid_map.get((sx, sy), 0)

    seed_order = [seed_idx]
    if n_pts > 1:
        # fall back to the highest-contrast subsets in the domain: contrast is
        # what actually determines whether a seed can converge
        contrast = np.array([
            ref_f64[max(0, int(domain_gy[i]) - 4):int(domain_gy[i]) + 5,
                    max(0, int(domain_gx[i]) - 4):int(domain_gx[i]) + 5].std()
            for i in range(0, n_pts, max(1, n_pts // 400))
        ])
        cand = np.arange(0, n_pts, max(1, n_pts // 400))[np.argsort(-contrast)]
        seed_order += [int(c) for c in cand[:12] if int(c) != seed_idx]

    seeded = False
    for cand_idx in seed_order:
        if cancel_flag[0]:
            break
        cx_, cy_ = int(domain_gx[cand_idx]), int(domain_gy[cand_idx])
        u0, v0, _ = ncc_initial_guess(ref_f64, cur_image_raw, cx_, cy_,
                                      params.subset_radius, params.search_radius,
                                      guess_u, guess_v)
        p0_guess = np.array([u0, v0, 0., 0., 0., 0.])
        sd0 = get_sd(cand_idx, cx_, cy_)
        if not sd0.valid:
            continue
        p0, cls0, conv0 = run_icgn(cur_interp, sd0, p0_guess,
                                   params.max_iter, params.conv_tol,
                                   intensity_scale=intensity_scale)
        # The seed sets the initial guess for every other point in the domain,
        # so hold it to a stricter standard than ordinary points.
        if conv0 and cls0 < min(params.corr_cutoff, 0.5):
            commit(cand_idx, p0, cls0)
            seed_idx = cand_idx
            seeded = True
            break

    if not seeded:
        return dict(u=u_f, v=v_f, du_dx=du_dx, du_dy=du_dy,
                    dv_dx=dv_dx, dv_dy=dv_dy, corr=corr_f, analyzed=ana)

    with lock:
        shared_state["done"] += 1

    heap = [(best_cls[seed_idx], seed_idx, p_global_copy(u_f, v_f, du_dx, du_dy,
                                                         dv_dx, dv_dy,
                                                         int(domain_gx[seed_idx]),
                                                         int(domain_gy[seed_idx])))]

    local_steps = 0
    report_interval = max(1, shared_state["total"] // 100)
    MAX_ATTEMPTS = 3

    def flush_progress(force=False):
        nonlocal local_steps
        if local_steps and (force or local_steps >= report_interval):
            with lock:
                shared_state["done"] += local_steps
                curr = shared_state["done"]
            _report(progress_cb, curr / shared_state["total"],
                    f"Analysing … {curr}/{shared_state['total']}")
            local_steps = 0

    # ---------------- reliability-guided propagation ----------------
    while heap and not cancel_flag[0]:
        cls_p, pidx, p_par = heapq.heappop(heap)
        if cls_p > best_cls[pidx] + 1e-12:
            continue  # stale heap entry, a better estimate for this point exists
        px, py = int(domain_gx[pidx]), int(domain_gy[pidx])

        for nx, ny in [(px + step, py), (px - step, py), (px, py + step), (px, py - step)]:
            nbidx = grid_map.get((nx, ny))
            if nbidx is None or solved[nbidx]:
                continue
            # A point that failed from one parent may still succeed from a
            # better one. Marking it done on first failure is what turns an
            # isolated bad subset into a wall the wavefront cannot cross.
            if attempts[nbidx] >= MAX_ATTEMPTS:
                continue
            attempts[nbidx] += 1

            ddx, ddy = float(nx - px), float(ny - py)
            u_i = p_par[0] + p_par[2] * ddx + p_par[3] * ddy
            v_i = p_par[1] + p_par[4] * ddx + p_par[5] * ddy
            p_i = np.array([u_i, v_i, p_par[2], p_par[3], p_par[4], p_par[5]])

            good, p_opt, cls_opt = try_point(nbidx, p_i, cutoff_disp)
            local_steps += 1

            if good:
                commit(nbidx, p_opt, cls_opt)
                heapq.heappush(heap, (cls_opt, nbidx, p_opt.copy()))

            flush_progress()

    # ---------------- hole healing ----------------
    # Points still unsolved but bordering solved neighbours get one more chance,
    # seeded from their single most reliable neighbour. This recovers subsets
    # that were only ever attempted from a poor parent, and it repeats until no
    # further progress is made so recovered points can heal their own borders.
    for _sweep in range(6):
        if cancel_flag[0]:
            break
        pending = np.where(~solved)[0]
        if len(pending) == 0:
            break
        healed = 0
        for idx in pending:
            x, y = int(domain_gx[idx]), int(domain_gy[idx])
            best_nb, best_nb_cls = None, np.inf
            for nx, ny in [(x + step, y), (x - step, y), (x, y + step), (x, y - step)]:
                j = grid_map.get((nx, ny))
                if j is not None and solved[j] and best_cls[j] < best_nb_cls:
                    best_nb, best_nb_cls = j, best_cls[j]
            if best_nb is None:
                continue
            bx, by = int(domain_gx[best_nb]), int(domain_gy[best_nb])
            p_par = np.array([u_f[by, bx], v_f[by, bx], du_dx[by, bx],
                              du_dy[by, bx], dv_dx[by, bx], dv_dy[by, bx]])
            ddx, ddy = float(x - bx), float(y - by)
            p_i = np.array([p_par[0] + p_par[2] * ddx + p_par[3] * ddy,
                            p_par[1] + p_par[4] * ddx + p_par[5] * ddy,
                            p_par[2], p_par[3], p_par[4], p_par[5]])
            good, p_opt, cls_opt = try_point(idx, p_i, cutoff_disp)
            local_steps += 1
            if good:
                commit(idx, p_opt, cls_opt)
                healed += 1
            flush_progress()
        if healed == 0:
            break

    flush_progress(force=True)

    return dict(u=u_f, v=v_f, du_dx=du_dx, du_dy=du_dy,
                dv_dx=dv_dx, dv_dy=dv_dy, corr=corr_f, analyzed=ana)


def p_global_copy(u_f, v_f, du_dx, du_dy, dv_dx, dv_dy, x, y):
    return np.array([u_f[y, x], v_f[y, x], du_dx[y, x],
                     du_dy[y, x], dv_dx[y, x], dv_dy[y, x]])


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