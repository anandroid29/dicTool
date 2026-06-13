# src/core/analysis.py
"""
analysis.py — DICAnalysis with strain rate computation, frame-sync support, and batched GPU execution.
Fixed: Survival rate denominator uses valid ROI subset count to prevent false Auto-Fallback triggers.
"""
from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

try:
    from PIL import Image as PILImage; _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False
try:
    import cv2; _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates, spline_filter
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

from .rg_dic import DICParams, DICResult, run_rg_dic
from .roi_loader import load_roi_mask


@dataclass
class PairResult:
    image_path: str
    u:     np.ndarray
    v:     np.ndarray
    Exx:   np.ndarray
    Exy:   np.ndarray
    Eyy:   np.ndarray
    Eeff:  np.ndarray
    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray
    corr:  np.ndarray
    Vx:    Optional[np.ndarray] = None
    Vy:    Optional[np.ndarray] = None
    Veff:  Optional[np.ndarray] = None
    dVx_dx: Optional[np.ndarray] = None
    dVx_dy: Optional[np.ndarray] = None
    dVy_dx: Optional[np.ndarray] = None
    dVy_dy: Optional[np.ndarray] = None
    Exx_rate:  Optional[np.ndarray] = None
    Exy_rate:  Optional[np.ndarray] = None
    Eyy_rate:  Optional[np.ndarray] = None
    Eeff_rate: Optional[np.ndarray] = None
    elapsed: float = 0.0


class DICAnalysis:
    def __init__(self) -> None:
        self.ref_path:  Optional[str]      = None
        self.def_paths: List[str]          = []
        self._ref_image: Optional[np.ndarray] = None
        self._roi_mask:  Optional[np.ndarray] = None
        self.params:  DICParams      = DICParams()
        self.results: List[PairResult] = []
        self.fps: float = 1.0
        self._cancel: list = [False]

        self.last_video_directory: str = os.path.expanduser("~")
        self.last_image_directory: str = os.path.expanduser("~")
        self.last_hdf5_directory: str = os.path.expanduser("~")

        self.load_settings()

    def set_reference(self, path: str) -> None:
        self.ref_path = path
        self._ref_image = _load_image(path)
        self._roi_mask = None

    def add_deformed(self, path: str) -> None:
        self.def_paths.append(path)

    def clear_deformed(self) -> None:
        self.def_paths.clear()
        self.results.clear()

    def set_roi_mask(self, mask: np.ndarray) -> None:
        if self._ref_image is not None and mask.shape != self._ref_image.shape:
            raise ValueError(f"ROI mask shape {mask.shape} != reference {self._ref_image.shape}")
        self._roi_mask = mask.astype(bool)

    def set_roi_from_file(self, path: str) -> None:
        if self._ref_image is None:
            raise RuntimeError("Load reference image before setting ROI from file.")
        mask = load_roi_mask(path, expected_shape=self._ref_image.shape)
        self._roi_mask = mask

    def set_full_roi(self) -> None:
        if self._ref_image is None:
            raise RuntimeError("Load reference first.")
        self._roi_mask = np.ones(self._ref_image.shape, dtype=bool)

    @property
    def reference_image(self) -> Optional[np.ndarray]:
        return self._ref_image

    @property
    def roi_mask(self) -> Optional[np.ndarray]:
        return self._roi_mask

    @property
    def deformed_paths(self) -> List[str]:
        return self.def_paths

    def cancel(self) -> None:
        self._cancel[0] = True

    def run(
            self,
            progress_cb: Optional[Callable[[float, str], None]] = None,
            seed_xy: Optional[tuple] = None,
            use_gpu: bool = False
    ) -> None:

        if use_gpu:
            if not _HAS_CUPY:
                raise RuntimeError("GPU acceleration requested but CuPy is not installed or NVIDIA drivers are missing.")
            self._run_gpu(progress_cb, seed_xy)
            return

        self._cancel[0] = False
        self.results.clear()
        if self._ref_image is None:
            raise RuntimeError("No reference image.")
        if not self.def_paths:
            raise RuntimeError("No deformed images.")
        if self._roi_mask is None:
            self.set_full_roi()

        ref = self._ref_image
        mask = self._roi_mask
        n = len(self.def_paths)

        guess_u, guess_v = 0.0, 0.0

        for i, def_path in enumerate(self.def_paths):
            if self._cancel[0]:
                break

            def pair_cb(frac, msg, _i=i, _n=n):
                if progress_cb:
                    progress_cb(0.90 * (_i / _n) + frac * (0.90 / _n), f"[{_i + 1}/{_n}] {msg}")

            pair_cb(0.0, f"Loading {os.path.basename(def_path)}…")
            cur = _load_image(def_path)
            if cur.shape != ref.shape:
                raise ValueError(f"Shape mismatch: {def_path}")

            t0 = time.perf_counter()
            dic = run_rg_dic(
                ref, cur, mask, self.params,
                seed_xy=seed_xy, progress_cb=pair_cb, cancel_flag=self._cancel,
                guess_u=guess_u, guess_v=guess_v,
                use_gpu=use_gpu,
            )
            elapsed = time.perf_counter() - t0

            valid = dic.analyzed & ~np.isnan(dic.u)
            if valid.any():
                guess_u = float(np.median(dic.u[valid]))
                guess_v = float(np.median(dic.v[valid]))

            nan_arr = np.full_like(dic.u, np.nan)

            self.results.append(PairResult(
                image_path=def_path,
                u=dic.u, v=dic.v,
                Exx=nan_arr.copy(), Exy=nan_arr.copy(), Eyy=nan_arr.copy(), Eeff=nan_arr.copy(),
                du_dx=nan_arr.copy(), du_dy=nan_arr.copy(),
                dv_dx=nan_arr.copy(), dv_dy=nan_arr.copy(),
                corr=dic.corr, elapsed=elapsed,
            ))

        if not self._cancel[0] and self.results:
            self._compute_velocities_and_rates(progress_cb)

        if progress_cb:
            progress_cb(1.0, "Complete.")

    def _run_gpu(
            self,
            progress_cb: Optional[Callable[[float, str], None]] = None,
            seed_xy: Optional[tuple] = None
    ) -> None:
        """
        Executes the Wavefront GPU pipeline with intelligent Global Seed Tracking and Auto-Fallback.
        """
        self._cancel[0] = False
        self.results.clear()

        if self._ref_image is None or not self.def_paths:
            raise RuntimeError("Missing reference or deformed images.")
        if self._roi_mask is None:
            self.set_full_roi()

        n_frames = len(self.def_paths)

        if progress_cb:
            progress_cb(0.0, "Initializing GPU solver and precomputing reference...")

        try:
            from .icgn_gpu import GPUWavefrontDIC
            gpu_solver = GPUWavefrontDIC(self.params)
            gpu_solver.precompute_reference(self._ref_image, self._roi_mask)
        except Exception as e:
            raise RuntimeError(f"GPU initialization failed: {e}")

        if seed_xy is None:
            ys_roi, xs_roi = np.where(self._roi_mask)
            if len(xs_roi) == 0:
                raise ValueError("ROI mask is empty.")
            seed_xy = (int(xs_roi.mean()), int(ys_roi.mean()))

        dist_sq = (gpu_solver.gx_flat - seed_xy[0])**2 + (gpu_solver.gy_flat - seed_xy[1])**2
        seed_idx = int(np.argmin(dist_sq))
        actual_seed_x = int(gpu_solver.gx_flat[seed_idx])
        actual_seed_y = int(gpu_solver.gy_flat[seed_idx])

        # CORRECTED EXPECTED SUBSETS: Count only subsets strictly inside the ROI
        expected_subsets = int(gpu_solver.valid_mask.sum())

        from .ncc import ncc_initial_guess

        warm_start_active = False
        guess_u, guess_v = 0.0, 0.0

        for i, def_path in enumerate(self.def_paths):
            if self._cancel[0]: break
            t0 = time.perf_counter()

            if progress_cb:
                progress_cb(0.90 * (i / n_frames), f"[{i + 1}/{n_frames}] Loading {os.path.basename(def_path)}...")

            cur_image = _load_image(def_path)

            if not warm_start_active:
                if progress_cb:
                    progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.3, f"[{i + 1}/{n_frames}] Global NCC Search...")

                guess_u, guess_v, _ = ncc_initial_guess(
                    self._ref_image, cur_image, actual_seed_x, actual_seed_y,
                    self.params.subset_radius, self.params.search_radius,
                    guess_u, guess_v
                )
                seed_p = np.array([guess_u, guess_v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

                if progress_cb:
                    progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.6, f"[{i + 1}/{n_frames}] Growing Wavefront...")

                u_f, v_f, du_dx, du_dy, dv_dx, dv_dy, corr_f = gpu_solver.solve_frame(
                    cur_image, seed_idx=seed_idx, seed_p=seed_p, warm_start=False
                )
                warm_start_active = True

            else:
                if progress_cb:
                    progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.5, f"[{i + 1}/{n_frames}] Batched temporal tracking...")

                u_f, v_f, du_dx, du_dy, dv_dx, dv_dy, corr_f = gpu_solver.solve_frame(
                    cur_image, warm_start=True
                )

                valid_count = np.count_nonzero(~np.isnan(u_f[self._roi_mask]))
                survival_rate = valid_count / max(1, expected_subsets)

                if survival_rate < 0.60:
                    print(f"\n[AUTO-FALLBACK] Frame {i+1} collapsed (Survival: {survival_rate*100:.1f}%). Re-running with targeted NCC...")

                    if progress_cb:
                        progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.7, f"[{i + 1}/{n_frames}] Jolt detected. Repairing via NCC...")

                    guess_u, guess_v, _ = ncc_initial_guess(
                        self._ref_image, cur_image, actual_seed_x, actual_seed_y,
                        self.params.subset_radius, self.params.search_radius,
                        guess_u, guess_v
                    )
                    seed_p = np.array([guess_u, guess_v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

                    u_f, v_f, du_dx, du_dy, dv_dx, dv_dy, corr_f = gpu_solver.solve_frame(
                        cur_image, seed_idx=seed_idx, seed_p=seed_p, warm_start=False
                    )

            if not np.isnan(u_f[actual_seed_y, actual_seed_x]):
                guess_u = float(u_f[actual_seed_y, actual_seed_x])
                guess_v = float(v_f[actual_seed_y, actual_seed_x])

            elapsed = time.perf_counter() - t0
            nan_arr = np.full_like(u_f, np.nan)

            self.results.append(PairResult(
                image_path=def_path,
                u=u_f, v=v_f,
                Exx=nan_arr.copy(), Exy=nan_arr.copy(), Eyy=nan_arr.copy(), Eeff=nan_arr.copy(),
                du_dx=du_dx, du_dy=du_dy, dv_dx=dv_dx, dv_dy=dv_dy,
                corr=corr_f, elapsed=elapsed
            ))

        if not self._cancel[0] and self.results:
            self._compute_velocities_and_rates(progress_cb)

        if progress_cb:
            progress_cb(1.0, "Complete.")

    def _compute_velocities_and_rates(self, progress_cb: Optional[Callable[[float, str], None]] = None) -> None:
        N = len(self.results)
        if N < 2: return
        dt = 1.0 / max(self.fps, 1e-9)

        u_stack = np.stack([r.u for r in self.results])
        v_stack = np.stack([r.v for r in self.results])

        v_u = np.zeros_like(u_stack)
        v_v = np.zeros_like(v_stack)

        v_u[1:-1] = (u_stack[2:] - u_stack[:-2]) / (2 * dt)
        v_v[1:-1] = (v_stack[2:] - v_stack[:-2]) / (2 * dt)

        v_u[0] = (u_stack[1] - u_stack[0]) / dt
        v_v[0] = (v_stack[1] - v_stack[0]) / dt
        v_u[-1] = (u_stack[-1] - u_stack[-2]) / dt
        v_v[-1] = (v_stack[-1] - v_stack[-2]) / dt

        for i, res in enumerate(self.results):
            res.Vx, res.Vy = v_u[i], v_v[i]
            res.Veff = np.sqrt(res.Vx ** 2 + res.Vy ** 2)

        from .strain import compute_velocity_strains
        mask = self._roi_mask if self._roi_mask is not None else np.ones_like(self.results[0].u, dtype=bool)

        for i, res in enumerate(self.results):
            if progress_cb:
                p = 0.90 + 0.07 * (i / max(1, N))
                progress_cb(p, f"[{i + 1}/{N}] Computing strain rates…")

            valid = mask & ~np.isnan(res.Vx) & ~np.isnan(res.Vy)
            rates = compute_velocity_strains(res.Vx, res.Vy, valid, self.params.strain_window)

            res.dVx_dx = rates["dVx_dx"]
            res.dVx_dy = rates["dVx_dy"]
            res.dVy_dx = rates["dVy_dx"]
            res.dVy_dy = rates["dVy_dy"]
            res.Exx_rate = rates["Exx_rate"]
            res.Exy_rate = rates["Exy_rate"]
            res.Eyy_rate = rates["Eyy_rate"]
            res.Eeff_rate = rates["Eeff_rate"]

        if N > 0:
            res0 = self.results[0]
            curr_Exx = np.zeros_like(res0.u)
            curr_Exy = np.zeros_like(res0.u)
            curr_Eyy = np.zeros_like(res0.u)

            for i in range(N):
                if progress_cb:
                    p = 0.97 + 0.03 * (i / max(1, N))
                    progress_cb(p, f"[{i + 1}/{N}] Integrating strains…")

                curr = self.results[i]
                valid = ~np.isnan(curr.u)

                if i > 0:
                    prev = self.results[i - 1]
                    rate_prev_xx = np.nan_to_num(prev.Exx_rate, nan=0.0)
                    rate_curr_xx = np.nan_to_num(curr.Exx_rate, nan=0.0)
                    curr_Exx += 0.5 * (rate_prev_xx + rate_curr_xx) * dt

                    rate_prev_xy = np.nan_to_num(prev.Exy_rate, nan=0.0)
                    rate_curr_xy = np.nan_to_num(curr.Exy_rate, nan=0.0)
                    curr_Exy += 0.5 * (rate_prev_xy + rate_curr_xy) * dt

                    rate_prev_yy = np.nan_to_num(prev.Eyy_rate, nan=0.0)
                    rate_curr_yy = np.nan_to_num(curr.Eyy_rate, nan=0.0)
                    curr_Eyy += 0.5 * (rate_prev_yy + rate_curr_yy) * dt

                curr.Exx = np.where(valid, curr_Exx, np.nan)
                curr.Exy = np.where(valid, curr_Exy, np.nan)
                curr.Eyy = np.where(valid, curr_Eyy, np.nan)

                with np.errstate(invalid='ignore'):
                    curr.Eeff = np.sqrt(np.maximum(
                        (2.0 / 3.0) * (curr.Exx ** 2 + curr.Eyy ** 2 + 2.0 * curr.Exy ** 2 - curr.Exx * curr.Eyy), 0.0
                    ))

    def get_trajectories(self, max_frame: int, step: int = 10) -> list[list[tuple[float, float]]]:
        if not self.results or max_frame < 0:
            return []

        valid = np.isfinite(self.results[0].u) & np.isfinite(self.results[0].v)

        y_lines = np.unique(np.where(valid)[0])
        x_lines = np.unique(np.where(valid)[1])

        y_sampled = y_lines[::step]
        x_sampled = x_lines[::step]

        xx, yy = np.meshgrid(x_sampled, y_sampled)
        xx = xx.ravel()
        yy = yy.ravel()

        valid_intersections = valid[yy, xx]
        x0 = xx[valid_intersections]
        y0 = yy[valid_intersections]

        N_particles = len(x0)
        active = np.ones(N_particles, dtype=bool)

        paths = [[(float(x), float(y))] for x, y in zip(x0, y0)]

        for i in range(0, max_frame + 1):
            if i >= len(self.results):
                break

            u_i = self.results[i].u[y0, x0]
            v_i = self.results[i].v[y0, x0]

            lost = ~np.isfinite(u_i) | ~np.isfinite(v_i)
            active[lost] = False

            for p_idx in np.where(active)[0]:
                paths[p_idx].append((float(x0[p_idx] + u_i[p_idx]), float(y0[p_idx] + v_i[p_idx])))

        return [p for p in paths if len(p) > 1]

    def get_global_range(self, field: str) -> tuple[float, float]:
        vmin, vmax = float('inf'), float('-inf')
        for res in self.results:
            arr = getattr(res, field, None)
            if arr is not None:
                valid = arr[np.isfinite(arr)]
                if valid.size > 0:
                    vmin = min(vmin, float(valid.min()))
                    vmax = max(vmax, float(valid.max()))
        return (vmin, vmax) if vmin != float('inf') else (0.0, 1.0)

    def export_csv(self, result_index: int, directory: str) -> None:
        res = self.results[result_index]
        base = os.path.splitext(os.path.basename(res.image_path))[0]
        fields = ("u", "v", "Exx", "Exy", "Eyy", "Eeff",
                  "Vx", "Vy", "Veff",
                  "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                  "Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate", "corr")
        for name in fields:
            arr = getattr(res, name, None)
            if arr is not None:
                np.savetxt(os.path.join(directory, f"{base}_{name}.csv"), arr, delimiter=",")

    def export_hdf5(self, path: str) -> None:
        import h5py
        with h5py.File(path, "w") as f:
            # 1. Save Global Attributes
            f.attrs.update(dict(
                reference_image=self.ref_path or "",
                subset_radius=self.params.subset_radius,
                subset_spacing=self.params.subset_spacing,
                strain_window=self.params.strain_window,
                fps=self.fps,
            ))

            # 2. Save the ROI Mask (CRITICAL FIX)
            if self._roi_mask is not None:
                f.create_dataset(
                    "roi_mask",
                    data=self._roi_mask.astype(bool),
                    compression="gzip",
                    compression_opts=4
                )

            # 3. Save Frame Data
            for i, res in enumerate(self.results):
                g = f.create_group(f"frame_{i:04d}")
                g.attrs["image_path"] = res.image_path
                g.attrs["elapsed_s"] = res.elapsed
                fields = ("u", "v", "Exx", "Exy", "Eyy", "Eeff",
                          "Vx", "Vy", "Veff",
                          "du_dx", "du_dy", "dv_dx", "dv_dy",
                          "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                          "Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate", "corr")
                for name in fields:
                    arr = getattr(res, name, None)
                    if arr is not None:
                        g.create_dataset(name, data=arr.astype(np.float32),
                                         compression="gzip", compression_opts=4)

    def load_hdf5(self, path: str) -> None:
        import h5py
        self.results.clear()
        self.def_paths.clear()

        with h5py.File(path, "r") as f:
            # 1. Restore Global Attributes
            self.ref_path = f.attrs.get("reference_image", "")
            try:
                if self.ref_path and os.path.exists(self.ref_path):
                    self._ref_image = _load_image(self.ref_path)
            except Exception:
                pass

            self.params.subset_radius = int(f.attrs.get("subset_radius", self.params.subset_radius))
            self.params.subset_spacing = int(f.attrs.get("subset_spacing", self.params.subset_spacing))
            self.params.strain_window = int(f.attrs.get("strain_window", self.params.strain_window))
            self.fps = float(f.attrs.get("fps", 1.0))

            # 2. Restore the ROI Mask (CRITICAL FIX)
            if "roi_mask" in f:
                self._roi_mask = f["roi_mask"][:].astype(bool)
            else:
                self._roi_mask = None

            # 3. Restore Frame Data
            for k in sorted([key for key in f.keys() if key.startswith("frame_")]):
                g = f[k]
                ipath = g.attrs.get("image_path", "")
                self.def_paths.append(ipath)

                res = PairResult(
                    image_path=ipath,
                    u=g["u"][:] if "u" in g else np.zeros(0),
                    v=g["v"][:] if "v" in g else np.zeros(0),
                    Exx=g["Exx"][:] if "Exx" in g else np.zeros(0),
                    Exy=g["Exy"][:] if "Exy" in g else np.zeros(0),
                    Eyy=g["Eyy"][:] if "Eyy" in g else np.zeros(0),
                    Eeff=g["Eeff"][:] if "Eeff" in g else np.zeros(0),
                    du_dx=g["du_dx"][:] if "du_dx" in g else np.zeros(0),
                    du_dy=g["du_dy"][:] if "du_dy" in g else np.zeros(0),
                    dv_dx=g["dv_dx"][:] if "dv_dx" in g else np.zeros(0),
                    dv_dy=g["dv_dy"][:] if "dv_dy" in g else np.zeros(0),
                    corr=g["corr"][:] if "corr" in g else np.zeros(0),
                    elapsed=float(g.attrs.get("elapsed_s", 0.0))
                )
                extra_fields = ("Vx", "Vy", "Veff", "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                                "Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate")
                for rate in extra_fields:
                    if rate in g:
                        setattr(res, rate, g[rate][:])
                self.results.append(res)

    def _get_settings_path(self) -> str:
        import os
        return os.path.join(os.path.expanduser("~"), ".pydic_settings.json")

    def load_settings(self) -> None:
        import json, os
        path = self._get_settings_path()

        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                for k, v in data.items():
                    if hasattr(self.params, k):
                        setattr(self.params, k, v)

                # Load all specialized directories
                dirs = ["last_video_directory", "last_image_directory", "last_hdf5_directory"]
                for d in dirs:
                    if d in data and os.path.exists(data[d]):
                        setattr(self, d, data[d])

            except Exception as e:
                print(f"[Warning] Failed to load settings: {e}")
        else:
            self.save_settings()

    def save_settings(self) -> None:
        import json, os
        path = self._get_settings_path()

        try:
            data = {
                "subset_radius": self.params.subset_radius,
                "subset_spacing": self.params.subset_spacing,
                "strain_window": self.params.strain_window,
                "max_iter": self.params.max_iter,
                "conv_tol": self.params.conv_tol,
                "corr_cutoff": self.params.corr_cutoff,
                "search_radius": self.params.search_radius,

                # Save all specialized directories
                "last_video_directory": getattr(self, "last_video_directory", os.path.expanduser("~")),
                "last_image_directory": getattr(self, "last_image_directory", os.path.expanduser("~")),
                "last_hdf5_directory": getattr(self, "last_hdf5_directory", os.path.expanduser("~")),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[Error] Failed to save settings to {path}: {e}")

def _load_image(path: str) -> np.ndarray:
    if _HAVE_CV2:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read: {path}")
        mx = float(np.iinfo(img.dtype).max) if img.dtype.kind == "u" else 1.0
        return img.astype(np.float64) / mx
    elif _HAVE_PIL:
        return np.asarray(PILImage.open(path).convert("L"), np.float64) / 255.0
    else:
        raise ImportError("Install opencv-python or Pillow.")