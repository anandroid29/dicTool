"""
analysis.py — DICAnalysis with strain rate computation and frame-sync support.
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
        self.load_settings()

    # -- configuration --
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

    # -- analysis --
    def cancel(self) -> None:
        self._cancel[0] = True

    def run(
            self,
            progress_cb: Optional[Callable[[float, str], None]] = None,
            seed_xy: Optional[tuple] = None,
    ) -> None:
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

        # Track previous displacements to prevent losing fast-moving subsets
        guess_u, guess_v = 0.0, 0.0

        for i, def_path in enumerate(self.def_paths):
            if self._cancel[0]:
                break

            def pair_cb(frac, msg, _i=i, _n=n):
                if progress_cb:
                    # Allocate 90% of total progress to DIC tracking
                    progress_cb(0.90 * (_i / _n) + frac * (0.90 / _n), f"[{_i + 1}/{_n}] {msg}")

            pair_cb(0.0, f"Loading {os.path.basename(def_path)}…")
            cur = _load_image(def_path)
            if cur.shape != ref.shape:
                raise ValueError(f"Shape mismatch: {def_path}")

            t0 = time.perf_counter()
            dic = run_rg_dic(
                ref, cur, mask, self.params,
                seed_xy=seed_xy, progress_cb=pair_cb, cancel_flag=self._cancel,
                guess_u=guess_u, guess_v=guess_v
            )
            elapsed = time.perf_counter() - t0

            # Update guesses for the next frame using robust median displacement
            valid = dic.analyzed & ~np.isnan(dic.u)
            if valid.any():
                guess_u = float(np.median(dic.u[valid]))
                guess_v = float(np.median(dic.v[valid]))

            # We no longer calculate strains from displacement here.
            # Fill with NaNs temporarily; they will be computed via integration.
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
            self._compute_velocities_and_rates(progress_cb)  # <-- Pass progress_cb here

        if progress_cb:
            progress_cb(1.0, "Complete.")

    def _compute_velocities_and_rates(self, progress_cb: Optional[Callable[[float, str], None]] = None) -> None:
        N = len(self.results)
        dt = 1.0 / max(self.fps, 1e-9)

        # 1. Compute Velocities via temporal finite difference
        for i, res in enumerate(self.results):
            if N == 1:
                nan = np.full_like(res.u, np.nan)
                res.Vx = res.Vy = res.Veff = nan.copy()
                continue

            if i == 0:
                prev, nxt, dt_tot = res, self.results[1], dt
            elif i == N - 1:
                prev, nxt, dt_tot = self.results[i - 1], res, dt
            else:
                prev, nxt, dt_tot = self.results[i - 1], self.results[i + 1], 2 * dt

            def _fd(a, b):
                with np.errstate(invalid="ignore"):
                    return (b - a) / dt_tot

            res.Vx = _fd(prev.u, nxt.u)
            res.Vy = _fd(prev.v, nxt.v)
            with np.errstate(invalid="ignore"):
                res.Veff = np.sqrt(res.Vx ** 2 + res.Vy ** 2)

        # 2. Compute Strain Rates via spatial gradient of Velocity
        from .strain import compute_velocity_strains
        mask = self._roi_mask if self._roi_mask is not None else np.ones_like(self.results[0].u, dtype=bool)

        for i, res in enumerate(self.results):
            if progress_cb:
                # Map to 90% - 97% overall progress
                p = 0.90 + 0.07 * (i / max(1, N))
                progress_cb(p, f"[{i + 1}/{N}] Computing strain rates…")

            if N == 1:
                nan = np.full_like(res.u, np.nan)
                res.dVx_dx = res.dVx_dy = res.dVy_dx = res.dVy_dy = nan.copy()
                res.Exx_rate = res.Exy_rate = res.Eyy_rate = res.Eeff_rate = nan.copy()
                continue

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

        # 3. Integrate Strain Rates to compute Cumulative Strains (Trapezoidal Rule)
        if N > 0:
            res0 = self.results[0]
            # Initialize accumulators at 0
            curr_Exx = np.zeros_like(res0.u)
            curr_Exy = np.zeros_like(res0.u)
            curr_Eyy = np.zeros_like(res0.u)

            for i in range(N):
                if progress_cb:
                    # Map to 97% - 100% overall progress
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

        # Extract unique valid grid lines used by the DIC solver
        y_lines = np.unique(np.where(valid)[0])
        x_lines = np.unique(np.where(valid)[1])

        # Decimate purely in 2D space to guarantee an orthogonal grid
        y_sampled = y_lines[::step]
        x_sampled = x_lines[::step]

        # Create perfect grid intersections
        xx, yy = np.meshgrid(x_sampled, y_sampled)
        xx = xx.ravel()
        yy = yy.ravel()

        # Filter out intersections that fall outside the active ROI
        valid_intersections = valid[yy, xx]
        x0 = xx[valid_intersections]
        y0 = yy[valid_intersections]

        N_particles = len(x0)
        active = np.ones(N_particles, dtype=bool)

        paths = [[(float(x), float(y))] for x, y in zip(x0, y0)]

        # Start at 0 to explicitly include the first deformed frame
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
            f.attrs.update(dict(
                reference_image=self.ref_path or "",
                subset_radius=self.params.subset_radius,
                subset_spacing=self.params.subset_spacing,
                strain_window=self.params.strain_window,
                fps=self.fps,
            ))
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

    def load_settings(self) -> None:
        import json, os
        path = os.path.join(os.getcwd(), "pydic_settings.json")

        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(self.params, k):
                        setattr(self.params, k, v)
            except Exception:
                pass
        else:
            self.save_settings()

    def save_settings(self) -> None:
        import json, os
        path = os.path.join(os.getcwd(), "pydic_settings.json")

        try:
            data = {
                "subset_radius": self.params.subset_radius,
                "subset_spacing": self.params.subset_spacing,
                "strain_window": self.params.strain_window,
                "max_iter": self.params.max_iter,
                "conv_tol": self.params.conv_tol,
                "corr_cutoff": self.params.corr_cutoff,
                "search_radius": self.params.search_radius
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass

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