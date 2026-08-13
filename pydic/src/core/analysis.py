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
from .strain_accum import StrainAccumulator, total_strain
from .roi_loader import load_roi_mask
from .units import Calibration


# Native (uncalibrated) unit of each exportable field, used to label exports.
# Kept beside the data rather than in the UI so a headless export says the same
# thing the results view does.
_FIELD_BASE_UNIT = {
    "u": "px", "v": "px",
    "u_inc": "px", "v_inc": "px", "mag_inc": "px",
    "Vx": "px/s", "Vy": "px/s", "Veff": "px/s",
    "Exx": "", "Exy": "", "Eyy": "", "Eeff": "",
    "Exx_rate": "1/s", "Exy_rate": "1/s", "Eyy_rate": "1/s", "Eeff_rate": "1/s",
    "dVx_dx": "1/s", "dVx_dy": "1/s", "dVy_dx": "1/s", "dVy_dy": "1/s",
    "du_dx": "", "du_dy": "", "dv_dx": "", "dv_dy": "",
    "corr": "ZNSSD",
}


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
    # Frame-to-frame displacement: how far each material point moved since the
    # PREVIOUS frame, as opposed to u/v which accumulate from the reference.
    # Both are Lagrangian -- indexed by reference-frame material point -- so the
    # increment is a plain difference at matching indices.
    u_inc:   Optional[np.ndarray] = None
    v_inc:   Optional[np.ndarray] = None
    mag_inc: Optional[np.ndarray] = None
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
    # True where u/v/strain are live, trustworthy accumulated values for THIS
    # frame. A point whose history broke (occlusion, dropout, ROI loss) is
    # False here even though its arrays may still hold a frozen last-known
    # number -- kept for backward compatibility with anything indexing the
    # raw arrays, but display/export code should mask by this field, not by
    # isnan(u) alone, since a frozen value is not NaN.
    valid: Optional[np.ndarray] = None
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
        # Spatial calibration. Uncalibrated by default: the solver is pixel-native
        # and stays that way -- this only affects how results are presented.
        self.calibration: Calibration = Calibration()
        self._cancel: list = [False]

        # Human-readable notes about anything load_settings had to correct. The
        # console print alone was too easy to miss -- a silently repaired
        # setting is exactly the kind of thing that should be surfaced, since
        # the stored value was producing bad results. The UI drains this once at
        # startup.
        self.settings_notices: List[str] = []
        # Cleared if the settings file exists but cannot be parsed, which blocks
        # save_settings from overwriting it with defaults.
        self._settings_readable: bool = True

        # Operator overrides for the dynamic ROI, in reference-frame coordinates.
        # Arrays, so they live here rather than in the JSON-serialised params.
        self.dynamic_include_mask: Optional[np.ndarray] = None
        self.dynamic_exclude_mask: Optional[np.ndarray] = None

        self.last_video_directory: str = os.path.expanduser("~")
        self.last_image_directory: str = os.path.expanduser("~")
        self.last_hdf5_directory: str = os.path.expanduser("~")

        self.load_settings()

    def make_dynamic_roi(self) -> "DynamicROI":
        """Build the dynamic ROI from params + operator overrides.

        Single construction point so the CPU and GPU paths cannot drift apart
        in how they interpret the threshold and include/exclude masks.
        """
        return DynamicROI(
            self.params.dynamic_roi,
            keep_min_area_frac=getattr(self.params, "dynamic_roi_min_area_frac", 0.02),
            threshold=getattr(self.params, "dynamic_roi_threshold", None),
            include_mask=self.dynamic_include_mask,
            exclude_mask=self.dynamic_exclude_mask,
            roi_mask=self._roi_mask,
            fill_holes=getattr(self.params, "dynamic_roi_fill_holes", True),
        )

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

        # Calibrate the texture threshold ONCE on the reference frame. The old
        # per-frame _compute_dynamic_mask recalibrated its scale and Otsu
        # threshold on every image, so "enough texture to correlate" drifted
        # frame to frame and the mask edge flickered -- see DynamicROI's
        # docstring, which the GPU path already follows.
        dyn_roi = self.make_dynamic_roi()
        dyn_roi.calibrate(ref)

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

            d_mask = dyn_roi.mask(cur)
            if d_mask is not None and valid.any():
                y_ref, x_ref = np.where(valid)
                x_cur = np.round(x_ref + dic.u[valid]).astype(int)
                y_cur = np.round(y_ref + dic.v[valid]).astype(int)
                H_i, W_i = cur.shape
                
                kept = (x_cur >= 0) & (x_cur < W_i) & (y_cur >= 0) & (y_cur < H_i)
                in_bnd_idx = np.where(kept)[0]
                kept[in_bnd_idx] = d_mask[y_cur[in_bnd_idx], x_cur[in_bnd_idx]]
                
                lost = ~kept
                y_lost, x_lost = y_ref[lost], x_ref[lost]
                
                dic.u[y_lost, x_lost] = np.nan
                dic.v[y_lost, x_lost] = np.nan
                dic.du_dx[y_lost, x_lost] = np.nan
                dic.du_dy[y_lost, x_lost] = np.nan
                dic.dv_dx[y_lost, x_lost] = np.nan
                dic.dv_dy[y_lost, x_lost] = np.nan
                dic.corr[y_lost, x_lost] = np.nan
                
                valid[y_lost, x_lost] = False

            if valid.any():
                guess_u = float(np.median(dic.u[valid]))
                guess_v = float(np.median(dic.v[valid]))

            # Strain was previously never computed on this path -- every strain
            # field was handed out as an all-NaN placeholder, so selecting Exx /
            # Eyy / Exy / Eeff in the results view showed nothing at all. This
            # path correlates every frame directly against the fixed reference,
            # so the displacement field IS the total field and finite strain
            # follows from it directly.
            st = total_strain(dic.u, dic.v, valid, self.params.effective_strain_window())

            self.results.append(PairResult(
                image_path=def_path,
                u=dic.u, v=dic.v,
                Exx=np.where(valid, st["Exx"], np.nan),
                Exy=np.where(valid, st["Exy"], np.nan),
                Eyy=np.where(valid, st["Eyy"], np.nan),
                # Hencky equivalent strain. Green-Lagrange overstates badly past
                # ~20%, which is routine here.
                Eeff=np.where(valid, st["Eeff_log"], np.nan),
                du_dx=np.where(valid, dic.du_dx, np.nan),
                du_dy=np.where(valid, dic.du_dy, np.nan),
                dv_dx=np.where(valid, dic.dv_dx, np.nan),
                dv_dy=np.where(valid, dic.dv_dy, np.nan),
                corr=dic.corr, valid=valid.copy(), elapsed=elapsed,
            ))

        if not self._cancel[0] and self.results:
            self._compute_incremental_displacements()
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

        # --- Updated Lagrangian: accumulate total displacement from incremental ---
        # Always start from None. Keying this off hasattr(self, 'H_ref') meant a
        # SECOND run on the same DICAnalysis began with a stale all-NaN array
        # instead, so the first frame took a different code path than it did on
        # the first run.
        total_u = None
        total_v = None
        # Position hints handed to the solver: last-known offsets for every
        # started point, including ones that dropped out. Distinct from
        # total_u/total_v, which are masked to valid points for reporting.
        hint_u = None
        hint_v = None
        total_du_dx = None; total_du_dy = None
        total_dv_dx = None; total_dv_dy = None
        prev_image = self._ref_image  # first reference is frame 0
        accum = None
        dyn_roi = self.make_dynamic_roi()
        dyn_roi.calibrate(self._ref_image)

        H_img, W_img = self._ref_image.shape
        self.H_ref, self.W_ref = H_img, W_img

        for i, def_path in enumerate(self.def_paths):
            if self._cancel[0]: break
            t0 = time.perf_counter()

            if progress_cb:
                progress_cb(0.90 * (i / n_frames), f"[{i + 1}/{n_frames}] Loading {os.path.basename(def_path)}...")

            cur_image = _load_image(def_path)

            if hint_u is not None and np.isfinite(hint_u[actual_seed_y, actual_seed_x]):
                current_seed_x = int(round(actual_seed_x + hint_u[actual_seed_y, actual_seed_x]))
                current_seed_y = int(round(actual_seed_y + hint_v[actual_seed_y, actual_seed_x]))
            else:
                current_seed_x = actual_seed_x
                current_seed_y = actual_seed_y

            # The seed can be carried outside the frame by accumulated motion;
            # ncc_initial_guess would then build an empty search window and
            # silently return the previous guess forever.
            r_pad = int(self.params.subset_radius)
            current_seed_x = int(np.clip(current_seed_x, r_pad, W_img - r_pad - 1))
            current_seed_y = int(np.clip(current_seed_y, r_pad, H_img - r_pad - 1))

            if not warm_start_active:
                if progress_cb:
                    progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.3, f"[{i + 1}/{n_frames}] Global NCC Search...")

                guess_u, guess_v, _ = ncc_initial_guess(
                    prev_image, cur_image, current_seed_x, current_seed_y,
                    self.params.subset_radius, self.params.search_radius,
                    guess_u, guess_v
                )
                seed_p = np.array([guess_u, guess_v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

                if progress_cb:
                    progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.6, f"[{i + 1}/{n_frames}] Growing Wavefront...")

                inc_u, inc_v, inc_du_dx, inc_du_dy, inc_dv_dx, inc_dv_dy, corr_f = gpu_solver.solve_frame(
                    cur_image, seed_idx=seed_idx, seed_p=seed_p, warm_start=False, total_u=hint_u, total_v=hint_v
                )
                warm_start_active = True

            else:
                if progress_cb:
                    progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.5, f"[{i + 1}/{n_frames}] Batched temporal tracking...")

                inc_u, inc_v, inc_du_dx, inc_du_dy, inc_dv_dx, inc_dv_dy, corr_f = gpu_solver.solve_frame(
                    cur_image, warm_start=True, total_u=hint_u, total_v=hint_v
                )

                valid_count = np.count_nonzero(~np.isnan(inc_u[self._roi_mask]))
                survival_rate = valid_count / max(1, expected_subsets)

                if survival_rate < 0.60:
                    print(f"\n[AUTO-FALLBACK] Frame {i+1} collapsed (Survival: {survival_rate*100:.1f}%). Re-running with targeted NCC...")

                    if progress_cb:
                        progress_cb(0.90 * (i / n_frames) + (0.90 / n_frames) * 0.7, f"[{i + 1}/{n_frames}] Jolt detected. Repairing via NCC...")

                    guess_u, guess_v, _ = ncc_initial_guess(
                        prev_image, cur_image, current_seed_x, current_seed_y,
                        self.params.subset_radius, self.params.search_radius,
                        guess_u, guess_v
                    )
                    seed_p = np.array([guess_u, guess_v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

                    inc_u, inc_v, inc_du_dx, inc_du_dy, inc_dv_dx, inc_dv_dy, corr_f = gpu_solver.solve_frame(
                        cur_image, seed_idx=seed_idx, seed_p=seed_p, warm_start=False, total_u=hint_u, total_v=hint_v
                    )

            # --- Accumulate incremental -> total displacement and strain ---
            # Delegated to StrainAccumulator, which (a) never replaces an
            # existing accumulated total with a bare increment, and (b) re-bases
            # a point that returns after a gap from its live neighbours instead
            # of retiring it permanently.
            if accum is None:
                accum = StrainAccumulator((H_img, W_img), self.params.effective_strain_window())
            accum.add_frame(inc_u, inc_v, inc_du_dx, inc_du_dy, inc_dv_dx, inc_dv_dy)

            # Dynamic ROI uses TOTAL displacement for coordinate mapping. Applied
            # BEFORE the frame's outputs are snapshotted -- doing it afterwards
            # left PairResult.valid disagreeing with the strain arrays computed
            # from accum.results() in the same iteration.
            d_mask = dyn_roi.mask(cur_image)
            if d_mask is not None:
                vmask = accum.valid
                if vmask.any():
                    y_ref, x_ref = np.where(vmask)
                    x_cur = np.round(x_ref + accum.u[vmask]).astype(int)
                    y_cur = np.round(y_ref + accum.v[vmask]).astype(int)
                    H_i, W_i = cur_image.shape
                    kept = (x_cur >= 0) & (x_cur < W_i) & (y_cur >= 0) & (y_cur < H_i)
                    ib = np.where(kept)[0]
                    kept[ib] = d_mask[y_cur[ib], x_cur[ib]]
                    lost_mask = np.zeros((H_i, W_i), bool)
                    lost_mask[y_ref[~kept], x_ref[~kept]] = True
                    # Drops the point for THIS frame only. Its accumulated total
                    # is retained so that if the mask picks it up again the
                    # accumulator can resume it rather than start it over.
                    accum.mark_lost(lost_mask)

            # accum.u/v deliberately retain the last known total for points that
            # dropped out this frame, so they are unsafe to publish as-is: a
            # consumer (notably the frame-to-frame difference in
            # _compute_velocities_and_rates) would see an ordinary-looking
            # number that had quietly stopped updating. Mask on the way out.
            frame_valid = accum.valid
            total_u = np.where(frame_valid, accum.u, np.nan)
            total_v = np.where(frame_valid, accum.v, np.nan)
            total_du_dx = np.where(frame_valid, inc_du_dx, np.nan)
            total_du_dy = np.where(frame_valid, inc_du_dy, np.nan)
            total_dv_dx = np.where(frame_valid, inc_dv_dx, np.nan)
            total_dv_dy = np.where(frame_valid, inc_dv_dy, np.nan)

            # Unmasked last-known positions, for the next frame's search. Keep
            # these separate from the reported totals: feeding the masked (NaN)
            # version back made the solver look for a dropped point at its
            # frame-0 position, which is why a dropout never recovered.
            hint_u, hint_v = accum.position_hint()

            # Track seed displacement for NCC initial guess (incremental, small)
            if np.isfinite(inc_u[actual_seed_y, actual_seed_x]):
                guess_u = float(inc_u[actual_seed_y, actual_seed_x])
                guess_v = float(inc_v[actual_seed_y, actual_seed_x])

            # --- Updated Lagrangian: swap reference to current frame for next iteration ---
            gpu_solver.update_reference_image(cur_image)
            prev_image = cur_image

            elapsed = time.perf_counter() - t0

            st = accum.results()
            self.results.append(PairResult(
                image_path=def_path,
                u=total_u.copy(), v=total_v.copy(),
                # Accumulated finite strain. Eeff is the PATH-INTEGRATED
                # equivalent strain -- the correct measure through a shear zone
                # with large rotation, and the one comparable to the
                # gamma/sqrt(3) figures quoted in the machining literature.
                Exx=st["Exx"].copy(), Exy=st["Exy"].copy(),
                Eyy=st["Eyy"].copy(), Eeff=st["Eeff_path"].copy(),
                du_dx=total_du_dx.copy(), du_dy=total_du_dy.copy(),
                dv_dx=total_dv_dx.copy(), dv_dy=total_dv_dy.copy(),
                corr=corr_f, valid=frame_valid.copy(), elapsed=elapsed
            ))

        if not self._cancel[0] and self.results:
            self._compute_incremental_displacements()
            self._compute_velocities_and_rates(progress_cb)

        if progress_cb:
            progress_cb(1.0, "Complete.")

    def _compute_incremental_displacements(self) -> None:
        """Per-frame displacement increments, u_inc[i] = u[i] - u[i-1].

        u/v accumulate against the fixed reference, so at frame 900 they report
        the whole history rather than what happened between frames 899 and 900.
        Both fields index the same reference-frame material points, which is what
        makes the plain difference the correct material increment.

        Frame 0 has no predecessor: its increment is its own displacement from
        the reference, which is exactly the reference -> frame-0 step.

        A point needs valid data at BOTH ends for its increment to mean anything.
        Where it does not, the increment is NaN rather than a difference taken
        against a frozen or missing value.
        """
        def _val(r) -> np.ndarray:
            m = np.isfinite(r.u) & np.isfinite(r.v)
            return (m & r.valid) if r.valid is not None else m

        prev = None
        prev_val = None
        for res in self.results:
            here_val = _val(res)
            if prev is None:
                u_i = np.where(here_val, res.u, np.nan)
                v_i = np.where(here_val, res.v, np.nan)
            else:
                both = here_val & prev_val
                u_i = np.where(both, res.u - prev.u, np.nan)
                v_i = np.where(both, res.v - prev.v, np.nan)

            res.u_inc, res.v_inc = u_i, v_i
            with np.errstate(invalid="ignore"):
                res.mag_inc = np.sqrt(u_i ** 2 + v_i ** 2)

            prev, prev_val = res, here_val

    def _compute_velocities_and_rates(self, progress_cb: Optional[Callable[[float, str], None]] = None) -> None:
        N = len(self.results)
        if N < 2: return
        dt = 1.0 / max(self.fps, 1e-9)

        # A point can be valid at frames i-1 and i+1 but NaN at i (a single
        # dropped frame that healed), or vice-versa. Differencing blindly
        # propagates one bad/missing frame into its TWO neighbours' velocity.
        # Use the same central/forward/backward difference as before, but pick
        # it per-point from whichever neighbours are actually finite.
        #
        # Only a 3-frame sliding window is held. Stacking all N frames needed
        # 2 * N * H * W * 8 bytes -- at 1280x720 that is ~15 MB per frame per
        # component, so a 1700-frame sequence asked for ~50 GB up front and
        # simply raised MemoryError before any velocity was produced.
        def _val(k):
            r = self.results[k]
            m = np.isfinite(r.u) & np.isfinite(r.v)
            return (m & r.valid) if r.valid is not None else m

        prev_val = None
        here_val = _val(0)
        for i in range(N):
            next_val = _val(i + 1) if i < N - 1 else None

            u_here, v_here = self.results[i].u, self.results[i].v
            Vx = np.full(u_here.shape, np.nan)
            Vy = np.full(u_here.shape, np.nan)

            zero = np.zeros_like(here_val)
            has_prev = prev_val if prev_val is not None else zero
            has_next = next_val if next_val is not None else zero

            central = here_val & has_prev & has_next
            fwd_only = here_val & ~central & has_next
            bwd_only = here_val & ~central & ~fwd_only & has_prev

            if central.any():
                u_p, v_p = self.results[i - 1].u, self.results[i - 1].v
                u_n, v_n = self.results[i + 1].u, self.results[i + 1].v
                Vx[central] = (u_n[central] - u_p[central]) / (2 * dt)
                Vy[central] = (v_n[central] - v_p[central]) / (2 * dt)
            if fwd_only.any():
                u_n, v_n = self.results[i + 1].u, self.results[i + 1].v
                Vx[fwd_only] = (u_n[fwd_only] - u_here[fwd_only]) / dt
                Vy[fwd_only] = (v_n[fwd_only] - v_here[fwd_only]) / dt
            if bwd_only.any():
                u_p, v_p = self.results[i - 1].u, self.results[i - 1].v
                Vx[bwd_only] = (u_here[bwd_only] - u_p[bwd_only]) / dt
                Vy[bwd_only] = (v_here[bwd_only] - v_p[bwd_only]) / dt
            # A point valid only at frame i, with neither neighbour usable, has
            # no basis for a velocity estimate -- left as NaN rather than 0.0,
            # which previously looked like "material genuinely at rest".

            res = self.results[i]
            res.Vx, res.Vy = Vx, Vy
            with np.errstate(invalid="ignore"):
                res.Veff = np.sqrt(Vx ** 2 + Vy ** 2)

            prev_val, here_val = here_val, next_val

        from .strain import compute_velocity_strains
        mask = self._roi_mask if self._roi_mask is not None else np.ones_like(self.results[0].u, dtype=bool)

        for i, res in enumerate(self.results):
            if progress_cb:
                p = 0.90 + 0.07 * (i / max(1, N))
                progress_cb(p, f"[{i + 1}/{N}] Computing strain rates…")

            valid = mask & ~np.isnan(res.Vx) & ~np.isnan(res.Vy)
            rates = compute_velocity_strains(res.Vx, res.Vy, valid, self.params.effective_strain_window())

            res.dVx_dx = rates["dVx_dx"]
            res.dVx_dy = rates["dVx_dy"]
            res.dVy_dx = rates["dVy_dx"]
            res.dVy_dy = rates["dVy_dy"]
            res.Exx_rate = rates["Exx_rate"]
            res.Exy_rate = rates["Exy_rate"]
            res.Eyy_rate = rates["Eyy_rate"]
            res.Eeff_rate = rates["Eeff_rate"]

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

    # ------------------------------------------------------------------
    # Marker-seeded trajectories
    # ------------------------------------------------------------------
    # Displacement fields are only populated at correlation grid points, so a
    # marker dropped at an arbitrary pixel needs local interpolation rather than
    # a direct lookup.

    def _sample_sparse(self, arr: np.ndarray, x: float, y: float,
                       search: Optional[int] = None) -> float:
        """Inverse-distance sample of a sparse (NaN-filled) field at a float position."""
        if arr is None:
            return float("nan")
        H, W = arr.shape
        if search is None:
            search = max(4, int(getattr(self.params, "subset_spacing", 3)) * 3)
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - search), min(W, xi + search + 1)
        y0, y1 = max(0, yi - search), min(H, yi + search + 1)
        if x1 <= x0 or y1 <= y0:
            return float("nan")
        win = arr[y0:y1, x0:x1]
        fin = np.isfinite(win)
        if not fin.any():
            return float("nan")
        ys, xs = np.nonzero(fin)
        vals = win[fin]
        d2 = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
        k = min(4, len(d2))
        idx = np.argpartition(d2, k - 1)[:k]
        d2k, vk = d2[idx], vals[idx]
        if d2k.min() < 1e-9:
            return float(vk[int(np.argmin(d2k))])
        w = 1.0 / d2k
        return float((vk * w).sum() / w.sum())

    def reference_from_current(self, x: float, y: float, frame_idx: int
                              ) -> Optional[tuple[tuple[float, float], float]]:
        """
        Map a click on the DISPLAYED (deformed) frame back to reference coordinates.

        Markers are stored in reference coordinates so that a marker follows the
        same material point for the whole sequence. Without this, a marker placed
        while scrubbed to frame 200 would be interpreted as a reference-frame
        position and would track the wrong material.

        Returns ((x_ref, y_ref), residual_px) or None.
        """
        if not self.results:
            return None
        idx = max(0, min(int(frame_idx), len(self.results) - 1))
        res = self.results[idx]
        v = np.isfinite(res.u) & np.isfinite(res.v)
        if not v.any():
            return None
        ys, xs = np.nonzero(v)
        cx = xs + res.u[ys, xs]
        cy = ys + res.v[ys, xs]
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        i = int(np.argmin(d2))
        return (float(xs[i]), float(ys[i])), float(np.sqrt(d2[i]))

    def marker_positions(self, seeds, frame_idx: int) -> list[Optional[tuple[float, float]]]:
        """Where each reference-frame marker sits on the displayed frame."""
        if not self.results or not seeds:
            return [None] * len(seeds)
        idx = max(0, min(int(frame_idx), len(self.results) - 1))
        res = self.results[idx]
        out = []
        for (sx, sy) in seeds:
            u = self._sample_sparse(res.u, sx, sy)
            v = self._sample_sparse(res.v, sx, sy)
            out.append(None if not (np.isfinite(u) and np.isfinite(v))
                       else (float(sx + u), float(sy + v)))
        return out

    def get_trajectories_from_seeds(self, seeds, max_frame: int, trail: int = 0
                                   ) -> list[dict]:
        """
        Trace one trajectory per user-placed marker.

        seeds      : list of (x, y) in REFERENCE-frame coordinates
        max_frame  : trace up to and including this displayed frame
        trail      : 0 = full history, else keep only the last `trail` segments

        Each entry: {"points": [(x,y), ...], "lost_at": int|None, "seed": (x,y)}
        Strictly speaking these are pathlines (trajectories of individual
        material points). They coincide with streaklines only in steady flow --
        worth keeping in mind for segmented-chip conditions.
        """
        if not self.results or not seeds:
            return []
        last = min(int(max_frame), len(self.results) - 1)
        out: list[dict] = []
        for (sx, sy) in seeds:
            pts: list[tuple[float, float]] = [(float(sx), float(sy))]
            lost_at = None
            for i in range(0, last + 1):
                u = self._sample_sparse(self.results[i].u, sx, sy)
                v = self._sample_sparse(self.results[i].v, sx, sy)
                if not (np.isfinite(u) and np.isfinite(v)):
                    lost_at = i
                    break
                pts.append((float(sx + u), float(sy + v)))
            if trail and trail > 0 and len(pts) > trail + 1:
                pts = pts[-(trail + 1):]
            out.append({"points": pts, "lost_at": lost_at,
                        "seed": (float(sx), float(sy))})
        return out

    # ------------------------------------------------------------------
    # Frame-pair analysis
    # ------------------------------------------------------------------
    # A "pair" is two frames (i, j) treated as one measurement interval. Every
    # quantity below is derived from that interval alone, so several pairs drawn
    # from different parts of a sequence can be averaged into one field.
    #
    # Cumulative strain is deliberately absent. Exx/Eyy/Exy/Eeff measure total
    # deformation since the reference, so they carry the whole history before
    # the pair even begins -- averaging them across pairs would average that
    # shared history, not the pairs. Rates and displacements are interval
    # quantities and average correctly.

    PAIR_FIELDS = ("u", "v", "u_inc", "v_inc", "mag_inc",
                   "Vx", "Vy", "Veff",
                   "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                   "Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate")

    def pair_interval(self, i: int, j: int) -> float:
        """Elapsed time between two frames, in seconds."""
        return abs(j - i) / max(self.fps, 1e-9)

    def pair_kinematics(self, i: int, j: int) -> "PairResult":
        """Displacement, velocity and strain rate between frames i and j.

        u/v and u_inc/v_inc both carry the pair's own displacement: an isolated
        interval has no "cumulative" value distinct from its own increment.
        """
        n = len(self.results)
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"Frame pair ({i}, {j}) out of range 0..{n - 1}")
        if i == j:
            raise ValueError("A frame pair needs two different frames.")
        if j < i:
            i, j = j, i

        a, b = self.results[i], self.results[j]

        def _val(r):
            m = np.isfinite(r.u) & np.isfinite(r.v)
            return (m & r.valid) if r.valid is not None else m

        ok = _val(a) & _val(b)
        du = np.where(ok, b.u - a.u, np.nan)
        dv = np.where(ok, b.v - a.v, np.nan)
        with np.errstate(invalid="ignore"):
            mag = np.sqrt(du ** 2 + dv ** 2)

        dt = self.pair_interval(i, j)
        Vx, Vy = du / dt, dv / dt
        with np.errstate(invalid="ignore"):
            Veff = np.sqrt(Vx ** 2 + Vy ** 2)

        # Rates come from this pair's own mean velocity field, so they describe
        # the interval rather than borrowing a neighbouring frame's derivative.
        from .strain import compute_velocity_strains
        roi = self._roi_mask if self._roi_mask is not None else np.ones(du.shape, dtype=bool)
        rate_valid = roi & np.isfinite(Vx) & np.isfinite(Vy)
        rates = compute_velocity_strains(
            Vx, Vy, rate_valid, self.params.effective_strain_window())

        nan = np.full(du.shape, np.nan)
        return PairResult(
            image_path=f"pair {i + 1}→{j + 1}",
            u=du, v=dv,
            u_inc=du, v_inc=dv, mag_inc=mag,
            Exx=nan.copy(), Exy=nan.copy(), Eyy=nan.copy(), Eeff=nan.copy(),
            du_dx=nan.copy(), du_dy=nan.copy(),
            dv_dx=nan.copy(), dv_dy=nan.copy(),
            corr=np.where(ok, b.corr, np.nan),
            Vx=Vx, Vy=Vy, Veff=Veff,
            dVx_dx=rates["dVx_dx"], dVx_dy=rates["dVx_dy"],
            dVy_dx=rates["dVy_dx"], dVy_dy=rates["dVy_dy"],
            Exx_rate=rates["Exx_rate"], Exy_rate=rates["Exy_rate"],
            Eyy_rate=rates["Eyy_rate"], Eeff_rate=rates["Eeff_rate"],
            valid=ok, elapsed=dt,
        )

    def average_pairs(self, pairs) -> "PairResult":
        """Element-wise mean across several frame pairs.

        Averaging ignores NaN per pixel, so a point that dropped out during one
        pair still contributes through the pairs where it was tracked. A point
        present in no pair stays NaN rather than becoming zero.
        """
        pairs = [tuple(p) for p in pairs]
        if not pairs:
            raise ValueError("Select at least one frame pair.")

        per_pair = [self.pair_kinematics(i, j) for i, j in pairs]

        out = per_pair[0]
        if len(per_pair) > 1:
            for name in self.PAIR_FIELDS:
                stack = [getattr(p, name) for p in per_pair]
                stack = [s for s in stack if s is not None]
                if not stack:
                    continue
                with np.errstate(invalid="ignore"):
                    # all-NaN pixels warn under plain nanmean; they are expected
                    # here and must stay NaN.
                    arr = np.stack(stack)
                    counts = np.sum(np.isfinite(arr), axis=0)
                    summed = np.nansum(arr, axis=0)
                    mean = np.where(counts > 0, summed / np.maximum(counts, 1), np.nan)
                setattr(out, name, mean)
            out.valid = np.any(np.stack([p.valid for p in per_pair]), axis=0)
            out.elapsed = float(np.mean([p.elapsed for p in per_pair]))

        label = ", ".join(f"{i + 1}→{j + 1}" for i, j in pairs)
        out.image_path = (f"average of {len(pairs)} pairs [{label}]"
                          if len(pairs) > 1 else f"pair {label}")
        return out

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

    def export_csv(self, result_index, directory: str) -> None:
        """Write the selected frame's fields as CSV.

        Accepts a frame index or a PairResult directly, so a frame-pair average
        exports through exactly the same path as a single frame.
        """
        res = (self.results[result_index] if isinstance(result_index, (int, np.integer))
               else result_index)
        base = os.path.splitext(os.path.basename(res.image_path))[0]
        # Pair labels ("average of 3 pairs [1→2, ...]") are not filenames.
        base = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in base).strip("_")
        base = base or "export"
        fields = ("u", "v", "u_inc", "v_inc", "mag_inc",
                  "Exx", "Exy", "Eyy", "Eeff",
                  "Vx", "Vy", "Veff",
                  "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                  "Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate", "corr")
        # CSV is a one-way export, so write the values as displayed and state the
        # unit in the header. (HDF5 is different: it must round-trip, so it keeps
        # raw pixels plus the calibration as metadata.) With no calibration set
        # this is byte-identical to the old behaviour.
        for name in fields:
            arr = getattr(res, name, None)
            if arr is None:
                continue
            # A field with nothing finite in it has no data to export. This is
            # how cumulative strain is skipped for a frame-pair average, where
            # it is deliberately undefined -- writing a grid of "nan" would
            # read as a failed measurement rather than an excluded one.
            if arr.size == 0 or not np.any(np.isfinite(arr)):
                continue
            base_unit = _FIELD_BASE_UNIT.get(name, "")
            out, unit = self.calibration.convert(name, arr, base_unit)
            header = f"{name} [{unit}]" if unit else name
            if self.calibration.calibrated:
                header += f"  ({self.calibration.describe()})"
            np.savetxt(os.path.join(directory, f"{base}_{name}.csv"), out,
                       delimiter=",", header=header)

    def export_hdf5(self, path: str, progress_cb: Optional[Callable[[float], None]] = None) -> None:
        import h5py
        with h5py.File(path, "w") as f:
            # 1. Save Global Attributes
            f.attrs.update(dict(
                reference_image=self.ref_path or "",
                subset_radius=self.params.subset_radius,
                subset_spacing=self.params.subset_spacing,
                strain_window=self.params.strain_window,
                fps=self.fps,
                # Datasets stay in the solver's native pixel units so a file
                # round-trips exactly; the calibration rides along as metadata
                # and is what the viewer applies on display.
                metres_per_pixel=(self.calibration.metres_per_pixel
                                  if self.calibration.calibrated else 0.0),
                display_unit=self.calibration.display_unit,
                length_units="pixels",
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
                if progress_cb:
                    progress_cb(i / len(self.results))
                g = f.create_group(f"frame_{i:04d}")
                g.attrs["image_path"] = res.image_path
                g.attrs["elapsed_s"] = res.elapsed
                fields = ("u", "v", "u_inc", "v_inc", "mag_inc",
                          "Exx", "Exy", "Eyy", "Eeff",
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

            mpp = float(f.attrs.get("metres_per_pixel", 0.0) or 0.0)
            self.calibration = Calibration(
                mpp if mpp > 0 else None,
                str(f.attrs.get("display_unit", "mm")))

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
                extra_fields = ("u_inc", "v_inc", "mag_inc",
                                "Vx", "Vy", "Veff", "dVx_dx", "dVx_dy", "dVy_dx", "dVy_dy",
                                "Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate")
                for rate in extra_fields:
                    if rate in g:
                        setattr(res, rate, g[rate][:])
                self.results.append(res)

        # Files written before incremental displacement existed carry only the
        # cumulative fields. Deriving them here costs one subtraction per frame
        # and spares the user a full re-run to see the new view.
        if self.results and any(r.u_inc is None for r in self.results):
            self._compute_incremental_displacements()

    def _get_settings_path(self) -> str:
        import os
        # PYDIC_SETTINGS_PATH lets a test (or a second instance) point at its own
        # file. Relying on HOME/USERPROFILE for that is unreliable -- expanduser
        # consults several variables in a platform-specific order, so a test can
        # believe it is sandboxed and still write over the real settings.
        override = os.environ.get("PYDIC_SETTINGS_PATH")
        if override:
            return override
        return os.path.join(os.path.expanduser("~"), ".pydic_settings.json")

    def load_settings(self) -> None:
        import json, os
        path = self._get_settings_path()

        if os.path.exists(path):
            try:
                # utf-8-sig, not the platform default. A settings file carrying a
                # UTF-8 BOM (anything that has touched it from PowerShell, an
                # editor, or a sync tool) otherwise raised JSONDecodeError, the
                # except branch below quietly kept the defaults, and the next
                # save_settings() wrote those defaults straight over the user's
                # real values. A read failure must never be able to destroy the
                # file -- see the guard in the handler.
                with open(path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)

                # Settings written before the tolerance/cutoff semantics changed
                # must not be replayed: conv_tol is now in pixels of subset-edge
                # motion (1e-6 px is unreachable) and corr_cutoff=2.0 means
                # "accept everything" on the [0,4] ZNSSD scale. Drop those keys
                # and let the current defaults stand.
                SCHEMA = 2
                stale = int(data.get("schema_version", 1)) < SCHEMA
                DROP_ON_MIGRATE = {"conv_tol", "corr_cutoff"}
                migrated = []
                if "calibration" in data:
                    self.calibration = Calibration.from_dict(data.get("calibration"))

                for k, v in data.items():
                    if k in ("schema_version", "calibration"):
                        continue
                    if stale and k in DROP_ON_MIGRATE:
                        migrated.append(k)
                        self.settings_notices.append(
                            f"“{k}” was written under older semantics and has been "
                            f"reset to the current default "
                            f"({getattr(self.params, k, '?')}).")
                        continue
                    if hasattr(self.params, k):
                        setattr(self.params, k, v)
                # The schema gate above only fires once, so a value that is
                # unreachable under the CURRENT semantics can survive in a file
                # already stamped with the current schema. conv_tol is the one
                # that matters: it is a subset-edge motion in pixels, so
                # anything below ~1e-4 px can never be met and every subset
                # burns all max_iter iterations before giving up -- several
                # times slower for no gain in accuracy. Clamp on load.
                CONV_TOL_MIN, CONV_TOL_MAX = 1e-4, 1e-1
                ct = float(self.params.conv_tol)
                if not (CONV_TOL_MIN <= ct <= CONV_TOL_MAX):
                    clamped = min(max(ct, CONV_TOL_MIN), CONV_TOL_MAX)
                    print(f"[Settings] conv_tol={ct:g} px is outside the usable "
                          f"range [{CONV_TOL_MIN:g}, {CONV_TOL_MAX:g}]; using {clamped:g}.")
                    self.params.conv_tol = clamped
                    migrated.append("conv_tol")
                    self.settings_notices.append(
                        f"Convergence tolerance was {ct:g} px, which no subset can "
                        f"ever reach — every one ran the full iteration budget. "
                        f"Raised to {clamped:g} px.")

                if migrated:
                    print(f"[Settings] Migrated to schema {SCHEMA}; reset to new "
                          f"defaults: {', '.join(sorted(set(migrated)))}")
                    self.save_settings()

                # Load all specialized directories
                dirs = ["last_video_directory", "last_image_directory", "last_hdf5_directory"]
                for d in dirs:
                    if d in data and os.path.exists(data[d]):
                        setattr(self, d, data[d])

            except Exception as e:
                # Do NOT let a later save overwrite a file we could not read.
                # Failing to parse means the user's real settings are still in
                # there; silently replacing them with defaults turns a recoverable
                # read problem into permanent data loss.
                self._settings_readable = False
                print(f"[Warning] Failed to load settings: {e}")
                self.settings_notices.append(
                    f"Your settings file could not be read ({e}). Defaults are in "
                    f"use for this session and the file has been left untouched: "
                    f"{path}")
        else:
            self.save_settings()

    def save_settings(self) -> None:
        import json, os
        path = self._get_settings_path()

        # Refuse to write over a file we failed to parse -- the values in it are
        # the user's, and defaults would clobber them irrecoverably.
        if not getattr(self, "_settings_readable", True):
            print(f"[Settings] Not overwriting unreadable settings file: {path}")
            return

        try:
            data = {
                "schema_version": 2,
                "subset_radius": self.params.subset_radius,
                "subset_spacing": self.params.subset_spacing,
                "strain_window": self.params.strain_window,
                "max_iter": self.params.max_iter,
                "conv_tol": self.params.conv_tol,
                "corr_cutoff": self.params.corr_cutoff,
                "search_radius": self.params.search_radius,
                "dynamic_roi": getattr(self.params, "dynamic_roi", "None"),
                "dynamic_roi_threshold": getattr(self.params, "dynamic_roi_threshold", None),
                "dynamic_roi_min_area_frac": getattr(self.params, "dynamic_roi_min_area_frac", 0.02),
                "dynamic_roi_fill_holes": getattr(self.params, "dynamic_roi_fill_holes", True),
                "calibration": self.calibration.to_dict(),
                "shape_order": getattr(self.params, "shape_order", 1),
                "mask_subsets_to_roi": getattr(self.params, "mask_subsets_to_roi", True),

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

class DynamicROI:
    """
    Texture-based dynamic ROI with a STATIONARY decision rule.

    The previous implementation recomputed both the normalisation and an Otsu
    threshold from every individual frame. Two consequences:

      * per-frame max normalisation meant one specular flash or the tool edge
        rescaled the whole metric, moving every pixel relative to the threshold;
      * per-frame Otsu moved the threshold itself as the scene evolved.

    Together they make the ROI boundary jitter frame to frame, which is what
    produces a ragged, frame-varying mask edge -- and because a masked-out point
    loses its accumulated history, that jitter compounds over a sequence.

    Here the scale and threshold are calibrated ONCE from the reference frame and
    then held fixed, so "enough texture to correlate" means the same thing in
    frame 500 as in frame 1.
    """

    def __init__(self, method: str, keep_min_area_frac: float = 0.02,
                 threshold: Optional[float] = None,
                 include_mask: Optional[np.ndarray] = None,
                 exclude_mask: Optional[np.ndarray] = None,
                 roi_mask: Optional[np.ndarray] = None,
                 fill_holes: bool = True):
        self.method = method
        self.keep_min_area_frac = keep_min_area_frac
        # Fill regions that the texture metric rejected but which are completely
        # enclosed by kept material. A hole in the middle of valid material is
        # almost always a local dropout -- a glare spot, a patch where the
        # speckle is momentarily washed out -- not an absence of specimen, and
        # leaving it out fragments the field for no physical reason. Morphological
        # closing alone only ever fixed holes smaller than its kernel.
        self.fill_holes = bool(fill_holes)
        # The static ROI the operator drew. The dynamic mask REFINES that region
        # rather than competing with it: nothing outside it is ever kept, and --
        # just as importantly -- the threshold is calibrated only from pixels
        # inside it. Calibrating over the whole frame let background and tooling
        # dominate the statistics, so the threshold that came back described the
        # scene rather than the specimen.
        self.roi_mask = None if roi_mask is None else roi_mask.astype(bool)
        # Normalised texture threshold in [0, 1]. None means "pick it with Otsu
        # on the reference frame", which is the old behaviour and stays the
        # default; the dynamic-ROI editor sets it explicitly when the user
        # drags the slider.
        self.threshold = threshold
        # Hard user overrides, in REFERENCE-frame coordinates. include wins over
        # exclude wins over the texture metric, so a region the operator marked
        # is never silently reinterpreted frame to frame.
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask
        self.scale = None
        self.thresh = None
        self.auto_thresh = None
        self.enabled = method not in ("None", None) and _HAVE_CV2
        if method not in ("None", None) and not _HAVE_CV2:
            print("[Warning] cv2 not available for dynamic ROI. Ignoring.")

    def _metric(self, img: np.ndarray) -> Optional[np.ndarray]:
        if self.method == "Contrast":
            mean = cv2.blur(img, (9, 9))
            var = cv2.blur(img ** 2, (9, 9)) - mean ** 2
            return np.sqrt(np.maximum(var, 0))
        if self.method == "Edge Detection":
            gxx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            gyy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(gxx ** 2 + gyy ** 2)
        if self.method == "Hybrid":
            mean = cv2.blur(img, (9, 9))
            var = cv2.blur(img ** 2, (9, 9)) - mean ** 2
            std = np.sqrt(np.maximum(var, 0))
            gxx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            gyy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            grad = np.sqrt(gxx ** 2 + gyy ** 2)
            # Put both terms on a comparable footing before adding them; the raw
            # sum was dominated by the unnormalised Sobel response.
            sd = std / max(std.std(), 1e-12)
            gd = grad / max(grad.std(), 1e-12)
            return sd + gd
        return None

    def calibrate(self, ref_image: np.ndarray) -> None:
        if not self.enabled:
            return
        m = self._metric(ref_image)
        if m is None:
            self.enabled = False
            return
        # Scale and Otsu threshold from ROI pixels only -- see roi_mask above.
        sel = m[self.roi_mask] if (self.roi_mask is not None and self.roi_mask.any()) else m
        self.scale = float(np.percentile(sel, 99.0)) or 1.0
        m8 = self._metric8(ref_image, m)
        sel8 = m8[self.roi_mask] if (self.roi_mask is not None and self.roi_mask.any()) else m8
        thr, _ = cv2.threshold(sel8.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.auto_thresh = float(thr)
        self.thresh = (self.auto_thresh if self.threshold is None
                       else float(np.clip(self.threshold, 0.0, 1.0)) * 255.0)

    def _metric8(self, img: np.ndarray, m: Optional[np.ndarray] = None):
        """Texture metric rescaled to uint8 against the calibrated scale."""
        if m is None:
            m = self._metric(img)
        if m is None:
            return None
        s = self.scale if self.scale else (float(np.percentile(m, 99.0)) or 1.0)
        m8 = np.clip(m / s * 255.0, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(m8, (5, 5), 0)

    # ---- helpers for the dynamic-ROI editor -------------------------------
    def metric_normalised(self, img: np.ndarray) -> Optional[np.ndarray]:
        """The texture metric as a 0..1 image, for preview and histogramming."""
        m8 = self._metric8(img)
        return None if m8 is None else m8.astype(np.float64) / 255.0

    def auto_threshold_normalised(self) -> Optional[float]:
        return None if self.auto_thresh is None else self.auto_thresh / 255.0

    def mask(self, cur_image: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        if self.thresh is None:
            self.calibrate(cur_image)
        m8 = self._metric8(cur_image)
        if m8 is None:
            return None
        mask = (m8 >= self.thresh).astype(np.uint8) * 255

        # Operator overrides are applied BEFORE the morphology and
        # connected-component steps so that a hand-included region takes part in
        # closing and cannot be discarded as a too-small component.
        if self.exclude_mask is not None:
            mask[self.exclude_mask] = 0
        if self.include_mask is not None:
            mask[self.include_mask] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Keep every sufficiently large connected region, not just the largest.
        # In machining the chip and the workpiece are separate bodies (and the
        # tool splits the field); keeping only the largest contour discards one
        # of them outright, and which one is "largest" can flip between frames.
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_lab <= 1:
            out = mask > 0
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            min_area = self.keep_min_area_frac * float(areas.max())
            keep = np.zeros(n_lab, bool)
            keep[1:] = areas >= min_area
            out = keep[labels]

        # Clip to the static ROI before filling, so "enclosed" means enclosed
        # within the region being analysed rather than within the whole frame.
        if self.roi_mask is not None:
            out = out & self.roi_mask

        # Contour fill: anything fully surrounded by kept material becomes kept.
        # Unlike the morphological close above, this has no size limit -- a hole
        # is filled because it is enclosed, not because it is small.
        if self.fill_holes and out.any():
            from scipy.ndimage import binary_fill_holes
            out = binary_fill_holes(out)
            if self.roi_mask is not None:
                out = out & self.roi_mask

        # Re-apply the operator's decisions LAST. Setting them before morphology
        # was not enough to make them stick, and both were being partly undone:
        #
        #   * closing is a dilate-then-erode, so it grew neighbouring kept
        #     material back over the edge of an excluded region;
        #   * component pruning deleted a force-included patch whenever it was
        #     smaller than keep_min_area_frac of the largest region -- measured
        #     at 0 of 900 px surviving, i.e. include did nothing at all.
        #
        # Doing it here also means an explicit Exclude beats the hole fill, which
        # is the escape hatch when an enclosed gap is a genuine void rather than
        # a texture dropout. Precedence: include > exclude > fill > metric.
        if self.exclude_mask is not None:
            out = out & ~self.exclude_mask
        if self.include_mask is not None:
            out = out | self.include_mask
            if self.roi_mask is not None:
                out = out & self.roi_mask
        return out


# The old _compute_dynamic_mask() wrapper is gone. It recalibrated scale and
# threshold on whichever frame it was handed, which is precisely the per-frame
# drift DynamicROI exists to avoid, and it had no way to receive the static ROI
# or the operator's include/exclude regions. Build the object through
# DICAnalysis.make_dynamic_roi() and calibrate it once on the reference frame.
