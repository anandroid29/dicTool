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
from .strain import compute_strains


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
        """
        Load a binary ROI mask from an image file, .npy array, or Ncorr MAT file.
        Supported formats: PNG, TIF, JPG, BMP (white=ROI), .npy, .mat/.h5.
        The mask must match the reference image dimensions.
        """
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

        for i, def_path in enumerate(self.def_paths):
            if self._cancel[0]:
                break
            off, scale = i / n, 0.95 / n

            def pair_cb(frac, msg, _i=i, _n=n):
                if progress_cb:
                    progress_cb(_i/n + frac*0.95/_n, f"[{_i+1}/{_n}] {msg}")

            pair_cb(0, f"Loading {os.path.basename(def_path)}…")
            cur = _load_image(def_path)
            if cur.shape != ref.shape:
                raise ValueError(f"Shape mismatch: {def_path}")

            t0 = time.perf_counter()
            dic: DICResult = run_rg_dic(
                ref, cur, mask, self.params,
                seed_xy=seed_xy, progress_cb=pair_cb, cancel_flag=self._cancel,
            )
            elapsed = time.perf_counter() - t0

            pair_cb(0.98, "Strains…")
            valid = dic.analyzed & ~np.isnan(dic.u)
            sf = compute_strains(dic.u, dic.v, valid, self.params.strain_window)

            self.results.append(PairResult(
                image_path=def_path,
                u=dic.u, v=dic.v,
                Exx=sf["Exx"], Exy=sf["Exy"], Eyy=sf["Eyy"], Eeff=sf["Eeff"],
                du_dx=sf["du_dx"], du_dy=sf["du_dy"],
                dv_dx=sf["dv_dx"], dv_dy=sf["dv_dy"],
                corr=dic.corr, elapsed=elapsed,
            ))

        if not self._cancel[0] and self.results:
            if progress_cb:
                progress_cb(0.97, "Computing strain rates…")
            self._compute_strain_rates()

        if progress_cb:
            progress_cb(1.0, "Complete.")

        N = len(self.results)
        dt = 1.0 / max(self.fps, 1e-9)

        for i, res in enumerate(self.results):
            if N == 1:
                nan = np.full_like(res.Exx, np.nan)
                res.Exx_rate = res.Exy_rate = res.Eyy_rate = res.Eeff_rate = nan.copy()
                continue

            if i == 0:
                prev, nxt, dt_tot = res, self.results[1], dt
            elif i == N - 1:
                prev, nxt, dt_tot = self.results[i-1], res, dt
            else:
                prev, nxt, dt_tot = self.results[i-1], self.results[i+1], 2*dt

            def _fd(a, b):
                with np.errstate(invalid="ignore"):
                    return (b - a) / dt_tot

            res.Exx_rate  = _fd(prev.Exx,  nxt.Exx)
            res.Exy_rate  = _fd(prev.Exy,  nxt.Exy)
            res.Eyy_rate  = _fd(prev.Eyy,  nxt.Eyy)
            res.Eeff_rate = _fd(prev.Eeff, nxt.Eeff)

    # -- export --
    def export_csv(self, result_index: int, directory: str) -> None:
        res = self.results[result_index]
        base = os.path.splitext(os.path.basename(res.image_path))[0]
        for name in ("u","v","Exx","Exy","Eyy","Eeff",
                     "Exx_rate","Exy_rate","Eyy_rate","Eeff_rate","corr"):
            arr = getattr(res, name, None)
            if arr is not None:
                np.savetxt(os.path.join(directory,f"{base}_{name}.csv"), arr, delimiter=",")

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
                g.attrs["elapsed_s"]  = res.elapsed
                for name in ("u","v","Exx","Exy","Eyy","Eeff",
                             "Exx_rate","Exy_rate","Eyy_rate","Eeff_rate",
                             "du_dx","du_dy","dv_dx","dv_dy","corr"):
                    arr = getattr(res, name, None)
                    if arr is not None:
                        g.create_dataset(name, data=arr.astype(np.float32),
                                         compression="gzip", compression_opts=4)



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
