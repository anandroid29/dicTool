"""Thin ctypes boundary to PyDIC's native C++/CUDA runtime.

The compiled library owns every numerical operation used by CUDA mode,
including host-side NCC/wavefront scheduling. Python supplies contiguous input
arrays and receives dense result arrays; it never owns a device pointer or
launches a kernel.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import threading
from typing import Optional

import numpy as np


class NativeCudaError(RuntimeError):
    pass


_LIB = None
_LOAD_ERROR: Optional[str] = None
_LOAD_LOCK = threading.Lock()

_F64_P = C.POINTER(C.c_double)
_U8_P = C.POINTER(C.c_uint8)


def _library_candidates() -> list[Path]:
    repository = Path(__file__).resolve().parents[3]
    build = repository / "build" / "native_cuda" / "bin"
    names = ("pydic_cuda.dll", "libpydic_cuda.so", "libpydic_cuda.dylib")
    candidates: list[Path] = []
    override = os.environ.get("PYDIC_CUDA_LIBRARY")
    if override:
        candidates.append(Path(override).expanduser())
    for directory in (build, build / "Release", build / "Debug"):
        candidates.extend(directory / name for name in names)
    return candidates


def _configure(lib) -> None:
    lib.pydic_cuda_version.argtypes = []
    lib.pydic_cuda_version.restype = C.c_char_p
    lib.pydic_cuda_abi_version.argtypes = []
    lib.pydic_cuda_abi_version.restype = C.c_uint32
    lib.pydic_cuda_last_error.argtypes = []
    lib.pydic_cuda_last_error.restype = C.c_char_p
    lib.pydic_cuda_device_count.argtypes = []
    lib.pydic_cuda_device_count.restype = C.c_int
    lib.pydic_cuda_synchronize.argtypes = []
    lib.pydic_cuda_synchronize.restype = C.c_int
    lib.pydic_cuda_memory_info.argtypes = [
        C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    lib.pydic_cuda_memory_info.restype = C.c_int
    lib.pydic_cuda_solver_create.argtypes = [
        C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_double, C.c_double,
        C.c_int]
    lib.pydic_cuda_solver_create.restype = C.c_void_p
    lib.pydic_cuda_solver_destroy.argtypes = [C.c_void_p]
    lib.pydic_cuda_solver_destroy.restype = None
    lib.pydic_cuda_solver_precompute.argtypes = [
        C.c_void_p, _F64_P, _U8_P, C.c_int, C.c_int]
    lib.pydic_cuda_solver_precompute.restype = C.c_int
    lib.pydic_cuda_solver_ncc.argtypes = [
        C.c_void_p, _F64_P, C.c_int, C.c_double, C.c_double,
        C.POINTER(C.c_double), C.POINTER(C.c_double),
        C.POINTER(C.c_double)]
    lib.pydic_cuda_solver_ncc.restype = C.c_int
    lib.pydic_cuda_solver_icgn.argtypes = [
        C.c_void_p, _F64_P, C.c_int, _F64_P, _F64_P,
        C.POINTER(C.c_double), C.POINTER(C.c_uint8)]
    lib.pydic_cuda_solver_icgn.restype = C.c_int
    lib.pydic_cuda_solver_solve.argtypes = [
        C.c_void_p, _F64_P, C.c_int, C.c_int, C.c_double, C.c_double,
        _F64_P, _F64_P, _F64_P, _F64_P, _F64_P, _F64_P, _F64_P]
    lib.pydic_cuda_solver_solve.restype = C.c_int
    lib.pydic_cuda_solver_update_reference.argtypes = [C.c_void_p, _F64_P]
    lib.pydic_cuda_solver_update_reference.restype = C.c_int
    lib.pydic_cuda_plane_fit.argtypes = [
        _F64_P, _F64_P, _U8_P, C.c_int, C.c_int, C.c_int,
        _F64_P, _F64_P, _F64_P, _F64_P]
    lib.pydic_cuda_plane_fit.restype = C.c_int


def _load_library():
    global _LIB, _LOAD_ERROR
    if _LIB is not None:
        return _LIB
    with _LOAD_LOCK:
        if _LIB is not None:
            return _LIB
        errors = []
        for candidate in _library_candidates():
            if not candidate.is_file():
                continue
            try:
                lib = C.CDLL(str(candidate))
                _configure(lib)
                abi = int(lib.pydic_cuda_abi_version())
                if abi != 2:
                    raise OSError(
                        f"native CUDA ABI {abi} is incompatible; expected 2")
                _LIB = lib
                _LOAD_ERROR = None
                return lib
            except (OSError, AttributeError) as exc:
                errors.append(f"{candidate}: {exc}")
        _LOAD_ERROR = ("; ".join(errors) if errors else
                       "Native CUDA library not found. Run scripts/build_cuda.py "
                       "with an NVIDIA CUDA Toolkit installation.")
        raise NativeCudaError(_LOAD_ERROR)


def _native_error(lib) -> str:
    raw = lib.pydic_cuda_last_error()
    return raw.decode("utf-8", errors="replace") if raw else "Unknown native CUDA error."


def native_cuda_available(*, refresh: bool = False) -> bool:
    global _LIB, _LOAD_ERROR
    if refresh:
        with _LOAD_LOCK:
            _LIB = None
            _LOAD_ERROR = None
    try:
        return _load_library().pydic_cuda_device_count() > 0
    except Exception:
        return False


def native_cuda_library_present() -> bool:
    """Cheap UI-time check that never loads CUDA or creates a device context."""
    return any(candidate.is_file() for candidate in _library_candidates())


def native_cuda_diagnostic() -> str:
    try:
        lib = _load_library()
        count = int(lib.pydic_cuda_device_count())
        if count <= 0:
            return _native_error(lib)
        version = (lib.pydic_cuda_version() or b"unknown").decode("ascii", "replace")
        return f"Native CUDA {version}; {count} CUDA device(s) available"
    except Exception as exc:
        return str(exc)


def native_cuda_memory_info() -> tuple[int, int]:
    """Return (free_bytes, total_bytes) from the native CUDA runtime."""
    lib = _load_library()
    free_bytes, total_bytes = C.c_uint64(), C.c_uint64()
    if lib.pydic_cuda_memory_info(C.byref(free_bytes), C.byref(total_bytes)) != 0:
        raise NativeCudaError(_native_error(lib))
    return int(free_bytes.value), int(total_bytes.value)


def _f64(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def _u8(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.uint8)


def _f64_ptr(array: Optional[np.ndarray]):
    return None if array is None else array.ctypes.data_as(_F64_P)


class NativeCudaSolver:
    """Python lifetime wrapper around one native solver session."""

    FRESH = 0
    WARM_START = 1
    RECOVER_FAILED = 2

    def __init__(self, params) -> None:
        self.params = params
        if int(getattr(params, "shape_order", 1)) != 1:
            raise NativeCudaError(
                "Native CUDA currently supports only first-order affine "
                "shape functions; select CPU mode for second-order DIC.")
        self._lib = _load_library()
        if self._lib.pydic_cuda_device_count() <= 0:
            raise NativeCudaError(_native_error(self._lib))
        self._handle = self._lib.pydic_cuda_solver_create(
            int(params.subset_radius), int(params.subset_spacing),
            int(params.search_radius), int(getattr(params, "rescue_radius", 12)),
            int(params.max_iter), float(params.conv_tol), float(params.corr_cutoff),
            int(bool(getattr(params, "mask_subsets_to_roi", True))))
        if not self._handle:
            raise NativeCudaError(_native_error(self._lib))
        self._initialized = False
        self._current_image = None

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            self._lib.pydic_cuda_solver_destroy(handle)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def precompute_reference(self, reference: np.ndarray, roi_mask: np.ndarray) -> None:
        reference = _f64(reference)
        roi = _u8(roi_mask)
        if reference.ndim != 2 or roi.shape != reference.shape:
            raise ValueError("Reference image and ROI must be same-sized 2-D arrays.")
        self.H, self.W = reference.shape
        r, spacing = int(self.params.subset_radius), int(self.params.subset_spacing)
        ys = np.arange(r, self.H - r, spacing, dtype=np.int32)
        xs = np.arange(r, self.W - r, spacing, dtype=np.int32)
        gx, gy = np.meshgrid(xs, ys)
        self.grid_shape = (len(ys), len(xs))
        self.gx_flat = gx.ravel()
        self.gy_flat = gy.ravel()
        self.valid_mask = np.asarray(roi[self.gy_flat, self.gx_flat], dtype=bool)
        status = self._lib.pydic_cuda_solver_precompute(
            self._handle, _f64_ptr(reference), roi.ctypes.data_as(_U8_P),
            self.H, self.W)
        if status != 0:
            raise NativeCudaError(_native_error(self._lib))
        self._initialized = True
        self._current_image = None

    def _outputs(self):
        return tuple(np.empty((self.H, self.W), dtype=np.float64) for _ in range(7))

    def ncc_guess(self, current: np.ndarray, grid_index: int,
                  guess_u: float = 0.0,
                  guess_v: float = 0.0) -> tuple[float, float, float]:
        """Run only the native integer-pixel ZNCC seed stage."""
        if not self._initialized:
            raise NativeCudaError("Must precompute the native CUDA reference first.")
        current = _f64(current)
        if current.shape != (self.H, self.W):
            raise ValueError("Current image shape does not match the reference.")
        u, v, zncc = C.c_double(), C.c_double(), C.c_double()
        status = self._lib.pydic_cuda_solver_ncc(
            self._handle, _f64_ptr(current), int(grid_index),
            float(guess_u), float(guess_v),
            C.byref(u), C.byref(v), C.byref(zncc))
        if status != 0:
            raise NativeCudaError(_native_error(self._lib))
        self._current_image = current
        return float(u.value), float(v.value), float(zncc.value)

    def solve_subset(self, current: np.ndarray, grid_index: int,
                     initial_parameters: np.ndarray
                     ) -> tuple[np.ndarray, float, bool]:
        """Run only one native IC-GN solve for stage-parity validation."""
        if not self._initialized:
            raise NativeCudaError("Must precompute the native CUDA reference first.")
        current = _f64(current)
        initial = _f64(np.asarray(initial_parameters).reshape(6))
        if current.shape != (self.H, self.W):
            raise ValueError("Current image shape does not match the reference.")
        output = np.empty(6, dtype=np.float64)
        znssd, accepted = C.c_double(), C.c_uint8()
        status = self._lib.pydic_cuda_solver_icgn(
            self._handle, _f64_ptr(current), int(grid_index),
            _f64_ptr(initial), _f64_ptr(output), C.byref(znssd),
            C.byref(accepted))
        if status != 0:
            raise NativeCudaError(_native_error(self._lib))
        return output, float(znssd.value), bool(accepted.value)

    def _solve(self, current: Optional[np.ndarray], mode: int, seed_idx: int,
               guess_u: float, guess_v: float):
        if not self._initialized:
            raise NativeCudaError("Must precompute the native CUDA reference first.")
        current_array = None if current is None else _f64(current)
        if current_array is not None and current_array.shape != (self.H, self.W):
            raise ValueError("Current image shape does not match the reference.")
        outputs = self._outputs()
        status = self._lib.pydic_cuda_solver_solve(
            self._handle, _f64_ptr(current_array), int(mode), int(seed_idx),
            float(guess_u), float(guess_v),
            *(_f64_ptr(output) for output in outputs))
        if status != 0:
            raise NativeCudaError(_native_error(self._lib))
        if current_array is not None:
            self._current_image = current
        return outputs

    def solve_frame(self, current: np.ndarray, seed_idx: int = -1,
                    seed_p: Optional[np.ndarray] = None,
                    warm_start: bool = False, *, recovery_seeds=None,
                    seed_guess: Optional[tuple[float, float]] = None):
        if recovery_seeds is not None:
            guess = seed_guess or (0.0, 0.0)
            return self._solve(None, self.RECOVER_FAILED, -1, *guess)
        if warm_start:
            return self._solve(current, self.WARM_START, -1, 0.0, 0.0)
        if seed_guess is not None:
            guess_u, guess_v = seed_guess
        elif seed_p is not None:
            guess_u, guess_v = float(seed_p[0]), float(seed_p[1])
        else:
            guess_u = guess_v = 0.0
        return self._solve(current, self.FRESH, seed_idx, guess_u, guess_v)

    def recover_failed(self, guess_u: float = 0.0, guess_v: float = 0.0):
        return self._solve(None, self.RECOVER_FAILED, -1, guess_u, guess_v)

    def update_reference_image(self, new_reference: np.ndarray) -> None:
        promote = self._current_image is new_reference
        array = None if promote else _f64(new_reference)
        if array is not None and array.shape != (self.H, self.W):
            raise ValueError("Updated reference shape does not match the solver.")
        status = self._lib.pydic_cuda_solver_update_reference(
            self._handle, _f64_ptr(array))
        if status != 0:
            raise NativeCudaError(_native_error(self._lib))
        self._current_image = None

    @staticmethod
    def release_temporary_memory() -> None:
        lib = _load_library()
        if lib.pydic_cuda_synchronize() != 0:
            raise NativeCudaError(_native_error(lib))


def native_plane_fit(vx: np.ndarray, vy: np.ndarray, component_mask: np.ndarray,
                     radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lib = _load_library()
    if lib.pydic_cuda_device_count() <= 0:
        raise NativeCudaError(_native_error(lib))
    vx = _f64(vx)
    vy = _f64(vy)
    mask = _u8(component_mask)
    if vx.ndim != 2 or vy.shape != vx.shape or mask.shape != vx.shape:
        raise ValueError("Plane-fit arrays must have matching 2-D shapes.")
    outputs = tuple(np.empty_like(vx) for _ in range(4))
    status = lib.pydic_cuda_plane_fit(
        _f64_ptr(vx), _f64_ptr(vy), mask.ctypes.data_as(_U8_P),
        vx.shape[0], vx.shape[1], int(radius),
        *(_f64_ptr(output) for output in outputs))
    if status != 0:
        raise NativeCudaError(_native_error(lib))
    return outputs
