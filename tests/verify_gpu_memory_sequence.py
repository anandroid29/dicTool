"""Manual 20-frame GPU memory regression for full-resolution result history."""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "output" / "verification"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["PYDIC_SETTINGS_PATH"] = str(OUT / "memory_test_settings.json")

from pydic.core.analysis import DICAnalysis
from pydic.core.cuda_native import native_cuda_memory_info


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_bytes() -> int:
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong)
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError()
    return int(counters.WorkingSetSize)


def retained_array_bytes(results) -> int:
    seen = set()
    total = 0
    for result in results:
        for value in vars(result).values():
            if isinstance(value, np.ndarray) and id(value) not in seen:
                seen.add(id(value))
                total += value.nbytes
    return total


def main() -> None:
    paths = [ROOT / "sample video_frames" / f"frame_{i:06d}.png"
             for i in range(21)]
    ref = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise RuntimeError(f"Cannot read {paths[0]}")
    h, w = ref.shape

    # A compact textured ROI keeps runtime practical while preserving the real
    # full-frame result dimensions that caused history RAM to grow per frame.
    local_std = cv2.blur(ref.astype(np.float32) ** 2, (17, 17))
    local_std -= cv2.blur(ref.astype(np.float32), (17, 17)) ** 2
    local_std[:80] = -1
    local_std[max(80, h - 80):] = -1
    local_std[:, :80] = -1
    local_std[:, max(80, w - 80):] = -1
    sy, sx = np.unravel_index(int(np.argmax(local_std)), local_std.shape)
    roi = np.zeros_like(ref, dtype=bool)
    half = 70
    roi[max(0, sy-half):min(h, sy+half+1),
        max(0, sx-half):min(w, sx+half+1)] = True

    analysis = DICAnalysis()
    analysis.set_reference(str(paths[0]))
    analysis.set_roi_mask(roi)
    for path in paths[1:]:
        analysis.add_deformed(str(path))
    analysis.fps = 2412.0
    analysis.params.subset_radius = 7
    analysis.params.subset_spacing = 8
    analysis.params.strain_window = 3
    analysis.params.max_iter = min(analysis.params.max_iter, 12)
    analysis.params.search_radius = min(analysis.params.search_radius, 24)
    analysis.params.dynamic_roi = "None"

    samples = []
    last_frame = None

    def sample(frame, message):
        nonlocal last_frame
        if frame == last_frame:
            return
        last_frame = frame
        free_bytes, total_bytes = native_cuda_memory_info()
        samples.append({
            "completed_frames": frame,
            "rss_mib": rss_bytes() / (1024 ** 2),
            "gpu_used_mib": (total_bytes - free_bytes) / (1024 ** 2),
            "gpu_free_mib": free_bytes / (1024 ** 2),
            "gpu_total_mib": total_bytes / (1024 ** 2),
            "message": message,
        })

    def progress(_frac, message):
        if message.startswith("[") and "] Loading " in message:
            current = int(message[1:message.index("/")])
            sample(current - 1, message)

    sample(0, "before run")
    analysis.run(progress_cb=progress, seed_xy=(int(sx), int(sy)), use_gpu=True)
    sample(len(analysis.results), "complete")

    unique_bytes = retained_array_bytes(analysis.results)
    evidence = {
        "frames": len(analysis.results),
        "shape": [h, w],
        "roi_pixels": int(roi.sum()),
        "seed": [int(sx), int(sy)],
        "retained_result_mib": unique_bytes / (1024 ** 2),
        "retained_result_mib_per_frame": unique_bytes / max(1, len(analysis.results)) / (1024 ** 2),
        "all_numeric_result_arrays_float32": all(
            value.dtype == np.float32
            for result in analysis.results for value in vars(result).values()
            if isinstance(value, np.ndarray) and value.dtype.kind == "f"),
        "displacement_aliases_shared": all(
            result.u_inc is result.u and result.v_inc is result.v
            for result in analysis.results),
        "samples": samples,
    }
    path = OUT / "gpu_memory_20_frames.json"
    path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
