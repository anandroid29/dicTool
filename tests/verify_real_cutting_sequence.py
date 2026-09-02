"""Run CPU and GPU on a late cutting-video segment and write evidence JSON.

This is intentionally a manual/integration verification, not part of the fast
unit-test suite. It uses frames where the curled chip is clipped by the image's
top boundary and compares the two backends on the same image-derived ROI.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["STRAINX_SETTINGS_PATH"] = str(ROOT / "output" / "verification" / "settings.json")

from strainx.core.analysis import DICAnalysis


def build_roi(ref_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read {ref_path}")
    h, w = img.shape
    region = np.zeros((h, w), np.uint8)
    # Late in this recording the chip occupies the upper-middle region and is
    # partially clipped by y=0. Exclude the workpiece and tool.
    region[0:min(260, h), max(520, w // 3):min(900, w)] = 1
    texture = (img > 28).astype(np.uint8)
    texture = cv2.morphologyEx(texture, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    texture = cv2.dilate(texture, np.ones((9, 9), np.uint8), iterations=1)
    mask = (region & texture).astype(bool)

    # Seed at the valid point with greatest local standard deviation.
    f = img.astype(np.float32)
    mean = cv2.blur(f, (17, 17))
    std = np.sqrt(np.maximum(cv2.blur(f * f, (17, 17)) - mean * mean, 0.0))
    score = np.where(mask, std, -1.0)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return mask, (int(x), int(y))


def summarize(analysis: DICAnalysis, elapsed: float) -> dict:
    valid_counts = []
    max_inf = []
    p999_inf = []
    max_gl = []
    max_increment_inf = []
    previous = None
    for result in analysis.results:
        valid = np.isfinite(result.u) & np.isfinite(result.v)
        valid_counts.append(int(valid.sum()))
        inf = result.Eeff_inf[np.isfinite(result.Eeff_inf)]
        gl = result.Eeff_gl[np.isfinite(result.Eeff_gl)]
        max_inf.append(float(inf.max()) if inf.size else None)
        p999_inf.append(float(np.percentile(inf, 99.9)) if inf.size else None)
        max_gl.append(float(gl.max()) if gl.size else None)
        if previous is None:
            increment = result.Eeff_inf
        else:
            both = np.isfinite(result.Eeff_inf) & np.isfinite(previous)
            increment = np.where(both, result.Eeff_inf - previous, np.nan)
        finite_increment = increment[np.isfinite(increment)]
        max_increment_inf.append(
            float(finite_increment.max()) if finite_increment.size else None)
        previous = result.Eeff_inf
    return {
        "elapsed_s": elapsed,
        "frames_completed": len(analysis.results),
        "valid_counts": valid_counts,
        "max_accumulated_Eeff_inf": max_inf,
        "p99_9_accumulated_Eeff_inf": p999_inf,
        "max_accumulated_Eeff_gl": max_gl,
        "max_frame_increment_Eeff_inf": max_increment_inf,
    }


def run_backend(name: str, use_gpu: bool, reference: Path,
                deformed: list[Path], mask: np.ndarray, seed: tuple[int, int]):
    analysis = DICAnalysis()
    analysis.set_reference(str(reference))
    analysis.set_roi_mask(mask)
    for path in deformed:
        analysis.add_deformed(str(path))
    analysis.fps = 2412.0
    analysis.params.dynamic_roi = "None"
    analysis.params.subset_radius = 10
    analysis.params.subset_spacing = 10
    analysis.params.strain_window = 20
    analysis.params.search_radius = 24
    analysis.params.rescue_radius = 8
    analysis.params.max_iter = 25
    analysis.params.conv_tol = 1e-3
    analysis.params.corr_cutoff = 0.5
    analysis.params.mask_subsets_to_roi = False
    start = time.perf_counter()
    analysis.run(seed_xy=seed, use_gpu=use_gpu)
    elapsed = time.perf_counter() - start
    return analysis, summarize(analysis, elapsed)


def compare(cpu: DICAnalysis, gpu: DICAnalysis) -> dict:
    rows = []
    for i, (a, b) in enumerate(zip(cpu.results, gpu.results)):
        common = (np.isfinite(a.u) & np.isfinite(a.v) &
                  np.isfinite(b.u) & np.isfinite(b.v))
        strain_common = common & np.isfinite(a.Eeff_inf) & np.isfinite(b.Eeff_inf)
        disp_err = np.hypot(a.u - b.u, a.v - b.v)
        strain_err = np.abs(a.Eeff_inf - b.Eeff_inf)
        rows.append({
            "frame_offset": i + 1,
            "common_displacement_points": int(common.sum()),
            "median_displacement_difference_px": (
                float(np.median(disp_err[common])) if common.any() else None),
            "p95_displacement_difference_px": (
                float(np.percentile(disp_err[common], 95)) if common.any() else None),
            "common_strain_points": int(strain_common.sum()),
            "median_Eeff_inf_difference": (
                float(np.median(strain_err[strain_common]))
                if strain_common.any() else None),
            "p95_Eeff_inf_difference": (
                float(np.percentile(strain_err[strain_common], 95))
                if strain_common.any() else None),
        })
    return {"per_frame": rows}


def main():
    frames_dir = ROOT / "sample video_frames"
    # The last eleven frames visibly contain a chip clipped by the top border.
    reference = frames_dir / "frame_001694.png"
    deformed = [frames_dir / f"frame_{i:06d}.png" for i in range(1695, 1699)]
    mask, seed = build_roi(reference)
    evidence = {
        "dataset": str(frames_dir),
        "reference": reference.name,
        "deformed": [p.name for p in deformed],
        "roi_pixels": int(mask.sum()),
        "seed_xy": seed,
    }

    analyses = {}
    for name, use_gpu in (("cpu", False), ("gpu", True)):
        try:
            analyses[name], evidence[name] = run_backend(
                name, use_gpu, reference, deformed, mask, seed)
            evidence[name]["status"] = "passed"
        except Exception as exc:
            evidence[name] = {"status": "failed", "error": repr(exc)}

    if "cpu" in analyses and "gpu" in analyses:
        evidence["cpu_gpu_comparison"] = compare(analyses["cpu"], analyses["gpu"])

    out = ROOT / "output" / "verification" / "real_cutting_cpu_gpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
