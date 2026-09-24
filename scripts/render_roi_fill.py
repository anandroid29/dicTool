"""Render a verification sheet for the contour-filled sweep ROI."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from strainx.core.analysis import DynamicROI, _load_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    with h5py.File(args.source, "r") as handle:
        static_mask = handle["roi_mask"][:].astype(bool)
        reference_path = str(handle.attrs["reference_image"])
        include = (handle["dynamic_include_mask"][:].astype(bool)
                   if "dynamic_include_mask" in handle else None)
        exclude = (handle["dynamic_exclude_mask"][:].astype(bool)
                   if "dynamic_exclude_mask" in handle else None)
        method = str(handle.attrs.get("dynamic_roi", "None"))
        threshold = float(handle.attrs.get("dynamic_roi_threshold", np.nan))
        threshold = threshold if np.isfinite(threshold) else None
        min_area = float(handle.attrs.get("dynamic_roi_min_area_frac", 0.02))

    reference = _load_image(reference_path)
    dynamic_roi = DynamicROI(
        method, keep_min_area_frac=min_area, threshold=threshold,
        include_mask=include, exclude_mask=exclude, roi_mask=static_mask,
        fill_holes=True, hysteresis=0.0)
    dynamic_roi.calibrate(reference)
    mask = dynamic_roi.mask(reference, reference_frame=True).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    filled &= static_mask.astype(np.uint8)
    added = filled.astype(bool) & ~mask.astype(bool)

    frame = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise FileNotFoundError(reference_path)
    base = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def panel(region: np.ndarray, colour: tuple[int, int, int], title: str):
        image = base.copy()
        layer = np.empty_like(image)
        layer[:] = colour
        selected = region.astype(bool)
        if selected.any():
            image[selected] = cv2.addWeighted(
                image[selected], 0.45, layer[selected], 0.55, 0.0)
        cv2.putText(
            image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            (255, 255, 255), 2, cv2.LINE_AA)
        return image

    sheet = np.concatenate((
        panel(static_mask, (220, 110, 30), "Static ROI (not used)"),
        panel(mask, (30, 160, 220), "Saved Dynamic ROI"),
        panel(filled, (30, 180, 60),
              f"Filled Dynamic ROI (+{added.sum():,})"),
    ), axis=1)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    verification = args.output_directory / "roi_fill_verification.png"
    cv2.imwrite(str(verification), sheet)
    np.save(args.output_directory / "roi_filled.npy", filled.astype(bool))
    print(
        f"static={static_mask.sum():,} dynamic={mask.sum():,} "
        f"filled={filled.sum():,} added={added.sum():,} "
        f"external_contours={len(contours)} method={method} "
        f"threshold={threshold}")
    print(verification.resolve())


if __name__ == "__main__":
    main()
