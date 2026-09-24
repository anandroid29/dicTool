"""
ncc.py
------
Normalized Cross-Correlation (NCC) initial guess for DIC.
"""

from __future__ import annotations
import numpy as np

import cv2

def ncc_initial_guess(
    ref_image: np.ndarray,
    cur_image: np.ndarray,
    center_x: int,
    center_y: int,
    subset_radius: int,
    search_radius: int = 30,
    guess_u: float = 0.0,
    guess_v: float = 0.0,
) -> tuple[float, float, float]:

    H, W = ref_image.shape
    r = subset_radius

    # ---- Template from reference image ----
    r1 = max(0, center_y - r)
    r2 = min(H, center_y + r + 1)
    c1 = max(0, center_x - r)
    c2 = min(W, center_x + r + 1)

    template = ref_image[r1:r2, c1:c2].astype(np.float32)
    th, tw = template.shape
    if th < 3 or tw < 3:
        return guess_u, guess_v, 0.0

    # ---- Shifted search region in current image ----
    tgt_x = int(round(center_x + guess_u))
    tgt_y = int(round(center_y + guess_v))

    sr1 = max(0, tgt_y - r - search_radius)
    sr2 = min(H, tgt_y + r + search_radius + 1)
    sc1 = max(0, tgt_x - r - search_radius)
    sc2 = min(W, tgt_x + r + search_radius + 1)

    search = cur_image[sr1:sr2, sc1:sc2].astype(np.float32)

    if search.shape[0] < th or search.shape[1] < tw:
        return guess_u, guess_v, 0.0

    # OpenCV is a required application dependency. Use zero-mean NCC.
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (col0, row0) = cv2.minMaxLoc(result)

    match_row = sr1 + row0
    match_col = sc1 + col0

    u0 = float(match_col - c1)
    v0 = float(match_row - r1)

    return u0, v0, float(score)
