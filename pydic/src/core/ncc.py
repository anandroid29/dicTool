"""
ncc.py
------
Normalized Cross-Correlation (NCC) initial guess for DIC.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ncc_initial_guess(
    ref_image: np.ndarray,
    cur_image: np.ndarray,
    center_x: int,
    center_y: int,
    subset_radius: int,
    search_radius: int = 30,
) -> tuple[float, float, float]:
    """
    Find the integer-pixel displacement (u0, v0) of a subset by NCC.

    Parameters
    ----------
    ref_image, cur_image : (H, W) float arrays
        Reference and current greyscale images.
    center_x, center_y : int
        Pixel coordinates of the subset centre in the reference image.
    subset_radius : int
        Radius of the circular subset (a square of side 2r+1 is used here
        for the template, consistent with Ncorr's NCC pass).
    search_radius : int
        Half-extent of the search window beyond the template boundary.

    Returns
    -------
    u0, v0 : float
        Integer-pixel x and y displacements.
    ncc_score : float
        Peak NCC value (close to 1.0 = good match).
    """
    H, W = ref_image.shape
    r = subset_radius

    # ---- Template from reference image (square bounding box of circle) ----
    r1 = max(0, center_y - r)
    r2 = min(H, center_y + r + 1)
    c1 = max(0, center_x - r)
    c2 = min(W, center_x + r + 1)

    template = ref_image[r1:r2, c1:c2].astype(np.float32)
    th, tw = template.shape
    if th < 3 or tw < 3:
        return 0.0, 0.0, 0.0

    # ---- Search region in current image -----------------------------------
    sr1 = max(0, r1 - search_radius)
    sr2 = min(H, r2 + search_radius)
    sc1 = max(0, c1 - search_radius)
    sc2 = min(W, c2 + search_radius)

    search = cur_image[sr1:sr2, sc1:sc2].astype(np.float32)

    if search.shape[0] < th or search.shape[1] < tw:
        return 0.0, 0.0, 0.0

    # ---- Correlation ------------------------------------------------------
    if _HAVE_CV2:
        result = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        col0, row0 = max_loc 
    else:
        result = _fft_ncc(search, template)
        idx = np.unravel_index(np.argmax(result), result.shape)
        row0, col0 = idx
        score = float(result[row0, col0])
    match_row = sr1 + row0  
    match_col = sc1 + col0

    u0 = float(match_col - c1)
    v0 = float(match_row - r1)

    return u0, v0, float(score)

def _fft_ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Compute NCC between *image* and *template* using FFT cross-correlation.
    Returns correlation map with the same valid-region shape as cv2.matchTemplate.
    """
    ih, iw = image.shape
    th, tw = template.shape
    t = template - template.mean()
    t_norm = np.sqrt((t ** 2).sum())
    if t_norm < 1e-12:
        return np.zeros((ih - th + 1, iw - tw + 1), dtype=np.float32)

    t_norm_inv = 1.0 / t_norm

    pad_h = ih
    pad_w = iw
    F = np.fft.rfft2(image, s=(pad_h, pad_w))
    T = np.fft.rfft2(np.flipud(np.fliplr(t)), s=(pad_h, pad_w))
    cross = np.fft.irfft2(F * T, s=(pad_h, pad_w))
    from scipy.ndimage import uniform_filter
    img2 = image ** 2
    local_sum = uniform_filter(image.astype(np.float64), size=(th, tw)) * th * tw
    local_sum2 = uniform_filter(img2.astype(np.float64), size=(th, tw)) * th * tw
    local_std = np.sqrt(np.maximum(local_sum2 - local_sum ** 2 / (th * tw), 0.0))
    local_std = np.maximum(local_std, 1e-12)

    r_h = ih - th + 1
    r_w = iw - tw + 1
    cross_valid = cross[th - 1:th - 1 + r_h, tw - 1:tw - 1 + r_w]
    std_valid   = local_std[th // 2: th // 2 + r_h, tw // 2: tw // 2 + r_w]

    ncc = cross_valid * t_norm_inv / (std_valid * np.sqrt(th * tw) + 1e-12)
    return ncc.astype(np.float32)
