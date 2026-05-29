"""
roi_loader.py
-------------
Load a Region-of-Interest mask from various file formats — mirrors Ncorr's
"Load ROI from file" functionality (ncorr_gui_setrois.m, set_roi_ref_source).

Supported sources
-----------------
Image file   (.png, .tif, .tiff, .jpg, .bmp)
    Any grayscale or colour image.  White / bright pixels = inside ROI,
    black / dark pixels = outside.  Threshold = 50 % of max (matches
    MATLAB's im2bw default of 0.5).

NumPy array  (.npy)
    A boolean or uint8 (H, W) array saved with np.save.

Ncorr MAT   (.mat, .h5)
    MATLAB v7.3 save file from Ncorr.  The reference-image ROI mask is
    extracted automatically from the HDF5 tree.

The loaded mask must match the reference image size.  A ValueError is raised
if it does not.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_roi_mask(path: str,
                  expected_shape: Optional[Tuple[int, int]] = None,
                  ) -> np.ndarray:
    """
    Load an ROI mask from *path* and return a bool (H, W) numpy array.

    Parameters
    ----------
    path : str
        Path to the file (image, .npy, or .mat / .h5).
    expected_shape : (H, W) tuple, optional
        If given, raises ValueError when the loaded mask does not match.

    Returns
    -------
    mask : (H, W) bool ndarray
        True = inside ROI.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        mask = _load_npy(path)
    elif ext in (".mat", ".h5", ".hdf5"):
        mask = _load_ncorr_mat(path)
    else:
        mask = _load_image(path)

    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(
            f"Loaded ROI shape {mask.shape} does not match reference image "
            f"shape {expected_shape}. The mask must be the same size as the "
            f"reference image."
        )
    return mask


def save_roi_mask(mask: np.ndarray, path: str) -> None:
    """
    Save an ROI mask as a PNG file (white = ROI, black = background).
    The saved file can be reloaded with load_roi_mask().
    """
    try:
        from PIL import Image
        arr = (mask.astype(np.uint8)) * 255
        Image.fromarray(arr, mode="L").save(path)
        return
    except ImportError:
        pass
    try:
        import cv2
        arr = (mask.astype(np.uint8)) * 255
        cv2.imwrite(path, arr)
        return
    except ImportError:
        pass
    raise ImportError("Install Pillow or opencv-python to save ROI images.")


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray:
    """Load image file and threshold at 50 % of max (matches MATLAB im2bw)."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float64)
        return arr > (arr.max() * 0.5)
    except ImportError:
        pass
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read: {path}")
        return img > int(img.max() * 0.5)
    except ImportError:
        pass
    raise ImportError("Install Pillow or opencv-python to load image ROI files.")


def _load_npy(path: str) -> np.ndarray:
    """Load .npy file as bool mask."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"NumPy ROI array must be 2-D, got shape {arr.shape}.")
    return arr.astype(bool)


def _load_ncorr_mat(path: str) -> np.ndarray:
    """
    Extract the reference-image ROI mask from a Ncorr v7.3 MAT file.

    Ncorr stores the ROI as a uint8 binary array somewhere in the HDF5 tree.
    We search for all binary arrays that have the same shape as the reference
    image and pick the one referenced by data_dic_save/displacements/roi_dic.
    Falls back to the largest binary array if the reference cannot be resolved.
    """
    import h5py

    with h5py.File(path, "r") as f:
        # Try the canonical path first
        try:
            roi_refs = f["data_dic_save/displacements/roi_ref_formatted"]
            # roi_ref_formatted[0] is the reference ROI for image index 0
            roi_obj = f[roi_refs[0, 0]]
            # This may be an HDF5 group with a 'mask' dataset, or a dataset
            if isinstance(roi_obj, h5py.Dataset):
                arr = roi_obj[:]
                if arr.dtype == np.uint8 and arr.ndim == 2:
                    return arr.astype(bool)
        except Exception:
            pass

        # Fallback: scan all datasets for binary 2-D uint8 arrays
        candidates = {}

        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.dtype == np.uint8 and obj.ndim == 2:
                    arr = obj[:]
                    if set(np.unique(arr)).issubset({0, 1}):
                        candidates[name] = arr

        f.visititems(_visit)

        if not candidates:
            raise ValueError(
                "No binary ROI mask found in the Ncorr MAT file. "
                "Try exporting the ROI as a PNG from Ncorr instead."
            )

        # Pick the largest mask (most likely to be the full ROI)
        best = max(candidates, key=lambda k: candidates[k].sum())
        mask = candidates[best].astype(bool)

        # Ncorr stores masks in (x, y) = (col, row) = (W, H) MATLAB layout.
        # The actual image is (H, W). Transpose if the mask looks transposed.
        # Heuristic: if mask is taller than wide and image appears landscape, transpose.
        # We return it as-is and let the caller check shape vs reference image.
        return mask
