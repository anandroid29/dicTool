"""
units.py — spatial calibration for reporting results in physical units.

The solver works entirely in pixels: displacements come out of IC-GN in pixels,
velocities in pixels per second. That is the honest native unit and nothing in
the core changes here. This module only converts for presentation and export.

Only LENGTH-dimensioned fields are affected. Strain is a ratio and strain rate
is a ratio per second, so both are already independent of how big a pixel is --
scaling them by a calibration would be flatly wrong.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

# Metres per unit.
LENGTH_UNITS = {
    "m":  1.0,
    "mm": 1e-3,
    "µm": 1e-6,
    "nm": 1e-9,
}
# Order for dropdowns: coarse to fine.
LENGTH_UNIT_ORDER = ["m", "mm", "µm", "nm"]

# Which fields carry a length dimension, and how.
#   "length"   -> px      becomes  <unit>
#   "velocity" -> px/s    becomes  <unit>/s
# Anything absent is dimensionless (strain) or already physical (strain rate)
# and is passed through untouched.
FIELD_DIMENSION = {
    "u": "length",
    "v": "length",
    "u_inc": "length",
    "v_inc": "length",
    "mag_inc": "length",
    "Vx": "velocity",
    "Vy": "velocity",
    "Veff": "velocity",
}


class Calibration:
    """Pixel size plus the unit results should be reported in.

    Uncalibrated (metres_per_pixel is None) is the default and means everything
    stays in pixels, exactly as before.
    """

    def __init__(self, metres_per_pixel: Optional[float] = None,
                 display_unit: str = "mm") -> None:
        self.metres_per_pixel = metres_per_pixel
        self.display_unit = display_unit if display_unit in LENGTH_UNITS else "mm"

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_pixel_size(cls, value: float, unit: str) -> "Calibration":
        """e.g. from_pixel_size(2.5, 'µm') -> one pixel is 2.5 µm across."""
        if value is None or value <= 0 or unit not in LENGTH_UNITS:
            return cls(None, unit if unit in LENGTH_UNITS else "mm")
        return cls(float(value) * LENGTH_UNITS[unit], unit)

    @classmethod
    def from_known_length(cls, pixels: float, length: float, unit: str) -> "Calibration":
        """Calibrate from a feature of known size spanning `pixels` pixels."""
        if not pixels or pixels <= 0:
            return cls(None, unit)
        return cls.from_pixel_size(float(length) / float(pixels), unit)

    @property
    def calibrated(self) -> bool:
        return self.metres_per_pixel is not None and self.metres_per_pixel > 0

    def pixel_size_in(self, unit: str) -> Optional[float]:
        if not self.calibrated or unit not in LENGTH_UNITS:
            return None
        return self.metres_per_pixel / LENGTH_UNITS[unit]

    # -- conversion -----------------------------------------------------------
    def factor_and_unit(self, field_key: str, base_unit: str = "") -> Tuple[float, str]:
        """Multiplier and label to turn a native (pixel) field into display units.

        Returns (1.0, base_unit) whenever there is nothing to convert, so callers
        can use this unconditionally.
        """
        dim = FIELD_DIMENSION.get(field_key)
        if dim is None or not self.calibrated:
            return 1.0, base_unit
        per_unit = LENGTH_UNITS[self.display_unit]
        factor = self.metres_per_pixel / per_unit
        if dim == "length":
            return factor, self.display_unit
        return factor, f"{self.display_unit}/s"      # velocity

    def compact_factor_and_unit(
            self, field_key: str, magnitude: float, base_unit: str = "",
            threshold: float = 100.0) -> Tuple[float, str]:
        """Display a calibrated length field in a readable engineering unit.

        ``display_unit`` remains the unit in which pixel size was entered and
        persisted.  For presentation only, values at or above ``threshold``
        are promoted toward the next larger available unit.  For example,
        100 µm/s is shown as 0.1 mm/s instead of a three-digit µm/s value.
        """
        factor, unit = self.factor_and_unit(field_key, base_unit)
        dim = FIELD_DIMENSION.get(field_key)
        if (dim is None or not self.calibrated or
                not math.isfinite(float(magnitude)) or threshold <= 0):
            return factor, unit

        index = LENGTH_UNIT_ORDER.index(self.display_unit)
        shown = abs(float(magnitude)) * factor
        while shown >= threshold and index > 0:
            index -= 1
            promoted = LENGTH_UNIT_ORDER[index]
            factor = self.metres_per_pixel / LENGTH_UNITS[promoted]
            shown = abs(float(magnitude)) * factor

        promoted = LENGTH_UNIT_ORDER[index]
        return factor, (promoted if dim == "length" else f"{promoted}/s")

    def convert(self, field_key: str, arr, base_unit: str = ""):
        """(scaled_array, unit_label). The array is returned as-is when no
        scaling applies, so this does not copy needlessly."""
        factor, unit = self.factor_and_unit(field_key, base_unit)
        if arr is None or factor == 1.0:
            return arr, unit
        return arr * factor, unit

    def convert_value(self, field_key: str, value: float, base_unit: str = ""):
        factor, unit = self.factor_and_unit(field_key, base_unit)
        return (None if value is None else value * factor), unit

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"metres_per_pixel": self.metres_per_pixel,
                "display_unit": self.display_unit}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Calibration":
        if not d:
            return cls()
        return cls(d.get("metres_per_pixel"), d.get("display_unit", "mm"))

    def describe(self) -> str:
        if not self.calibrated:
            return "Uncalibrated — results in pixels"
        px = self.pixel_size_in(self.display_unit)
        return f"1 px = {px:.6g} {self.display_unit}"
