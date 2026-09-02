"""
stats.py — robust summary statistics for DIC result fields.

DIC fields are not clean. A correlation that lands on a specimen edge, a
reflection, or a decorrelated patch can return a displacement of hundreds of
pixels or a strain of several hundred percent while every neighbour reads a
fraction of a percent. Those points are a small fraction of the field and they
are not signal, but they dominate any statistic built on extremes -- min, max,
and the colour limits derived from them.

Everything here is therefore either explicitly robust (percentile-based, so a
bounded fraction of bad points cannot move the answer) or explicitly labelled
as a true extreme. Nothing modifies the data; these only describe it.

This lives in core rather than beside the renderer because both the display
layer and the analysis layer need the same definitions, and core must not
depend on the UI.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .compact_field import finite_values

# Below this many points a percentile is not meaningfully different from the
# extremes it is meant to guard against, so the true range is used instead.
MIN_SAMPLES_FOR_PERCENTILE = 20


def robust_limits(values: np.ndarray,
                  coverage: float = 100.0) -> Optional[Tuple[float, float]]:
    """Limits spanning the central `coverage` percent of the finite values.

    100 returns the true min/max. Anything less trims symmetric tails: 98 spans
    the 1st to 99th percentile, ignoring the most extreme 1% at each end.

    Returns None when there is nothing finite to describe, so callers can
    distinguish "no data" from "a valid range that happens to be narrow".
    """
    finite = finite_values(values)
    if finite.size == 0:
        return None

    cov = float(np.clip(coverage, 1.0, 100.0))
    if cov >= 100.0 or finite.size < MIN_SAMPLES_FOR_PERCENTILE:
        return float(finite.min()), float(finite.max())

    tail = (100.0 - cov) / 2.0
    lo, hi = np.percentile(finite, [tail, 100.0 - tail])
    return float(lo), float(hi)


def field_summary(values: np.ndarray) -> Optional[dict]:
    """Descriptive statistics for one field, robust and non-robust side by side.

    Mean and standard deviation are reported because they are what people
    expect, but a single decorrelated subset moves both without bound. Median
    and the interquartile range are reported alongside precisely so that a
    disagreement between the two is visible: when mean and median diverge, the
    field has outliers and the mean is not describing the material.

    `p_low`/`p_high` are the 1st and 99th percentiles -- practical extremes that
    survive a handful of bad points, as opposed to `minimum`/`maximum`, which
    are the true extremes and may well be those bad points.
    """
    finite = finite_values(values)
    if finite.size == 0:
        return None

    finite = finite.astype(np.float64, copy=False)

    # One percentile call for every quantile needed, not one per group. Each
    # call runs its own O(n) partition over the whole field, and this runs on
    # every rendered frame -- two calls doubled the cost of the statistics
    # panel for no benefit, since the quantiles come from the same data.
    if finite.size >= MIN_SAMPLES_FOR_PERCENTILE:
        p_low, q1, med, q3, p_high = np.percentile(finite, [1, 25, 50, 75, 99])
    elif finite.size >= 4:
        q1, med, q3 = np.percentile(finite, [25, 50, 75])
        p_low, p_high = float(finite.min()), float(finite.max())
    else:
        q1 = q3 = np.nan
        med = float(np.median(finite))
        p_low, p_high = float(finite.min()), float(finite.max())

    return {
        "count":   int(finite.size),
        "mean":    float(finite.mean()),
        "std":     float(finite.std()),
        "median":  float(med),
        "iqr":     float(q3 - q1) if np.isfinite(q1) else float("nan"),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "p_low":   float(p_low),
        "p_high":  float(p_high),
    }
