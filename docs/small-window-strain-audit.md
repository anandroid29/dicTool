# Small-window strain-rate investigation

Reported configuration: displayed frame 13, subset radius 5 px, grid spacing
1 px, strain half-window 1 px, temporal span 1. Investigated the r05/g01
correlation and derived files in both `External Testing vids/Parametric analysis`
and `External Testing vids/good_vid_frames`. Files were read without modification.

## Findings

The strain fit does not introduce the observed fine texture independently of
the displacement data. With these settings it differentiates a 3-by-3 local
group containing at most nine displacement measurements (six are required).
At 100 frames/s, small spatial variations in displacement become much larger
strain-rate variations.

For frame 13 in the first sweep:

- CPU versus native CUDA maximum displacement-gradient difference: 3.6e-15.
- Parametric integral-image fit versus CPU maximum difference: 8.9e-9.
- Independent `numpy.linalg.lstsq` fits at 1,000 sampled centres versus CPU:
  maximum difference 1.3e-15.
- No finite-support disagreements between the fits or cached rate.
- Cached rate versus recomputed rate: maximum absolute difference 0.000226 /s,
  at rates reaching 5,009 /s, consistent with the cache's float32 precision.

The second sweep gives the same conclusion (GPU difference below 1.8e-15,
independent-fit difference below 1.6e-15).

There are also suspect correlation matches, a separate problem from small-window
noise amplification. In the first sweep's reported frame, horizontal displacement
has a median of 3.22 px but extremes of -45.2 and +51.2 px; vertical displacement
ranges from -42.7 to +65.2 px. Such points can create extreme derivative spikes.
These ranges identify candidates for examining correspondence quality; they do
not justify indiscriminately rejecting large real motion or smoothing across a
physical shear discontinuity.

On frame 13, lowering the correlation-cost cutoff from 0.30 to 0.05 removes
some extreme spikes but also rejects 22% of measured centres in the broad
shear-band region sampled for this audit. The maximum remaining rate is still
119 /s. A global stricter cutoff therefore is not a safe automatic repair.

The viewer's cyan/magenta colours are explicit lower/upper colour-range clipping
flags. A 99% automatic range flags part of the data even when those samples are
valid. They are not a separate tensor value or an indication of solver failure.

## Controlled zero-strain experiment

Native CUDA, 100-by-112 synthetic texture, Gaussian blur sigma 0.8, random seed
71, rigid shift `(u,v)=(2.4,-0.35)` px generated with cubic interpolation,
50 iterations, tolerance 0.001, cutoff 0.3, grid 1, 100 frames/s. Measurements
exclude a 20-pixel image boundary. Ground-truth strain is zero. This experiment
includes image resampling error as well as correlation error.

| Subset radius | Displacement RMS error | Rate RMS, window 1 | Window 3 | Window 5 | Window 9 |
|---|---:|---:|---:|---:|---:|
| 5 px | 0.01165 px | 0.7341 /s | 0.2451 /s | 0.1381 /s | 0.0512 /s |
| 12 px | 0.00440 px | 0.1876 /s | 0.0746 /s | 0.0462 /s | 0.0243 /s |

This demonstrates that visible fine strain-rate texture can exist without a
defect in the gradient calculation. It does not establish that all features of
the real specimen are noise, or validate every accepted displacement match.

Ncorr's [strain-calculation explanation](https://ncorr.com/index.php/dic-algorithms)
also describes the amplification of displacement noise by differentiation and
the tradeoff between spatial resolution and strain-window size.

## Outcome

No production algorithm or saved analysis was changed on the basis of the
screenshot. Added independent local least-squares regression checks for
translation, affine and nonlinear fields at the minimum window, for CPU, native
CUDA, and parametric fits, including missing points and grid spacings 1 and 3.

Diagnostic outputs are in `output/verification/frame13_strain_audit.json`,
`rigid_translation_strain_audit.json`, and `frame13_common_scale.png`. The figure
uses a fixed 0–30 /s scale for all four windows and omits the image background
and clipping flags to make the field itself comparable.
