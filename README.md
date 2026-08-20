# PyDIC — 2D Digital Image Correlation

PyDIC measures full-field displacement, strain, velocity and strain rate from a
sequence of images of a deforming specimen. It replicates the algorithmic core
of Ncorr (Blaber, Adair & Antoniou, 2015) in a free, self-contained Python
package, and adds direct full-field **strain rate** — the capability that
motivated the project.

Built at IIT Kanpur. No MATLAB, no licence, no scripting required: the whole
pipeline runs through a six-step GUI.

---

## Contents

- [What it produces](#what-it-produces)
- [Installation](#installation)
- [The workflow](#the-workflow)
- [Getting accurate numbers](#getting-accurate-numbers) ← **read this before trusting a result**
- [Output fields](#output-fields)
- [Reading the results view](#reading-the-results-view)
- [Exports](#exports)
- [How the algorithms work](#how-the-algorithms-work)
- [GPU acceleration](#gpu-acceleration)
- [Project layout](#project-layout)
- [Validation and known limits](#validation-and-known-limits)
- [References](#references)

---

## What it produces

For every frame in a sequence, on the full correlation grid:

| Quantity | Fields | Notes |
|---|---|---|
| Displacement | `u`, `v`, `mag_inc` | Motion over **one frame interval**, not since the start |
| Velocity | `Vx`, `Vy`, `Veff` | Displacement ÷ Δt, in px/s or calibrated units/s |
| Strain rate | `Exx_rate`, `Eyy_rate`, `Exy_rate`, `Gxy_rate`, `Eeff_rate` | Rate-of-deformation tensor, s⁻¹ |
| Accumulated strain | `Exx_gl`, `Eyy_gl`, `Exy_gl`, `Eeff_gl` | Green–Lagrange, summed across the sequence |
| Velocity gradients | `dVx_dx`, `dVx_dy`, `dVy_dx`, `dVy_dy` | s⁻¹ |
| Correlation quality | `corr` | ZNSSD cost per subset |

Plus: interactive ROI drawing, automatic texture-based ROI tracking, marker
trajectories (streaklines), frame-pair averaging, video/image-sequence export,
and CSV/HDF5 output.

---

## Installation

```bash
git clone https://github.com/anandroid29/dicTool.git
cd dicTool
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
python pydic/main.py
```

Requires Python 3.10+. Dependencies are NumPy, SciPy, PyQt6, OpenCV, Matplotlib,
Pillow, scikit-image and h5py.

GPU acceleration is optional and needs CuPy plus an NVIDIA driver; see
[GPU acceleration](#gpu-acceleration). Everything works without it.

---

## The workflow

**1 · Import.** Load an image folder, or import a video and extract frames.
Colour images are converted to greyscale automatically.

> Video import asks for the **true capture rate**, which for high-speed footage
> is *not* the video's playback rate. See [Frame rate](#1-frame-rate-the-most-common-error).

**2 · ROI.** Draw the region to analyse with polygon, rectangle or circle tools,
erase parts of the mask, or load an existing binary mask. Everything outside the
ROI is ignored.

**3 · Dynamic ROI** *(skipped when off).* Tracks the specimen automatically as
it moves and deforms, so subsets that leave the material are dropped rather than
correlating against background. Modes are Contrast, Edge, Hybrid (default) or
None. The texture threshold is calibrated once on the reference frame so the
mask edge does not flicker frame to frame.

**4 · Parameters.** Solver settings — see the table below.

**5 · Analysis.** Runs the correlation. Progress is per frame and cancellable;
frames already completed are kept.

**6 · Results.** Field viewer with a temporal scrubber, colour-scale controls,
statistics, a value probe, marker trajectories, frame-pair averaging and export.

---

## Getting accurate numbers

Four settings determine whether the numbers mean anything. Everything else is
refinement.

### 1 · Frame rate — the most common error

Velocity and strain rate are divided by **Δt = 1 / frame rate**. Get the rate
wrong and every rate-dimensioned result is wrong by exactly that factor.

The trap: a video file stores a *playback* rate, and high-speed footage is
almost always written out slowed down. A 2000 Hz recording saved as a 50 fps
file will report 50 fps to any tool that asks the container — giving a Δt that
is **40× too large** and velocities **40× too small**.

PyDIC therefore asks for the true capture rate at import, pre-filled from the
container but always editable, and shows the resulting Δt so the number can be
sanity-checked before it propagates. If you subsample during extraction, the
effective sample rate is `capture rate ÷ step`, and that is what gets stored.

Displacement and strain are **unaffected** — they carry no time dimension.

### 2 · Spatial calibration

Without it, results are in pixels — the honest native unit. Set the pixel size
(Results view, `1 px =`) and displacement and velocity are reported in physical
units. Strain and strain rate are ratios and never change.

### 3 · Subset radius and spacing

| Larger subset radius | Larger spacing |
|---|---|
| Less noise | Faster |
| Blurs sharp strain gradients | Coarser grid |
| Needs more speckle inside it | Fewer points for the strain fit |

The strain window must contain enough grid points to fit a plane: the exact
count per axis is `2·⌊strain_window / subset_spacing⌋ + 1`, and fewer than three
gives an all-NaN strain field. PyDIC clamps this automatically and tells you it
did rather than returning empty results.

### 4 · Colour scale coverage

The colour scale is a *display* choice, but a bad one hides real data. See
[Colour scaling](#colour-scaling).

### Parameter reference

| Parameter | Default | What it controls |
|---|---|---|
| Subset radius | 21 px | Correlation window size |
| Subset spacing | 3 px | Distance between subset centres |
| Strain window | 15 px | Half-width of the least-squares plane fit |
| Shape order | 1 (affine) | 1 = 6-parameter affine; 2 = 12-parameter quadratic |
| IC-GN max iterations | 50 | Convergence limit |
| Convergence tolerance | 1×10⁻³ px | Subset-edge motion per iteration |
| ZNSSD cutoff | 0.30 | Max correlation cost; 0.30 ⇒ ZNCC ≥ 0.85 |
| NCC search radius | 50 px | Half-width of the per-frame seed search |
| Dynamic ROI | Hybrid | Contrast / Edge / Hybrid / None |

Second-order shape functions capture curvature inside the subset and reduce
systematic error where the strain gradient is high, at the cost of markedly
higher noise sensitivity. They need a well-textured, reasonably large subset to
be worth using.

The ZNSSD cutoff is worth understanding: ZNSSD of two unit-normalised subsets
lies in [0, 4] and equals 2(1 − ZNCC). A cutoff of 2.0 accepts ZNCC ≥ 0 — that
is, everything. The default 0.30 accepts ZNCC ≥ 0.85.

---

## Output fields

### Displacement is per-interval, strain is accumulated

This is the single most important thing to know when reading results.

Correlation runs **immediate-frame**: each frame is correlated against the one
before it, not against the original reference. So:

- **`u` and `v` are the motion over one frame interval.** At frame 900 they tell
  you what happened between frames 899 and 900 — not the total since the start.
  This is what makes velocity and strain rate meaningful at every frame.
- **Strain is accumulated** by a `StrainAccumulator` that sums each interval's
  contribution. `Exx_gl` and friends are the *total* deformation since the
  reference.

Mixing the two up is the easiest way to misread a result: a small `u` late in a
test does not mean the specimen barely moved, it means it barely moved *in that
interval*.

`u_inc` / `v_inc` exist as aliases of `u` / `v` for file and API compatibility.

### Equivalent strain and strain rate

`Eeff_gl` and `Eeff_rate` are von Mises equivalent measures assuming plastic
incompressibility, so the out-of-plane term is fixed at ε_zz = −(ε_xx + ε_yy):

```
ε_eq = sqrt( 2/3 · (ε_xx² + ε_yy² + ε_zz² + 2ε_xy²) )
```

which reduces to the textbook results — uniaxial → ε_xx, pure shear →
2ε_xy/√3, equibiaxial → 2ε_xx. Restoring ε_zz matters: omitting it leaves
equibiaxial strain reading **59% low**, because equibiaxial deformation is
carried almost entirely by the thickness change a 2-D tensor cannot see.

---

## Reading the results view

### Colour scaling

DIC fields reliably contain a few subsets that converged onto noise at an edge
or a dropout, with values orders of magnitude outside the real range. On a raw
min/max scale those few pixels own the entire colourbar and the actual field
flattens to one colour.

**Coverage** sets the central share of the data the scale must span. The default
99% ignores the extreme 0.5% at each tail; 100% restores true min/max. This
affects the *mapping only* — statistics and exports always report true values.

Because a trimmed scale hides data, it says so:

- Values above the range render **magenta**, below it **cyan**, instead of
  sitting at the end colour where they read as legitimate extremes
- The colourbar grows caps in those colours, on screen and in exports
- The fraction lying outside the range is stated numerically

Scale modes are **Auto** (this frame), **Global** (whole sequence, so frames stay
comparable) and **Range** (limits you type, in display units). **Sym** centres
the scale on zero, which is what you want for signed quantities.

The colormap is applied exactly as defined — nothing adjusts the colours
afterwards — so a colour on the image and the same colour on the bar mean the
same value.

### Statistics

Mean and standard deviation sit beside median, IQR and P1–P99. Both are shown on
purpose: when mean and median disagree the field contains outliers and the mean
is not describing the material. PyDIC flags the mean in amber when it parts from
the median by more than half the IQR. Min/Max remain, labelled as true extremes —
compare them with P1–P99 to see how far the tails reach.

### Value probe

Hover the image to read the stored value at that pixel, direct from the array
rather than inferred from the colour. It distinguishes *between subset centres*
(no data because the grid is sparse) from *not correlated here* (correlation
failed) — a distinction that matters when judging coverage.

### Marker trajectories

Place markers and PyDIC traces each one's path through the sequence, following
the material point. Trail length is configurable, and a marker whose history
breaks is drawn dashed and reported as lost.

### Frame-pair averaging

Select any number of frame pairs and average the displacement, velocity and
strain rate measured across them. Pairs can be added individually, or in bulk as
sequential (1→2, 2→3, …) or non-overlapping (1→2, 3→4, …) intervals.

This is the tool for noisy data. Displacement between two adjacent high-speed
frames is often a fraction of a pixel, so correlation noise is a large share of
it; averaging K independent pairs reduces that noise roughly as √K while leaving
the underlying motion intact.

Accumulated strain is **deliberately excluded** — it carries the whole history
preceding each pair, so averaging it across pairs would average that shared
history rather than the pairs. Those fields are disabled while averaging is
active.

---

## Exports

| Format | Contents |
|---|---|
| **CSV** | Every field of the current frame (or the current pair average), one file per field, with units in the header. Fields containing nothing finite are skipped rather than written as a grid of `nan`. |
| **HDF5** | The whole sequence, gzip-compressed, plus ROI mask, parameters, frame rate and calibration. Round-trips back into PyDIC. |
| **Video / image sequence** | Multi-panel mosaics — fields, raw frames, streaklines — with colourbars and labels. Rendered through the same code as the on-screen view, so exports cannot drift from what you saw. |

HDF5 stores results in native pixel units with the calibration as metadata, so a
file reopens identically regardless of the display unit chosen at export time.
CSV writes values as displayed, with the unit stated in the header.

---

## How the algorithms work

The implementation follows Ncorr closely. What follows is what actually happens
per frame.

### B-spline interpolation

Both images are prefiltered into quintic (5th-order) B-spline coefficient arrays
via `scipy.ndimage.spline_filter`, an IIR recursive filter numerically equivalent
to the FFT deconvolution Ncorr uses. Sub-pixel intensity is then

```
g(x̃, ỹ) = [1 Δx Δx² Δx³ Δx⁴ Δx⁵] · [QK] · C[xf-2:xf+3, yf-2:yf+3] · [QK]ᵀ · [1 Δy …]ᵀ
```

where [QK] is the 6×6 quintic kernel, C the coefficient array, xf = ⌊x̃⌋ and
Δx = x̃ − xf. Gradients come from the same coefficient array rather than
finite-differencing raw pixels, which would introduce a systematic bias.

### Seeding by NCC

One subset per frame gets an integer-pixel initial guess from Normalised Cross
Correlation, searched within `search_radius` of the previous frame's result. All
other subsets inherit their guess from a converged neighbour.

### IC-GN optimiser

Sub-pixel displacement minimises the Zero-mean Normalised Sum of Squared
Differences:

```
C_LS = Σ_(i,j)∈S [ f̃(x_ref,i, y_ref,j)/‖f̃‖ − g̃(x_cur,i, y_cur,j)/‖g̃‖ ]²
```

Deformation is parameterised by **p** = [u, v, ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]ᵀ
(affine) or a 12-element quadratic vector, warping reference coordinates to
current ones. Each iteration:

1. Steepest-descent images **SD** = [f_x, f_y, f_x·Δx, f_x·Δy, f_y·Δx, f_y·Δy] — once per subset
2. Hessian **H** = **SD**ᵀ**SD** — once per subset
3. Warp the current image by **p** to get g̃
4. Solve **H**Δ**p** = **SD**ᵀ(g̃ − f̃) by Cholesky
5. Compositional update **M**(**p**) ← **M**(**p**) ∘ **M**(Δ**p**)⁻¹
6. Exit when the implied subset-edge motion falls below the tolerance

Step 5 is what distinguishes IC-GN: applying the correction in the reference
frame keeps the precomputed Hessian valid across iterations.

### Reliability-guided propagation (RG-DIC)

Solutions propagate outward from the seed, each converged subset warm-starting
its neighbours. The guess is **extrapolated**, not copied:

```
u_init = u_p + (∂u/∂x)(x_n − x_p) + (∂u/∂y)(y_n − y_p)
v_init = v_p + (∂v/∂x)(x_n − x_p) + (∂v/∂y)(y_n − y_p)
```

Copying `u_p, v_p` verbatim introduces an O(spacing) px error — enough to drop
IC-GN into the wrong local minimum where gradients are high.

Converged subsets enter a min-heap keyed by C_LS, so the best results propagate
first and poor subsets near a stress concentration are processed last, where
they cannot poison their neighbours' guesses. A neighbour whose solution jumps
more than `spacing + 1` px relative to its parent is rejected as spurious.

### Strain

Per-subset displacement gradients from IC-GN are noisy, so strain uses Ncorr's
strain-window approach: for each subset centre, fit a least-squares plane to
u(x,y) and v(x,y) over the surrounding neighbourhood, then form Green–Lagrange
strain from the smoothed gradients:

```
E_xx = ∂u/∂x + ½[(∂u/∂x)² + (∂v/∂x)²]
E_yy = ∂v/∂y + ½[(∂u/∂y)² + (∂v/∂y)²]
E_xy = ½[∂u/∂y + ∂v/∂x + (∂u/∂x)(∂u/∂y) + (∂v/∂x)(∂v/∂y)]
```

These are the finite-strain expressions, not linearised engineering strains —
which matters once deformation is large.

The plane fit is restricted to one connected material component, so a
neighbourhood never borrows points across a crack or a dropout, and it is
evaluated with separable 1-D correlations, keeping it O(N) in window size.

Per-interval strains are then accumulated across the sequence.

### Strain rate

Rate is computed from the **velocity field**, not by differentiating strain in
time. Each frame's velocity is its interval displacement over Δt; fitting
velocity gradients by the same plane-fit machinery gives the velocity gradient
tensor **L**, and the rate-of-deformation tensor is its symmetric part:

```
D = ½(L + Lᵀ)
Ė_xx = ∂Vx/∂x     Ė_yy = ∂Vy/∂y     Ė_xy = ½(∂Vx/∂y + ∂Vy/∂x)
```

with `Gxy_rate = 2·Ė_xy` as engineering shear rate. This is the Eulerian rate of
deformation and is well-defined at every frame, including the first and last —
unlike a finite difference of accumulated strain, which needs neighbours on both
sides and inherits the accumulated field's drift.

---

## GPU acceleration

With CuPy and an NVIDIA driver installed, the solver runs a batched wavefront
pipeline on the GPU, tracking a global seed across frames. If a frame's subset
survival rate falls below 60% the run automatically falls back to a targeted NCC
re-seed for that frame rather than propagating a collapsed solution.

The GPU path has its own small per-subset integer-shift rescue sweep
(`rescue_radius`, default 12 px), kept deliberately narrow: a wide sweep over
quasi-periodic speckle locks onto false ZNSSD minima.

CPU and GPU paths share the same ROI construction, strain accumulation and rate
computation, so results are directly comparable.

---

## Project layout

```
dicTool/
├── pydic/
│   ├── main.py                     Entry point
│   └── src/
│       ├── core/
│       │   ├── analysis.py         DICAnalysis: pipeline, frame pairs, export, settings
│       │   ├── rg_dic.py           DICParams and reliability-guided propagation
│       │   ├── icgn.py             IC-GN optimiser (CPU)
│       │   ├── icgn_gpu.py         Batched wavefront IC-GN (CuPy)
│       │   ├── bspline.py          Quintic B-spline coefficients and interpolation
│       │   ├── ncc.py              Normalised cross-correlation seeding
│       │   ├── shape_order.py      Affine and quadratic shape functions
│       │   ├── strain.py           Plane fit, velocity gradients, von Mises equivalent
│       │   ├── strain_accum.py     Cross-frame strain accumulation and repair
│       │   ├── stats.py            Robust limits and field summaries
│       │   ├── units.py            Spatial calibration
│       │   └── roi_loader.py       ROI mask loading
│       └── ui/
│           ├── wizard.py           Window and step navigation
│           ├── render.py           Field → RGBA, colour ranges, panels (shared by view and export)
│           ├── image_canvas.py     Interactive canvas, ROI tools, markers
│           ├── video_importer.py   Frame extraction and capture-rate entry
│           ├── video_export.py     Mosaic rendering and encoding
│           ├── theme.py            Dark stylesheet
│           ├── components.py       Shared widgets
│           └── pages/              welcome · roi · dynamic_roi · params · analysis · results
│                                   frame_pair_dialog · video_export_dialog
└── requirements.txt
```

Field colouring lives in `ui/render.py` and is called by both the on-screen
overlay and the video exporter, so the two cannot drift apart.

---

## Validation and known limits

**Against Ncorr.** The reference-based implementation was validated on a real
CFRP tensile dataset: R² = 0.991 on vertical displacement across ten loading
frames, Bland–Altman agreement within ±1.34 px. Strain rates agreed with rates
derived from Ncorr's per-frame strain means to within 0.2×10⁻³ s⁻¹ for the
dominant E_yy component.

> That validation was measured on the earlier reference-to-current scheme. The
> solver has since moved to immediate-frame correlation with accumulated strain,
> which changes error propagation across long sequences. Treat the figures as
> evidence for the correlation core, not as a current end-to-end certificate,
> and re-validate against your own reference data for published work.

**Known limits.**

- 2D only. Out-of-plane motion appears as apparent in-plane strain; keep the
  camera normal to the surface and the specimen in focus.
- Result fields are stored as float32 to keep long sequences in memory. That is
  ~7 significant digits — far below DIC measurement uncertainty, but worth
  knowing if you post-process. Frame-pair sums and averages accumulate in
  float64.
- Accumulated strain inherits drift over very long sequences, in the way any
  incremental scheme does. Frame-pair averaging and per-interval rates avoid it.
- Equivalent strain assumes plastic incompressibility. For elastic-dominated
  deformation with ν ≠ 0.5 it is not the right measure.
- Speckle quality dominates everything. No solver setting recovers a pattern
  that is too coarse, too fine, blurred or blown out.

---

## References

Blaber, J., Adair, B., & Antoniou, A. (2015). Ncorr: Open-Source 2D Digital
Image Correlation Matlab Software. *Experimental Mechanics*, 55(6), 1105–1122.
https://doi.org/10.1007/s11340-015-0009-1

Pan, B., Li, K., & Tong, W. (2013). Fast, robust and accurate digital image
correlation calculation without redundant computation. *Experimental Mechanics*,
53, 1277–1289.

Pan, B., Qian, K., Xie, H., & Asundi, A. (2009). Two-dimensional digital image
correlation for in-plane displacement and strain measurement: a review.
*Measurement Science and Technology*, 20, 062001.

Baker, S., & Matthews, I. (2004). Lucas-Kanade 20 Years On: A Unifying
Framework. *International Journal of Computer Vision*, 56(3), 221–255.

Sutton, M. A., Orteu, J. J., & Schreier, H. W. (2009). *Image Correlation for
Shape, Motion and Deformation Measurements*. Springer, New York.

Pan, B. (2016). Recent progress in digital image correlation. *Experimental
Mechanics*, 56, 67–73.

Dong, Y. C., & Pan, B. (2017). A review of speckle pattern fabrication and
assessment for digital image correlation. *Experimental Mechanics*, 57,
1161–1181.

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing
agreement between two methods of clinical measurement. *The Lancet*, 327(8476),
307–310.
