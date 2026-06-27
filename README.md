# StrainX — Python Digital Image Correlation

StrainX is a 2D Digital Image Correlation tool written in Python. We built it as part of our work at IIT Kanpur to replicate the algorithmic core of Ncorr (Blaber, Adair & Antoniou, 2015) in a freely available, fully automated package and to add one capability Ncorr doesn't have: direct full-field strain rate computation.

The algorithms follow the Ncorr paper closely enough that we validated StrainX against Ncorr on a real CFRP tensile dataset and got R² = 0.991 on vertical displacement across ten loading frames, with Bland–Altman agreement within ±1.34 px. The strain rates are a new addition and are described at the bottom of this file.

---

## What it does

- Tracks speckle patterns between a reference image and one or more deformed images
- Outputs full-field displacement (u, v) and Green–Lagrangian strain (Exx, Exy, Eyy) maps
- Computes strain rates dExx/dt, dExy/dt, dEyy/dt directly from the image sequence
- Runs the whole pipeline through a five-step GUI no scripting needed

---

## Installation

```bash
git clone https://github.com/anandroid29/StrainX.git
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
cd StrainX
python main.py
```

Dependencies are NumPy, SciPy, PyQt6, OpenCV, and Matplotlib. No MATLAB required.

---

## How to use it

The GUI walks you through five steps in order.

**Step 1 — Load images.** Add your reference image first, then your deformed images in temporal order. Colour images are converted to greyscale automatically.

**Step 2 — Draw the ROI.** Use the polygon, rectangle, or circle tools to mark the region you want to analyse. You can also erase parts of the mask, or load a binary mask directly if you already have one. Hit "Preview Mask" to check it before moving on.

**Step 3 — Set parameters.** The main ones to think about are:

| Parameter | What it controls | Typical values used in our validation |
|---|---|---|
| Subset radius r (px) | Size of the correlation window | 21 |
| Subset spacing s (px) | Distance between subset centres | 1 |
| Strain window radius r_E (px) | Neighbourhood for least-squares plane fit | 15 |
| IC-GN max iterations | Convergence limit | 50 |
| Convergence tolerance ‖Δp‖ | Exit criterion for IC-GN | 1×10⁻⁶ |
| ZNSSD cutoff C_LS | Maximum acceptable correlation cost | 2.0 |

There is a trade-off between spatial resolution and noise that is worth being aware of: a larger subset radius smooths out noise but blurs sharp gradients. A larger strain window has the same effect on the strain fields. We used r = 21 px and r_E = 15 px for the CFRP validation, reasonable starting points for most quasi-static tests.

**Step 4 — Run.** Click "Run Analysis". A progress bar shows completion per image. You can cancel at any time without losing work already done on earlier frames.

**Step 5 — View results.** The results viewer opens automatically. Tabs switch between u, v, Exx, Exy, Eyy, and strain rate fields. A slider lets you step through the deformed image sequence. Export to CSV, PNG, or HDF5 from there.

---

## How the algorithms work

We tried to stay as close to the Ncorr implementation as possible. Here is what is happening under the hood.

### B-spline interpolation

Before anything else, the grayscale values of both the reference and deformed images are prefiltered into quintic (5th-order) B-spline coefficient arrays. We use `scipy.ndimage.spline_filter` for this, which applies an IIR recursive filter that is numerically equivalent to the FFT-based deconvolution Ncorr uses. Once the coefficients are computed, any sub-pixel intensity value g(x̃, ỹ) is evaluated as:

```
g(x̃, ỹ) = [1  Δx  Δx²  Δx³  Δx⁴  Δx⁵] · [QK] · C[xf-2:xf+3, yf-2:yf+3] · [QK]ᵀ · [1  Δy  ...]ᵀ
```

where [QK] is the 6×6 quintic kernel matrix, C is the B-spline coefficient array, xf = floor(x̃), and Δx = x̃ − xf. The spatial gradients ∂g/∂x and ∂g/∂y needed for the Hessian are computed analytically from the same coefficient array and not by finite-differencing the raw pixel values, which would introduce a systematic bias.

### Initial guess via NCC

For the seed subset, an integer-pixel initial displacement is found by Normalized Cross-Correlation (NCC): the reference subset is padded to full image size, convolved with the deformed image, and the peak of the resulting correlation map gives the integer-pixel (u⁽ᵍ⁾, v⁽ᵍ⁾) initial guess. All other subsets get their initial guess from a neighbour (see RG-DIC below) and skip the NCC step entirely.

### IC-GN optimiser

The sub-pixel displacement is refined by minimising the Zero-mean Normalised Sum of Squared Differences (ZNSSD) cost:

```
C_LS = Σ_{(i,j)∈S} [ f̃(x_ref,i, y_ref,j) / ‖f̃‖  −  g̃(x_cur,i, y_cur,j) / ‖g̃‖ ]²
```

where f̃ and g̃ are the zero-mean reference and deformed subset intensities, and the summation is over all points (i,j) in the circular subset S.

The deformation is parameterised by a six-element vector **p** = [u, v, ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]ᵀ. The warp function maps reference subset coordinates to current ones:

```
x_cur = x_ref + u + (∂u/∂x)(x_ref − x_c) + (∂u/∂y)(y_ref − y_c)
y_cur = y_ref + v + (∂v/∂x)(x_ref − x_c) + (∂v/∂y)(y_ref − y_c)
```

The IC-GN method solves for a small incremental warp Δ**p** that minimises C_LS in the reference image frame (which lets the Hessian be precomputed and reused across iterations). Each iteration:

1. Precompute steepest-descent images: **SD**_k = [f_x, f_y, f_x·Δx, f_x·Δy, f_y·Δx, f_y·Δy] at each pixel k — done once per subset
2. Precompute Hessian **H** = **SD**ᵀ **SD** — also done once per subset  
3. Warp the deformed image with the current **p**_old to get g̃
4. Compute the gradient ∇C_LS and solve **H** Δ**p** = **SD**ᵀ(g̃ − f̃) via Cholesky decomposition
5. Apply the compositional warp update: **M**(**p**) ← **M**(**p**) ∘ **M**(Δ**p**)⁻¹
6. Exit when ‖Δ**p**‖ < 10⁻⁶

The compositional update in step 5 is what makes IC-GN different from a standard Gauss-Newton solver. It applies the correction in the reference frame, keeping the Hessian valid across iterations. This is the same update Ncorr uses.

### Reliability-Guided DIC (RG-DIC)

Rather than analysing every subset independently, RG-DIC propagates solutions outward from a single seed point, using each converged subset's result to warm-start its neighbours. The seed is the only point that uses NCC for an initial guess. All other subsets inherit the initial guess from whichever of their processed neighbours has the lowest C_LS value.

Crucially, the initial guess is not just copied verbatim — it is extrapolated using a first-order Taylor expansion. If the parent subset at (x_p, y_p) has converged to **p**_p = [u_p, v_p, ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]ᵀ, the initial displacement given to a neighbour at (x_n, y_n) is:

```
u_init = u_p + (∂u/∂x)(x_n − x_p) + (∂u/∂y)(y_n − y_p)
v_init = v_p + (∂v/∂x)(x_n − x_p) + (∂v/∂y)(y_n − y_p)
```

Skipping this Taylor correction (just copying u_p, v_p) introduces an O(s) px error in the initial guess — large enough to send IC-GN into the wrong local minimum in high-gradient regions. We implement it exactly as Ncorr does.

A processed subset is added to a min-heap keyed by its C_LS value. At each step the best (lowest C_LS) point is popped and its four neighbours are queued for analysis. This ordering means high-quality results propagate first and bad subsets near the stress concentration are processed last, preventing them from poisoning their neighbours' initial guesses.

If a neighbour's IC-GN solution produces a displacement jump of more than s + 1 px relative to the parent, it is rejected as spurious rather than added to the heap.

### Green–Lagrangian strains

The displacement gradients coming directly out of IC-GN are noisy (they are computed per-subset and sensitive to any per-subset error). Instead, we follow the strain window approach from Ncorr: for each subset centre (x_c, y_c), we collect all valid displacement points within a circular neighbourhood of radius r_E and fit a least-squares plane to u(x,y) and v(x,y) separately:

```
u_plane(x, y) = a_u + (∂u/∂x)_plane · x + (∂u/∂y)_plane · y
v_plane(x, y) = a_v + (∂v/∂x)_plane · x + (∂v/∂y)_plane · y
```

This is solved as an over-determined linear system. The smoothed gradients (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y) from the plane fit are then used to compute the Green–Lagrangian strain components:

```
E_xx = ∂u/∂x + ½[(∂u/∂x)² + (∂v/∂x)²]
E_yy = ∂v/∂y + ½[(∂u/∂y)² + (∂v/∂y)²]
E_xy = ½[∂u/∂y + ∂v/∂x + (∂u/∂x)(∂u/∂y) + (∂v/∂x)(∂v/∂y)]
```

These are the full Green–Lagrangian (finite-strain) expressions, not the linearised engineering strains, important if your specimen undergoes large deformations.

### Strain rates

This is the main thing StrainX does that Ncorr does not. Once strain fields are computed for all N frames, we differentiate them in time using a finite-difference scheme. For an image sequence acquired at frame rate f_ps (frames per second), Δt = 1/f_ps, and:

```
dE_ij/dt at frame 0:       (E_ij,1 − E_ij,0) / Δt
dE_ij/dt at frame n (interior):  (E_ij,n+1 − E_ij,n-1) / (2Δt)
dE_ij/dt at frame N-1:     (E_ij,N-1 − E_ij,N-2) / Δt
```

Central differences in the interior, one-sided at the boundaries. The result is a full-field strain rate map at every frame, output alongside the displacement and strain fields.

We validated this against rates derived by applying the same scheme to Ncorr's per-frame strain means. For the dominant E_yy component, the two agreed within 0.2 × 10⁻³ s⁻¹ over the middle loading frames. Integrating the computed rates back via the trapezoidal rule recovers the original strain time history to within about 12–13% over ten steps, which is what you would expect from accumulated truncation error.

---

## File structure

```
StrainX/
├── main.py
├── requirements.txt
├── README.md
└── src/
    ├── core/
    │   ├── bspline.py       B-spline coefficient computation and sub-pixel interpolation
    │   ├── ncc.py           Normalised cross-correlation for integer-pixel seed guess
    │   ├── icgn.py          IC-GN optimiser (precomputed Hessian, compositional update)
    │   ├── rg_dic.py        Reliability-guided propagation with Taylor extrapolation
    │   ├── strain.py        Least-squares plane fit and Green–Lagrangian strains
    │   └── analysis.py      Top-level DICAnalysis class, DICParams, strain rate computation
    └── ui/
        ├── theme.py         Dark stylesheet
        ├── main_window.py   Main window and step navigation
        ├── image_canvas.py  Interactive image canvas with ROI drawing tools
        ├── param_panel.py   Parameter input panel
        └── results_panel.py Colourmap viewer, temporal scrubber, export
```

---

## References

Blaber, J., Adair, B., & Antoniou, A. (2015). Ncorr: Open-Source 2D Digital Image Correlation Matlab Software. *Experimental Mechanics*, 55(6), 1105–1122. https://doi.org/10.1007/s11340-015-0009-1

Pan, B., Li, K., & Tong, W. (2013). Fast, robust and accurate digital image correlation calculation without redundant computation. *Experimental Mechanics*, 53, 1277–1289.

Pan, B., Qian, K., Xie, H., & Asundi, A. (2009). Two-dimensional digital image correlation for in-plane displacement and strain measurement: a review. *Measurement Science and Technology*, 20, 062001.

Baker, S., & Matthews, I. (2004). Lucas-Kanade 20 Years On: A Unifying Framework. *International Journal of Computer Vision*, 56(3), 221–255.

Sutton, M. A., Orteu, J. J., & Schreier, H. W. (2009). *Image Correlation for Shape, Motion and Deformation Measurements*. Springer, New York.

Pan, B. (2016). Recent progress in digital image correlation. *Experimental Mechanics*, 56, 67–73.

Dong, Y. C., & Pan, B. (2017). A review of speckle pattern fabrication and assessment for digital image correlation. *Experimental Mechanics*, 57, 1161–1181.

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307–310.
