# PyDIC — Python Digital Image Correlation Suite

A professional, open-source 2D Digital Image Correlation (DIC) application written in Python. Implements the full algorithmic framework described in **Blaber, Adair & Antoniou (2015)** — the Ncorr paper — with a modern, intuitive graphical interface designed to eliminate the tedious workflow of legacy DIC tools.

---

## Features

### Core Algorithms (faithful to the Ncorr paper)
- **Biquintic B-spline interpolation** — sub-pixel accuracy via 5th-order spline interpolation with FFT-based coefficient computation
- **Normalized Cross-Correlation (NCC)** — robust integer-pixel initial guess per seed subset
- **Inverse Compositional Gauss-Newton (IC-GN)** — fast sub-pixel refinement via compositional warp updates and precomputed Hessians
- **Reliability-Guided DIC (RG-DIC)** — propagates deformation from lowest-error subsets first, using neighbor deformation as initial guess
- **Green-Lagrangian strains** — Exx, Exy, Eyy computed from displacement gradients via least-squares plane fitting over a configurable strain window
- **Effective strain** — Eeff = √(2/3 · eᵢⱼeᵢⱼ), deviatoric component
- **Temporal analysis** — full displacement and strain fields for an ordered sequence of deformed images

### User Interface Highlights
- **Step-guided workflow** — five logical stages (Images → ROI → Parameters → Analyse → Results) with no hidden steps or buried menus
- **Interactive ROI tools** — polygon (click-to-add), rectangle, circle; mask eraser; mask preview overlay
- **Live progress** — per-subset progress with estimated time remaining
- **Rich results viewer** — tabbed colormaps for u, v, Exx, Exy, Eyy, Eeff; adjustable range; custom colormaps
- **Temporal scrubber** — slider and frame picker for stepping through a deformed image sequence
- **Export** — CSV, PNG, and HDF5 output; side-by-side image+result figure

---

## Installation

### 1. Clone or download
```bash
git clone https://github.com/yourname/pydic.git
cd pydic
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
python main.py
```

---

## Workflow

### Step 1 — Load Images
Click **Add Reference Image** and select a single grayscale or colour image (colour images are automatically converted to greyscale). Then click **Add Deformed Images** to load one or more deformed images in temporal order. Images are listed in the left panel with a thumbnail.

### Step 2 — Define ROI
Use the ROI toolbar to draw the region of interest on the reference image:
- **Polygon** — click to place vertices, double-click to close
- **Rectangle** — click-drag
- **Circle** — click-drag from centre
- **Erase** — paint to remove areas from the mask

Click **Preview Mask** to see the current ROI highlighted.

### Step 3 — Set Parameters
| Parameter | Description | Typical range |
|-----------|-------------|---------------|
| Subset radius (px) | Radius of the circular correlation window | 10–40 |
| Subset spacing (px) | Centre-to-centre step between subsets | 1–10 |
| Strain window (px) | Half-width of least-squares strain window | 5–20 |
| Max iterations | IC-GN convergence limit | 50 |
| Convergence tol | ‖Δp‖ threshold for IC-GN exit | 1×10⁻⁴ |
| Correlation cutoff | Maximum CLS value to accept a point | 0.8 |
| Search radius (px) | NCC initial guess search extent | 20–50 |

### Step 4 — Analyse
Click **Run Analysis**. The progress bar shows completion per image pair. Analysis can be cancelled at any time.

### Step 5 — View Results
The results panel opens automatically. Switch between quantities using the tab bar. Adjust the colormap range with the sliders. Use the temporal scrubber to step through deformed images. Click **Export** to save.

---

## Algorithm Details

### Biquintic B-Spline Interpolation
Gray-scale values at sub-pixel locations are evaluated using quintic B-splines (order 5). The B-spline coefficients are computed by applying `scipy.ndimage.spline_filter` (IIR-based deconvolution, equivalent to the FFT-based approach in Ncorr). Interpolation is then performed with `scipy.ndimage.map_coordinates`.

### IC-GN Optimization
For each subset centred at (xc, yc) with deformation vector **p** = [u, v, ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y]ᵀ:

1. Precompute steepest-descent images: **SD**ₖ = [fx, fy, fx·Δx, fx·Δy, fy·Δx, fy·Δy] per pixel k
2. Precompute Hessian **H** = **SD**ᵀ**SD** / σ²_f (computed once per subset)
3. Each iteration:
   - Warp the current image with **p**_old → g̃
   - Compute residual: f̃ − g̃ (ZNSSD criterion)
   - Solve **H** Δ**p** = **SD**ᵀ(g̃ − f̃) / σ_f via Cholesky decomposition
   - Compositional update: **M**_new = **M**_old · **M**(Δ**p**)⁻¹
4. Exit when ‖Δ**p**‖ < tolerance

### Reliability-Guided DIC
A seed point is analysed first (NCC initial guess). Its result is added to a min-heap keyed by CLS. At each iteration the best point is popped and its four neighbours are analysed using the parent's deformation as the initial guess. This prevents bad points from polluting neighbours and avoids redundant NCC calls.

### Green-Lagrangian Strains
Displacement gradients are obtained by fitting a least-squares plane to u(x,y) and v(x,y) over a circular strain window:

    Exx = ∂u/∂x + ½[(∂u/∂x)² + (∂v/∂x)²]
    Eyy = ∂v/∂y + ½[(∂u/∂y)² + (∂v/∂y)²]
    Exy = ½[∂u/∂y + ∂v/∂x + ∂u/∂x·∂u/∂y + ∂v/∂x·∂v/∂y]

The plane fit is vectorised using `scipy.ndimage.convolve` for efficiency.

---

## File Structure

```
pydic/
├── main.py                  Entry point
├── requirements.txt
├── README.md
└── src/
    ├── core/
    │   ├── bspline.py       Biquintic B-spline interpolation
    │   ├── ncc.py           Normalized cross-correlation (initial guess)
    │   ├── icgn.py          Inverse compositional Gauss-Newton optimizer
    │   ├── rg_dic.py        Reliability-Guided DIC engine
    │   ├── strain.py        Green-Lagrangian strain computation
    │   └── analysis.py      High-level DICAnalysis class and DICParams
    └── ui/
        ├── theme.py         QSS dark stylesheet
        ├── main_window.py   Main QMainWindow
        ├── image_canvas.py  Interactive image + ROI canvas
        ├── param_panel.py   Parameter controls panel
        └── results_panel.py Results colourmap viewer
```

---

## References

Blaber, J., Adair, B., & Antoniou, A. (2015). Ncorr: Open-Source 2D Digital Image Correlation Matlab Software. *Experimental Mechanics*, 55(6), 1105–1122. https://doi.org/10.1007/s11340-015-0009-1

Baker, S., & Matthews, I. (2004). Lucas-Kanade 20 Years On: A Unifying Framework. *International Journal of Computer Vision*, 56(3), 221–255.

Pan, B. (2009). Reliability-guided digital image correlation for image deformation measurement. *Applied Optics*, 48(8).

Pan, B., Li, K., & Tong, W. (2013). Fast, robust and accurate digital image correlation calculation without redundant computation. *Experimental Mechanics*, 53, 1277–1289.

---

## Licence
MIT
