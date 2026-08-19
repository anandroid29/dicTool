# PyDIC

PyDIC is a desktop application for two-dimensional, full-field Digital Image
Correlation (DIC). It tracks a speckled surface through an image sequence and
reports displacement, velocity, strain, and strain-rate fields through a PyQt6
workflow. CPU analysis is always available; an optional CuPy backend accelerates
supported analyses on NVIDIA CUDA GPUs.

The application is intended for planar, in-plane measurements. It does not
perform stereo DIC or reconstruct out-of-plane motion.

## Current result semantics

PyDIC uses an updated-Lagrangian sequence: every image is correlated against the
immediately previous image. This distinction is important when interpreting the
results.

| Result family | Meaning |
|---|---|
| Displacement `u`, `v`, magnitude | Motion from the immediately previous frame to the current frame; it is not accumulated |
| Velocity `Vx`, `Vy`, effective velocity | Immediate displacement divided by the frame interval |
| Strain `Exx`, `Eyy`, `Exy` | Green–Lagrange strain increments accumulated frame by frame |
| Equivalent strain | Sum of the non-negative equivalent magnitude of each strain increment |
| Strain rate | Instantaneous symmetric spatial gradient of the velocity field; it is not obtained by differentiating accumulated strain in time |

`Exy` is tensor shear strain, not engineering shear (`Gxy = 2 Exy`). The
equivalent strain and strain-rate fields use the same von Mises-style magnitude,
with the out-of-plane term inferred from plastic incompressibility.

## Features

- Import an ordered image folder, extract frames from a video, or reopen an HDF5
  analysis session.
- Draw or load a static ROI using rectangle, circle, polygon, eraser, and full-ROI
  tools, then place the propagation seed.
- Optionally refine the static ROI on every frame with Contrast, Edge Detection,
  or Hybrid dynamic ROI masking.
- Run reliability-guided IC-GN DIC on the CPU or use optional CuPy acceleration.
- Choose first-order affine or second-order quadratic shape functions. The GPU
  solver currently uses first order; select CPU for a true second-order run.
- Recover points after temporary dynamic-ROI or tracking dropout instead of
  making every loss permanent.
- Display calibrated displacement and velocity in `m`, `mm`, `µm`, or `nm`.
- Inspect fields with an FEA-style `turbo` colourmap by default, plus alternative
  sequential and diverging maps.
- Use per-frame, sequence-global, symmetric, or manually entered colour limits.
- Place trajectory markers, display streaklines, and average selected frame
  pairs for displacement, velocity, and strain rate.
- Export the current frame to CSV, save/load HDF5 result sessions, or render a
  configurable multi-panel video or PNG image sequence.

## Requirements and installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/anandroid29/PyDIC.git
cd PyDIC
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages and start the application from the repository
root:

```bash
python -m pip install -r requirements.txt
python pydic/main.py
```

The required packages are NumPy, SciPy, OpenCV, Matplotlib, PyQt6, Pillow,
scikit-image, and h5py. MATLAB is not required.

### Optional GPU acceleration

GPU analysis requires an NVIDIA CUDA-capable GPU, a compatible CUDA driver, and
a CuPy package matching the installed CUDA version. CuPy is deliberately not in
`requirements.txt`, because the correct package depends on the local CUDA
installation. Install it using the official CuPy instructions, then select
**Use GPU Acceleration (CuPy)** on the Parameters page.

The GPU is an acceleration backend, not a different measurement convention.
Both backends feed the same validity, gradient, strain accumulation, velocity,
and strain-rate post-processing. Very small floating-point differences are
normal.

## Application workflow

The wizard contains six labelled stages. The Dynamic ROI stage is automatically
skipped when its mode is **None**.

### 1. Import

Choose one of the following sources:

- **Video:** select a video, choose frame extraction settings and the reference
  frame, and optionally set the physical size represented by one pixel.
- **Image folder:** images are sorted by filename. Configure sampling, frame
  rate, an optional ROI-mask file, and pixel-to-length calibration. A file with
  `roi` in its name is offered as the ROI mask rather than as a specimen frame.
- **HDF5:** reopen a previously saved analysis directly in Results.

The effective frame rate determines the interval used for velocity and strain
rate. When no reliable rate is available, PyDIC uses an interval of one second.

### 2. Static ROI

Draw the material region that is eligible for analysis, load a binary mask, or
use the complete image. Place the seed in a well-textured part of that ROI.
Subset centres outside the static ROI are never admitted by Dynamic ROI.

Choose the Dynamic ROI mode here:

- **None:** use the static ROI on every frame and skip the next stage.
- **Contrast:** retain locally textured regions using local intensity variation.
- **Edge Detection:** retain regions using Sobel gradient magnitude.
- **Hybrid:** combine normalized contrast and edge scores.

### 3. Dynamic ROI (optional)

Dynamic ROI is a per-frame texture filter for the material points selected by the
static ROI. Its scoring normalization and automatic Otsu threshold are calibrated
on the reference frame, then the same rule is applied at each point's advected
position on later frames; the original ROI is not reused as a stationary image
crop. The threshold does not drift to fit each image independently.

Available controls are:

- **Automatic (Otsu):** derive the normalized threshold from the reference
  image. Turn it off to set the threshold manually.
- **Texture threshold:** a higher value keeps only stronger texture or edges.
- **Minimum region size:** remove connected regions smaller than this fraction
  of the largest retained region.
- **Fill enclosed holes:** restore rejected islands completely enclosed by a
  retained material region.
- **Include / Exclude overrides:** draw persistent, material-attached rectangle,
  polygon, or circle overrides. Include wins if both channels overlap.

The reference preview shown here is also used by the Parameters preview and by
the first Analysis-page ROI display. Later Analysis frames show their own
measured dynamic masks. A point hidden or lost on one frame may become valid
again later.

### 4. Parameters

The current defaults are:

| Parameter | Default | Meaning |
|---|---:|---|
| Subset radius | 21 px | Half-size of the correlation subset |
| Grid spacing | 3 px | Distance between neighbouring subset centres |
| Strain window | 15 px | Half-width of the local displacement/velocity plane fit |
| Maximum iterations | 50 | IC-GN iteration limit per subset |
| Convergence tolerance | 0.001 px | Subset-edge motion threshold for stopping IC-GN |
| Correlation cutoff | 0.30 | Maximum accepted ZNSSD cost |
| Shape order | 1 | First-order affine, six-parameter warp |
| NCC search radius | 50 px | Half-width of the integer-pixel seed search |

For unit-normalized subsets, ZNSSD lies in `[0, 4]` and equals
`2 × (1 − ZNCC)`. The default cutoff `0.30` therefore corresponds to
`ZNCC >= 0.85`; lower cutoffs are stricter.

The strain window is measured in image pixels while samples occur only at DIC
grid points. Its regular support per axis is:

```text
2 * floor(strain_window / grid_spacing) + 1
```

At least three grid points per axis are required. The UI warns when the entered
window is too small, and the analysis uses the minimum valid window rather than
silently producing an all-NaN strain field.

Larger subsets are usually more robust and less noisy but reduce spatial
resolution. A wider strain window reduces gradient noise but smooths local strain
features. Parameters must be selected for the specimen texture, expected motion,
and feature scale; the defaults are starting points, not universal settings.

### 5. Analysis

PyDIC tracks each current frame from the immediately previous one. Integer-pixel
NCC provides or repairs an initial guess, IC-GN refines it to sub-pixel precision,
and a reliability-guided wavefront propagates good solutions through the ROI.
The previous interval also supplies temporal warm starts.

Before gradients or strain accumulation, measurements that are outside the
current frame, rejected by Dynamic ROI, non-finite, or otherwise invalid are
converted to missing values. Gradient fitting and dropout recovery are restricted
to connected valid material regions so smoothing does not bridge cuts or bleed
background values into the specimen.

Analysis may be cancelled. Completed frames remain available in the current
session.

### 6. Results

Select a result family and field, scrub or play the sequence, inspect statistics,
reset the view, and control overlay opacity. Available families are:

- **Displacement:** `u`, `v`, and magnitude.
- **Velocity:** `Vx`, `Vy`, and effective velocity.
- **Strain rate:** `Exx`, `Eyy`, tensor shear `Exy`, and effective rate.
- **Strain:** accumulated Green–Lagrange `Exx`, `Eyy`, tensor shear `Exy`, and
  equivalent strain magnitude.

The display range can be automatic per frame, global across the sequence, or
manual. A symmetric-about-zero option is useful for signed component fields.
Statistics, colourbar values, plots, and exports ignore invalid and non-finite
pixels rather than treating them as zeros.

## Spatial and temporal calibration

The solver always works in pixels. Pixel-to-length calibration changes only
presentation and export units:

- displacement: `px` becomes the selected length unit;
- velocity: `px/s` becomes the selected length unit per second;
- strain remains dimensionless;
- strain rate remains `s^-1`.

Enter the physical size of one pixel during video extraction, while importing an
image folder, or on the Results page. The calibration is cached in application
settings, written to extracted-frame metadata, and stored in HDF5 sessions.
Entering `0` leaves the analysis uncalibrated.

## Export

### CSV

**CSV (this frame)** exports the fields for the selected result frame. Values are
written with their applicable calibrated units.

### HDF5

**HDF5 (all frames)** stores the complete result sequence, ROI, Dynamic ROI
configuration and overrides, analysis backend, core grid/strain parameters,
frame-rate information, calibration, and result-semantics metadata.
HDF5 files can be loaded from the Import page without rerunning correlation.
Source images are referenced by path rather than embedded, so keep them available
if image backdrops are needed after moving the HDF5 file. Compatibility aliases
are retained for older sessions and integrations, but the UI presents only the
current result convention.

### Video and image-sequence export

The exporter can create a `1 x 1` through `4 x 4` panel layout. Each cell can be
configured independently as a result field, raw frame, streaklines only, or an
empty panel. Applicable field panels can choose their field, colourmap, colour
range, symmetric scale, colourbar, label, background, and optional streakline
overlay. Controls that do not apply to raw-frame, streakline-only, or empty
panels are hidden.

Supported outputs are MP4 (`mp4v`), AVI (`XVID`, `MJPG`, or lossless `FFV1`), and
a numbered PNG image sequence. Backgrounds may use the deformed frame, reference
frame, a solid colour, or transparency. True alpha transparency is preserved
only by the PNG image-sequence output; transparent pixels become black in the
available video codecs.

## Algorithm overview

1. Input images are converted to grayscale and normalized.
2. The CPU path constructs quintic B-spline coefficients for sub-pixel intensity
   interpolation. The GPU backend uses its CUDA/CuPy interpolation path.
3. NCC finds an integer-pixel seed estimate when a global or rescue search is
   needed.
4. Inverse-compositional Gauss–Newton minimizes ZNSSD for each circular subset.
5. Reliability-guided propagation prioritizes the best converged subsets and
   warm-starts neighbouring points.
6. Local least-squares plane fits are applied to immediate displacement and
   velocity fields. Invalid pixels are excluded, and separate material
   components are fitted independently.
7. Green–Lagrange strain increments are accumulated. Velocity and strain rate
   remain interval quantities.

For an immediate displacement gradient, the accumulated increment uses:

```text
Exx = du/dx + 0.5 * ((du/dx)^2 + (dv/dx)^2)
Eyy = dv/dy + 0.5 * ((du/dy)^2 + (dv/dy)^2)
Exy = 0.5 * (du/dy + dv/dx + du/dx*du/dy + dv/dx*dv/dy)
```

This implementation adds these component increments through time. It does not
currently compose a total deformation gradient multiplicatively across frames.

## Testing

Run the automated test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The `tests/` directory also contains manual and hardware-dependent verification
scripts for off-screen UI/export checks, real image sequences, GPU behaviour,
memory use, CPU/GPU consistency, and HDF5 compatibility. GPU checks require a
working CuPy/CUDA installation; data-dependent checks require their local sample
frames.

## Important limitations

- PyDIC is 2D DIC. Out-of-plane displacement, perspective change, defocus, and
  major illumination changes can appear as false in-plane motion or correlation
  loss.
- Reliable measurement requires a suitable random, high-contrast speckle pattern
  and enough subset support inside the ROI.
- Dynamic ROI is a texture heuristic, not material segmentation. Inspect its
  reference preview and use manual overrides where needed.
- A point that appears late or returns without enough intact neighbouring history
  may be rebased. It can contribute again, but its accumulated history may not
  represent an unbroken path from the original reference frame.
- Second-order shape functions are CPU-only in the current implementation.
- Long, high-resolution sequences retain result fields for every completed frame
  and can still require substantial system memory. GPU temporary-memory pools are
  released between frames, but GPU hardware and driver limits still apply.
- Scientific results should be validated for the camera, optics, calibration,
  specimen, loading mode, and expected deformation range before use in critical
  decisions.

## Project structure

```text
PyDIC/
|-- README.md
|-- requirements.txt
|-- tests/
`-- pydic/
    |-- main.py
    `-- src/
        |-- core/
        |   |-- analysis.py       Sequence orchestration, results, and persistence
        |   |-- bspline.py        CPU B-spline interpolation
        |   |-- icgn.py           CPU IC-GN solver
        |   |-- icgn_gpu.py       CuPy GPU solver
        |   |-- ncc.py            Integer-pixel initial search
        |   |-- rg_dic.py         CPU reliability-guided DIC and parameters
        |   |-- strain.py         Gradient fitting and strain-rate calculation
        |   |-- strain_accum.py   Immediate-frame tracking and strain accumulation
        |   `-- units.py          Spatial calibration and unit conversion
        `-- ui/
            |-- wizard.py         Six-stage application workflow
            |-- image_canvas.py   ROI, zoom, marker, and overlay canvas
            |-- render.py         Shared results/export rendering
            |-- video_export.py   Headless multi-panel rendering and encoding
            `-- pages/            Import, ROI, Dynamic ROI, Parameters, Analysis,
                                 Results, frame-pair, and export screens
```

## References

- Blaber, J., Adair, B., & Antoniou, A. (2015). Ncorr: Open-Source 2D Digital
  Image Correlation Matlab Software. *Experimental Mechanics*, 55(6), 1105–1122.
  https://doi.org/10.1007/s11340-015-0009-1
- Pan, B., Li, K., & Tong, W. (2013). Fast, robust and accurate digital image
  correlation calculation without redundant computation. *Experimental
  Mechanics*, 53, 1277–1289.
- Pan, B., Qian, K., Xie, H., & Asundi, A. (2009). Two-dimensional digital image
  correlation for in-plane displacement and strain measurement: a review.
  *Measurement Science and Technology*, 20, 062001.
- Baker, S., & Matthews, I. (2004). Lucas-Kanade 20 Years On: A Unifying
  Framework. *International Journal of Computer Vision*, 56(3), 221–255.
- Sutton, M. A., Orteu, J. J., & Schreier, H. W. (2009). *Image Correlation for
  Shape, Motion and Deformation Measurements*. Springer.
