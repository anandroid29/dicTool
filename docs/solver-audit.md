# Solver audit — 8 September 2026

Confirmed GPU recovery defects were fixed without changing the correlation cutoff
or the displacement-jump acceptance gate. Native CUDA was rebuilt locally as
`2.2.0-native` (ABI 2, architecture 120) and exercised on the RTX 5060 GPU.

## Findings and fixes

| Priority | Finding | Resolution |
|---|---|---|
| High | GPU NCC chose one geometric centre per failed component. An unmeasurable centre could prevent all surrounding measurable subsets from being tried; unchanged passes selected the same centre again. | NCC searches each failed subset, using accepted neighbouring motion/affine terms where available. Every candidate still passes IC-GN and the existing cutoff. |
| High | Recovery selected only the largest 32 components. Permanently bad components could continually exclude a smaller, measurable component. | Removed the cap, ranking, and distance-transform seed machinery. |
| High | A fresh GPU solve had one global seed. Disconnected material, a bad seed, or complete temporal dropout could leave a component without any way to restart under neighbour recovery. | Independently initialize components with no accepted solution after fresh/warm solves. Flat NCC templates fail as measurements rather than aborting the complete solve. |
| High | Integer rescue evaluated translation alone and could replace a useful affine guess before IC-GN evaluated it. | Refine the supplied affine warp first. Run integer rescue only when that solve fails acceptance; retain the best evaluated candidate. |
| Medium | CPU and GPU normalized-intensity Jacobians omitted the derivative of the subset mean. | Subtract the mean raw Jacobian before the normalization projection on both backends. A finite-difference derivative regression verifies this independently. |
| Medium | GPU convergence added squared pixel translations to squared dimensionless affine coefficients. CPU scaled affine terms by subset radius. | GPU now uses the CPU's subset-edge-motion units. Cutoff remains 0.30 by default. |
| Medium | Neighbour recovery reopened border failures but could leave deeper exhausted points closed to the new wavefront. | Reopen failed eligible points and reset retry counts for each neighbour pass. |
| Medium | Early rejection in the isolated CUDA subset API left returned parameters uninitialized. | Initialize output parameters from the supplied guess before rejection is possible. |
| Medium | HDF5 sessions omitted cutoff, convergence tolerance, iteration/search limits, shape order, and subset ROI clipping. A loaded session could silently use another machine's settings. | Save and restore these parameters; older files retain their previous compatibility behavior. Existing saved files cannot retrospectively establish their missing settings. |

The Jacobian correction differentiates `q = (f - mean(f))/||f - mean(f)||`:

`dq/dp = [J - mean(J) - q (qᵀ J)] / ||f - mean(f)||`.

The regression uses an intensity ramp with texture; the old derivative differed
from numerical differentiation by up to 0.0064872. This concerns the optimizer's
derivative, not a relaxation of its correlation criterion. The [Ncorr algorithm
paper](https://www.ncorr.com/download/publications/blaberncorr.pdf) provides the
IC-GN/normalized-correlation context; the derivative check is independently
computed from this repository's interpolator.

## Verification

The original suite passed all 134 tests before changes. Five focused regressions
were added for normalized derivatives, bad component seeds, component starvation,
restart after complete dropout, and solver-parameter persistence. Existing CUDA
parity expectations were updated for the corrected derivative and convergence
units, rather than preserving the previous port's errors.

Final verification: **139 tests passed**, including the hardware-dependent CUDA
tests, in 5.96 seconds. `git diff --check` passed. Production source has a net
reduction of 138 lines, excluding documentation and tests.

The synthetic stationary-image reproduction used identical images, ROI, seed,
and cutoff on both backends:

| Solver | Accepted subset centres |
|---|---:|
| Original GPU, after NCC | 0 |
| Corrected GPU, first pass / after NCC | 377 / 377 |
| CPU | 303 |

The extra GPU points in this case are stationary measurements validated against
the known zero-motion ground truth. The CPU's independently seeded row domains
also leave some measurable points unreached; CPU coverage is not a ground-truth
validity mask.

Three real image pairs were checked with radius 21, spacing 3, 50 iterations,
tolerance 0.001, cutoff 0.30, NCC search radius 50, and NCC hole recovery. Both
GPU versions used identical settings. CPU used the corrected code. These were
fresh pair solves with the saved static ROI, excluding dynamic masking and
strain processing so correlation could be compared directly.

| Saved sequence / pair | Original GPU | Corrected GPU | CPU | Corrected GPU–CPU displacement difference, 95th percentile |
|---|---:|---:|---:|---:|
| external video, 0 → 1 | 7,574 | 7,574 | 7,574 | 0.00884 px |
| external video, 250 → 251 | 8,044 | 8,044 | 8,044 | 0.01102 px |
| cutting video, 2279 → 2280 | 3,660 | 3,660 | 3,660 | 0.02554 px |

These real pairs did **not** reproduce the reported recurring holes, including
with the old DLL. The synthetic failures prove specific defects; they do not
prove that every missing point in the user's complete sequence has the same
cause. Complete warm-start sequences and the friend's exact data/settings/build
were not compared. `tests/verify_solver_audit.py` records reproducible pair
comparisons; local evidence is under `output/verification/audit_*.json`.

Recovery now does more useful work but can cost more time. Single-run GPU timings
before/after were 1.74/3.15 s, 2.03/2.56 s, and 0.84/0.41 s for these pairs.
These are diagnostic timings, not a controlled benchmark. Exhaustive NCC on many
failed points is the principal remaining recovery-performance tradeoff.

## Scientific interpretation limits retained

**First-arrival strain is a historical map.**
`DICAnalysis._transport_accumulated_strain` deliberately prevents an encountered
cell from receiving a later value. Existing tests explicitly require this. An
already populated cell therefore cannot show subsequent loading, unloading, or
a different material particle's current strain. The README now states this in
the result definitions. A current-material strain display would need a distinct
result convention and corresponding session/export semantics.

**Rate and equivalent strain assume small steps.** The code uses the symmetric
part of `H/dt`, where `H` is the pair displacement gradient on the source grid.
The exact spatial relation is `L = F_dot F^-1`, with `D = sym(L)`; it gives zero
deformation rate for rigid rotation. See [velocity-gradient
kinematics](https://www.continuummechanics.org/velocitygradient.html).

For a finite rotation, `H = R - I`. Consequently, this code's equivalent increment
is `2(1 - cos(theta))`, rather than zero. A direct `StrainPathTracker` experiment
at 10 degrees produced **0.03038449** equivalent strain while Green–Lagrange
`Exx` was approximately **5.55e-17**. The multiplicative deformation-gradient
composition is correct; the rate approximation is the limitation. The README
now makes the approximation explicit. An objective finite-step rate estimator
requires choosing and validating an interpolation of motion within each frame
interval; two images alone do not determine an instantaneous rate exactly.

**Plastic incompressibility is an assumption.** The equivalent calculation
infers the out-of-plane rate from zero trace and assumes unmeasured out-of-plane
shear is absent. It cannot determine plastic strain separately from elastic
strain using in-plane displacement alone. Integrating a non-negative equivalent
rate also accumulates measurement noise. These limits apply to both backends.

**Dark/saturated pixels are only an occlusion heuristic.** Both solvers reject
subsets using near-black/near-white fractions and asymmetry. Legitimate high
contrast speckles can satisfy those rules. Changing them without representative
images could admit tool/background matches, so the thresholds were retained.

## Bloat and simplification

Removed the component ranking/distance-transform recovery machinery, unused GPU
seed variables and validity counters, a stale CPU heap-entry branch that cannot
fire because accepted points are never replaced, an unused backend argument,
an ignored recovery-seed argument, and an unused synchronous trajectory cache
method. The active asynchronous trajectory cache remains in use.

Removed the FFT NCC fallback: OpenCV is a required dependency and the application
already imports it. That fallback also contained an extra square-root-of-area
normalization factor, so its scores were not true ZNCC. Removing the unreachable
application path is preferable to maintaining a second correlation implementation.

Removed calls to a misleading `release_temporary_memory` wrapper that only
synchronized CUDA and freed nothing. Solver outputs were already copied back
synchronously. Device allocations are still released by solver destruction.

Further refactor candidates remain: `analysis.py` repeats strict lattice sampling
and first-arrival deposition logic; CPU domain workers allocate seven full-image
float64 fields each even for narrow domains; native subset reference statistics
and Hessians are rebuilt on retries. These require targeted storage/performance
work and are not dead code. Compatibility loaders, optional GPU fallback, and
Qt callbacks were not classified as unused merely because calls are indirect.
