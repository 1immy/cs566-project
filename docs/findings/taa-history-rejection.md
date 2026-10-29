# Track 3: TAA history rejection

Implements issues #10, #11, and #12 in `src/taa.py`. The implementation adds
neighborhood history clamping, motion-adaptive blending, and a per-pixel
history-retention confidence map. The Track 2 alignment and baseline code
are unchanged.

## Reproduce locally

From the repository root on macOS/Linux:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python -m unittest discover -s tests -v
venv/bin/python scripts/run_sample_taa.py --group-size 64
```

On Windows, use `venv\Scripts\python.exe` for the virtual-environment Python.
The recorded run used Python 3.14.3, NumPy 2.5.4, and Pillow 13.0.0.
The dependency versions tested here are pinned in `requirements.txt`.
No additional data download is needed for these two development samples.

The runner writes ignored local artifacts to `results/sample_taa/`:

- `*_reconstructions.npz`: float32 naive, aligned, TAA, disabled-TAA control,
  and ground-truth images, plus frame counts, discounted history weights,
  last-update confidence, and last-update validity masks.
- `*_comparison.png`: labeled naive/aligned/TAA/ground-truth/confidence panels.
- `*_naive.png`, `*_aligned.png`, `*_taa.png`, `*_gt.png`,
  `*_confidence.png`, `*_valid_update.png`: individual display images.
- `manifest.json`: parameters, versions, input and source hashes, artifact
  hashes, scene/burst identifiers from filenames, frame ranges, crops, and
  diagnostic measurements. External paths are recorded as filenames only.

RGB display images all use a direct [0,1] to [0,255] mapping. Use the float
arrays for evaluation. No intensity normalization, ground-truth tuning, or
metric computation is performed. The runner checks the disabled-TAA control
against the unmodified baseline and rejects input/output path collisions.
The manifest explicitly distinguishes photon-probability reconstructions from
the provided RGB ground truth. The comparison panel labels confidence as
"Last retention" to identify its temporal scope.

The control uses `rtol=0` and an absolute roundoff allowance of
`max(1e-7, (number_of_groups - 1) * float32_epsilon)`, recorded as
`baseline_control_atol`. Many sequential mean updates can exceed a fixed
`1e-7` threshold purely through rounding. Coverage and disabled-control weights
must still equal the baseline counts exactly. See the independent verification
below for the reproduced failure and regression check.

## Interface and reference policy

The agreed public interface is unchanged:

```python
reject_and_blend(warped_frame, history, motion_field)
# -> merged_frame [H,W,3] float32, confidence_map [H,W] float32
```

Both RGB inputs are in photon-detection-probability space [0,1] and already
share coordinates. History must be finite. Motion is the aligner's backward
offset `(dy, dx)` in pixels, not velocity. If any RGB channel or either motion
component is nonfinite, that pixel is an invalid update. Invalid updates
preserve history and its weight, and return confidence zero. Invalid pixels
are also excluded from neighboring clamp bounds. Inputs are never modified.

The three-argument function treats its two inputs as equal nominal-weight
estimates and keeps no hidden state. Repeated calls alone do not implement
frame-count-weighted burst accumulation. For that, use:

```python
from src.taa import TAAConfig, reconstruct_taa

result = reconstruct_taa(cube, group_size=64, config=TAAConfig())
image = result.reconstruction
confidence = result.confidence_map
```

This wrapper uses the same kernel with explicit accumulation weights. It
partitions the burst into consecutive groups and calls the existing
`align_warp(group, reference)` with a fixed first-group mean, exactly as the
baseline does. The evolving reconstruction is a separate history buffer;
it is never substituted for that alignment reference. A short final group
uses its actual frame count. These concrete conventions are ready for Ethan
and Jack to review, without changing their interfaces or implementation.

## Method

At each valid pixel, let `F` be the current warped group mean and `H` the
previous reconstruction. Compute per-channel minimum and maximum over the
valid current pixels in a 3x3 neighborhood, expand both bounds by 0.05, and
restrict them to [0,1]. Neighborhoods are truncated at image edges. Let `C`
be `H` clipped to those bounds, and let `d` be the maximum absolute channel
difference between `H` and `C`.

The retained fraction of history is:

```text
c = exp(-d / 0.1) / (1 + hypot(dy, dx) / 8)
retained_weight = c * previous_weight
new_weight = retained_weight + current_group_frame_count
new_history = (retained_weight * C + current_group_frame_count * F) / new_weight
```

These are prototype constants, not calibrated noise parameters or fitted
quality optima. `TAAConfig` exposes the neighborhood radius, margin,
disagreement scale, displacement scale, and two component switches.

The returned confidence `c` is a history-retention fraction in [0,1]. It is
not the final interpolation coefficient and is not a calibrated probability
of correct alignment. More displacement reduces history retention, but
displacement relative to the first group does not measure instantaneous speed.
Discounted `history_weights` are algorithmic weights, not statistical effective
sample sizes. `frame_counts` records geometrically valid frames before rejection
and is directly comparable to the baseline's counts.

The wrapper returns the confidence and validity mask for the **last** update.
An invalid final update has confidence zero even if previous history is useful.
A single-group burst has zero confidence and a false validity mask because
there was no temporal comparison; its reconstruction is the group mean.

## Component controls

```bash
# Clipping with ordinary frame-weighted accumulation.
venv/bin/python scripts/run_sample_taa.py --no-adaptive --out results/sample_taa/clamp_only
# Neighborhood-disagreement and displacement retention, without image clipping.
venv/bin/python scripts/run_sample_taa.py --no-clamp --out results/sample_taa/adaptive_only
# Baseline control: no clipping or adaptive retention.
venv/bin/python scripts/run_sample_taa.py --no-clamp --no-adaptive --out results/sample_taa/disabled
```

`--no-clamp` disables clipping, but the adaptive confidence calculation still
uses disagreement with neighborhood bounds. `--no-adaptive` sets confidence
to one for valid updates. Disabling both recovers the aligned baseline within
float32 rounding. These switches support later component analysis; they do
not constitute completed held-out ablation experiments.

## Sample run and verification

Run on October 28, 2026 with the default configuration, four 64-frame groups,
sample-local frames 0 through 255, and crop `(336,464,336,464)` for each burst:

| Scene / burst | Disabled-TAA vs baseline max absolute difference | Valid pixels in final update | Mean final confidence on valid pixels |
|---|---:|---:|---:|
| kitchen1 / 000000 | 0 | 0.947754 | 0.763800 |
| kitchen1 / 000001 | 0 | 0.934204 | 0.596340 |

These are implementation diagnostics from `manifest.json`, not reconstruction
quality metrics. The sample format does not store original temporal indices;
the committed sample generator takes the first N frames, so local and original
indices coincide for these two inputs. Arbitrary external sample provenance
must be checked separately.

Tests cover the fixed API, consistent and stale history, per-channel spatial
bounds, displacement direction symmetry, invalid observations, read-only inputs,
component switches, partial groups, per-pixel missing/reappearing observations,
fixed-reference behavior, real translated input, and the disabled control on
both committed samples. Runner tests cover outputs, provenance, parameters,
and input protection.

The original three-commit implementation passed all 37 tests. Full TAA,
clipping-only, and adaptive-without-clipping sample runs completed on both committed bursts; each run's
disabled-TAA control matched the aligned baseline exactly at group size 64.

Independent verification of peer-review findings F1 through F5 is recorded in
[taa-peer-review-verification.md](taa-peer-review-verification.md), including
seeded synthetic MSE measurements, a runner roundoff correction, and additional
regression checks. These characterize failure regimes without changing the
TAA defaults or establishing held-out quality improvement.

## Limitations and Track 4 handoff

PSNR and SSIM are **not yet run**. These two samples are development inputs,
not a held-out benchmark. Visual inspection shows that all photon-domain
reconstructions are substantially darker than the provided RGB ground truth.
The intensity-response and reference-time conventions need to be settled with
Track 4 before interpreting metrics, using identical preprocessing, crops, and
support for every method.

Clamping can discard valid detail when the current group is noisy or wrong.
Adaptive weighting can reduce noise averaging when apparent motion comes from
photon noise, or when large displacement is nevertheless aligned correctly.
The clamp only catches history outside the current local range; a wrong match
within that range can pass. The displacement term is a heuristic and is not an
occlusion detector. Small or low-photon groups may require a noise-aware margin;
the present fixed margin does not adapt to group size or photon rate.

Lower confidence on burst 000001 does not demonstrate better reconstruction.
No quality improvement over the baseline is established by these checks.
Jack can consume the saved float arrays and validity/count diagnostics for the
shared evaluation harness. The TAA Approach contribution is the method above;
report conclusions must be based on the subsequent measured comparisons,
including negative results. Larger motion sweeps and held-out component
ablations remain part of the later team milestones.
