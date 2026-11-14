# Track 2 sample baseline

Run from the repository root:

```powershell
py -m pip install -r requirements.txt
py -m unittest discover -s tests -v
py scripts/run_sample_baselines.py --group-size 64 --reference last
```

The installation command uses the pinned project dependencies. The output
manifest records the Python, NumPy, and Pillow versions actually used.

The runner reads the two committed `kitchen1` samples and by default writes to
`results/sample_baselines/`. Each sample produces naive, aligned, and ground
truth PNGs, plus a compressed archive containing float32 reconstructions and
per-pixel contributing-frame counts. PNGs use a direct intensity mapping,
without exposure normalization. Use the float arrays for evaluation.
`manifest.json` records the samples, sample-local frame range, spatial crops,
settings, reference selection and its inclusive sample-local frame range,
group frame counts, dependency versions, and SHA-256 hashes of inputs and
implementation. The float archive also stores `reference` and the inclusive
sample-local `reference_frame_indices`.
The original frame range is not stored in the NPZ format. The current sample
generator takes the first N frames, so sample-local indices coincide with the
original frame indices for these two committed samples. External inputs are
identified by filename and hash to avoid recording personal absolute paths.
The runner rejects duplicate sample stems and output paths that overwrite
input files.

## Method

`src/baseline.py` divides each burst into consecutive groups (64 frames by
default). `aligned_average(cube, group_size=64, reference="last")` uses the
last existing group mean as its fixed reference. The keyword-only `reference`
option accepts `"first"` to reproduce the original reference policy. Group boundaries
always start at frame 0. For 256 frames grouped by 96, the groups are
`[0:96]`, `[96:192]`, and `[192:256]`; the last reference contains 64 frames,
not a newly selected 96-frame window. No input frames are reversed.

The reference group seeds the accumulation once without warping. All other
groups are visited in original order, aligned through
`src/alignment.py::align_warp(burst, history)`, and merged with weights equal
to their frame counts. Invalid warped pixels contribute neither intensity nor
frame count. Every reference pixel starts with the reference group's actual
frame count, even when that group is short. No history rejection or
motion-adaptive blending is applied.

Alignment averages each group, uses channel means for block matching, and
searches coarse to fine over at most three pyramid levels. Blocks are 8x8
pixels at each level. The coarse search radius is four pixels; finer levels
search within two pixels of the doubled previous displacement. Matching uses
mean squared error with at least half a block overlapping the source.
At the finest level, a zero-centered two-pixel fallback search also checks
small motions when the coarse prediction has sufficient overlap. Integer
winners are refined locally at half-, quarter-, and eighth-pixel steps on
common support, then RGB is sampled bilinearly. See
[alignment-subpixel-confidence.md](alignment-subpixel-confidence.md) for
the refinement and optional confidence contract. This changes reconstructed
pixel values relative to the original integer-only aligner, for both methods.
If no candidate overlaps sufficiently, the coarse displacement is retained
so that out-of-bounds pixels remain invalid instead of reverting to unaligned
content. This is geometric validity only; an in-bounds match can still be wrong.

## Track 3 interface

- Input `burst`: binary uint8/bool array `[T,H,W,3]` representing one group.
- Input `history`: finite reference RGB image `[H,W,3]` in `[0,1]`, in the
  same photon-detection-probability domain as the group mean.
- Output `warped_frame`: float32 RGB image in reference coordinates.
- Output `motion_field`: float32 `[H,W,2]` backward offsets `(dy,dx)` in pixels:
  `warped[y,x]` samples `group_mean` bilinearly at `(y+dy,x+dx)`.
- Optional `align_warp_with_confidence(burst, history)` returns those same two
  arrays plus an alignment-confidence map. This is distinct from TAA retention.
- Out-of-bounds warped pixels are `NaN`; consumers must mask them explicitly.
  Do not pass a history containing NaNs back to the aligner.

Both baseline and TAA comparisons must use the same alignment settings and
reference policy. The current Track 3 wrapper still uses the first group.
Its sample runner, characterization script, shared evaluation runner, and
baseline-control test explicitly request `reference="first"` to preserve
their original behavior. Their TAA arithmetic and evaluation rules are
unchanged. Last-reference TAA requires a coordinated Track 3 update; the
last-reference baseline must not be compared directly with first-reference
TAA as a rejection-only experiment. The current shared evaluation runner
therefore continues to evaluate all methods using its existing first-reference
policy. `align_warp(burst, history)` and its two returned arrays are unchanged.

## Issue #19 validation

On November 8, 2026, both commands completed on the two committed samples:

```powershell
py scripts/run_sample_baselines.py --reference first --group-size 64 --out results/issue19/first
py scripts/run_sample_baselines.py --group-size 64 --out results/issue19/last
```

The second command checks the default last reference. Outputs are ignored
local artifacts. These #19 checks preceded the #20 subpixel implementation;
their source hashes identify the earlier aligner. Their manifests identify
crop `(336,464,336,464)`, local
frames 0 through 255, groups of 64, and reference ranges 0 through 63 or
192 through 255. Existing tracked sample outputs retain their original
first-reference provenance and are not replaced by these runs.

These samples contain only the first 256 frames of each original 1,024-frame
burst. The last available group is therefore not the last group of the full
burst. These runs verify execution and reference bookkeeping, not agreement
with the full burst's ground-truth pose or improved reconstruction quality.

### Additional-scene verification completed

On November 8, 2026, the downloaded Drive slice was used to check
`library-homeoffice`, bursts 0 and 1. Both source files were verified as
uint8 `(1024,800,100,3)` packed arrays with matching ground-truth PNGs.
The same predetermined central crop `(272,528,272,528)`, all 1,024 frames,
64-frame groups, and updated subpixel aligner were used for both references.
The first reference contains frames 0 through 63; the last contains 960
through 1023. Both bursts were checked without selecting the better burst
or changing crop or settings in response to the scores.

The reproducible verification script reads the downloaded scene directly
and crops before unpacking; no raw data is copied into the repository:

```powershell
py scripts/verify_reference_scene.py --scene-dir ../../cs566-drive-slice/library-homeoffice --out results/issue19/library-validation
```

Adjust `--scene-dir` to the local downloaded scene. Committed code contains
no personal absolute paths. Ignored outputs include a manifest with input
and source hashes, CSV scores, raw float reconstructions and contribution
counts, and comparison PNGs. Metrics use the existing Track 4
`evaluate_channels` with `clip(2.7 * reconstruction,0,1)` for every method,
unchanged GT, and the entire identical crop. No gain is fitted to the GT,
no method-specific scoring mask is selected, and no reconstruction is shifted
to improve its score. This is an additional-scene development check, not a
held-out benchmark or a TAA comparison.

| Method | Burst 0 PSNR (dB) | Burst 0 SSIM | Burst 1 PSNR (dB) | Burst 1 SSIM |
|---|---:|---:|---:|---:|
| Naive average | 15.2007 | 0.2361 | 14.9467 | 0.2147 |
| Aligned, first reference | 14.7673 | 0.1841 | 14.2724 | 0.1623 |
| Aligned, last reference | 15.7095 | 0.4026 | 15.6810 | 0.3833 |

The existing phase-correlation diagnostic estimates the shift required to
bring each reference group into the GT coordinates. For burst 0, `(dy,dx)`
changes from `(1.3,-6.0)` for the first group to `(0.0,-0.2)` for the last;
the respective peak values are 0.09 and 0.20. For burst 1, the shifts are
`(4.0,-13.7)` and `(0.1,-0.6)`, with peaks 0.05 and 0.17. This is approximate
global translation, with noisy estimates at low peaks, rather than a claim
that all regions share a single exact displacement. The aligned-last
reconstruction has the same measured residual shift as its reference in
both bursts. Inspection of both comparison images also shows the shelf and
globe contours closer to GT with the last reference.

These measurements satisfy #19's additional-scene verification requirement
for the tested settings. They support the last-reference default without
establishing universal improvement across scenes or solving the remaining
intensity-response mismatch. The installed Python 3.14.7, NumPy 2.5.3, and
Pillow 12.3.0 versions are recorded in the manifest; this does not claim
reproduction of the pinned environment. GitHub issue closure remains a
separate review action.

## Limits and evaluation status

This version estimates fractional displacements at 1/8-pixel search resolution
and constant motion within each block. This resolution is not an accuracy
guarantee. It does not model motion within a group or calibrated occlusion
confidence. Coarse-to-fine search can converge to incorrect local
matches, including for integer shifts that do not align with the pyramid grid.
Independent photon noise in flat or smooth regions can produce apparent
motion in a static scene. Motion-field magnitude is therefore not a calibrated
physical-motion measurement or a confidence score. Track 3 should account for
unreliable correspondences; adding an alignment regularizer later must be done
equally for both methods.

Tests cover identity, shift direction, opposite local motions, independent
photon groups, missing source borders, real alignment followed by weighted
merging, partial groups, input validation, and runner artifact/provenance
checks. Sample smoke tests check shapes and ranges. These checks do not
establish improved reconstruction quality on the real samples.

PSNR/SSIM for the original 256-frame first/last sample runs have not been
rerun. The full-burst additional-scene measurements above cover the updated
aligner. Earlier
first-reference sample metrics are documented in `docs/midTerm-evaluation.md`
and the mid-term report. Track 4 should evaluate any new comparisons against
the same ground truth and spatial support, using its agreed shared intensity
rule. The two development samples are not a held-out benchmark. These
128x128, 256-frame runs do not reproduce the report's full-burst, 300x304,
per-image-gain table.
