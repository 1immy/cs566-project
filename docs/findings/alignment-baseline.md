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

The issue also requires confirmation on at least one more real scene. That
check is **not yet run**: only `kitchen1` is available locally. After obtaining
a different scene from the shared Drive slice, create a local, untracked
sample covering the full burst and run both reference modes with identical
frame and crop selections. For example, if the shared scene is downloaded
under `data/raw/library-homeoffice`:

```powershell
py scripts/make_sample.py --scene data/raw/library-homeoffice --indices 0 --frames 1024 --size 128 --out data/local_samples
py scripts/run_sample_baselines.py --samples data/local_samples/library-homeoffice_000000.npz --reference first --out results/issue19/library-first
py scripts/run_sample_baselines.py --samples data/local_samples/library-homeoffice_000000.npz --reference last --out results/issue19/library-last
```

Record that scene, burst, frame range, crop, reference selection, and observed
ground-truth pose agreement before closing #19. A second `kitchen1` burst or
a synthetic test does not satisfy the additional-scene requirement.

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

PSNR/SSIM for these new first/last runs have not yet been run. Earlier
first-reference sample metrics are documented in `docs/midTerm-evaluation.md`
and the mid-term report. Track 4 should evaluate any new comparisons against
the same ground truth and spatial support, using its agreed shared intensity
rule. The two development samples are not a held-out benchmark. These
128x128, 256-frame runs do not reproduce the report's full-burst, 300x304,
per-image-gain table.
