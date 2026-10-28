# Track 2 sample baseline

Run from the repository root:

```powershell
py -m pip install numpy Pillow
py -m unittest discover -s tests -v
py scripts/run_sample_baselines.py --group-size 64
```

The installation command is for a fresh Python environment; it does not pin
versions. The output manifest records the Python, NumPy, and Pillow versions
actually used.

The runner reads the two committed `kitchen1` samples and writes to
`results/sample_baselines/`. Each sample produces naive, aligned, and ground
truth PNGs, plus a compressed archive containing float32 reconstructions and
per-pixel contributing-frame counts. PNGs use a direct intensity mapping,
without exposure normalization. Use the float arrays for evaluation.
`manifest.json` records the samples, sample-local frame range, spatial crops,
settings, dependency versions, and SHA-256 hashes of inputs and implementation.
The original frame range is not stored in the NPZ format. The current sample
generator takes the first N frames, so sample-local indices coincide with the
original frame indices for these two committed samples. External inputs are
identified by filename and hash to avoid recording personal absolute paths.
The runner rejects duplicate sample stems and output paths that overwrite
input files.

## Method

`src/baseline.py` divides each burst into consecutive groups (64 frames by
default). The first group mean defines a fixed reference. Subsequent groups
are aligned through `src/alignment.py::align_warp(burst, history)` and merged
with weights equal to their frame counts. Invalid warped pixels contribute
neither intensity nor frame count. The final short group receives its actual
weight. No history rejection or motion-adaptive blending is applied.

Alignment averages each group, uses channel means for block matching, and
searches coarse to fine over at most three pyramid levels. Blocks are 8x8
pixels at each level. The coarse search radius is four pixels; finer levels
search within two pixels of the doubled previous displacement. Matching uses
mean squared error with at least half a block overlapping the source.
If no candidate overlaps sufficiently, the coarse displacement is retained
so that out-of-bounds pixels remain invalid instead of reverting to unaligned
content. This is geometric validity only; an in-bounds match can still be wrong.

## Track 3 interface

- Input `burst`: binary uint8/bool array `[T,H,W,3]` representing one group.
- Input `history`: finite reference RGB image `[H,W,3]` in `[0,1]`, in the
  same photon-detection-probability domain as the group mean.
- Output `warped_frame`: float32 RGB image in reference coordinates.
- Output `motion_field`: float32 `[H,W,2]` backward offsets `(dy,dx)` in pixels:
  `warped[y,x] = group_mean[y+dy,x+dx]`.
- Out-of-bounds warped pixels are `NaN`; consumers must mask them explicitly.
  Do not pass a history containing NaNs back to the aligner.

Both baseline and TAA comparisons must use the same alignment settings and
reference policy. These conventions still need confirmation with Track 3.

## Limits and evaluation status

This version estimates integer displacements and constant motion within each
block. It does not model motion within a group, subpixel displacement, or
occlusion confidence. Coarse-to-fine search can converge to incorrect local
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

PSNR/SSIM have not yet been run. Track 4 should evaluate both saved float
reconstructions against the same ground truth and spatial support. The two
development samples are not a held-out benchmark. The averaged photon signal
and provided RGB ground truth may have different intensity responses; any
evaluation preprocessing must be agreed with Track 4 and applied equally.
