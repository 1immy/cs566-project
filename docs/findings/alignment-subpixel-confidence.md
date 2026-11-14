# Track 2: subpixel motion and alignment confidence

Issue #20 requests fractional offsets and a per-block confidence diagnostic
for Track 3. The shared aligner now refines its integer block matches down to
1/8-pixel search resolution and uses bilinear RGB sampling. The original
`align_warp(burst, history)` signature and its two returned arrays are preserved.
Both baseline and TAA use this same updated aligner.

## Refinement

The original three-level, 8x8 block matcher provides integer predictions.
The finest level also checks offsets within two pixels of zero when the
propagated prediction overlaps at least half the block. This allows a small
motion to recover from a misleading coarse estimate. If the prediction lacks
that overlap, the additional search is omitted so missing source content is
not replaced by unrelated in-bounds edge pixels.

Each non-exact integer winner is refined with 3x3 searches at steps 0.5,
0.25, and 0.125 pixels. Fractional offsets stay within half a pixel of the
integer winner on each axis. Every candidate uses the same reference support
and a one-pixel source halo. Refinement is skipped if fewer than half the
block's pixels have that support. A candidate is accepted only when it lowers
SSD on the common support. Exact integer matches remain integer and retain
their exact sampling behavior.

RGB warping samples `(y+dy,x+dx)` bilinearly. Out-of-source coordinates are
NaN. Exact integer coordinates on the last row/column remain valid, since
they do not require an out-of-bounds neighbor. Motion remains constant within
each finest-level block; motion within a frame group is not modeled.

## Optional confidence API

```python
from src.alignment import align_warp_with_confidence
from src.burst_io import load_sample, naive_average

cube = load_sample("data/samples/kitchen1_000000.npz")
reference_image = naive_average(cube[192:256])
warped, motion, match_confidence = align_warp_with_confidence(cube[:64], reference_image)
```

The first two arrays match `align_warp(cube[:64], reference_image)` exactly,
including NaNs. `match_confidence` is float32 `[H,W]` in `[0,1]`, constant per
8x8 block except that invalid warped pixels receive zero. All outputs share
the reference grid. Inputs are not modified.

Confidence is computed from the integer search evidence before fractional
refinement, using channel-mean images. Let:

- `e` be the integer winner's mean squared residual on its valid overlap.
- `e_alt` be the smallest error among searched alternatives more than one
  integer pixel away in Chebyshev distance. Alternatives must cover the same
  reference support as the winner and are rescored on that support.
- `v` be the smaller spatial variance of the reference and matched patches.
- `a` be the winning overlap area divided by the block area.

The score is:

```text
separation = max(0, e_alt - e) / max(e_alt, 1e-12)
agreement = v / (v + e + 1e-12)
match_confidence = clip(separation * agreement * a, 0, 1)
```

No feasible winner or no comparable distant alternative gives zero. Flat
patches and exact repetitive ties give zero. A clean, unique match can score
high even at a large displacement: there is no explicit displacement penalty.
The subsequent warp validity mask sets missing observations to zero.

This is a heuristic score of the local integer match, not a calibrated
probability, a visibility classifier, a statistical sample count, or the
confidence that the fractional refinement is accurate. It can be conservative
on fractional motion and unreliable under photon noise, repeated structures,
intensity changes, or a locally convincing wrong match. Unsearched matches
are not considered. No threshold changes the baseline's contributions.

## Track 3 handoff

The current TAA wrapper calls the original two-output API and therefore uses
fractional alignment automatically. It does not yet consume `match_confidence`.
Adrian can use the optional diagnostic API when designing how registration
quality should influence history retention. Its direction and use must be
chosen explicitly; multiplying history by the score without examining which
observation is unreliable is not an established rejection rule.

Track 3's current `confidence_map` is the last update's retained fraction of
history. That quantity has different semantics from this alignment score.
The fixed `reject_and_blend` signature is unchanged, and no TAA equation or
default has been altered here. Baseline and TAA comparisons must still share
the reference policy, group partition, aligner, and evaluation preprocessing.

## Reproduction and validation

From the repository root:

```powershell
py -m unittest discover -s tests -v
py scripts/run_alignment_diagnostics.py
py scripts/run_sample_baselines.py --reference last --out results/issue20/baseline-last
py scripts/run_sample_taa.py --out results/issue20/taa-first
```

The diagnostic runner exports one NPZ and confidence PNG for each non-reference
group, plus a manifest with source/input hashes, frame ranges, crops, and
versions. Each NPZ stores `warped_frame`, `motion_field`,
`alignment_confidence`, and inclusive sample-local source/reference frame
ranges. The reference group is not matched against itself and receives no
artificial confidence measurement. `--reference first` is available.
All generated files under `results/issue20/` are ignored.

On November 8, 2026, all 60 tests passed in the installed environment. Tests
cover positive and negative fractional shifts, bilinear values and borders,
exact integer translations, missing-source strips, independent photon groups,
unique/flat/repetitive match confidence, displacement-independent confidence
on clean matches, blockwise scores and invalid masks, read-only inputs, both
reference policies, and exported diagnostic provenance. The fractional
geometry fixture uses deterministic binary counts from known smooth functions;
it is separate from the independent photon-noise test. On that fixture,
fractional motion has lower interior reconstruction error than rounding its
motion to integers. This is not evidence of real-scene quality improvement.

Both committed `kitchen1` samples completed all three runners above, with
64-frame groups, 128x128 crop `(336,464,336,464)`, local frames 0 through 255.
Diagnostics and baseline use the last available group (192 through 255);
TAA retains its first-reference policy (0 through 63). The disabled-TAA
control still matches the first-reference baseline. Those two different
reference modes are operational checks, not a fair baseline-versus-TAA
quality comparison. The manifests record the installed dependency versions;
this run does not claim reproduction of the pinned environment.

Subsequent full-burst development validation on `library-homeoffice` bursts
0 and 1 is recorded in [alignment-baseline.md](alignment-baseline.md), including
the first-versus-last pose diagnostic and PSNR/SSIM under the shared fixed
gain. This satisfies the additional-scene check for #19 on the updated aligner.
No held-out benchmark or subpixel-versus-integer real-scene quality study
has been completed. Historical report metrics and the #19 pre-subpixel outputs
describe the earlier integer method. Recompute all methods together before
reporting quantitative comparisons with this version.
