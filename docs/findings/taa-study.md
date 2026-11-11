# TAA comparison and ablation study

This is the Track 3 implementation and measurement record for issues
[#21](https://github.com/1immy/cs566-project/issues/21) and
[#27](https://github.com/1immy/cs566-project/issues/27).
The full kitchen1 comparison on bursts 0, 1, and 10 is **not yet run**.
The full `.npy`/`.png` pairs are unavailable in this checkout and the connected
Drive account could not read the documented shared slice on November 11, 2026.
The two committed samples do not contain burst 10 or the final 768 frames
of either available burst. Neither issue is complete until the full-data
tables below can be produced and interpreted.

## Shared protocol

Track 4 selected a fixed reconstruction gain of 2.7 in
[issue #18](https://github.com/1immy/cs566-project/issues/18#issuecomment-6552346154).
The new `scripts/run_taa_study.py` calls the existing `evaluate_channels`
without modifying Track 4's code. It evaluates `clip(2.7 * reconstruction, 0, 1)`
against unchanged RGB ground truth, using every pixel of the identical crop
for every method. PSNR and SSIM are computed per RGB channel and averaged;
SSIM uses the existing 7x7 uniform window with reflect padding.

The full study defaults to all 1,024 frames, groups of 64, the last group
as fixed reference, and the predetermined central 300x304 crop
`(250,550,248,552)`. The report specified those dimensions but did not record
the coordinates; these coordinates explicitly center that rectangle.
The baseline, full TAA, and each ablation use the same unmodified aligner,
reference, grouping and support. The reference group initializes history
once and all other groups are processed in their original order. Disabling
TAA must reproduce aligned averaging within recorded float32 rounding and
match its frame counts and accumulation weights exactly.

This updates the protocol consistently; it does not reproduce the historical
midterm Table 1, which used reversed bursts and fitted per-image gains.
Kitchen1 is development data, not a held-out benchmark. No held-out quality
claim is made by this study.

## Reproduce the full comparison and ablations

Place only the approved Drive slice under a local ignored directory, with
`000000.npy`, `000000.png`, `000001.npy`, `000001.png`, `000010.npy`, and
`000010.png` in the scene directory. No NAS access is needed.

```bash
venv/bin/python scripts/run_taa_study.py \
  --scene-dir data/drive-slice/kitchen1 \
  --indices 0 1 10 --crop 250 550 248 552 \
  --group-size 64 --reference last --gain 2.7 \
  --save-arrays --out results/sample_taa/study/full
```

The runner requires the full packed shape `(1024,800,100,3)` and uses the
existing crop-aware loader, so it need not unpack the complete image.
It produces:

- `results.csv`: per-channel and mean PSNR/SSIM for all seven methods.
- `findings.md`: main table and component differences from aligned averaging.
- `manifest.json`: input/source hashes, crop/frame/reference metadata,
  configurations, dependency versions, and disabled-control checks.
- Per-burst comparison figures and optional float reconstruction archives
  with retention, validity and frame-count diagnostics.

All seven methods are run on the same burst before moving to the next:
naive, aligned, disabled TAA, clamp only, blend only, confidence only, and
full TAA. The [component definitions](taa-component-controls.md) distinguish
displacement weighting from neighborhood-disagreement confidence. They do
not introduce an alignment-confidence model or tune parameters against GT.
The final retention map describes the penultimate group when the last group
is the reference; it does not summarize every update.

## Executed sample validation

Run on November 11, 2026:

```bash
venv/bin/python scripts/run_taa_study.py \
  --samples data/samples/kitchen1_000000.npz data/samples/kitchen1_000001.npz \
  --out results/sample_taa/study/samples
```

The [measured sample table](assets/taa-study/findings.md),
[per-channel CSV](assets/taa-study/results.csv), and
[manifest](assets/taa-study/manifest.json) are committed for review.
Each sample contains 256 frames and the crop `(336,464,336,464)`.
Its last group is frames 192 through 255, which is still far from the
original full burst's end-of-burst ground-truth pose. The sample result must
not substitute for issue #21's full-burst comparison.

Disabled TAA matched the aligned baseline exactly on both samples. Among
the isolated components, displacement-only blending gave the largest
sample gains over alignment: 0.0048 dB / 0.000518 SSIM on burst 0, and
0.0095 dB / 0.002390 SSIM on burst 1. Full TAA gained 0.0076 dB / 0.000873
and 0.0100 dB / 0.002713, respectively. These small, single-scene development
differences do not establish a general benefit or resolve issue #27.
The confidence-only and clamp-only changes were smaller and also positive
on these particular samples. The synthetic
[failure gallery](taa-failure-gallery.md) records regimes with negative
results separately; its photon-domain scores are not pooled with SPC scores.

## Remaining completion evidence

| Required run | Status |
|---|---|
| Full kitchen1 burst 0, all seven methods | Not yet run: full input unavailable |
| Full kitchen1 burst 1, all seven methods | Not yet run: full input unavailable |
| Full kitchen1 burst 10, all seven methods | Not yet run: full input unavailable |

Once inputs are available, run the command above, retain the actual generated
table and manifest, compare all component deltas on those same three bursts,
and replace this pending status with the measured findings. Do not infer the
missing values from the committed samples or the historical midterm table.
