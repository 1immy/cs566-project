# TAA comparison and ablation study

This is the Track 3 implementation and measurement record for issues
[#21](https://github.com/1immy/cs566-project/issues/21) and
[#27](https://github.com/1immy/cs566-project/issues/27).
The full kitchen1 comparison on bursts 0, 1, and 10 was completed on
November 11, 2026 using the downloaded team Drive slice. All seven methods
were run on each complete burst with the shared protocol below. Alignment
improves on naive averaging on both metrics for every burst. Full TAA changes
PSNR in mixed directions and lowers SSIM on all three bursts. The component
ablations do not identify a method that improves both metrics on every burst.

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

## Full-burst results: issue #21

All rows below use the 300x304 crop and all 1,024 frames. The last reference
group spans frames 960 through 1023, and the last processed TAA update spans
896 through 959. The full-data inputs for bursts 0 and 1 reproduce their
committed binary samples and ground-truth crops exactly.

| Scene / burst | Method | PSNR (dB) | SSIM |
|---|---|---:|---:|
| kitchen1 / 000000 | naive | 15.1106 | 0.298136 |
| kitchen1 / 000000 | aligned | 16.2930 | 0.430630 |
| kitchen1 / 000000 | taa | 16.2443 | 0.391412 |
| kitchen1 / 000001 | naive | 14.5170 | 0.303463 |
| kitchen1 / 000001 | aligned | 16.9756 | 0.407346 |
| kitchen1 / 000001 | taa | 17.1746 | 0.369056 |
| kitchen1 / 000010 | naive | 14.6750 | 0.358450 |
| kitchen1 / 000010 | aligned | 15.8333 | 0.484429 |
| kitchen1 / 000010 | taa | 15.8007 | 0.454473 |

Full TAA loses 0.0487 dB on burst 0 and 0.0326 dB on burst 10 relative to
aligned averaging, while gaining 0.1989 dB on burst 1. Its SSIM differences
are -0.039218, -0.038289, and -0.029955, respectively. A PSNR improvement
alone on burst 1 therefore does not establish an overall quality improvement.
The [real-motion crop](taa-failure-gallery.md#real-kitchen1-motion) displays
the corresponding reconstructions and TAA diagnostics.

The [complete seven-method table](assets/taa-study-full/findings.md),
[per-channel CSV](assets/taa-study-full/results.csv), and
[manifest](assets/taa-study-full/manifest.json) provide the unrounded scores,
input/source hashes, configuration and disabled-control evidence. Comparison
figures are available for [burst 0](assets/taa-study-full/kitchen1_000000_comparison.png),
[burst 1](assets/taa-study-full/kitchen1_000001_comparison.png), and
[burst 10](assets/taa-study-full/kitchen1_000010_comparison.png).
Raw bursts and float reconstruction archives remain in ignored local storage.

## Component analysis: issue #27

These are single-component variants on exactly the same three bursts and
scoring rule as the table above. Values are differences from aligned averaging.

| Burst | Component | Delta PSNR (dB) | Delta SSIM |
|---|---|---:|---:|
| 000000 | Clamp only | -0.0032 | +0.000800 |
| 000000 | Blend only | -0.0483 | -0.039218 |
| 000000 | Confidence only | -0.0016 | +0.000789 |
| 000001 | Clamp only | +0.0382 | +0.007475 |
| 000001 | Blend only | +0.2006 | -0.038200 |
| 000001 | Confidence only | +0.0502 | +0.006174 |
| 000010 | Clamp only | -0.0024 | +0.000431 |
| 000010 | Blend only | -0.0323 | -0.029941 |
| 000010 | Confidence only | -0.0015 | +0.000432 |

There is no single best ingredient across both metrics. Displacement-only
blending gives the largest PSNR gain on burst 1, but reduces PSNR on the
other two bursts and SSIM on all three. Clamp-only and neighborhood-confidence-only
retention slightly improve SSIM on all three bursts; their PSNR changes are
small losses on bursts 0 and 10 and gains on burst 1. Clamp-only has the
highest SSIM among isolated variants on bursts 0 and 1, while confidence-only
is marginally higher on burst 10. Confidence-only exceeds clamp-only PSNR on
each burst, so the metrics favor different components.

The full-TAA and blend-only results are close, suggesting that displacement
weighting dominates the changes in this configuration. Displacement alone
does not identify incorrect alignment and can discard useful history from
correctly registered groups. These observations motivate further study;
they do not justify tuning the reported defaults against these three crops.
Single-component ablations do not quantify all interactions between components.
No held-out or motion-speed-sweep conclusion follows from one development scene.

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
differences do not establish a general benefit. Issue #27's full-burst
component comparison is reported separately above.
The confidence-only and clamp-only changes were smaller and also positive
on these particular samples. The synthetic
[failure gallery](taa-failure-gallery.md) records regimes with negative
results separately; its photon-domain scores are not pooled with SPC scores.

## Completion evidence

| Required run | Status |
|---|---|
| Full kitchen1 burst 0, all seven methods | Measured, with scores and provenance committed |
| Full kitchen1 burst 1, all seven methods | Measured, with scores and provenance committed |
| Full kitchen1 burst 10, all seven methods | Measured, with scores and provenance committed |

The disabled-TAA reconstruction agrees with the aligned baseline within the
recorded float32 tolerance on all three bursts. Geometric frame counts match
for every variant, and disabled-TAA accumulation weights match baseline counts
exactly. The saved arrays were independently rescored with `evaluate_channels`
to check every per-channel and mean value in the CSV. All fixed public
interfaces and the shared alignment/evaluation implementations remain unchanged.
