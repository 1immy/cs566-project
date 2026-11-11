# TAA comparison and component ablations

Development samples only. Full kitchen1 bursts 0, 1, and 10: **not yet run**.

Shared gain: 2.7; reference: last; group size: 64. Every pixel of the same input crop is scored. See manifest.json for input/source hashes, crop and frame indices, and all component settings.

| Scene / burst | Method | PSNR (dB) | SSIM |
|---|---|---:|---:|
| kitchen1 / 000000 | naive | 11.9306 | 0.158893 |
| kitchen1 / 000000 | aligned | 12.0182 | 0.159436 |
| kitchen1 / 000000 | taa_disabled | 12.0182 | 0.159436 |
| kitchen1 / 000000 | clamp_only | 12.0208 | 0.159751 |
| kitchen1 / 000000 | blend_only | 12.0230 | 0.159953 |
| kitchen1 / 000000 | confidence_only | 12.0206 | 0.159709 |
| kitchen1 / 000000 | taa | 12.0259 | 0.160309 |
| kitchen1 / 000001 | naive | 9.7773 | 0.139671 |
| kitchen1 / 000001 | aligned | 9.7949 | 0.139030 |
| kitchen1 / 000001 | taa_disabled | 9.7949 | 0.139030 |
| kitchen1 / 000001 | clamp_only | 9.7963 | 0.139518 |
| kitchen1 / 000001 | blend_only | 9.8044 | 0.141420 |
| kitchen1 / 000001 | confidence_only | 9.7960 | 0.139367 |
| kitchen1 / 000001 | taa | 9.8049 | 0.141743 |

Clamp-only clips history with ordinary frame weighting. Blend-only discounts history using displacement. Confidence-only discounts history using neighborhood disagreement. Full TAA combines all three. The confidence map is a history-retention heuristic, not alignment-match confidence or a calibrated occlusion probability.

## Measured component differences from aligned averaging

| Scene / burst | Component | Delta PSNR (dB) | Delta SSIM |
|---|---|---:|---:|
| kitchen1 / 000000 | clamp_only | +0.0025 | +0.000315 |
| kitchen1 / 000000 | blend_only | +0.0048 | +0.000518 |
| kitchen1 / 000000 | confidence_only | +0.0024 | +0.000274 |
| kitchen1 / 000000 | taa | +0.0076 | +0.000873 |
| kitchen1 / 000001 | clamp_only | +0.0014 | +0.000488 |
| kitchen1 / 000001 | blend_only | +0.0095 | +0.002390 |
| kitchen1 / 000001 | confidence_only | +0.0011 | +0.000337 |
| kitchen1 / 000001 | taa | +0.0100 | +0.002713 |
