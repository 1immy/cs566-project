# TAA comparison and component ablations

Full-burst development comparison; these scenes are not held out.

Shared gain: 2.7; reference: last; group size: 64. Every pixel of the same input crop is scored. See manifest.json for input/source hashes, crop and frame indices, and all component settings.

| Scene / burst | Method | PSNR (dB) | SSIM |
|---|---|---:|---:|
| kitchen1 / 000000 | naive | 15.1106 | 0.298136 |
| kitchen1 / 000000 | aligned | 16.2930 | 0.430630 |
| kitchen1 / 000000 | taa_disabled | 16.2930 | 0.430630 |
| kitchen1 / 000000 | clamp_only | 16.2898 | 0.431430 |
| kitchen1 / 000000 | blend_only | 16.2448 | 0.391412 |
| kitchen1 / 000000 | confidence_only | 16.2914 | 0.431419 |
| kitchen1 / 000000 | taa | 16.2443 | 0.391412 |
| kitchen1 / 000001 | naive | 14.5170 | 0.303463 |
| kitchen1 / 000001 | aligned | 16.9756 | 0.407346 |
| kitchen1 / 000001 | taa_disabled | 16.9756 | 0.407346 |
| kitchen1 / 000001 | clamp_only | 17.0138 | 0.414821 |
| kitchen1 / 000001 | blend_only | 17.1762 | 0.369145 |
| kitchen1 / 000001 | confidence_only | 17.0258 | 0.413520 |
| kitchen1 / 000001 | taa | 17.1746 | 0.369056 |
| kitchen1 / 000010 | naive | 14.6750 | 0.358450 |
| kitchen1 / 000010 | aligned | 15.8333 | 0.484429 |
| kitchen1 / 000010 | taa_disabled | 15.8333 | 0.484429 |
| kitchen1 / 000010 | clamp_only | 15.8309 | 0.484859 |
| kitchen1 / 000010 | blend_only | 15.8010 | 0.454487 |
| kitchen1 / 000010 | confidence_only | 15.8318 | 0.484861 |
| kitchen1 / 000010 | taa | 15.8007 | 0.454473 |

Clamp-only clips history with ordinary frame weighting. Blend-only discounts history using displacement. Confidence-only discounts history using neighborhood disagreement. Full TAA combines all three. The confidence map is a history-retention heuristic, not alignment-match confidence or a calibrated occlusion probability.

## Measured component differences from aligned averaging

| Scene / burst | Component | Delta PSNR (dB) | Delta SSIM |
|---|---|---:|---:|
| kitchen1 / 000000 | clamp_only | -0.0032 | +0.000800 |
| kitchen1 / 000000 | blend_only | -0.0483 | -0.039218 |
| kitchen1 / 000000 | confidence_only | -0.0016 | +0.000789 |
| kitchen1 / 000000 | taa | -0.0487 | -0.039218 |
| kitchen1 / 000001 | clamp_only | +0.0382 | +0.007475 |
| kitchen1 / 000001 | blend_only | +0.2006 | -0.038200 |
| kitchen1 / 000001 | confidence_only | +0.0502 | +0.006174 |
| kitchen1 / 000001 | taa | +0.1989 | -0.038289 |
| kitchen1 / 000010 | clamp_only | -0.0024 | +0.000431 |
| kitchen1 / 000010 | blend_only | -0.0323 | -0.029941 |
| kitchen1 / 000010 | confidence_only | -0.0015 | +0.000432 |
| kitchen1 / 000010 | taa | -0.0326 | -0.029955 |
