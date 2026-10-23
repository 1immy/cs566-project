# Motion inside one 1024-frame burst

Question (issue: "Check how much motion there is inside one 1024-frame burst"): does the scene move *within* a burst, so that alignment has to happen inside a burst, or only between bursts?

**Short answer: yes, there is real motion inside a burst, so alignment has to happen within a burst.** It is mostly horizontal, and its size varies a lot from burst to burst.

## Method
`scripts/measure_burst_motion.py`. For one burst, average four consecutive 256-frame windows (central 500x496 crop, all 3 channels), then estimate the global translation of each later window against the first by phase correlation with sub-pixel refinement. A single binary frame is too noisy to align on its own (mean photon rate about 0.157), which is why windows are used.

## Result (`train/kitchen1`, 3 bursts)
Shift of each later window relative to window 1, as (dy, dx) in pixels, with the correlation peak in brackets (higher is more trustworthy):

| Burst | window 2 | window 3 | window 4 |
|---|---|---|---|
| 000000 | (-0.1, 2.0) [0.24] | (-0.3, 5.1) [0.13] | (-0.8, 9.4) [0.07] |
| 000001 | (-0.5, 5.8) [0.08] | (-1.9, 13.7) [0.04] | (-2.3, 20.2) [0.03] |
| 000010 | (-0.8, 1.4) [0.14] | (-2.2, 2.1) [0.09] | (-3.1, 2.1) [0.06] |

- Total drift across a burst ranges from about 2 px to about 20 px, mostly along x.
- Between consecutive bursts of this scene the ground-truth image shifts about 28 px along x (burst 0 to 1), so motion continues across bursts as well.
- Burst 000000 accelerates (2, 3.2, 4.3 px per window). Burst 000010 nearly stops.
- The bursts give a natural range of motion speeds for the failure analysis, in addition to the visionsim sweeps.

## Implications
- **Track 2:** a global translation per window is a reasonable first motion model, but sub-pixel accuracy matters (shifts of 1-2 px per 256 frames). Align groups of frames (for example 32 to 128 frames), not single binary frames.
- **Track 3:** the largest drifts (burst 000001) are where history rejection should matter most. Use that burst as an early stress case.
- Peaks fall below about 0.05 in the fastest cases, so those estimates are noisy.

## Caveats
One scene, three bursts, global translation only, central crop only. Per-region and non-translational motion (disocclusion, parallax) was not measured. Rerun on more scenes before putting a number in the report.
