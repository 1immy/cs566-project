# Synthetic motion-speed sweep (issue #6)

Question: how do naive averaging, block alignment and our TAA behave as the scene moves faster, when the true motion is known and controlled?

## What was built
`src/synthetic.py` and `scripts/run_synthetic_sweep.py`. A seeded numpy simulator, **not the lab's visionsim**. Given a ground-truth image it makes a 1024-frame binary burst:

- Radiance is `gt / 2.7`, so the fixed evaluation gain of 2.7 maps the long-run photon average exactly back to the ground truth.
- The content translates linearly along one direction (default +x) and the **last frame is at the ground-truth pose**, as in the real data.
- Each binary pixel is an independent Bernoulli draw, seeded.

Every method then runs through the same code as the real-data runs (`src/motion_study.py`): same cube, last-group reference, groups of 64, fixed gain 2.7. Records use the same format as `scripts/run_full_scale.py`, so `scripts/analyze_motion_bins.py` reads both.

```
python scripts/run_synthetic_sweep.py
python scripts/analyze_motion_bins.py --results results/synthetic_sweep
```

## Not modelled
Sensor effects other than photon shot noise, rolling shutter, disocclusion, parallax, non-translational motion, lighting changes, and a light-level sweep. Absolute PSNR is optimistic compared with real bursts because the intensity rule matches the simulator exactly. Use the sweep for trends against speed, not for absolute numbers.

## Run recorded in `results/synthetic_sweep/`
Ground truth: `data/samples/kitchen1_000000_gt.png` (128x128 crop), seed 566, 1024 frames, groups of 64, one image, one seed. Speeds are the nominal total drift over the burst. "Measured" is the drift between the first and last 256-frame window (`src/motion.py`), which spans 768 of the 1023 frame steps, so it is about 3/4 of the nominal value.

| nominal px | measured px | PSNR naive | PSNR aligned | PSNR TAA | SSIM naive | SSIM aligned | SSIM TAA |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 32.84 | 33.06 | 32.63 | 0.889 | 0.918 | 0.898 |
| 2 | 1.7 | 24.67 | 32.26 | 31.68 | 0.813 | 0.913 | 0.892 |
| 5 | 3.8 | 20.29 | 31.52 | 30.73 | 0.685 | 0.906 | 0.880 |
| 10 | 7.5 | 18.10 | 30.15 | 29.28 | 0.580 | 0.895 | 0.858 |
| 20 | 14.9 | 16.38 | 27.16 | 26.55 | 0.484 | 0.862 | 0.820 |
| 40 | 29.9 | 14.73 | 22.42 | 23.23 | 0.401 | 0.743 | 0.735 |

PSNR in dB. Plots: `results/motion_analysis/synthetic/psnr_vs_drift.png`, `ssim_vs_drift.png`, `delta_by_bin.png`.

## Observations (synthetic data only)
- Naive averaging collapses within a few pixels of drift (about 8 dB lost at 2 px). Block alignment degrades slowly.
- TAA is 0.4 to 0.9 dB below plain alignment in PSNR from 0 to 20 px and 0.02 to 0.04 lower in SSIM. At the fastest speed (30 px measured) TAA is 0.8 dB higher in PSNR but still slightly lower in SSIM (0.735 against 0.743), so the PSNR crossover lies between 15 and 30 px.
- This agrees with the mid-term and PR #44 results on real bursts (TAA matches alignment in PSNR and loses SSIM) and says nothing new about why. It is one image and one seed, with two bursts per bin, so no confidence intervals are reported.

## Rendered sweep with visionsim (in progress, not yet run)
The numpy simulator is deliberately small. A visionsim sweep adds rendered frames and the real sensor emulation step.
1. Build the scenes: `blender -b --factory-startup -P scripts/blender/make_pan_scenes.py -- --out data/blender --speeds 0 2 5 10 20 40`. This writes one `.blend` per speed with an emission-only textured plane and a camera that pans by that many pixels over 1024 frames (the last frame is the ground-truth pose). `data/` is not tracked.
2. Render a short test first: `visionsim blender.render-animation data/blender/pan_v10.blend data/blender/render_v10 --frame-end 8 --config.device-type cpu --config.max-samples 1 --config.no-use-denoising --config.no-previews`, and check what it wrote.
3. Binary frames: `visionsim emulate.spad --input-dir ... --output-dir ... --factor ...` (Bernoulli sampling of the linear frames). The factor and the conversion into the `data/samples` format are decided after step 2; they are not written yet.
The scene script has not been run in Blender yet.
