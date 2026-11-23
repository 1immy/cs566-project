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

## Rendered sweep with Blender frames
The numpy simulator translates one image. This sweep instead uses frames rendered by Blender Cycles, so the motion comes from a moving camera rather than from resampling a still image.
1. Build the scenes: `blender -b --factory-startup -P scripts/blender/make_pan_scenes.py -- --out data/blender --speeds 0 2 5 10 20 40`. One `.blend` per speed: an emission-only noise-texture plane (contrast stretched with a color ramp) under an orthographic camera that pans by that many pixels over 1024 frames. The last frame is the ground-truth pose. `data/` is not tracked.
2. Render the frames: `blender -b data/blender/pan_v10.blend -P scripts/blender/render_frames.py -- --out data/blender/frames_v10`, once per speed. Output is linear 16-bit grayscale PNG (`Raw` view transform), one per frame. Add `--frame-end 8` for a quick test.
3. Run the sweep: `python scripts/run_rendered_sweep.py --frames-root data/blender`. Each burst is a seeded Bernoulli draw from the frames divided by the gain 2.7 (`src/rendered.py`), then naive, aligned and TAA are scored as in the real-data runs. Records go to `results/rendered_sweep/` and are read by `scripts/analyze_motion_bins.py` as source "rendered".

Why not visionsim's renderer: on this setup (visionsim 0.2.0, Blender 5.3) `visionsim blender.render-animation` rendered the first frame and then disconnected without an error, exit code 0. Direct Blender rendering of the same scene works, so the frames are rendered that way. `visionsim emulate.spad` is not used either; the Bernoulli step is the same one as in the numpy sweep, so the two sweeps differ only in where the frames come from.

Not modelled, as in the numpy sweep: any noise but photon shot noise, rolling shutter, disocclusion and parallax (the scene is a flat plane panned by the camera). Cycles runs at 1 sample per pixel, so frames carry small pixel-filter noise; the ground truth is the last frame and has it too.

### Rendered results
One scene (noise texture, 128 x 128 px), one seed (566), one burst per speed, last-group reference, fixed gain 2.7. PSNR in dB / SSIM. "Measured" is the phase-correlation drift between the last and first 256-frame windows (`src/motion.py`), about three quarters of the nominal pan, as in the real data.

| Nominal px | Measured px | Naive | Aligned | TAA |
|---:|---:|---|---|---|
| 0 | 0.0 | 30.03 / 0.989 | 28.20 / 0.982 | 28.06 / 0.982 |
| 2 | 1.4 | 15.36 / 0.651 | 25.25 / 0.964 | 24.78 / 0.959 |
| 5 | 3.9 | 12.93 / 0.265 | 24.09 / 0.953 | 23.49 / 0.945 |
| 10 | 7.5 | 12.83 / 0.156 | 21.65 / 0.917 | 21.32 / 0.910 |
| 20 | 15.0 | 12.78 / 0.088 | 18.08 / 0.804 | 17.91 / 0.800 |
| 40 | 30.0 | 12.75 / 0.046 | 14.80 / 0.499 | 14.65 / 0.546 |

What the table supports:
- Naive averaging collapses as soon as there is motion (30 dB static, 15 dB at 2 px). Alignment recovers most of that and then degrades smoothly with speed.
- TAA is 0.1 to 0.6 dB below aligned in PSNR at every speed here, and up to 0.07 lower in SSIM, with one exception: at 40 px TAA's SSIM is higher (0.546 vs 0.499) while its PSNR is still lower. So on this scene TAA does not beat plain alignment in PSNR at any speed, and the fast-motion gap does not open up.
- The numpy sweep on kitchen1 had TAA ahead of aligned at 40 px (23.23 vs 22.42 dB) and behind at slower speeds. The two sweeps disagree on that one point, and they differ in texture and in how the motion is produced, so absolute numbers are not comparable between them.
- At 0 px aligned is 1.8 dB below naive. With nothing to correct, the 64-frame groups and sub-pixel resampling only cost sharpness on this fine texture.

What it does not support: any statement about intervals (one seed, one burst per speed), other textures, or real scenes. A pan across a flat plane also brings new content into view at the image edge, so part of the last frame was never seen in earlier frames; at 40 px that is nearly a third of the width of a 128 px image. That probably accounts for part of the low scores at the fast end, but I have not tested it (for example by scoring only the interior region).

