# Final comparison figures: naive vs aligned vs TAA (issue #30)

Two figures for the slides and the webpage, both from real full-scale bursts (1024 frames, groups of 64, last-group reference, fixed gain 2.7). Rebuild with:

```
python scripts/make_comparison_figures.py --results results/full_scale --per-burst results/motion_analysis/real/per_burst.csv --out docs/findings/figures
```

- `figures/comparison_full_frames.png`: ground truth, naive, aligned and TAA at full frame, one row per burst, with PSNR and SSIM under each reconstruction.
- `figures/comparison_crops.png`: the same bursts, the same fixed 280 px window in every tile (the same position in all four columns of every row), so fine detail and noise can be compared.

**Which rows.** Not hand-picked. In each motion bin the script takes the burst where TAA gains most over alignment in PSNR and the burst where it loses most. The rows are therefore the extremes of the 31 measured bursts, not typical ones.

| Row | Bin | Drift | TAA - aligned (PSNR) |
|---|---|---:|---:|
| diner 0 | slow | 4.0 px | +0.41 dB |
| simplekitchen 1 | slow | 3.2 px | -0.88 dB |
| barbershop 1 | medium | 21.9 px | +1.11 dB |
| bathroom5 1 | medium | 20.7 px | -1.35 dB |
| bathroom3 1 | fast | 50.0 px | +2.08 dB |
| ultramodern 0 | fast | 55.5 px | -2.04 dB |

**What the figure shows.**
- Naive averaging smears moving structure wherever the motion is visible (for example `simplekitchen` 1, `barbershop` 1 and `bathroom3` 1); on `diner` 0 it is hard to tell the three reconstructions apart by eye. Aligned and TAA recover structure.
- `bathroom3` 1 is the clearest TAA case: aligned shows blocky tile artifacts on the fixture and the basin, TAA is smoother and closer to the ground truth, and PSNR is 2.1 dB higher. SSIM is nevertheless lower for TAA (0.285 against 0.329).
- `bathroom5` 1 and `ultramodern` 0 are mostly noise for all three methods at this photon level, and TAA is the noisier of the two reconstructions that align.
- `simplekitchen` 1 is a slow burst where TAA loses both ways: 0.88 dB in PSNR and 0.158 in SSIM.
- Across all 31 bursts TAA has lower SSIM than alignment in every one (see `final-results.md`), including rows where it looks better to the eye. A plausible reason is the grain TAA keeps, which SSIM penalises more than PSNR does; this was not tested.

**Limits.** Six bursts out of 31, chosen as extremes. Motion bins use a measured drift that is unreliable for most medium and fast bursts, so the "fast" label means large measured drift. The window of the crop figure is fixed, so it sometimes lands on a plain region (for example `ultramodern` 0).
