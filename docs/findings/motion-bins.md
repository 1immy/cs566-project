# Motion bins, resumable full-scale runs and failure analysis (issues #23, #24, #25)

Status: **run on the real training scenes (31 bursts).** Results and caveats are in the last section. A synthetic run of the same pipeline is in [synthetic-sweep.md](synthetic-sweep.md).

In the commands below, replace `DATA_ROOT` with the folder that holds one sub-folder per scene (for example the `train` folder, or the shared Drive slice). Only small text and PNG outputs are written; raw data is never copied.

## 1. Measure and bin (issue #23)
```
python scripts/bin_scenes_by_motion.py --data-root DATA_ROOT --bursts 0 1 --per-bin 3
```
- Measures, per scene and burst, the global drift between the first and last 256-frame window (`src/motion.py`, same method as `scripts/measure_burst_motion.py`).
- `drift_px` is the length of that shift. It spans the 768 frames between window centres, so it is about 3/4 of the drift over the whole burst.
- Bins are tertiles of the measured drift (slow, medium, fast), or fixed with `--edges SLOW_MEDIUM MEDIUM_FAST`.
- Writes `results/motion_bins/motion.csv` (one row per burst) and `results/motion_bins/bins.json` (edges, counts, every burst's bin, and the selected set: `--per-bin` bursts per bin, closest to the bin median, distinct scenes first). Bursts with a correlation peak below 0.05 are listed as unreliable.
- Resumable: bursts already in `motion.csv` are not measured again.

The result of the run is in "Real-data results" below.

## 2. Full-resolution runs (issue #24)
```
python scripts/run_full_scale.py --data-root DATA_ROOT --plan results/motion_bins/bins.json
```
- Runs naive, aligned and TAA on each selected burst: 1024 frames, groups of 64, last-group reference, fixed gain 2.7 (the protocol in `evaluation-intensity.md`).
- Writes one JSON record (scores, motion, settings) and one comparison PNG per burst under `results/full_scale/<scene>/`, plus `summary.csv`.
- Resumable: a burst is skipped when its record exists with the same settings. A changed setting, a corrupt record or a leftover `.tmp` file re-runs it. A burst that fails writes `<burst>.error.txt`, the run continues, and the exit status is 1. Run the same command again to retry only the unfinished bursts.
- `--dry-run` lists what would run. `--all-planned` runs every binned burst (all 100) instead of the 9 selected ones; use it only deliberately, it takes hours. `--limit N` caps one session. Memory: one full burst unpacks to about 2 GB; `--crop Y0 Y1 X0 X1` (x bounds multiples of 8) reduces it.

## 3. Failure analysis (issue #25)
```
python scripts/analyze_motion_bins.py --results results/full_scale --bins results/motion_bins/bins.json
```
- Per bin: n, mean and standard deviation of PSNR and SSIM for each method, and the paired difference TAA minus aligned with a seeded bootstrap 95% interval over bursts (only when a bin has at least 3 bursts).
- Plots: `psnr_vs_drift.png`, `ssim_vs_drift.png`, `delta_by_bin.png`. Real and synthetic records are analysed separately and never pooled.
- Report the plain result, including bins where TAA does worse than alignment.

## Tests
`python -m unittest discover -s tests`. The new tests build small scenes in the real on-disk format and check the binning, the skip and retry behaviour of the runner, and the analysis.

## Real-data results
Run on `train` (the NAS data, 50 scenes): bursts 0 and 1 of every scene were measured (100 bursts), then 31 were reconstructed at full resolution (1024 frames, group size 64, last-group reference, fixed gain 2.7). The 31 are the selection of `--per-bin 10` (30 bursts: closest to each bin's median, distinct scenes first) plus `attic` 0, which was run by mistake with `--all-planned` before that flag was stopped. It is a real finished run, so it is included, which makes the medium bin n = 11. An earlier pass with `--per-bin 3` (9 bursts plus `attic` 0 and 1) is a subset of these 31 and is not reported separately.

**Bins.** Tertile edges on measured drift: 12.77 px and 37.82 px. Counts over all 100 bursts: 33 slow, 33 medium, 34 fast. Measured drift ranges from 0.9 to 106 px. Plots are in `docs/findings/figures/`.

**Reliability of the motion measure.** 80 of the 100 bursts have a smallest correlation peak below 0.05, which the code flags as unreliable; 26 of the 31 reconstructed bursts are flagged. Every burst with a peak at or above 0.05 has a measured drift below about 19 px, so the flag and the speed are tied together. Two readings fit this and the data cannot separate them: large motion smears the 256-frame window means and lowers the peak (the drift is real, measured coarsely), or the large values are partly noise. Read the bins as approximate.

**Per-bin means** (same cube, reference and aligner for every method). Different scenes sit in different bins, so compare methods within a row, not one method across rows.

| Bin | n | Drift px | PSNR naive | PSNR aligned | PSNR TAA | SSIM naive | SSIM aligned | SSIM TAA |
|---|---|---|---|---|---|---|---|---|
| slow | 10 | 3.0 to 6.5 | 15.02 | 15.49 | 15.45 | 0.330 | 0.437 | 0.381 |
| medium | 11 | 18.7 to 32.0 | 14.46 | 15.26 | 15.26 | 0.293 | 0.395 | 0.325 |
| fast | 10 | 48.9 to 56.4 | 13.59 | 14.64 | 14.93 | 0.296 | 0.361 | 0.269 |

**TAA minus aligned**, mean over the bursts in the bin, with the seeded bootstrap 95% interval:

| Bin | n | PSNR dB | SSIM |
|---|---|---|---|
| slow | 10 | -0.04 [-0.26, +0.12] | -0.056 [-0.085, -0.033] |
| medium | 11 | +0.00 [-0.36, +0.34] | -0.070 [-0.092, -0.049] |
| fast | 10 | +0.29 [-0.39, +0.95] | -0.092 [-0.118, -0.063] |

**Per burst (PSNR in dB).**

| Bin | Burst | Drift px | PSNR naive | PSNR aligned | PSNR TAA | TAA - aligned |
|---|---|---:|---:|---:|---:|---:|
| slow | diningroom 0 | 3.0 | 13.86 | 14.59 | 14.56 | -0.03 |
| slow | simplekitchen 1 | 3.19 | 17.40 | 17.99 | 17.12 | -0.87 |
| slow | stone-shower 0 | 3.45 | 16.11 | 16.33 | 16.34 | +0.01 |
| slow | classroom 0 | 3.52 | 13.24 | 13.84 | 13.94 | +0.10 |
| slow | diner 0 | 4.0 | 14.51 | 15.06 | 15.46 | +0.40 |
| slow | staircase 0 | 4.1 | 14.17 | 14.17 | 14.10 | -0.07 |
| slow | library-homeoffice 0 | 4.71 | 14.49 | 14.79 | 14.76 | -0.03 |
| slow | country-kitchen 0 | 5.36 | 13.69 | 14.29 | 14.32 | +0.03 |
| slow | cozykitchen 0 | 5.4 | 13.88 | 14.92 | 14.92 | +0.00 |
| slow | bathroom4 1 | 6.5 | 18.81 | 18.93 | 19.00 | +0.07 |
| medium | bathroom3 0 | 18.7 | 13.29 | 15.80 | 16.30 | +0.50 |
| medium | lazienka 0 | 20.21 | 13.56 | 14.59 | 14.58 | -0.01 |
| medium | kitchen1 1 | 20.33 | 16.26 | 16.47 | 16.16 | -0.31 |
| medium | bathroom5 1 | 20.66 | 12.16 | 13.88 | 12.53 | -1.35 |
| medium | designer-bedroom 1 | 21.03 | 14.45 | 15.00 | 14.91 | -0.09 |
| medium | barbershop 1 | 21.9 | 12.74 | 13.35 | 14.46 | +1.11 |
| medium | interior-scene 1 | 22.36 | 13.91 | 14.64 | 14.67 | +0.03 |
| medium | wooden-staircase 0 | 22.69 | 14.42 | 14.33 | 14.04 | -0.29 |
| medium | cocina-ii 1 | 24.64 | 13.84 | 14.50 | 14.46 | -0.04 |
| medium | domestic-office-table 1 | 25.06 | 13.17 | 13.64 | 13.60 | -0.04 |
| medium | attic 0 | 32.01 | 21.28 | 21.66 | 22.16 | +0.50 |
| fast | bathroom1 1 | 48.85 | 13.28 | 14.08 | 14.08 | +0.00 |
| fast | bathtime 1 | 49.7 | 11.45 | 12.56 | 13.57 | +1.01 |
| fast | bathroom3 1 | 50.04 | 11.59 | 13.65 | 15.73 | +2.08 |
| fast | kitchenpack 0 | 50.16 | 11.53 | 12.62 | 13.10 | +0.48 |
| fast | kitchen3 0 | 52.2 | 12.03 | 12.43 | 12.09 | -0.34 |
| fast | bath 1 | 53.08 | 14.00 | 14.21 | 14.09 | -0.12 |
| fast | mesa-concept 0 | 53.1 | 13.64 | 14.15 | 14.00 | -0.15 |
| fast | ultramodern 0 | 55.52 | 12.66 | 14.65 | 12.61 | -2.04 |
| fast | attic 1 | 56.21 | 22.01 | 22.02 | 22.33 | +0.31 |
| fast | bedroom2 1 | 56.38 | 13.72 | 16.05 | 17.71 | +1.66 |

**What this supports.**
- TAA and alignment are indistinguishable in mean PSNR in every bin: all three intervals include zero.
- TAA has lower SSIM than alignment in every bin, and every interval excludes zero. The deficit is -0.056, -0.070 and -0.092 from slow to fast. The slow and fast intervals overlap, so "it grows with speed" is suggestive, not established. It is of the same size as the 0.03 to 0.04 in the kitchen1 harness.
- Per-burst results are very uneven in the fast bin: TAA is ahead of alignment by 1.0 to 2.1 dB on three bursts (bathroom3 1, bedroom2 1, bathtime 1), behind by 2.0 dB on one (ultramodern 0) and within about 0.5 dB on the other six. The medium bin has bursts at +1.11 (barbershop 1) and -1.35 (bathroom5 1). The fast-bin mean of +0.29 dB is the product of these swings and its interval spans zero.
- Alignment over naive averaging: PSNR gain 0.47, 0.80 and 1.05 dB from slow to fast, so it grows with measured drift; SSIM gain goes the other way (0.107, 0.102, 0.065). The earlier 11-burst pass looked flat and I read it as noise; with 31 bursts the PSNR gain does rise, which fits the measured speed carrying some real information.

**What it does not support.**
- A statement about true speed. The measure is weak for medium and fast bursts, and the alignment gain is far smaller than in the synthetic sweeps, where naive averaging collapses within 2 px. Either most of this motion is not whole-frame translation, or the measured drift is partly noise, or the per-scene content differences dominate.
- A claim that TAA fails (or wins) at high speed. The only consistent finding is the SSIM deficit; the PSNR difference depends on the burst.
- Precise intervals. 10 or 11 bursts per bin is better than 3, but the spread between bursts is large, and two scenes (attic, bathroom3) appear twice.
- Absolute quality. PSNR is 11 to 22 dB here, with `attic` at about 21 to 22 dB for all methods, and fixed gain 2.7 may fit some scenes poorly, which compresses differences.

**Possible follow-up** (not done): a better motion measure (chain adjacent windows, filter before correlating, local tiles) and re-binning, so that the bins track speed more reliably; the 31 records can be re-analysed under new bins without re-running them.
