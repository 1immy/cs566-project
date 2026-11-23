# Motion bins, resumable full-scale runs and failure analysis (issues #23, #24, #25)

Status: **run on the real training scenes (11 bursts).** Results and caveats are in the last section. A synthetic run of the same pipeline is in [synthetic-sweep.md](synthetic-sweep.md).

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
Run on `train`: 50 scenes, bursts 0 and 1 each (100 bursts), full resolution, 1024 frames, group size 64, last-group reference, fixed gain 2.7.

**Bins.** Tertile edges on measured drift: 12.77 px and 37.82 px. Counts: 33 slow, 33 medium, 34 fast. Measured drift ranges from 0.9 to 106 px.

**Reliability of the motion measure.** 80 of the 100 bursts have a smallest correlation peak below 0.05, which the code flags as unreliable. Every burst with a peak at or above 0.05 has a measured drift below about 19 px, so the flag and the speed are tied together: the bursts the measure is least sure about are the fast ones. Two readings fit this and the data cannot separate them: large motion smears the 256-frame window means and lowers the peak (the drift is real, measured coarsely), or the large values are noise. The bin edges should be read as approximate for that reason.

**Selected bursts (3 per bin, closest to the bin median, distinct scenes).**

| Bin | Burst | Measured drift (px) | Smallest peak |
|---|---|---:|---:|
| slow | library-homeoffice 0 | 4.71 | 0.06 |
| slow | country-kitchen 0 | 5.36 | 0.05 |
| slow | staircase 0 | 4.10 | 0.01 |
| medium | interior-scene 1 | 22.36 | 0.02 |
| medium | wooden-staircase 0 | 22.69 | 0.01 |
| medium | barbershop 1 | 21.90 | 0.01 |
| fast | mesa-concept 0 | 53.10 | 0.01 |
| fast | bath 1 | 53.08 | 0.01 |
| fast | kitchen3 0 | 52.20 | 0.01 |

Two further bursts, attic 0 (32.0 px, medium) and attic 1 (56.2 px, fast), were run by mistake with `--all-planned` before it was stopped. They are real, finished runs, so they are included, which makes n = 4 in the medium and fast bins. They are not part of the selected set.

**Per-bin means** (11 bursts, fixed gain 2.7, same cube and aligner for every method). Different scenes sit in different bins, so compare methods within a row, not a method across rows.

| Bin | n | Drift px | PSNR naive | PSNR aligned | PSNR TAA | SSIM naive | SSIM aligned | SSIM TAA |
|---|---|---|---|---|---|---|---|---|
| slow | 3 | 4.1 to 5.4 | 14.11 | 14.42 | 14.40 | 0.363 | 0.460 | 0.390 |
| medium | 4 | 21.9 to 32.0 | 15.59 | 15.99 | 16.33 | 0.277 | 0.336 | 0.289 |
| fast | 4 | 52.2 to 56.2 | 15.42 | 15.70 | 15.62 | 0.314 | 0.365 | 0.271 |

**TAA minus aligned**, mean over the bursts in the bin, with the seeded bootstrap 95% interval:

| Bin | n | PSNR dB | SSIM |
|---|---|---|---|
| slow | 3 | -0.02 [-0.07, +0.03] | -0.070 [-0.097, -0.052] |
| medium | 4 | +0.34 [-0.13, +0.84] | -0.047 [-0.070, -0.016] |
| fast | 4 | -0.08 [-0.29, +0.19] | -0.095 [-0.134, -0.039] |

**What this supports.**
- TAA and alignment are indistinguishable in PSNR in every bin: all three intervals include zero.
- TAA has lower SSIM than alignment in every bin, by 0.05 to 0.10, about the same size as the 0.03 to 0.04 seen in the kitchen1 harness. It does not get larger with measured speed.
- Alignment beats naive averaging by 0.3 to 0.4 dB PSNR and 0.05 to 0.10 SSIM in every bin, and that gain does not grow with measured drift.
- So on these scenes TAA shows no failure at higher measured speed, and no advantage either; its SSIM deficit appears at all speeds.

**What it does not support.**
- Any claim about how methods behave as a function of true speed. The measure is unreliable in the medium and fast bins, and the flat alignment gain is what a noisy measure would also produce. The synthetic sweeps, where the speed is known, show naive averaging collapsing within 2 px; the real bins show nothing like it, which suggests the measured drift on the real scenes is not simple translation of the whole frame (for example local motion, flicker or noise), or that most of these bursts have little global motion.
- Tight intervals. With 3 to 4 bursts per bin a bootstrap can only reshuffle 3 to 4 values, so the intervals are close to the range of the observed differences and are too narrow to take as a population interval.
- Anything about scenes outside these 11 bursts, or about burst indices above 1.
- Absolute quality: PSNR is 12 to 16 dB here, lower than the 15 to 16 dB kitchen1 runs, and the fixed gain 2.7 may fit some scenes poorly, which compresses all differences.

Nine of the 11 bursts are on the low-peak list in `summary.md` (attic 0 and 1, barbershop 1, bath 1, interior-scene 1, kitchen3 0, mesa-concept 0, staircase 0, wooden-staircase 0).

**Possible follow-up** (not done): validate the measure on the real scenes, for example by checking whether bursts with a high peak and large drift exist, or by measuring drift with local windows instead of one global shift; and run more bursts per bin (`--per-bin 10`, about 5 min each) to get usable intervals.

