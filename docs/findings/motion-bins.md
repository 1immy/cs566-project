# Motion bins, resumable full-scale runs and failure analysis (issues #23, #24, #25)

Status: **code and tests done; not yet run on real data.** The real-data bins, scene set and plots are produced by the steps below on the machine that holds the dataset. Until then this page contains no real-data numbers. A synthetic run of the same pipeline is in [synthetic-sweep.md](synthetic-sweep.md).

All paths below are placeholders. Replace `DATA_ROOT` with the folder that holds one sub-folder per scene (for example the `train` folder, or the shared Drive slice). Only small text and PNG outputs are written; raw data is never copied.

## 1. Measure and bin (issue #23)
```
python scripts/bin_scenes_by_motion.py --data-root DATA_ROOT --bursts 0 1 --per-bin 3
```
- Measures, per scene and burst, the global drift between the first and last 256-frame window (`src/motion.py`, same method as `scripts/measure_burst_motion.py`).
- `drift_px` is the length of that shift. It spans the 768 frames between window centres, so it is about 3/4 of the drift over the whole burst.
- Bins are tertiles of the measured drift (slow, medium, fast), or fixed with `--edges SLOW_MEDIUM MEDIUM_FAST`.
- Writes `results/motion_bins/motion.csv` (one row per burst) and `results/motion_bins/bins.json` (edges, counts, every burst's bin, and the selected set: `--per-bin` bursts per bin, closest to the bin median, distinct scenes first). Bursts with a correlation peak below 0.05 are listed as unreliable.
- Resumable: bursts already in `motion.csv` are not measured again.

After the run, record here the edges, the counts per bin and the selected scenes: **not yet run**.

## 2. Full-resolution runs (issue #24)
```
python scripts/run_full_scale.py --data-root DATA_ROOT --plan results/motion_bins/bins.json
```
- Runs naive, aligned and TAA on each selected burst: 1024 frames, groups of 64, last-group reference, fixed gain 2.7 (the protocol in `evaluation-intensity.md`).
- Writes one JSON record (scores, motion, settings) and one comparison PNG per burst under `results/full_scale/<scene>/`, plus `summary.csv`.
- Resumable: a burst is skipped when its record exists with the same settings. A changed setting, a corrupt record or a leftover `.tmp` file re-runs it. A burst that fails writes `<burst>.error.txt`, the run continues, and the exit status is 1. Run the same command again to retry only the unfinished bursts.
- `--dry-run` lists what would run. `--all-planned` runs every binned burst instead of the selected ones. `--limit N` caps one session. Memory: one full burst unpacks to about 2 GB; `--crop Y0 Y1 X0 X1` (x bounds multiples of 8) reduces it.

## 3. Failure analysis (issue #25)
```
python scripts/analyze_motion_bins.py --results results/full_scale --bins results/motion_bins/bins.json
```
- Per bin: n, mean and standard deviation of PSNR and SSIM for each method, and the paired difference TAA minus aligned with a seeded bootstrap 95% interval over bursts (only when a bin has at least 3 bursts).
- Plots: `psnr_vs_drift.png`, `ssim_vs_drift.png`, `delta_by_bin.png`. Real and synthetic records are analysed separately and never pooled.
- Report the plain result, including bins where TAA does worse than alignment.

## Tests
`python -m unittest discover -s tests`. The new tests build small scenes in the real on-disk format and check the binning, the skip and retry behaviour of the runner, and the analysis.
