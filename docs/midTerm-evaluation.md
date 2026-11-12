# CS566 Track 4: Mid-term Evaluation

This script compares naive averaging, aligned averaging, and TAA using PSNR and SSIM. It reports scores for each RGB channel and the average.

## Running the tests

From the repository root:

```powershell
py -m unittest discover -s tests -v
```

## Full kitchen1 evaluation

Download the shared kitchen1 data and make sure the folder contains bursts 0, 1, and 10, along with their ground-truth PNGs.

The October 29 report used a 300×304 center crop, 1024 frames, groups of 64, and the last group as the reference. For the last-group runs, the burst was reversed before running the original alignment code.

To run the same five methods with these settings:

```powershell
py scripts/run_evaluation.py --scene-dir data/train/kitchen1 --bursts 0 1 10 --crop 250 550 248 552 --frames 1024 --group-size 64 --reference last --report-table --gain 2.7 --out results/evaluation
```

Change `--scene-dir` if your kitchen1 folder is somewhere else. The crop arguments are `y0 y1 x0 x1`. The x values must be divisible by 8 because the photon data is bit-packed.

The original report used a fitted gain for each reconstruction. We're now using a fixed gain of 2.7 for all methods, following issue #18. This means the new results use the same crop, frames, and reference settings as Table 1, but the PSNR values may differ. SSIM is also included.

## Sample evaluation

Run the two samples included in the repository:

```powershell
py scripts/run_evaluation.py --samples data/samples/kitchen1_000000.npz data/samples/kitchen1_000001.npz --gain 2.7 --out results/evaluation
```

To evaluate previously saved reconstructions:

```powershell
py scripts/run_evaluation.py --archives results/sample_taa/kitchen1_000000_reconstructions.npz results/sample_taa/kitchen1_000001_reconstructions.npz --out results/evaluation
```

Additional methods can be passed using `--pipeline package.module:function`.

## Output files

All results are saved in `results/evaluation/` by default.

- `results.csv` contains PSNR and SSIM scores for each method and burst.
- `protocol.json` records the settings used for the evaluation.
- `*_comparison.png` contains the ground truth and reconstructed images.

The full kitchen1 comparison produces 15 rows, covering five methods across three bursts.

The old `results/evaluation_check/` folder can be removed if it only contains duplicate results. Use `results/evaluation/` for new runs.

Each run overwrites the CSV and protocol files in the selected output folder.

## Intensity correction

The reconstructed images are darker than the ground truth, so we apply the same gain to every method:

```python
mapped = np.clip(2.7 * reconstruction, 0, 1)
```

Ground truth is converted to [0,1] without an additional gain. More details are in `docs/findings/evaluation-intensity.md`.

PSNR and SSIM are calculated in `src/evaluation.py`. SSIM uses a 7×7 window with reflect padding, and the existing `evaluate(reconstruction, ground_truth)` interface is unchanged.