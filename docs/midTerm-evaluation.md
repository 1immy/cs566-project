# CS566 Track 4: Mid-term Evaluation

This implementation adds PSNR and SSIM evaluation and a comparison script for naive averaging, aligned averaging, and TAA.

## Files

- `src/evaluation.py` — Computes PSNR and SSIM.
- `scripts/run_evaluation.py` — Runs reconstruction comparisons and saves results.
- `tests/test_evaluation.py` — Tests the evaluation functions.

## Usage

Run the tests from the project root:

```powershell
py -m unittest discover -s tests -v
```

Run the evaluation on the two sample bursts:

```powershell
py scripts/run_evaluation.py --samples data/samples/kitchen1_000000.npz data/samples/kitchen1_000001.npz --gain 2.7
```

To evaluate previously generated TAA reconstructions:

```powershell
py scripts/run_evaluation.py --archives results/sample_taa/kitchen1_000000_reconstructions.npz results/sample_taa/kitchen1_000001_reconstructions.npz --gain 2.7
```

Other reconstruction methods can be specified using `--pipeline package.module:function`. The function takes `(cube, group_size=64)` and returns an RGB image, a tuple containing the image, or an object with a `.reconstruction` attribute.

## Output

Results are saved in `results/evaluation_midterm/`:

- `results.csv` — Per-channel and mean PSNR/SSIM for each method.
- `protocol.json` — Evaluation settings.
- `*_comparison.png` — Comparison figures for each burst.

## Evaluation

All reconstruction methods use the same intensity adjustment:

```python
np.clip(2.7 * reconstruction, 0, 1)
```

The gain of 2.7 is used for the sample evaluation and is not calibrated to the camera response.

`evaluate(reconstruction, ground_truth)` returns mean PSNR and SSIM. `evaluate_channels()` also reports individual RGB channel values.

SSIM uses a 7×7 uniform window with reflect padding.

## Limitations

The sample evaluation uses 256 frames and a 128×128 crop for kitchen1 bursts 0 and 1. The existing mid-term report uses all 1024 frames, a 300×304 crop, and last-group alignment for bursts 0, 1, and 10.

The current baseline and TAA implementations align to the first group. Because the evaluation settings differ, the sample results should not be directly compared with the mid-term report's Table 1.

Reproducing the report's results requires the full kitchen1 bursts and a consistent last-group reference for both aligned averaging and TAA.