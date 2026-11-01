# Evaluation Intensity Rule

The reconstructed images are darker than the ground truth, so we use a fixed gain of 2.7 for all methods and scenes before calculating PSNR and SSIM.

```python
mapped = np.clip(2.7 * reconstruction, 0, 1)
```

The ground-truth images are converted to float RGB values in [0,1] by dividing by 255. No additional gain is applied to them. All methods use the same crop, frame selection, reference frame, and evaluation region.

The gain of 2.7 is based on our initial observations and is not a calibrated camera response. We use the same value for naive averaging, aligned averaging, and TAA rather than adjusting it separately for each method or scene. If we change the gain later, all results will need to be recalculated.

## Implementation

The intensity correction is implemented in `src/evaluation.py` through `apply_intensity_rule()` and used by `scripts/run_evaluation.py` before calculating PSNR and SSIM.

The existing `evaluate(reconstruction, ground_truth)` interface remains unchanged. Both inputs should already be in the same [0,1] intensity range.

The evaluation settings, including the gain, are saved in `results/evaluation/protocol.json`.

## Running the Evaluation

From the repository root:

```powershell
python -m unittest discover -s tests -p "test_intensity_rule.py" -v
```

```powershell
python scripts/run_evaluation.py --samples data/samples/kitchen1_000000.npz data/samples/kitchen1_000001.npz --out results/evaluation
```

## Notes

The October 29 mid-term report uses a different evaluation setup, including a least-squares exposure adjustment, last-group alignment, a 300×304 crop, and 1024 frames.

Our current sample evaluation uses a fixed gain of 2.7, first-group alignment, a 128×128 crop, and 256 frames. Because of these differences, the results should be treated separately from Table 1 in the mid-term report.