"""Verify issue #19 on full bursts from a locally downloaded Drive scene.

Run from the repo root with --scene-dir pointing at the downloaded scene.
All methods use the same full burst, crop, and fixed gain; no gain fitting,
scoring-mask selection, or ground-truth-driven alignment is performed.
This is development validation, not a held-out benchmark or a TAA comparison.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.measure_burst_motion import phase_corr
from scripts.run_evaluation import _figure
from scripts.run_sample_baselines import _sha256
from src.baseline import aligned_average
from src.burst_io import load_burst, load_gt, naive_average
from src.evaluation import evaluate_channels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--crop", type=int, nargs=4, default=(272, 528, 272, 528),
                        metavar=("Y0", "Y1", "X0", "X1"))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--gain", type=float, default=2.7,
                        help="one fixed shared gain, never fitted to ground truth")
    parser.add_argument("--out", type=Path, default=Path("results/issue19/scene-validation"))
    args = parser.parse_args(argv)
    y0, y1, x0, x1 = args.crop
    if not (0 <= y0 < y1 <= 800 and 0 <= x0 < x1 <= 800 and x0 % 8 == x1 % 8 == 0):
        parser.error("crop must lie within 800x800 with x bounds divisible by 8")
    if args.group_size <= 0 or not np.isfinite(args.gain) or args.gain <= 0:
        parser.error("group-size and gain must be positive and gain finite")
    if len(set(args.indices)) != len(args.indices) or any(index < 0 for index in args.indices):
        parser.error("burst indices must be distinct and nonnegative")
    inputs = set()
    for index in args.indices:
        for extension in ("npy", "png"):
            path = args.scene_dir / f"{index:06d}.{extension}"
            if not path.is_file():
                parser.error(f"Input file not found: {path}")
            inputs.add(path.resolve())
    outputs = [args.out / name for name in ("manifest.json", "results.csv")]
    for index in args.indices:
        stem = f"{args.scene_dir.name}_{index:06d}"
        outputs.extend(args.out / f"{stem}_{suffix}" for suffix in ("reconstructions.npz", "comparison.png"))
    if any(path.resolve() in inputs for path in outputs):
        parser.error("output files must not overwrite inputs")

    records, rows = [], []
    for index in args.indices:
        path = args.scene_dir / f"{index:06d}.npy"
        gt_path = path.with_suffix(".png")
        packed = np.load(path, mmap_mode="r")
        if packed.dtype != np.uint8 or packed.shape != (1024, 800, 100, 3):
            raise ValueError(f"{path.name}: expected a full packed 1024-frame RGB burst")
        del packed
        cube = load_burst(args.scene_dir, index=index, crop=tuple(args.crop))
        gt = load_gt(args.scene_dir, index=index, crop=tuple(args.crop))
        if gt.shape != cube.shape[1:]:
            raise ValueError("ground truth must match the selected crop")
        last_start = ((len(cube) - 1) // args.group_size) * args.group_size
        reference_ranges = {"first": [0, min(args.group_size, len(cube)) - 1],
                            "last": [last_start, len(cube) - 1]}
        reference_pose = {}
        for name, (start, end) in reference_ranges.items():
            reference = naive_average(cube[start:end + 1])
            dy, dx, peak = phase_corr(gt.mean(axis=2), reference.mean(axis=2))
            reference_pose[name] = {"shift_to_gt_dy_dx": [dy, dx],
                                    "shift_magnitude": float(np.hypot(dy, dx)), "peak": peak}
        print(f"{args.scene_dir.name}/{index:06d}: full burst loaded; reconstructing first reference", flush=True)
        first, first_counts = aligned_average(cube, args.group_size, reference="first")
        print(f"{args.scene_dir.name}/{index:06d}: reconstructing last reference", flush=True)
        last, last_counts = aligned_average(cube, args.group_size)
        methods = {"naive": naive_average(cube), "aligned_first": first, "aligned_last": last}
        reconstruction_pose, scores = {}, {}
        panels = [("Ground truth", gt)]
        for name, reconstruction in methods.items():
            mapped = np.clip(args.gain * reconstruction, 0, 1)
            score = evaluate_channels(mapped, gt)
            scores[name] = score
            row = {"scene": args.scene_dir.name, "burst": index, "method": name,
                   "frames": len(cube), "group_size": args.group_size, "gain": args.gain,
                   "psnr_mean": score["psnr"]["mean"], "ssim_mean": score["ssim"]["mean"]}
            rows.append(row)
            dy, dx, peak = phase_corr(gt.mean(axis=2), reconstruction.mean(axis=2))
            reconstruction_pose[name] = {"shift_to_gt_dy_dx": [dy, dx], "peak": peak}
            panels.append((name, mapped))
            print(f"  {name}: PSNR={row['psnr_mean']:.4f}, SSIM={row['ssim_mean']:.6f}", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        stem = f"{args.scene_dir.name}_{index:06d}"
        archive = args.out / f"{stem}_reconstructions.npz"
        figure = args.out / f"{stem}_comparison.png"
        np.savez_compressed(archive, **methods, ground_truth=gt,
                            first_frame_counts=first_counts, last_frame_counts=last_counts)
        _figure(panels, figure)
        records.append({
            "scene": args.scene_dir.name, "burst_index": index,
            "source_files": [f"{args.scene_dir.name}/{path.name}", f"{args.scene_dir.name}/{gt_path.name}"],
            "input_sha256": {path.name: _sha256(path), gt_path.name: _sha256(gt_path)},
            "shape": list(cube.shape), "frame_indices": [0, len(cube) - 1],
            "reference_frame_ranges_inclusive": reference_ranges,
            "reference_group_pose": reference_pose, "reconstruction_pose": reconstruction_pose,
            "scores": scores,
            "frame_counts_min_max": {"first": [float(first_counts.min()), float(first_counts.max())],
                                     "last": [float(last_counts.min()), float(last_counts.max())]},
            "archive": archive.name, "archive_sha256": _sha256(archive),
            "figure": figure.name, "figure_sha256": _sha256(figure),
        })
        del cube
    manifest = {
        "purpose": "issue #19 additional-scene development validation, not a held-out benchmark",
        "crop_y0_y1_x0_x1": list(args.crop), "group_size": args.group_size,
        "intensity_rule": "clip(gain * reconstruction,0,1); ground truth unchanged",
        "gain": args.gain, "gain_scope": "same fixed value for every method and burst",
        "scoring_support": "every pixel of the identical crop; no alignment or fitted gain during scoring",
        "metrics": "existing evaluate_channels: mean per-channel PSNR/SSIM, 7x7 uniform reflect SSIM",
        "pose_diagnostic": "existing phase_corr on channel means; shift to apply moving image to GT; rounded 0.1px",
        "pose_caveat": "global translation diagnostic only; low peaks and parallax limit interpretation",
        "randomness": "none", "samples": records,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__},
        "source_sha256": {name: _sha256(ROOT / name) for name in (
            "src/alignment.py", "src/baseline.py", "src/burst_io.py", "src/evaluation.py",
            "scripts/verify_reference_scene.py", "scripts/measure_burst_motion.py",
            "scripts/run_sample_baselines.py", "scripts/run_evaluation.py")},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved full-burst reference verification to {args.out}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
