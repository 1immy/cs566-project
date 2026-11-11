"""Track 3 comparison and ablations for issues #21 and #27.

Full study: python scripts/run_taa_study.py --scene-dir data/drive-slice/kitchen1
Smoke run: python scripts/run_taa_study.py --samples data/samples/kitchen1_000000.npz data/samples/kitchen1_000001.npz

Every method uses identical data, scoring support, reference, and fixed gain.
No parameters are fitted to ground truth. Full scene mode requires all 1024
frames; sample mode is explicitly development validation, not issue completion.
"""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_evaluation import _figure
from scripts.run_sample_taa import _record_path, _sha256
from src.baseline import aligned_average
from src.burst_io import load_burst, load_gt, load_sample, naive_average
from src.evaluation import evaluate_channels
from src.taa import TAAConfig, reconstruct_taa


CONFIGS = {
    "taa_disabled": TAAConfig(clamp=False, adaptive=False),
    "clamp_only": TAAConfig(adaptive=False),
    "blend_only": TAAConfig(clamp=False, disagreement_weighting=False),
    "confidence_only": TAAConfig(clamp=False, motion_weighting=False),
    "taa": TAAConfig(),
}


def reconstruct(cube, group_size, reference):
    """Run unchanged alignment for each method and assert coverage parity."""
    aligned, counts = aligned_average(cube, group_size, reference=reference)
    methods = {"naive": naive_average(cube), "aligned": aligned}
    diagnostics = {}
    updates = (len(cube) - 1) // group_size
    tolerance = max(1e-7, updates * float(np.finfo(np.float32).eps))
    for name, config in CONFIGS.items():
        result = reconstruct_taa(cube, group_size, config, reference=reference)
        if not np.array_equal(result.frame_counts, counts):
            raise RuntimeError(f"{name}: baseline/TAA coverage mismatch")
        if name == "taa_disabled":
            if (not np.allclose(result.reconstruction, aligned, atol=tolerance, rtol=0)
                    or not np.array_equal(result.history_weights, counts)):
                raise RuntimeError("disabled TAA does not match the aligned baseline")
        methods[name] = result.reconstruction
        diagnostics[name] = result
    control = {"max_abs_error": float(np.max(np.abs(methods["taa_disabled"] - aligned))),
               "atol": tolerance, "rtol": 0, "coverage_equal": True,
               "disabled_history_weights_equal_counts": True}
    return methods, diagnostics, counts, control


def _table(rows):
    lines = ["| Scene / burst | Method | PSNR (dB) | SSIM |",
             "|---|---|---:|---:|"]
    lines += [f"| {r['scene']} / {r['burst']:06d} | {r['method']} | "
              f"{r['psnr_mean']:.4f} | {r['ssim_mean']:.6f} |" for r in rows]
    return "\n".join(lines)


def _write_report(path, rows, full, reference, group_size, gain):
    status = ("Full-burst development comparison; these scenes are not held out."
              if full else "Development samples only; this run does not evaluate full kitchen1 bursts 0, 1, and 10.")
    text = ("# TAA comparison and component ablations\n\n" + status + "\n\n"
            f"Shared gain: {gain:g}; reference: {reference}; group size: {group_size}. "
            "Every pixel of the same input crop is scored. See manifest.json for "
            "input/source hashes, crop and frame indices, and all component settings.\n\n"
            + _table(rows) + "\n\n"
            "Clamp-only clips history with ordinary frame weighting. Blend-only "
            "discounts history using displacement. Confidence-only discounts history "
            "using neighborhood disagreement. Full TAA combines all three. The "
            "confidence map is a history-retention heuristic, not alignment-match "
            "confidence or a calibrated occlusion probability.\n\n")
    # Report both metrics independently; no post-hoc aggregate winner criterion.
    text += "## Measured component differences from aligned averaging\n\n"
    text += "| Scene / burst | Component | Delta PSNR (dB) | Delta SSIM |\n|---|---|---:|---:|\n"
    for base in (r for r in rows if r["method"] == "aligned"):
        for row in rows:
            if ((row["scene"], row["burst"]) == (base["scene"], base["burst"])
                    and row["method"] in ("clamp_only", "blend_only", "confidence_only", "taa")):
                text += (f"| {row['scene']} / {row['burst']:06d} | {row['method']} | "
                         f"{row['psnr_mean'] - base['psnr_mean']:+.4f} | "
                         f"{row['ssim_mean'] - base['ssim_mean']:+.6f} |\n")
    path.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene-dir", type=Path)
    source.add_argument("--samples", type=Path, nargs="+")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 10])
    parser.add_argument("--crop", type=int, nargs=4, default=(250, 550, 248, 552),
                        metavar=("Y0", "Y1", "X0", "X1"))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--reference", choices=("first", "last"), default="last")
    parser.add_argument("--gain", type=float, default=2.7)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/sample_taa/study"))
    args = parser.parse_args(argv)
    if args.group_size <= 0 or not np.isfinite(args.gain) or args.gain <= 0:
        parser.error("group-size and gain must be positive and gain finite")
    if len(set(args.indices)) != len(args.indices) or any(i < 0 for i in args.indices):
        parser.error("indices must be distinct and nonnegative")
    y0, y1, x0, x1 = args.crop
    if not (0 <= y0 < y1 <= 800 and 0 <= x0 < x1 <= 800 and x0 % 8 == x1 % 8 == 0):
        parser.error("crop must be within 800x800, with x bounds divisible by 8")
    if args.samples and (args.crop != (250, 550, 248, 552) or args.indices != [0, 1, 10]):
        parser.error("--crop and --indices apply only to --scene-dir; samples carry their own crop")
    inputs = ([(p, p.with_name(p.stem + "_gt.png")) for p in args.samples]
              if args.samples else [(args.scene_dir / f"{i:06d}.npy", args.scene_dir / f"{i:06d}.png")
                                    for i in args.indices])
    stems = [p.stem if args.samples else f"{args.scene_dir.name}_{p.stem}" for p, _ in inputs]
    if len(set(s.casefold() for s in stems)) != len(stems):
        parser.error("input names must be distinct")
    for path, gt_path in inputs:
        for item in (path, gt_path):
            if not item.is_file():
                parser.error(f"Input file not found: {item}")
    input_paths = {p.resolve() for pair in inputs for p in pair}
    outputs = [args.out / name for name in ("manifest.json", "results.csv", "findings.md")]
    outputs += [args.out / f"{s}_{suffix}" for s in stems
                for suffix in ("comparison.png", "reconstructions.npz")]
    if any(p.resolve() in input_paths for p in outputs):
        parser.error("outputs must not overwrite inputs")

    records, rows = [], []
    for (path, gt_path), stem in zip(inputs, stems):
        if args.samples:
            cube = load_sample(path)
            with Image.open(gt_path) as image:
                gt = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            with np.load(path) as sample:
                crop = sample["crop"].tolist() if "crop" in sample else None
            # Sample naming is part of provenance, never silently invent an index.
            scene, separator, index_text = path.stem.rpartition("_")
            if not separator or not scene or len(index_text) != 6 or not index_text.isdecimal():
                parser.error("sample names must be SCENE_NNNNNN.npz")
            index = int(index_text)
        else:
            packed = np.load(path, mmap_mode="r")
            if packed.dtype != np.uint8 or packed.shape != (1024, 800, 100, 3):
                raise ValueError(f"{path.name}: expected full packed shape (1024,800,100,3), uint8")
            del packed
            scene, index, crop = args.scene_dir.name, int(path.stem), list(args.crop)
            cube = load_burst(args.scene_dir, index=index, crop=args.crop)
            gt = load_gt(args.scene_dir, index=index, crop=args.crop)
        if cube.shape[1:] != gt.shape:
            raise ValueError(f"{stem}: ground truth shape mismatch")
        print(f"{stem}: reconstructing {len(cube)} frames with {args.reference} reference", flush=True)
        methods, diagnostics, counts, control = reconstruct(cube, args.group_size, args.reference)
        panels = [("Ground truth", gt)]
        for name, reconstruction in methods.items():
            mapped = np.clip(args.gain * reconstruction, 0, 1)
            scores = evaluate_channels(mapped, gt)
            row = {"scene": scene, "burst": index, "method": name, "frames": len(cube),
                   "height": gt.shape[0], "width": gt.shape[1], "group_size": args.group_size,
                   "reference": args.reference, "gain": args.gain}
            for metric, values in scores.items():
                row.update({f"{metric}_{channel}": value for channel, value in values.items()})
            rows.append(row)
            print(f"  {name}: PSNR={row['psnr_mean']:.4f}, SSIM={row['ssim_mean']:.6f}", flush=True)
            if name in ("naive", "aligned", "taa"):
                panels.append((name, mapped))
        args.out.mkdir(parents=True, exist_ok=True)
        figure = args.out / f"{stem}_comparison.png"
        _figure(panels, figure)
        artifacts = {figure.name: _sha256(figure)}
        if args.save_arrays:
            archive = args.out / f"{stem}_reconstructions.npz"
            result = diagnostics["taa"]
            np.savez_compressed(archive, **methods, ground_truth=gt, frame_counts=counts,
                                confidence_map=result.confidence_map,
                                valid_update_mask=result.valid_update_mask,
                                history_weights=result.history_weights)
            artifacts[archive.name] = _sha256(archive)
        starts = list(range(0, len(cube), args.group_size))
        reference_start = starts[0] if args.reference == "first" else starts[-1]
        update_starts = [s for s in starts if s != reference_start]
        last_start = update_starts[-1] if update_starts else None
        records.append({"scene": scene, "burst": index,
                        "inputs": [_record_path(path), _record_path(gt_path)],
                        "input_sha256": {path.name: _sha256(path), gt_path.name: _sha256(gt_path)},
                        "shape": list(cube.shape), "crop_y0_y1_x0_x1": crop,
                        "frame_indices_inclusive": [0, len(cube) - 1],
                        "frame_index_scope": "sample-local" if args.samples else "original burst",
                        "reference_frames_inclusive": [reference_start, min(reference_start + args.group_size, len(cube)) - 1],
                        "last_update_frames_inclusive": (None if last_start is None else
                                                         [last_start, min(last_start + args.group_size, len(cube)) - 1]),
                        "baseline_control": control, "artifacts_sha256": artifacts})
    manifest = {
        "purpose": "issues #21 and #27: shared-protocol comparison and single-component ablations",
        "data_status": ("full-burst development study; not held out" if args.scene_dir
                        else "development samples only; full kitchen1 0,1,10 excluded from this run"),
        "intensity_rule": "clip(gain * reconstruction, 0, 1); ground truth unchanged",
        "intensity_agreement": "https://github.com/1immy/cs566-project/issues/18#issuecomment-6552346154",
        "gain": args.gain, "gain_scope": "identical fixed gain for every scene and method",
        "metrics": "existing evaluate_channels; arithmetic mean per-channel PSNR/SSIM, 7x7 uniform reflect SSIM",
        "scoring_support": "all pixels of identical crop; no fitted gain or scoring alignment",
        "reference": args.reference, "group_size": args.group_size,
        "update_order": "reference initializes history once; other groups update in original chronological order",
        "confidence_map": "last processed update retention fraction; use valid_update_mask to distinguish missing evidence",
        "configs": {name: asdict(config) for name, config in CONFIGS.items()},
        "randomness": "none", "samples": records,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__},
        "source_sha256": {name: _sha256(ROOT / name) for name in (
            "scripts/run_taa_study.py", "scripts/run_sample_taa.py", "scripts/run_evaluation.py",
            "src/taa.py", "src/alignment.py", "src/baseline.py", "src/burst_io.py", "src/evaluation.py")},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_report(args.out / "findings.md", rows, bool(args.scene_dir), args.reference, args.group_size, args.gain)
    print(f"Wrote comparison, ablations, and provenance to {args.out}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
