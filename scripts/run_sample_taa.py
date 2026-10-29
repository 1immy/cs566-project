"""Generate comparable Track 3 sample reconstructions and confidence maps.

Run from the repo root: python scripts/run_sample_taa.py --group-size 64
Uses the existing baseline unchanged. No ground-truth tuning, normalization,
PSNR, or SSIM is performed here; float outputs are supplied to Track 4.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import aligned_average
from src.burst_io import load_sample, naive_average
from src.taa import TAAConfig, reconstruct_taa


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_path(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _display(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def _comparison(panels):
    """A labeled image grid with identical direct intensity mapping."""
    h, w = panels[0][1].shape[:2]
    panel_width = max(160, 2 * w)
    canvas = Image.new("RGB", (len(panels) * panel_width, 2 * h + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        x = index * panel_width
        draw.text((x + 6, 8), label, fill="black")
        picture = _display(array).convert("RGB").resize((2 * w, 2 * h), Image.Resampling.NEAREST)
        canvas.paste(picture, (x, 28))
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--samples", type=Path, nargs="+", default=[
        Path("data/samples/kitchen1_000000.npz"),
        Path("data/samples/kitchen1_000001.npz")])
    parser.add_argument("--out", type=Path, default=Path("results/sample_taa"))
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--clamp-margin", type=float, default=0.05)
    parser.add_argument("--rejection-scale", type=float, default=0.1)
    parser.add_argument("--motion-scale", type=float, default=8.0)
    parser.add_argument("--no-clamp", action="store_true",
                        help="disable history clipping; adaptive retention still uses neighborhood disagreement")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="disable both neighborhood-disagreement and displacement retention penalties")
    args = parser.parse_args(argv)
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    try:
        config = TAAConfig(radius=args.radius, clamp_margin=args.clamp_margin,
                           rejection_scale=args.rejection_scale, motion_scale=args.motion_scale,
                           clamp=not args.no_clamp, adaptive=not args.no_adaptive)
    except ValueError as error:
        parser.error(str(error))

    stems = [path.stem.casefold() for path in args.samples]
    if len(stems) != len(set(stems)):
        parser.error("--samples must have unique filename stems")
    input_paths = set()
    for path in args.samples:
        for input_path in (path, path.with_name(path.stem + "_gt.png")):
            if not input_path.is_file():
                parser.error(f"Input file not found: {input_path}")
            input_paths.add(input_path.resolve())
    suffixes = ("naive.png", "aligned.png", "taa.png", "gt.png", "confidence.png",
                "valid_update.png", "comparison.png", "reconstructions.npz")
    output_paths = [args.out / "manifest.json"]
    for path in args.samples:
        output_paths.extend(args.out / f"{path.stem}_{suffix}" for suffix in suffixes)
    if any(path.resolve() in input_paths for path in output_paths):
        parser.error("--out must not overwrite any sample or ground truth input")

    records = []
    for path in args.samples:
        cube = load_sample(path)
        gt_path = path.with_name(path.stem + "_gt.png")
        with Image.open(gt_path) as image:
            gt = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        if gt.shape != cube.shape[1:]:
            raise ValueError(f"Ground truth shape mismatch for {path.name}")
        naive = naive_average(cube)
        aligned, counts = aligned_average(cube, args.group_size)
        result = reconstruct_taa(cube, args.group_size, config)
        control = reconstruct_taa(cube, args.group_size, TAAConfig(clamp=False, adaptive=False))
        difference = float(np.max(np.abs(control.reconstruction - aligned)))
        group_counts = [min(args.group_size, len(cube) - start)
                        for start in range(0, len(cube), args.group_size)]
        # Repeated float32 mean updates round differently from sum-then-divide.
        # Allow one float32 epsilon per update on [0,1]; coverage and weights
        # must still agree exactly. Record the allowance alongside the error.
        control_atol = max(1e-7, (len(group_counts) - 1) * float(np.finfo(np.float32).eps))
        if (not np.allclose(control.reconstruction, aligned, atol=control_atol, rtol=0)
                or not np.array_equal(control.frame_counts, counts)
                or not np.array_equal(control.history_weights, counts)
                or not np.array_equal(result.frame_counts, counts)):
            raise RuntimeError("TAA and baseline alignment/coverage control failed")

        args.out.mkdir(parents=True, exist_ok=True)
        for name, array in (("naive", naive), ("aligned", aligned),
                            ("taa", result.reconstruction), ("gt", gt),
                            ("confidence", result.confidence_map),
                            ("valid_update", result.valid_update_mask)):
            _display(array).save(args.out / f"{path.stem}_{name}.png")
        _comparison([("Naive", naive), ("Aligned", aligned), ("TAA", result.reconstruction),
                     ("Ground truth", gt), ("Last retention", result.confidence_map)]).save(
                         args.out / f"{path.stem}_comparison.png")
        np.savez_compressed(
            args.out / f"{path.stem}_reconstructions.npz",
            naive=naive, aligned=aligned, taa=result.reconstruction,
            taa_disabled=control.reconstruction, ground_truth=gt,
            frame_counts=counts, history_weights=result.history_weights,
            confidence_map=result.confidence_map, valid_update_mask=result.valid_update_mask)
        with np.load(path) as sample:
            crop = sample["crop"].tolist() if "crop" in sample else None
        identity = re.fullmatch(r"(.+)_(\d{6})", path.stem)
        valid_confidence = result.confidence_map[result.valid_update_mask]
        records.append({
            "sample": _record_path(path), "ground_truth": _record_path(gt_path),
            "scene_from_filename": identity[1] if identity else None,
            "burst_index_from_filename": int(identity[2]) if identity else None,
            "sample_sha256": _sha256(path), "ground_truth_sha256": _sha256(gt_path),
            "shape": list(cube.shape), "crop_y0_y1_x0_x1": crop,
            "sample_local_frame_indices": [0, len(cube) - 1],
            "group_frame_counts": group_counts,
            "baseline_control_max_abs_error": difference,
            "baseline_control_atol": control_atol,
            "valid_frame_counts_min_max": [float(counts.min()), float(counts.max())],
            "history_weights_min_max": [float(result.history_weights.min()),
                                        float(result.history_weights.max())],
            "last_update_valid_fraction": float(result.valid_update_mask.mean()),
            "last_update_confidence_mean_on_valid": (
                float(valid_confidence.mean()) if valid_confidence.size else None),
            "artifacts_sha256": {suffix: _sha256(args.out / f"{path.stem}_{suffix}")
                                 for suffix in suffixes},
        })
        print(f"{path.stem}: saved TAA and confidence; "
              f"disabled-TAA/baseline max difference {difference:.3g}")

    manifest = {
        "method": "neighborhood history clamp + adaptive history-weight retention",
        "reference": "first group mean, fixed; shared unmodified Track 2 aligner",
        "group_size": args.group_size, "taa": asdict(config),
        "confidence": "last update's retained history fraction; invalid updates are zero",
        "history_weights": "discounted accumulation weights, not effective sample sizes",
        "frame_counts": "geometrically valid frames before history rejection",
        "image_domain": "naive/aligned/taa/taa_disabled are photon-detection probabilities in [0,1]",
        "ground_truth_domain": "provided RGB PNG / 255; response and reference time unresolved with Track 4",
        "display": "direct [0,1] mapping for all RGB outputs; no exposure normalization",
        "evaluation": "PSNR/SSIM not yet run; development samples only",
        "randomness": "none", "samples": records,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pillow": PIL.__version__},
        "source_sha256": {name: _sha256(ROOT / name) for name in (
            "src/taa.py", "src/alignment.py", "src/baseline.py", "src/burst_io.py",
            "scripts/run_sample_taa.py", "requirements.txt")},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)
                                          + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
