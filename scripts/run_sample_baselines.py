"""Reproduce Track 2 sample outputs from the repo root.

    py scripts/run_sample_baselines.py --group-size 64

PNG images use the same direct [0,1] to [0,255] mapping without exposure
normalization. Float reconstructions are preserved for Track 4 evaluation.
These development samples are not a held-out benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline import aligned_average
from src.burst_io import load_sample, naive_average


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_path(path):
    """Prefer repository-relative provenance without recording personal paths."""
    try:
        return path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        # External samples are identified by filename plus their content hash.
        return path.name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--samples", type=Path, nargs="+", default=[
        Path("data/samples/kitchen1_000000.npz"),
        Path("data/samples/kitchen1_000001.npz")])
    parser.add_argument("--out", type=Path, default=Path("results/sample_baselines"))
    args = parser.parse_args(argv)
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    stems = [path.stem.casefold() for path in args.samples]
    if len(stems) != len(set(stems)):
        parser.error("--samples must have unique filename stems to avoid overwriting outputs")
    input_paths = set()
    for path in args.samples:
        gt_path = path.with_name(path.stem + "_gt.png")
        for input_path in (path, gt_path):
            if not input_path.is_file():
                parser.error(f"Input file not found: {input_path}")
            input_paths.add(input_path.resolve())
    output_paths = [args.out / "manifest.json"]
    for path in args.samples:
        output_paths.extend(args.out / f"{path.stem}_{suffix}" for suffix in (
            "naive.png", "aligned.png", "gt.png", "reconstructions.npz"))
    if any(path.resolve() in input_paths for path in output_paths):
        parser.error("--out must not overwrite any sample or ground truth input")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for path in args.samples:
        cube = load_sample(path)
        gt_path = path.with_name(path.stem + "_gt.png")
        with Image.open(gt_path) as image:
            gt = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        if gt.shape != cube.shape[1:]:
            raise ValueError(f"Ground truth shape mismatch for {path}")
        naive = naive_average(cube)
        aligned, counts = aligned_average(cube, args.group_size)
        for name, array in (("naive", naive), ("aligned", aligned), ("gt", gt)):
            Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(
                args.out / f"{path.stem}_{name}.png")
        np.savez_compressed(args.out / f"{path.stem}_reconstructions.npz",
                            naive=naive, aligned=aligned, ground_truth=gt,
                            frame_counts=counts)
        with np.load(path) as sample:
            crop = sample["crop"].tolist() if "crop" in sample else None
        record = {"sample": _record_path(path), "ground_truth": _record_path(gt_path),
                  "sample_sha256": _sha256(path), "ground_truth_sha256": _sha256(gt_path),
                  "shape": list(cube.shape), "crop_y0_y1_x0_x1": crop,
                  "sample_local_frame_indices": [0, len(cube) - 1],
                  "group_size": args.group_size,
                  "contributing_frames_min": float(counts.min()),
                  "contributing_frames_max": float(counts.max())}
        records.append(record)
        print(f"{path.stem}: {len(cube)} frames, valid contributions "
              f"{counts.min():.0f}-{counts.max():.0f}; saved to {args.out}")
    manifest = {"method": "hierarchical block alignment + frame-count-weighted average",
                "reference": "first group mean, fixed",
                "alignment": {"block_size": 8, "max_pyramid_levels": 3,
                              "coarse_search_radius": 4, "refinement_radius": 2,
                              "motion": "integer backward offsets (dy,dx)",
                              "border_matching": "at least half-block valid overlap",
                              "unmatchable_edge_blocks": "retain coarse prediction",
                              "invalid_warp_pixels": "excluded from sum and counts"},
                "display": "direct [0,1] mapping; no intensity normalization",
                "evaluation": "PSNR/SSIM not yet run; development samples only",
                "randomness": "none", "samples": records,
                "versions": {"python": sys.version.split()[0],
                             "numpy": np.__version__, "pillow": PIL.__version__},
                "source_sha256": {name: _sha256(Path(__file__).resolve().parents[1] / name)
                                  for name in ("src/alignment.py", "src/baseline.py",
                                               "src/burst_io.py", "scripts/run_sample_baselines.py")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
