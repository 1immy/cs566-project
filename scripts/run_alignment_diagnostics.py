"""Export per-group fractional motion and match confidence on local samples.

Run from the repo root: py scripts/run_alignment_diagnostics.py
No ground truth is read, no quality metrics are computed, and no blend weights
are changed. Confidence is a local matching heuristic, not TAA retention.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_sample_baselines import _record_path, _sha256
from src.alignment import align_warp_with_confidence
from src.burst_io import load_sample, naive_average


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", type=Path, default=[
        Path("data/samples/kitchen1_000000.npz"), Path("data/samples/kitchen1_000001.npz")])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--reference", choices=("first", "last"), default="last")
    parser.add_argument("--out", type=Path, default=Path("results/issue20/diagnostics"))
    args = parser.parse_args(argv)
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    if len({path.stem.casefold() for path in args.samples}) != len(args.samples):
        parser.error("sample filename stems must be unique")
    for path in args.samples:
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    inputs = {path.resolve() for path in args.samples}
    if (args.out / "manifest.json").resolve() in inputs:
        parser.error("--out must not overwrite inputs")
    records = []
    for path in args.samples:
        cube = load_sample(path)
        if cube.ndim != 4 or cube.shape[-1] != 3 or min(cube.shape) == 0:
            raise ValueError("sample must have nonempty shape [T,H,W,3]")
        start = 0 if args.reference == "first" else ((len(cube) - 1) // args.group_size) * args.group_size
        stop = min(start + args.group_size, len(cube))
        reference = naive_average(cube[start:stop])
        groups = []
        for group_start in range(0, len(cube), args.group_size):
            if group_start == start:
                continue
            group_stop = min(group_start + args.group_size, len(cube))
            warped, motion, confidence = align_warp_with_confidence(cube[group_start:group_stop], reference)
            stem = f"{path.stem}_group_{group_start:06d}"
            archive = args.out / f"{stem}.npz"
            preview = args.out / f"{stem}_confidence.png"
            if any(output.resolve() in inputs for output in (archive, preview)):
                parser.error("--out must not overwrite inputs")
            args.out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(archive, warped_frame=warped, motion_field=motion,
                                alignment_confidence=confidence,
                                group_frame_indices=np.array([group_start, group_stop - 1]),
                                reference_frame_indices=np.array([start, stop - 1]))
            Image.fromarray(np.rint(255 * confidence).astype(np.uint8)).save(preview)
            valid = np.isfinite(warped).all(axis=2)
            groups.append({
                "sample_local_frame_indices": [group_start, group_stop - 1],
                "archive": archive.name, "archive_sha256": _sha256(archive),
                "confidence_preview": preview.name,
                "valid_fraction": float(valid.mean()),
                "fractional_motion_fraction_on_valid": (
                    float(np.any(motion[valid] != np.rint(motion[valid]), axis=1).mean())
                    if valid.any() else None),
                "mean_alignment_confidence_on_valid": float(confidence[valid].mean()) if valid.any() else None,
            })
        with np.load(path) as sample:
            crop = sample["crop"].tolist() if "crop" in sample else None
        records.append({"sample": _record_path(path), "sample_sha256": _sha256(path),
                        "shape": list(cube.shape), "crop_y0_y1_x0_x1": crop,
                        "reference_sample_local_frame_indices": [start, stop - 1],
                        "groups": groups})
        print(f"{path.stem}: exported {len(groups)} group diagnostics, {args.reference} reference [{start}:{stop}]")
    manifest = {
        "purpose": "alignment diagnostics only; no ground truth or PSNR/SSIM",
        "reference": args.reference, "group_size": args.group_size,
        "motion": "backward (dy,dx), 1/8-pixel local refinement; bilinear RGB sampling",
        "confidence": "integer match separation * texture/(texture+residual+1e-12) * overlap fraction; invalid=0",
        "confidence_scope": "per 8x8 block, not calibrated probability or Track 3 history retention",
        "reference_group": "not matched against itself; no synthetic confidence exported",
        "samples": records,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__},
        "source_sha256": {name: _sha256(ROOT / name) for name in (
            "src/alignment.py", "src/burst_io.py", "scripts/run_sample_baselines.py",
            "scripts/run_alignment_diagnostics.py", "requirements.txt")},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
