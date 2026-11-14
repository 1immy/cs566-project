"""Render seeded, synthetic Track 3 failure examples with the real aligner.

Run from the repository root: python scripts/make_taa_failure_gallery.py
These are controlled Bernoulli scenes, not SPC or visionsim benchmark results.
The target is the known probability image at the final instant. All methods
use the same input, crop and gain=1; no camera-response conversion is needed.
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import aligned_average
from src.burst_io import naive_average
from src.evaluation import evaluate_channels
from src.taa import TAAConfig, reconstruct_taa


SEED = 5662026
FRAMES = 256
GROUP_SIZE = 64
SHAPE = (96, 128)
# Chosen from scene geometry, before reconstruction or metric inspection.
CROPS = {"fast_motion": (24, 72, 40, 88), "disocclusion": (24, 72, 32, 80)}


def _texture(height, width):
    """Analytic RGB texture with edges, fine detail and no randomness."""
    y, x = np.indices((height, width))
    checker = ((x // 4 + y // 4) % 2).astype(np.float32)
    stripe = ((x // 9) % 2).astype(np.float32)
    circle = ((x % 64 - 32) ** 2 + (y - 48) ** 2 < 18 ** 2).astype(np.float32)
    return np.stack((0.08 + 0.5 * checker,
                     0.10 + 0.4 * stripe,
                     0.06 + 0.55 * circle), axis=2).astype(np.float32)


def probability_sequence(kind):
    """Return exact latent probabilities, including the final-instant target.

    Translation samples a wider scene, so no image wraps at the border.
    The foreground in the second scene moves independently of the background.
    Integer positions follow rounded linear motion at every binary exposure.
    """
    height, width = SHAPE
    if kind == "fast_motion":
        positions = np.rint(np.linspace(0, 48, FRAMES)).astype(int)
        scene = _texture(height, width + 48)
        probability = np.stack([scene[:, dx:dx + width] for dx in positions])
        details = {"camera_translation_px": [0, 48],
                   "motion_description": "camera x advances 48 pixels in 255 frame intervals; no wraparound",
                   "failure_mechanism": "motion inside each 64-frame mean is not modeled by one group warp"}
    elif kind == "disocclusion":
        positions = np.rint(np.linspace(24, 96, FRAMES)).astype(int)
        background = _texture(height, width)
        probability = np.broadcast_to(background, (FRAMES, *background.shape)).copy()
        for frame, left in enumerate(positions):
            # A moving opaque 32-pixel-wide slab reveals the fixed background.
            probability[frame, 12:84, left:min(left + 32, width)] = (0.68, 0.16, 0.13)
        details = {"foreground_left_x": [24, 96], "foreground_width": 32,
                   "foreground_y0_y1": [12, 84],
                   "motion_description": "opaque foreground advances 72 pixels over a stationary textured background",
                   "failure_mechanism": "block translation cannot establish visibility behind an independently moving occluder"}
    else:
        raise ValueError(f"unknown scene: {kind}")
    details["positions_per_frame"] = positions.tolist()
    return probability, details


def make_scene(kind, seed=SEED):
    probability, details = probability_sequence(kind)
    scene_id = ("fast_motion", "disocclusion").index(kind)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 26, scene_id]))
    cube = (rng.random(probability.shape) < probability).astype(np.uint8)
    return cube, probability[-1].copy(), probability[-GROUP_SIZE:].mean(axis=0), details


def _crop(image, bounds):
    y0, y1, x0, x1 = bounds
    return image[y0:y1, x0:x1]


def retention_rgb(confidence, valid):
    """A fixed 0..1 grayscale; magenta means no valid update, not rejection."""
    rgb = np.repeat(np.asarray(confidence)[..., None], 3, axis=2).copy()
    rgb[~np.asarray(valid, dtype=bool)] = (1, 0, 1)
    return rgb


def _image(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def render_case(kind, target, images, confidence, valid, output, seed=SEED):
    """Same crop and fixed display scale for every panel; no autocontrast."""
    bounds = CROPS[kind]
    panels = [("Final-instant truth", target), ("Naive average", images["naive"]),
              ("Aligned average", images["aligned"]), ("Full TAA", images["taa"]),
              ("Last history retention", retention_rgb(confidence, valid)),
              ("Last update validity", valid.astype(np.float32))]
    scale, panel_width, gap = 4, 216, 12
    canvas = Image.new("RGB", (len(panels) * panel_width + gap, 310), "white")
    draw = ImageDraw.Draw(canvas)
    title = "Fast camera motion" if kind == "fast_motion" else "Moving foreground and disocclusion"
    draw.text((gap, 10), f"{title} | synthetic Bernoulli, seed {seed} | 256 frames, groups of 64, last reference", fill="black")
    draw.text((gap, 29), f"Fixed crop (y0,y1,x0,x1)={bounds}; RGB: shared gain 1, no fitted exposure", fill="black")
    for index, (name, array) in enumerate(panels):
        x = gap + panel_width * index
        draw.text((x, 55), name, fill="black")
        crop = _crop(array, bounds)
        canvas.paste(_image(crop).resize((crop.shape[1] * scale, crop.shape[0] * scale),
                                        Image.Resampling.NEAREST), (x, 75))
        draw.rectangle((x - 1, 74, x + crop.shape[1] * scale, 75 + crop.shape[0] * scale),
                       outline=(160, 160, 160))
    draw.text((gap, 279), "Retention: black=0, white=1, magenta=invalid. Validity: white=valid. Both maps describe frames 128-191 updating the end-reference history.", fill="black")
    draw.text((gap, 295), "Truth is the last instantaneous probability image; motion blur within the last reference group remains part of the reconstruction error.", fill="black")
    canvas.save(output)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("docs/findings/assets/taa-failures"))
    parser.add_argument("--archives", type=Path, default=Path("results/sample_taa/failure_gallery"))
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    args.archives.mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for kind in CROPS:
        cube, target, reference_probability, details = make_scene(kind, args.seed)
        print(f"{kind}: reconstructing shared input with real alignment", flush=True)
        aligned, counts = aligned_average(cube, GROUP_SIZE, reference="last")
        taa = reconstruct_taa(cube, GROUP_SIZE, reference="last")
        disabled = reconstruct_taa(cube, GROUP_SIZE, TAAConfig(clamp=False, adaptive=False), reference="last")
        tolerance = 3 * float(np.finfo(np.float32).eps)
        np.testing.assert_allclose(disabled.reconstruction, aligned, atol=tolerance, rtol=0)
        np.testing.assert_array_equal(disabled.history_weights, counts)
        np.testing.assert_array_equal(taa.frame_counts, counts)
        images = {"naive": naive_average(cube), "aligned": aligned, "taa": taa.reconstruction}
        for method, reconstruction in images.items():
            score = evaluate_channels(_crop(reconstruction, CROPS[kind]), _crop(target, CROPS[kind]))
            rows.append({"scene": kind, "method": method, "psnr_mean": score["psnr"]["mean"],
                         "ssim_mean": score["ssim"]["mean"]})
            print(f"  {method}: {score['psnr']['mean']:.4f} dB, SSIM {score['ssim']['mean']:.6f}", flush=True)
        figure = args.out / f"{kind}.png"
        render_case(kind, target, images, taa.confidence_map, taa.valid_update_mask, figure, args.seed)
        archive = args.archives / f"{kind}.npz"
        np.savez_compressed(archive, ground_truth=target, reference_probability=reference_probability,
                            **images, confidence_map=taa.confidence_map, valid_update_mask=taa.valid_update_mask,
                            history_weights=taa.history_weights, frame_counts=counts)
        valid = _crop(taa.valid_update_mask, CROPS[kind])
        retention = _crop(taa.confidence_map, CROPS[kind])
        records.append({"scene": kind, "seed_sequence": [args.seed, 26, len(records)],
                        "shape": list(cube.shape), "crop_y0_y1_x0_x1": list(CROPS[kind]),
                        "input_cube_sha256": hashlib.sha256(cube.tobytes()).hexdigest(),
                        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                        "disabled_baseline_max_abs_error": float(np.max(np.abs(disabled.reconstruction - aligned))),
                        "disabled_baseline_atol": tolerance,
                        "crop_last_update_valid_fraction": float(valid.mean()),
                        "crop_last_retention_mean_on_valid": float(retention[valid].mean()) if valid.any() else None,
                        "reference_exposure_mse_vs_final_instant": float(np.mean((_crop(reference_probability, CROPS[kind]) - _crop(target, CROPS[kind])) ** 2)),
                        "figure": figure.name, "figure_sha256": _sha256(figure),
                        "archive": archive.name, "archive_sha256": _sha256(archive), **details})
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "purpose": "issue #26 synthetic failure illustrations, not held-out SPC evaluation or visionsim",
        "seed": args.seed, "frames": [0, FRAMES - 1], "group_size": GROUP_SIZE,
        "reference": "fixed last-group mean, frames 192-255, used identically by aligned and TAA",
        "taa_update_order": "seed with frames 192-255 once, then groups 0-63, 64-127, 128-191",
        "map_meaning": "last update only (frames 128-191); history-retention fraction, not occlusion probability or alignment confidence",
        "target": "known final-instant Bernoulli probability image at frame 255, not last-group exposure mean",
        "intensity_rule": "identity (gain=1) for all methods and targets in known probability space; distinct from SPC gain 2.7",
        "scoring_support": "all pixels of the predeclared crop, identical across methods; no validity-based masking",
        "metrics": "unmodified evaluate_channels; arithmetic RGB means, 7x7 uniform reflect-padded SSIM",
        "display": "fixed [0,1] RGB mapping, nearest-neighbor 4x crop enlargement; retention grayscale with magenta invalid pixels",
        "taa": asdict(TAAConfig()), "scenes": records,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__},
        "source_sha256": {name: _sha256(ROOT / name) for name in (
            "scripts/make_taa_failure_gallery.py", "src/taa.py", "src/baseline.py",
            "src/alignment.py", "src/burst_io.py", "src/evaluation.py")},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved figures, metrics and provenance to {args.out}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
