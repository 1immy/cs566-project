"""Add a predetermined real kitchen1 crop to the Track 3 failure gallery.

Consumes the full study's verified archives without reconstructing, fitting
exposure, or selecting a crop from method scores. Run after run_taa_study.py.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_taa_failure_gallery import _image, retention_rgb
from scripts.run_sample_taa import _sha256


SOURCE_CROP = (336, 464, 336, 464)
STEM = "kitchen1_000001"


def render(study_dir, out):
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(r for r in manifest["samples"] if r["scene"] == "kitchen1" and r["burst"] == 1)
    if record["shape"][0] != 1024 or record["frame_index_scope"] != "original burst":
        raise ValueError("the real-data gallery requires the full original burst")
    archive = study_dir / f"{STEM}_reconstructions.npz"
    digest = _sha256(archive)
    if digest != record["artifacts_sha256"][archive.name]:
        raise ValueError("archive hash does not match the study manifest")
    sy0, sy1, sx0, sx1 = record["crop_y0_y1_x0_x1"]
    y0, y1, x0, x1 = SOURCE_CROP
    if not (sy0 <= y0 < y1 <= sy1 and sx0 <= x0 < x1 <= sx1):
        raise ValueError("study does not contain the predetermined gallery crop")
    crop = np.s_[y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0]
    with np.load(archive) as arrays:
        confidence = arrays["confidence_map"][crop]
        valid = arrays["valid_update_mask"][crop]
        panels = [("Ground truth", arrays["ground_truth"][crop])]
        for name in ("naive", "aligned", "taa"):
            panels.append((name, np.clip(manifest["gain"] * arrays[name][crop], 0, 1)))
        panels.extend([("Last history retention", retention_rgb(confidence, valid)),
                       ("Last update validity", valid.astype(np.float32))])
    width, gap = 272, 12
    canvas = Image.new("RGB", (6 * width + gap, 374), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), "Real SPC kitchen1 / burst 000001 | 1024 frames | fixed crop (336,464,336,464)", fill="black")
    draw.text((gap, 29), f"Reconstruction gain {manifest['gain']:g}, unchanged GT | reference={manifest['reference']} | group size={manifest['group_size']}", fill="black")
    for index, (label, array) in enumerate(panels):
        x = gap + index * width
        draw.text((x, 53), label, fill="black")
        canvas.paste(_image(array).resize((256, 256), Image.Resampling.NEAREST), (x, 75))
        draw.rectangle((x - 1, 74, x + 256, 331), outline=(160, 160, 160))
    start, end = record["last_update_frames_inclusive"]
    draw.text((gap, 342), f"Retention: black=0, white=1, magenta=invalid. Validity: white=valid. Maps describe frames {start}-{end} only.", fill="black")
    draw.text((gap, 358), "Real motion example; disocclusion is not labeled or inferred from confidence. Crop fixed before inspecting full-burst results.", fill="black")
    out.mkdir(parents=True, exist_ok=True)
    figure = out / f"{STEM}.png"
    canvas.save(figure)
    provenance = {
        "purpose": "issue #26 real-motion illustration; complementary to controlled synthetic disocclusion",
        "scene": "kitchen1", "burst": 1, "frames": [0, 1023],
        "source_crop_y0_y1_x0_x1": record["crop_y0_y1_x0_x1"],
        "gallery_crop_y0_y1_x0_x1": list(SOURCE_CROP),
        "crop_selection": "predetermined central 128x128 crop; not selected by reconstruction scores",
        "gain": manifest["gain"], "ground_truth": "unchanged",
        "reference": manifest["reference"], "group_size": manifest["group_size"],
        "reference_frames_inclusive": record["reference_frames_inclusive"],
        "last_update_frames_inclusive": record["last_update_frames_inclusive"],
        "map_meaning": "last-update retained history fraction, invalid observations magenta; not calibrated occlusion probability",
        "valid_fraction": float(valid.mean()),
        "retention_mean_on_valid": float(confidence[valid].mean()) if valid.any() else None,
        "source_manifest_sha256": _sha256(manifest_path),
        "source_archive": archive.name, "source_archive_sha256": digest,
        "figure": figure.name, "figure_sha256": _sha256(figure),
        "source_sha256": {name: _sha256(ROOT / name) for name in (
            "scripts/render_taa_study_crop.py", "scripts/make_taa_failure_gallery.py",
            "scripts/run_sample_taa.py")},
    }
    (out / f"{STEM}.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {figure}")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("results/sample_taa/study/full"))
    parser.add_argument("--out", type=Path, default=Path("docs/findings/assets/taa-failures"))
    args = parser.parse_args(argv)
    return render(args.study_dir, args.out)


if __name__ == "__main__":
    main()
