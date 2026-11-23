"""Final comparison figures: naive vs aligned vs TAA on real full-scale bursts (issue #30).

  python scripts/make_comparison_figures.py --results results/full_scale \
      --per-burst results/motion_analysis/real/per_burst.csv --out docs/findings/figures

Rows are NOT hand-picked. In each motion bin (slow, medium, fast) the script takes the burst
where TAA gains most over alignment in PSNR and the burst where it loses most, so the figure
shows the spread of outcomes and not only flattering cases. Each row shows the ground truth,
naive, aligned and TAA at full frame, with the PSNR and SSIM of each method below its tile.
A second figure shows one fixed 280 px window of the same tiles, for the same bursts.

Inputs are what scripts/run_full_scale.py already wrote: <scene>/<burst>_comparison.png (four
tiles side by side) and <scene>/<burst>.json. Only small figures are written.
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

NAMES = ["Ground truth", "Naive", "Aligned", "TAA"]
KEYS = [None, "naive", "aligned", "taa"]
BINS = ("slow", "medium", "fast")
CROP = (260, 230, 280)       # y, x, size inside each 800 px tile; one fixed window for every row


def select_rows(per_burst_csv):
    """[(scene, burst, bin)] for the largest TAA gain and largest TAA loss (PSNR) in each bin."""
    with open(per_burst_csv, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    chosen = []
    for label in BINS:
        members = [r for r in rows if r["bin"] == label]
        if not members:
            continue
        ranked = sorted(members, key=lambda r: float(r["dpsnr_taa_aligned"]))
        for r in (ranked[-1], ranked[0]):
            item = (r["scene"], int(r["burst"]), label)
            if item not in chosen:
                chosen.append(item)
    return chosen


def _runs(mask, min_length=50):
    runs, start = [], None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return [r for r in runs if r[1] - r[0] >= min_length]


def split_tiles(image):
    """The four image tiles of a run_evaluation-style comparison figure (white background)."""
    nonwhite = image.astype(int).sum(axis=2) < 740
    columns = _runs(nonwhite[len(nonwhite) // 8:].mean(axis=0) > 0.5)
    rows = _runs(nonwhite.mean(axis=1) > 0.5)
    if len(columns) != 4 or len(rows) != 1:
        raise ValueError(f"expected four tiles in one row, found {len(columns)} columns and {len(rows)} rows")
    (y0, y1), = rows
    return [image[y0:y1, a:b] for a, b in columns]


def load(results, scene, burst):
    base = os.path.join(results, scene, f"{burst:06d}")
    with Image.open(base + "_comparison.png") as handle:
        tiles = split_tiles(np.asarray(handle.convert("RGB")))
    with open(base + ".json", encoding="utf-8") as file:
        return tiles, json.load(file)


def draw(rows, results, path, crop=None, dpi=110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 4, figsize=(11.2, 2.9 * len(rows)), squeeze=False)
    for r, (scene, burst, label) in enumerate(rows):
        tiles, record = load(results, scene, burst)
        for c in range(4):
            tile = tiles[c]
            if crop:
                y, x, size = crop
                tile = tile[y:y + size, x:x + size]
            ax = axes[r][c]
            ax.imshow(tile)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(NAMES[c], fontsize=12, fontweight="bold")
            if KEYS[c]:
                s = record["scores"][KEYS[c]]
                ax.set_xlabel(f"{s['psnr_mean']:.2f} dB   SSIM {s['ssim_mean']:.3f}", fontsize=9.5)
        delta = record["scores"]["taa"]["psnr_mean"] - record["scores"]["aligned"]["psnr_mean"]
        axes[r][0].set_ylabel(f"{scene} {burst}\n{label}, {record['motion']['drift_px']:g} px\nTAA vs aligned {delta:+.2f} dB",
                              fontsize=9.5, labelpad=8)
    fig.tight_layout(h_pad=1.0, w_pad=0.4)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results/full_scale")
    parser.add_argument("--per-burst", default="results/motion_analysis/real/per_burst.csv")
    parser.add_argument("--out", default="docs/findings/figures")
    args = parser.parse_args(argv)
    for path in (args.per_burst, args.results):
        if not os.path.exists(path):
            parser.error(f"not found: {path}")
    rows = select_rows(args.per_burst)
    if not rows:
        parser.error("no bursts in the per-burst table")
    draw(rows, args.results, os.path.join(args.out, "comparison_full_frames.png"))
    draw(rows, args.results, os.path.join(args.out, "comparison_crops.png"), crop=CROP)
    for scene, burst, label in rows:
        print(f"{label:6s} {scene} {burst}")
    return rows


if __name__ == "__main__":
    main()
