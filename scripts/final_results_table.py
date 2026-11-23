"""Final multi-scene results table and statistics (issue #29).

  python scripts/final_results_table.py --per-burst results/motion_analysis/real/per_burst.csv \
      --out docs/findings/final-results.md

Reads the per-burst table written by scripts/analyze_motion_bins.py and writes one Markdown page
(and a CSV of the same table) with, per motion bin and overall: mean PSNR and SSIM of naive,
aligned and TAA; the paired differences TAA - aligned and aligned - naive with a seeded bootstrap
95% interval over bursts; and sign counts with an exact two-sided sign test, over bursts and over
scenes (bursts of one scene averaged first). Every number is computed here from the per-burst
table, nothing is typed in.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from scripts.analyze_motion_bins import bootstrap_ci

BINS = ("slow", "medium", "fast")
METHODS = ("naive", "aligned", "taa")
LABEL = {"naive": "Naive", "aligned": "Aligned", "taa": "TAA"}


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    out = []
    for r in rows:
        item = {"scene": r["scene"], "burst": int(r["burst"]), "bin": r["bin"],
                "drift": float(r["drift_px"]), "reliable": r["reliable"] == "True"}
        for m in METHODS:
            item["psnr_" + m] = float(r["psnr_" + m])
            item["ssim_" + m] = float(r["ssim_" + m])
        out.append(item)
    return out


def sign_test(differences):
    """(wins, losses, ties, exact two-sided p) for H0: wins and losses are equally likely."""
    wins = sum(d > 0 for d in differences)
    losses = sum(d < 0 for d in differences)
    n = wins + losses
    if n == 0:
        return wins, losses, 0, 1.0
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return wins, losses, len(differences) - n, p


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2f} ({values.std(ddof=1):.2f})" if len(values) > 1 else f"{values.mean():.2f}"


def mean_sd_ssim(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} ({values.std(ddof=1):.3f})" if len(values) > 1 else f"{values.mean():.3f}"


def delta_cell(values, digits):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    if len(values) < 3:
        return f"{mean:+.{digits}f}"
    lo, hi = bootstrap_ci(values)
    return f"{mean:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def groups(rows):
    result = [(label, [r for r in rows if r["bin"] == label]) for label in BINS]
    result = [(label, g) for label, g in result if g]
    result.append(("all", rows))
    return result


def scene_means(rows, key_a, key_b):
    per_scene = defaultdict(list)
    for r in rows:
        per_scene[r["scene"]].append(r[key_a] - r[key_b])
    return [float(np.mean(v)) for v in per_scene.values()]


def build(rows):
    lines = []
    add = lines.append
    add("## Main table: PSNR (dB) and SSIM, mean (standard deviation) over bursts")
    add("")
    add("| Motion bin | n | Drift px | PSNR naive | PSNR aligned | PSNR TAA | SSIM naive | SSIM aligned | SSIM TAA |")
    add("|---|---:|---|---|---|---|---|---|---|")
    for label, g in groups(rows):
        drift = f"{min(r['drift'] for r in g):.1f} to {max(r['drift'] for r in g):.1f}"
        cells = [mean_sd([r["psnr_" + m] for r in g]) for m in METHODS] + [mean_sd_ssim([r["ssim_" + m] for r in g]) for m in METHODS]
        add(f"| {label} | {len(g)} | {drift} | " + " | ".join(cells) + " |")
    add("")
    add("## Paired differences, mean over bursts [bootstrap 95% interval]")
    add("")
    add("| Motion bin | n | TAA - aligned, PSNR dB | TAA - aligned, SSIM | Aligned - naive, PSNR dB | Aligned - naive, SSIM |")
    add("|---|---:|---|---|---|---|")
    for label, g in groups(rows):
        cells = [delta_cell([r["psnr_taa"] - r["psnr_aligned"] for r in g], 2),
                 delta_cell([r["ssim_taa"] - r["ssim_aligned"] for r in g], 3),
                 delta_cell([r["psnr_aligned"] - r["psnr_naive"] for r in g], 2),
                 delta_cell([r["ssim_aligned"] - r["ssim_naive"] for r in g], 3)]
        add(f"| {label} | {len(g)} | " + " | ".join(cells) + " |")
    add("")
    add("## Sign counts: how often TAA beats aligned")
    add("")
    add("| Metric | Unit | TAA better | TAA worse | Ties | Exact sign test p |")
    add("|---|---|---:|---:|---:|---|")
    for metric in ("psnr", "ssim"):
        for unit, diffs in (("bursts", [r[metric + "_taa"] - r[metric + "_aligned"] for r in rows]),
                            ("scenes", scene_means(rows, metric + "_taa", metric + "_aligned"))):
            w, l, t, p = sign_test(diffs)
            add(f"| {metric.upper()} | {len(diffs)} {unit} | {w} | {l} | {t} | {fmt_p(p)} |")
    add("")
    reliable = sum(r["reliable"] for r in rows)
    add(f"Motion estimates with a correlation peak of at least 0.05 (reliable): {reliable} of {len(rows)} bursts. "
        f"Scenes: {len({r['scene'] for r in rows})}.")
    add("")
    return lines


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["bin", "n", "metric", "method", "mean", "sd"])
        for label, g in groups(rows):
            for metric in ("psnr", "ssim"):
                for m in METHODS:
                    v = np.array([r[f"{metric}_{m}"] for r in g])
                    writer.writerow([label, len(g), metric, m, f"{v.mean():.4f}", f"{v.std(ddof=1) if len(v) > 1 else float('nan'):.4f}"])


HEADER = """# Final multi-scene results (issue #29)

Generated by `scripts/final_results_table.py` from `results/motion_analysis/real/per_burst.csv`; do not edit the numbers by hand.

**Setup.** {n} real bursts from {scenes} scenes of the training set, each reconstructed at full resolution with naive averaging, aligned averaging and TAA from the same photon cube: 1024 frames, groups of 64, last-group reference, fixed gain 2.7 with PSNR and SSIM as in `docs/findings/evaluation-intensity.md`. The bursts are the ones chosen by motion bin (`docs/findings/motion-bins.md`); the bins are tertiles of the measured drift over all 100 measured bursts, edges 12.77 px and 37.82 px.

"""

FOOTER = """
## How to read this
- Compare methods within a row. Different bins hold different scenes, so a method's value cannot be compared across rows.
- Intervals are seeded bootstrap 95% intervals over bursts. With 10 or 11 bursts per bin they are close to the range of the observed differences and are not population intervals.
- The sign test treats the bursts (or scenes) as independent draws and asks only whether TAA beats alignment more often than chance. It says nothing about the size of a difference.
- Most motion estimates are flagged unreliable (correlation peak below 0.05), so the bin assignment is approximate; the fast bin in particular should be read as "large measured drift", not as a known speed.
- This table is the broad real-data result. The kitchen1 harness table in the README and the report is a separate, smaller experiment (3 bursts of one scene) and is not mixed into these numbers.
- Fixed gain 2.7 may fit some scenes poorly, which compresses all differences; PSNR here is 11 to 22 dB.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-burst", default="results/motion_analysis/real/per_burst.csv")
    parser.add_argument("--out", default="docs/findings/final-results.md")
    args = parser.parse_args(argv)
    if not os.path.isfile(args.per_burst):
        parser.error(f"not found: {args.per_burst}")
    rows = read_rows(args.per_burst)
    if len(rows) < 3:
        parser.error("need at least 3 bursts")
    body = HEADER.format(n=len(rows), scenes=len({r["scene"] for r in rows})) + "\n".join(build(rows)) + FOOTER
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        file.write(body)
    write_csv(rows, os.path.splitext(args.out)[0] + ".csv")
    print(body)
    return rows


if __name__ == "__main__":
    main()
