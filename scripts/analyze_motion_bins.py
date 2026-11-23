"""PSNR and SSIM of naive, aligned and TAA across motion-speed bins, with plots (issue #25).

  python scripts/analyze_motion_bins.py --results results/full_scale --bins results/motion_bins/bins.json
  python scripts/analyze_motion_bins.py --results results/synthetic_sweep

Reads the JSON records written by scripts/run_full_scale.py (real bursts) and
scripts/run_synthetic_sweep.py (synthetic bursts). Real and synthetic records are analysed
separately and written to <out>/real and <out>/synthetic; they are never pooled.
Bins use the measured drift (src/motion.py). Edge precedence: --edges, then --bins
(a bins.json from bin_scenes_by_motion.py), then tertiles of that source's own drifts.
Per bin it reports n, mean and standard deviation of every metric, and the paired difference
TAA minus aligned with a seeded bootstrap 95% interval over bursts (only when n >= 3).
Nothing here is fitted to the ground truth; it only summarises existing records.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.motion import LABELS3, assign_bin, tertile_edges
from src.motion_study import METHODS, load_records

SEED = 566
RESAMPLES = 10000
COLORS = {"naive": "#7a7a7a", "aligned": "#1b6ca8", "taa": "#d1495b"}   # distinct in grayscale too


def bootstrap_ci(values, seed=SEED, resamples=RESAMPLES):
    """Percentile 95% interval of the mean over bursts; None when n < 3."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(value, digits=2):
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def analyse(records, edges):
    rows = []
    for r in records:
        row = {"scene": r["scene"], "burst": r["burst"], "drift_px": r["motion"]["drift_px"],
               "reliable": r["motion"].get("reliable", True),
               "bin": assign_bin(r["motion"]["drift_px"], edges, LABELS3)}
        for m in METHODS:
            row[f"psnr_{m}"] = r["scores"][m]["psnr_mean"]
            row[f"ssim_{m}"] = r["scores"][m]["ssim_mean"]
        row["dpsnr_taa_aligned"] = row["psnr_taa"] - row["psnr_aligned"]
        row["dssim_taa_aligned"] = row["ssim_taa"] - row["ssim_aligned"]
        rows.append(row)
    bins = []
    for label in LABELS3:
        members = [x for x in rows if x["bin"] == label]
        if not members:
            continue
        entry = {"bin": label, "n": len(members),
                 "drift_min": min(x["drift_px"] for x in members),
                 "drift_max": max(x["drift_px"] for x in members)}
        for key in [f"{k}_{m}" for k in ("psnr", "ssim") for m in METHODS]:
            vals = [x[key] for x in members]
            entry[f"{key}_mean"] = float(np.mean(vals))
            entry[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        for key in ("dpsnr_taa_aligned", "dssim_taa_aligned"):
            vals = [x[key] for x in members]
            entry[f"{key}_mean"] = float(np.mean(vals))
            entry[f"{key}_ci"] = bootstrap_ci(vals)
        bins.append(entry)
    return rows, bins


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, tuple):
        return f"{value[0]:.4f};{value[1]:.4f}"
    return value


def write_csv(path, rows):
    fields = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fields})


def markdown(source, edges, rows, bins):
    lines = [f"# Motion-bin results ({source})", "",
             f"Bursts: {len(rows)}. Bin edges on measured drift (px): {edges}. "
             "Fixed gain 2.7, same cube, reference and aligner for every method.", "",
             "| bin | n | drift px | PSNR naive | PSNR aligned | PSNR TAA | SSIM naive | SSIM aligned | SSIM TAA |",
             "|---|---|---|---|---|---|---|---|---|"]
    for b in bins:
        lines.append(f"| {b['bin']} | {b['n']} | {b['drift_min']:.1f} to {b['drift_max']:.1f} | "
                     + " | ".join(f"{b[f'psnr_{m}_mean']:.2f}" for m in METHODS) + " | "
                     + " | ".join(f"{b[f'ssim_{m}_mean']:.3f}" for m in METHODS) + " |")
    lines += ["", "TAA minus aligned, mean over bursts in the bin (bootstrap 95% interval when n >= 3):", "",
              "| bin | n | PSNR dB | SSIM |", "|---|---|---|---|"]
    for b in bins:
        def cell(key, digits):
            ci = b[f"{key}_ci"]
            text = f"{b[f'{key}_mean']:+.{digits}f}"
            return text + (f" [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]" if ci else " (n < 3, no interval)")
        lines.append(f"| {b['bin']} | {b['n']} | {cell('dpsnr_taa_aligned', 2)} | {cell('dssim_taa_aligned', 3)} |")
    unreliable = [(x["scene"], x["burst"]) for x in rows if not x["reliable"]]
    if unreliable:
        lines += ["", f"Bursts whose motion estimate has a low correlation peak (< 0.05), so their bin may be off: {unreliable}"]
    return "\n".join(lines) + "\n"


def plots(out, source, edges, rows, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = sorted(rows, key=lambda x: x["drift_px"])
    for metric, label, name in (("psnr", "PSNR (dB)", "psnr_vs_drift.png"), ("ssim", "SSIM", "ssim_vs_drift.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4), dpi=150)
        for m in METHODS:
            xs = [x["drift_px"] for x in ordered]
            ys = [x[f"{metric}_{m}"] for x in ordered]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.2 if source == "synthetic" else 0, color=COLORS[m], label=m)
        for edge in edges:
            ax.axvline(edge, color="#999999", lw=0.8, ls="--")
        ax.set_xlabel("measured drift between first and last window (px)")
        ax.set_ylabel(label)
        ax.set_title(f"{label.split()[0]} versus motion speed ({source}, n={len(rows)})")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, name))
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), dpi=150)
    for ax, key, label in ((axes[0], "dpsnr_taa_aligned", "TAA minus aligned, PSNR (dB)"),
                           (axes[1], "dssim_taa_aligned", "TAA minus aligned, SSIM")):
        labels = [b["bin"] for b in bins]
        means = [b[f"{key}_mean"] for b in bins]
        ci = [b[f"{key}_ci"] for b in bins]
        has_ci = all(ci)
        err = [[m - c[0] for m, c in zip(means, ci)], [c[1] - m for m, c in zip(means, ci)]] if has_ci else None
        ax.bar(labels, means, yerr=err, color=COLORS["taa"], capsize=3, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"TAA versus aligned by motion bin ({source})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "delta_by_bin.png"))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, help="folders of JSON records")
    parser.add_argument("--bins", help="bins.json from bin_scenes_by_motion.py (uses its edges)")
    parser.add_argument("--edges", nargs=2, type=float, metavar=("SLOW_MEDIUM", "MEDIUM_FAST"))
    parser.add_argument("--out", default="results/motion_analysis")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    records = [r for folder in args.results for r in load_records(folder)]
    if not records:
        parser.error("no records found; run scripts/run_full_scale.py or scripts/run_synthetic_sweep.py first")
    fixed = list(args.edges) if args.edges else None
    if fixed is None and args.bins:
        with open(args.bins, encoding="utf-8") as file:
            fixed = json.load(file)["edges_px"]
    written = {}
    for source in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == source]
        if len(subset) < 3 and fixed is None:
            print(f"{source}: only {len(subset)} records, need 3 for tertile bins (or pass --edges); skipped")
            continue
        edges = fixed or tertile_edges([r["motion"]["drift_px"] for r in subset])
        rows, bins = analyse(subset, edges)
        folder = os.path.join(args.out, source)
        os.makedirs(folder, exist_ok=True)
        write_csv(os.path.join(folder, "per_burst.csv"), rows)
        write_csv(os.path.join(folder, "per_bin.csv"), bins)
        with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as file:
            file.write(markdown(source, edges, rows, bins))
        if not args.no_plots:
            plots(folder, source, edges, rows, bins)
        written[source] = {"edges": edges, "rows": rows, "bins": bins}
        print(f"{source}: {len(rows)} bursts, edges {edges} px -> {folder}")
    return written


if __name__ == "__main__":
    main()
