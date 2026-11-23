"""Measure motion speed over many scenes and bin bursts slow / medium / fast (issue #23).

Example (Drive slice or the full dataset; pass your own folder, no path is hardcoded):
  python scripts/bin_scenes_by_motion.py --data-root path/to/train --bursts 0 1 --per-bin 3

For every scene folder under --data-root and every burst index in --bursts, the global
drift between the first and last 256-frame window is measured (src/motion.py). Bins are
tertiles of the measured drift by default, or fixed with --edges. Output (small text
files only, safe to share):
  results/motion_bins/motion.csv   one row per scene and burst
  results/motion_bins/bins.json    edges, labels, counts, every burst's bin, and the
                                   selected scene set (--per-bin per bin, distinct scenes first)
Measurements already in motion.csv are reused, so an interrupted run can continue.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.motion import LABELS3, assign_bin, measure_burst_motion, tertile_edges

FIELDS = ["scene", "burst", "dy", "dx", "drift_px", "speed_px_per_256", "path_px",
          "min_peak", "reliable"]


def read_rows(path):
    if not os.path.isfile(path):
        return {}
    with open(path, newline="", encoding="utf-8") as file:
        return {(r["scene"], int(r["burst"])): r for r in csv.DictReader(file)}


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({f: rows[key][f] for f in FIELDS})
    os.replace(tmp, path)


def select(entries, per_bin, labels):
    """Per bin, take the bursts closest to that bin's median drift; distinct scenes first."""
    chosen = []
    for label in labels:
        members = [e for e in entries if e["bin"] == label]
        if not members:
            continue
        median = float(np.median([e["drift_px"] for e in members]))
        ranked = sorted(members, key=lambda e: (abs(e["drift_px"] - median), e["scene"], e["burst"]))
        picked, seen = [], set()
        for e in ranked:                       # first pass: one burst per scene
            if e["scene"] not in seen and len(picked) < per_bin:
                picked.append(e)
                seen.add(e["scene"])
        for e in ranked:                       # second pass: fill with repeat scenes
            if e not in picked and len(picked) < per_bin:
                picked.append(e)
        chosen.extend(picked)
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="folder holding one sub-folder per scene")
    parser.add_argument("--scenes", nargs="+", help="scene folder names (default: every sub-folder with .npy files)")
    parser.add_argument("--bursts", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--edges", nargs=2, type=float, metavar=("SLOW_MEDIUM", "MEDIUM_FAST"),
                        help="fixed drift edges in px instead of tertiles")
    parser.add_argument("--per-bin", type=int, default=3, help="bursts to select per bin")
    parser.add_argument("--out", default="results/motion_bins")
    args = parser.parse_args(argv)
    if args.per_bin < 1:
        parser.error("--per-bin must be at least 1")
    if args.edges and not args.edges[0] < args.edges[1]:
        parser.error("--edges must be increasing")

    if not os.path.isdir(args.data_root):
        parser.error(f"data root not found: {args.data_root}")
    scenes = args.scenes or sorted(
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
        and any(f.endswith(".npy") for f in os.listdir(os.path.join(args.data_root, d))))
    if not scenes:
        parser.error("no scene folders with .npy files found")

    csv_path = os.path.join(args.out, "motion.csv")
    rows = read_rows(csv_path)
    skipped = []
    for scene in scenes:
        scene_dir = os.path.join(args.data_root, scene)
        for burst in args.bursts:
            if (scene, burst) in rows:
                continue
            if not os.path.isfile(os.path.join(scene_dir, f"{burst:06d}.npy")):
                skipped.append((scene, burst))
                continue
            motion = measure_burst_motion(scene_dir, burst)
            rows[(scene, burst)] = {"scene": scene, "burst": burst, **{k: motion[k] for k in FIELDS[2:]}}
            write_rows(csv_path, rows)                 # after each burst, so a stop loses nothing
            print(f"{scene} {burst:06d}: drift {motion['drift_px']} px, min peak {motion['min_peak']}")
    if skipped:
        print(f"missing bursts (not measured): {skipped}")

    wanted = {(s, b) for s in scenes for b in args.bursts}
    entries = [{"scene": s, "burst": b, "drift_px": float(r["drift_px"]),
                "speed_px_per_256": float(r["speed_px_per_256"]), "min_peak": float(r["min_peak"]),
                "reliable": str(r["reliable"]) == "True"}
               for (s, b), r in sorted(rows.items()) if (s, b) in wanted]
    if len(entries) < 3:
        parser.error(f"need at least 3 measured bursts to form bins, have {len(entries)}")
    edges = list(args.edges) if args.edges else tertile_edges([e["drift_px"] for e in entries])
    for e in entries:
        e["bin"] = assign_bin(e["drift_px"], edges, LABELS3)
    counts = {label: sum(e["bin"] == label for e in entries) for label in LABELS3}
    selected = select(entries, args.per_bin, LABELS3)
    summary = {"edges_px": edges, "labels": list(LABELS3),
               "method": "fixed" if args.edges else "tertiles of measured drift",
               "bursts_measured": len(entries), "counts": counts,
               "unreliable_bursts": [(e["scene"], e["burst"]) for e in entries if not e["reliable"]],
               "selected": selected, "all": entries}
    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "bins.json.tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    os.replace(tmp, os.path.join(args.out, "bins.json"))
    print(f"edges (px): {edges}; counts: {counts}; selected {len(selected)} bursts")
    return summary


if __name__ == "__main__":
    main()
