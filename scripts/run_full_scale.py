"""Resumable full-resolution run of naive, aligned and TAA over many bursts (issue #24).

Raw data never leaves the machine that runs this. Only small outputs are written:
one JSON record (scores, motion measurement, settings) and one comparison PNG per
burst, plus summary.csv. Those are what gets shared with the team.

Examples (pass your own paths; none are hardcoded):
  python scripts/run_full_scale.py --data-root path/to/train --scenes kitchen1 --bursts 0 1 10
  python scripts/run_full_scale.py --data-root path/to/train --plan results/motion_bins/bins.json

Resuming: a burst is skipped when its record already exists with the same settings
(frames, crop, group size, reference, gain). A changed setting, a corrupt record or a
leftover .tmp file means the burst is run again. A burst that raises an error is logged
to <scene>/<burst>.error.txt, the run continues, and the exit status is 1. Re-run the same
command to retry only the unfinished bursts.
Memory: one full burst unpacks to about 2 GB (1024 x 800 x 800 x 3 uint8). Use --crop
(x bounds multiples of 8) on a smaller machine.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from scripts.run_evaluation import _figure
from src.burst_io import list_bursts, load_burst, load_gt
from src.evaluation import DEFAULT_EVALUATION_GAIN, apply_intensity_rule
from src.motion import default_region, measure_cube_motion
from src.motion_study import (METHODS, config_key, load_records, make_record, read_record,
                              reconstruct_methods, score_methods, write_json_atomic)

SUMMARY_FIELDS = ["source", "scene", "burst", "drift_px", "min_peak"] + [
    f"{metric}_{method}" for method in METHODS for metric in ("psnr", "ssim")]


def write_summary(out):
    records = load_records(out)
    path = os.path.join(out, "summary.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in records:
            row = {"source": r["source"], "scene": r["scene"], "burst": r["burst"],
                   "drift_px": r["motion"]["drift_px"], "min_peak": r["motion"]["min_peak"]}
            for method in METHODS:
                row[f"psnr_{method}"] = r["scores"][method]["psnr_mean"]
                row[f"ssim_{method}"] = r["scores"][method]["ssim_mean"]
            writer.writerow(row)
    os.replace(tmp, path)
    return len(records)


def job_list(args, parser):
    """(scene, burst) pairs to run, in a stable order."""
    if args.plan:
        with open(args.plan, encoding="utf-8") as file:
            plan = json.load(file)
        key = "all" if args.all_planned else "selected"
        jobs = [(e["scene"], int(e["burst"])) for e in plan[key]]
    else:
        scenes = args.scenes or sorted(d for d in os.listdir(args.data_root)
                                       if os.path.isdir(os.path.join(args.data_root, d)))
        jobs = []
        for scene in scenes:
            scene_dir = os.path.join(args.data_root, scene)
            if not os.path.isdir(scene_dir):
                parser.error(f"scene folder not found: {scene_dir}")
            bursts = args.bursts if args.bursts is not None else list_bursts(scene_dir)
            jobs.extend((scene, b) for b in bursts)
    return jobs[:args.limit] if args.limit else jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="folder holding one sub-folder per scene")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--bursts", nargs="+", type=int, help="burst indices (default: every burst in each scene)")
    parser.add_argument("--plan", help="bins.json from bin_scenes_by_motion.py; runs its selected bursts")
    parser.add_argument("--all-planned", action="store_true", help="with --plan, run every binned burst, not just the selected ones")
    parser.add_argument("--limit", type=int, help="run at most this many bursts this time")
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--crop", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--reference", choices=("first", "last"), default="last")
    parser.add_argument("--gain", type=float, default=DEFAULT_EVALUATION_GAIN)
    parser.add_argument("--out", default="results/full_scale")
    parser.add_argument("--dry-run", action="store_true", help="list what would run and what would be skipped")
    args = parser.parse_args(argv)
    if not os.path.isdir(args.data_root):
        parser.error(f"data root not found: {args.data_root}")
    if args.frames < 512 or args.group_size < 1:
        parser.error("--frames must be at least 512 (motion needs two 256-frame windows) and --group-size positive")
    if args.crop and (args.crop[2] % 8 or args.crop[3] % 8):
        parser.error("--crop x bounds must be multiples of 8 (bit-packed width)")
    if args.plan and args.bursts:
        parser.error("--plan and --bursts are exclusive")

    key = config_key(args.frames, args.crop, args.group_size, args.reference, args.gain)
    jobs = job_list(args, parser)
    done, todo, failed = 0, 0, []
    for scene, burst in jobs:
        out_json = os.path.join(args.out, scene, f"{burst:06d}.json")
        existing = read_record(out_json)
        if existing and existing.get("config_key") == key:
            done += 1
            print(f"skip  {scene} {burst:06d} (done)")
            continue
        todo += 1
        if args.dry_run:
            print(f"would run {scene} {burst:06d}" + (" (settings changed)" if existing else ""))
            continue
        scene_dir = os.path.join(args.data_root, scene)
        start = time.time()
        try:
            cube = load_burst(scene_dir, burst, frames=args.frames, crop=tuple(args.crop) if args.crop else None)
            if len(cube) != args.frames:
                raise ValueError(f"burst has {len(cube)} frames, expected {args.frames}")
            gt = load_gt(scene_dir, burst, crop=tuple(args.crop) if args.crop else None)
            motion = measure_cube_motion(cube, crop=default_region(cube))
            methods = reconstruct_methods(cube, args.group_size, args.reference)
            del cube
            scores = score_methods(methods, gt, args.gain)
            record = make_record("real", scene, burst, motion, scores, frames=args.frames,
                                 crop=args.crop, group_size=args.group_size,
                                 reference=args.reference, gain=args.gain, seconds=time.time() - start)
            panels = [("Ground truth", gt)] + [(m, apply_intensity_rule(methods[m], args.gain)) for m in METHODS]
            os.makedirs(os.path.join(args.out, scene), exist_ok=True)
            _figure(panels, os.path.join(args.out, scene, f"{burst:06d}_comparison.png"))
            write_json_atomic(out_json, record)        # last, so a record means the burst finished
            error_file = os.path.join(args.out, scene, f"{burst:06d}.error.txt")
            if os.path.exists(error_file):
                os.remove(error_file)
            print(f"done  {scene} {burst:06d} in {record['seconds']} s, drift {motion['drift_px']} px, "
                  + ", ".join(f"{m} {scores[m]['psnr_mean']:.2f} dB" for m in METHODS))
        except Exception:
            failed.append((scene, burst))
            os.makedirs(os.path.join(args.out, scene), exist_ok=True)
            with open(os.path.join(args.out, scene, f"{burst:06d}.error.txt"), "w", encoding="utf-8") as file:
                file.write(traceback.format_exc())
            print(f"FAIL  {scene} {burst:06d} (see its .error.txt); continuing")
    if args.dry_run:
        print(f"{done} already done, {todo} would run")
        return 0
    total = write_summary(args.out) if os.path.isdir(args.out) else 0
    print(f"{done} skipped, {todo - len(failed)} run, {len(failed)} failed; {total} records in {args.out}/summary.csv")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
