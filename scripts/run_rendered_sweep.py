"""Motion-speed sweep on Blender-rendered pans (issue #6).

  blender ... make_pan_scenes.py, then render_frames.py per speed (see docs/findings/synthetic-sweep.md)
  python scripts/run_rendered_sweep.py --frames-root data/blender --speeds 0 2 5 10 20 40

For each speed it reads <frames-root>/frames_v<speed>/f_*.png (linear radiance, last frame =
ground-truth pose), draws a seeded binary burst (src/rendered.py), then runs naive, aligned
and TAA and scores them exactly as the real-data runs do (src/motion_study.py, fixed gain 2.7,
last-group reference). Records go in the same format as scripts/run_full_scale.py, under
source "rendered", so scripts/analyze_motion_bins.py reads them next to the numpy sweep.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from scripts.run_evaluation import _figure
from src.evaluation import DEFAULT_EVALUATION_GAIN, apply_intensity_rule
from src.motion import measure_cube_motion
from src.motion_study import (METHODS, make_record, reconstruct_methods, score_methods,
                              write_json_atomic)
from src.rendered import burst_from_frames, load_rendered_frames

DEFAULT_SPEEDS = [0, 2, 5, 10, 20, 40]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-root", default="data/blender", help="folder holding frames_v<speed>/")
    parser.add_argument("--speeds", nargs="+", type=float, default=DEFAULT_SPEEDS,
                        help="total pan in px over the burst, as given to make_pan_scenes.py")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--reference", choices=("first", "last"), default="last")
    parser.add_argument("--gain", type=float, default=DEFAULT_EVALUATION_GAIN)
    parser.add_argument("--seed", type=int, default=566)
    parser.add_argument("--out", default="results/rendered_sweep")
    args = parser.parse_args(argv)
    if any(s < 0 for s in args.speeds):
        parser.error("speeds must be nonnegative")

    scene = "rendered_pan"
    records = []
    for index, speed in enumerate(args.speeds):
        folder = os.path.join(args.frames_root, f"frames_v{speed:g}")
        if not os.path.isdir(folder):
            parser.error(f"frames folder not found: {folder}")
        start = time.time()
        frames = load_rendered_frames(folder)
        if len(frames) < 512:
            parser.error(f"{folder} has {len(frames)} frames, need at least 512")
        cube, gt = burst_from_frames(frames, args.gain, args.seed, tag=int(round(speed * 100)))
        motion = measure_cube_motion(cube)
        methods = reconstruct_methods(cube, args.group_size, args.reference)
        scores = score_methods(methods, gt, args.gain)
        record = make_record("rendered", scene, index, motion, scores, frames=len(frames), crop=None,
                             group_size=args.group_size, reference=args.reference, gain=args.gain,
                             seconds=time.time() - start, nominal_drift_px=speed, angle_deg=0.0,
                             seed=args.seed, simulator="Blender Cycles frames + Bernoulli (src/rendered.py)")
        write_json_atomic(os.path.join(args.out, scene, f"{index:06d}.json"), record)
        panels = [("Ground truth", gt)] + [(m, apply_intensity_rule(methods[m], args.gain)) for m in METHODS]
        os.makedirs(os.path.join(args.out, scene), exist_ok=True)
        _figure(panels, os.path.join(args.out, scene, f"{index:06d}_comparison.png"))
        records.append(record)
        print(f"nominal {speed:g} px, measured {motion['drift_px']} px: "
              + ", ".join(f"{m} {scores[m]['psnr_mean']:.2f} dB / {scores[m]['ssim_mean']:.3f}" for m in METHODS))
    return records


if __name__ == "__main__":
    main()
