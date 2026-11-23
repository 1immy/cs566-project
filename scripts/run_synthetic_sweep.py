"""Synthetic motion-speed sweep on a ground-truth image (issue #6).

  python scripts/run_synthetic_sweep.py
  python scripts/run_synthetic_sweep.py --gt data/samples/kitchen1_000001_gt.png --speeds 0 2 5 10 20 40

For each speed, src/synthetic.py builds a 1024-frame binary burst from the ground truth
whose content drifts by that many pixels over the burst (the last frame is the ground-truth
pose), then naive, aligned and TAA are run and scored exactly as in the real-data runs
(src/motion_study.py, fixed gain 2.7, last-group reference). Records are written in the same
format as scripts/run_full_scale.py, so scripts/analyze_motion_bins.py reads both.
This is a seeded numpy simulator, not visionsim: see docs/findings/synthetic-sweep.md for
what it does and does not model. --save-npz writes each burst in the data/samples format
(large; not for committing).
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from scripts.run_evaluation import _figure
from src.evaluation import DEFAULT_EVALUATION_GAIN, apply_intensity_rule
from src.motion import measure_cube_motion
from src.motion_study import (METHODS, make_record, reconstruct_methods, score_methods,
                              write_json_atomic)
from src.synthetic import synthesize_burst

DEFAULT_SPEEDS = [0, 2, 5, 10, 20, 40]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", default="data/samples/kitchen1_000000_gt.png", help="ground-truth PNG (the end-of-burst pose)")
    parser.add_argument("--speeds", nargs="+", type=float, default=DEFAULT_SPEEDS,
                        help="total drift in px over the 1024-frame burst")
    parser.add_argument("--angle", type=float, default=0.0, help="motion direction in degrees, 0 = along x")
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--reference", choices=("first", "last"), default="last")
    parser.add_argument("--gain", type=float, default=DEFAULT_EVALUATION_GAIN)
    parser.add_argument("--seed", type=int, default=566)
    parser.add_argument("--out", default="results/synthetic_sweep")
    parser.add_argument("--save-npz", help="folder to also write each burst as a data/samples-style .npz")
    args = parser.parse_args(argv)
    if not os.path.isfile(args.gt):
        parser.error(f"ground truth not found: {args.gt}")
    if args.frames < 512 or any(s < 0 for s in args.speeds):
        parser.error("--frames must be at least 512 and speeds nonnegative")

    gt = np.asarray(Image.open(args.gt).convert("RGB"), dtype=np.float32) / 255
    scene = "synthetic_" + os.path.splitext(os.path.basename(args.gt))[0].replace("_gt", "")
    records = []
    for index, speed in enumerate(args.speeds):
        start = time.time()
        cube = synthesize_burst(gt, speed, args.frames, args.gain, args.seed, args.angle)
        motion = measure_cube_motion(cube)
        methods = reconstruct_methods(cube, args.group_size, args.reference)
        scores = score_methods(methods, gt, args.gain)
        record = make_record("synthetic", scene, index, motion, scores, frames=args.frames, crop=None,
                             group_size=args.group_size, reference=args.reference, gain=args.gain,
                             seconds=time.time() - start, nominal_drift_px=speed, angle_deg=args.angle,
                             seed=args.seed, simulator="src/synthetic.py (numpy)")
        write_json_atomic(os.path.join(args.out, scene, f"{index:06d}.json"), record)
        panels = [("Ground truth", gt)] + [(m, apply_intensity_rule(methods[m], args.gain)) for m in METHODS]
        os.makedirs(os.path.join(args.out, scene), exist_ok=True)
        _figure(panels, os.path.join(args.out, scene, f"{index:06d}_comparison.png"))
        if args.save_npz:
            os.makedirs(args.save_npz, exist_ok=True)
            np.savez_compressed(os.path.join(args.save_npz, f"{scene}_v{speed:g}.npz"),
                                packed=np.packbits(cube, axis=2), width=gt.shape[1])
        records.append(record)
        print(f"nominal {speed:g} px, measured {motion['drift_px']} px: "
              + ", ".join(f"{m} {scores[m]['psnr_mean']:.2f} dB / {scores[m]['ssim_mean']:.3f}" for m in METHODS))
    return records


if __name__ == "__main__":
    main()
