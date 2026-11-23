"""Estimate global translation between 256-frame windows inside one photon cube.

Usage:
  python scripts/measure_burst_motion.py path/to/train/kitchen1 0 1 10
(scene dir, then one or more burst indices). Prints, per burst, the shift of each
later window relative to the first one, as (dy, dx, peak) in pixels, plus the
window-to-window shift. A low peak (< ~0.05) means the estimate is unreliable.
For a whole-dataset table and slow/medium/fast bins use scripts/bin_scenes_by_motion.py.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.burst_io import load_burst
from src.motion import REAL_CROP as CROP, WINDOW as WIN, phase_corr


def main():
    scene = sys.argv[1]
    indices = [int(i) for i in sys.argv[2:]] or [0]
    for idx in indices:
        wins = []
        for s in range(0, 1024, WIN):
            cube = load_burst(scene, idx, frames=slice(s, s + WIN), crop=CROP)
            wins.append(cube.mean(axis=(0, 3), dtype=np.float32))
        vs_first = [phase_corr(wins[0], w) for w in wins[1:]]
        step = [phase_corr(wins[i], wins[i + 1])[:2] for i in range(len(wins) - 1)]
        print(f"burst {idx:06d} vs first window: {vs_first}  window-to-window (dy,dx): {step}")


if __name__ == "__main__":
    main()
