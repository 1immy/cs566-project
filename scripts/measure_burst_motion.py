"""Estimate global translation between 256-frame windows inside one photon cube.

Usage:
  python scripts/measure_burst_motion.py "Z:/.../train/kitchen1" 0 1 10
(scene dir, then one or more burst indices). Prints, per burst, the shift of each
later window relative to the first one, as (dy, dx, peak) in pixels, plus the
window-to-window shift. A low peak (< ~0.05) means the estimate is unreliable.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from burst_io import load_burst

CROP = (150, 650, 152, 648)   # central region; x bounds must be multiples of 8
WIN = 256


def phase_corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fft2(a * w) * np.conj(np.fft.fft2(b * w))
    F /= np.abs(F) + 1e-9
    r = np.fft.ifft2(F).real
    y, x = np.unravel_index(r.argmax(), r.shape)
    h, wd = r.shape

    def refine(l, c, rr):
        d = l - 2 * c + rr
        return 0.0 if d == 0 else 0.5 * (l - rr) / d

    dy = (y - h if y > h // 2 else y) + refine(r[(y - 1) % h, x], r[y, x], r[(y + 1) % h, x])
    dx = (x - wd if x > wd // 2 else x) + refine(r[y, (x - 1) % wd], r[y, x], r[y, (x + 1) % wd])
    return round(float(dy), 1), round(float(dx), 1), round(float(r.max()), 2)


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
