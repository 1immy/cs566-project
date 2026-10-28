"""Write a small committable sample into data/samples/ from the full dataset.

Usage (run where the dataset lives):
  python scripts/make_sample.py --scene "Z:/UW-Madison/cs566-single-photon/train/kitchen1" \
      --out data/samples --indices 0 1 --frames 256 --size 128

Stores each burst still bit-packed (np.savez_compressed) plus the cropped GT PNG,
so a sample is ~1.5 MB. Load with src/burst_io.load_sample().
"""
import argparse, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from burst_io import load_burst, load_gt, FULL_H, FULL_W


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", default="data/samples")
    p.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--size", type=int, default=128, help="square crop, multiple of 8")
    p.add_argument("--y0", type=int, default=None)
    p.add_argument("--x0", type=int, default=None)
    a = p.parse_args()
    s = a.size
    y0 = (FULL_H - s) // 2 if a.y0 is None else a.y0
    x0 = ((FULL_W - s) // 2 // 8) * 8 if a.x0 is None else a.x0
    crop = (y0, y0 + s, x0, x0 + s)
    name = os.path.basename(os.path.normpath(a.scene))
    os.makedirs(a.out, exist_ok=True)
    for i in a.indices:
        cube = load_burst(a.scene, i, frames=a.frames, crop=crop)
        gt = load_gt(a.scene, i, crop=crop)
        stem = os.path.join(a.out, f"{name}_{i:06d}")
        np.savez_compressed(stem + ".npz", packed=np.packbits(cube, axis=2),
                            crop=np.array(crop), width=s)
        Image.fromarray((gt * 255).round().astype(np.uint8)).save(stem + "_gt.png")
        print(stem, cube.shape, f"{os.path.getsize(stem + '.npz') / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
