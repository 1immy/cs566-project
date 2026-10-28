"""Burst loading for the Single Photon Challenge reconstruction data.

On-disk format (verified on train/kitchen1/000000.npy):
  NNNNNN.npy  uint8, shape (1024, 800, 100, 3)
              = 1024 binary frames, 800 x 800 px, 3 color channels (RGB),
                bit-packed along the WIDTH axis (8 pixels per byte, np.packbits order).
  NNNNNN.png  800 x 800 RGB ground-truth image for that burst.
Each scene folder holds many numbered (npy, png) pairs.

Public API (the Track 1 interface everyone codes against):
    load_burst(scene_dir, index=0, frames=None, crop=None) -> cube[T, H, W, 3] uint8 in {0, 1}
    load_gt(scene_dir, index=0, crop=None)                 -> gt[H, W, 3] float32 in [0, 1]
"""
import os
import numpy as np
from PIL import Image

FULL_H, FULL_W = 800, 800


def _paths(scene_dir, index):
    stem = f"{index:06d}"
    return os.path.join(scene_dir, stem + ".npy"), os.path.join(scene_dir, stem + ".png")


def _check_crop(crop):
    """crop = (y0, y1, x0, x1) in pixels. x0 and x1 must be multiples of 8 (packed axis)."""
    y0, y1, x0, x1 = crop
    if x0 % 8 or x1 % 8:
        raise ValueError("crop x bounds must be multiples of 8 (width is bit-packed)")
    return y0, y1, x0, x1


def load_burst(scene_dir, index=0, frames=None, crop=None):
    """Load one photon cube as unpacked binary frames.

    frames: None for all 1024, an int N for the first N, or a slice.
    crop:   None or (y0, y1, x0, x1); x bounds multiples of 8.
    Memory: the full cube unpacks to ~2 GB, so use frames= / crop= when prototyping.
    """
    npy, _ = _paths(scene_dir, index)
    a = np.load(npy, mmap_mode="r")                     # (T, H, W/8, 3), packed
    if isinstance(frames, int):
        frames = slice(0, frames)
    if frames is not None:
        a = a[frames]
    if crop is not None:
        y0, y1, x0, x1 = _check_crop(crop)
        a = a[:, y0:y1, x0 // 8:x1 // 8, :]
    return np.unpackbits(np.asarray(a), axis=2)         # (T, H, W, 3) in {0,1}


def load_gt(scene_dir, index=0, crop=None):
    _, png = _paths(scene_dir, index)
    gt = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt = gt[y0:y1, x0:x1]
    return gt


def naive_average(cube):
    """Baseline: mean over time. Returns (H, W, 3) float32 in [0, 1]."""
    return cube.mean(axis=0, dtype=np.float32)


def list_bursts(scene_dir):
    return sorted(int(f[:6]) for f in os.listdir(scene_dir) if f.endswith(".npy"))


def load_sample(npz_path):
    """Load a sample written by scripts/make_sample.py -> cube[T, H, W, 3] in {0,1}."""
    d = np.load(npz_path)
    return np.unpackbits(d["packed"], axis=2)[:, :, : int(d["width"]), :]
