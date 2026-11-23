"""Photon bursts from rendered radiance frames (issue #6, rendered sweep).

scripts/blender/render_frames.py writes one linear 16-bit grayscale PNG per frame. Here those
frames become a binary burst the same way src/synthetic.py does it: radiance = frame / gain,
each binary pixel an independent seeded Bernoulli draw, the LAST frame is the ground-truth pose.
The frame images carry the motion, so there is no resampling step and no interpolation blur.
Not modelled: anything beyond photon shot noise, as in src/synthetic.py.
"""
import glob
import os

import numpy as np
from PIL import Image


def read_linear_png(path):
    """Float image [H, W] in [0, 1] from a 16-bit (or 8-bit) grayscale/RGB PNG."""
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    scale = 65535.0 if array.dtype == np.uint16 else 255.0
    return array.astype(np.float32) / scale


def load_rendered_frames(folder):
    """All f_*.png of a folder, in frame order, as a float array [T, H, W]."""
    paths = sorted(glob.glob(os.path.join(folder, "f_*.png")))
    if not paths:
        raise FileNotFoundError(f"no f_*.png frames in {folder}")
    frames = np.stack([read_linear_png(path) for path in paths])
    return frames


def burst_from_frames(frames, gain=2.7, seed=566, tag=0):
    """Binary cube [T, H, W, 3] uint8 and the ground truth [H, W, 3] (the last frame).

    The three channels get independent draws from the same radiance, so the cube has the
    layout the rest of the pipeline expects.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or len(frames) < 2 or frames.min() < 0 or frames.max() > 1 or gain <= 0:
        raise ValueError("frames must be [T>=2, H, W] in [0, 1] and gain > 0")
    radiance = np.clip(frames / gain, 0, 1)
    rng = np.random.default_rng([seed, int(tag)])
    cube = np.empty(frames.shape + (3,), dtype=np.uint8)
    for t in range(len(frames)):
        cube[t] = rng.random(frames.shape[1:] + (3,), dtype=np.float32) < radiance[t][..., None]
    gt = np.repeat(frames[-1][..., None], 3, axis=2)
    return cube, gt
