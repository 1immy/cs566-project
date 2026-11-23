"""Synthetic photon bursts with a known, controlled motion speed (issue #6).

A self-contained numpy simulator, not the lab's visionsim. It turns a ground-truth
image into a burst of binary frames:
  * radiance r = gt / gain, so the fixed evaluation gain (2.7) maps the long-run
    photon average exactly back to gt;
  * the content translates linearly along one direction; the LAST frame is at the
    ground-truth pose, matching the convention of the real data (the ground truth
    is the pose at the end of the burst);
  * each binary pixel is an independent Bernoulli(r) draw (seeded).
Not modelled: sensor noise sources other than photon shot noise, rolling shutter,
disocclusion, parallax, non-translational motion, scene lighting changes.
"""
import numpy as np


def shift_image(image, sy, sx):
    """Resample image so that out[y, x] = image[y + sy, x + sx] (bilinear, edge clamped)."""
    h, w = image.shape[:2]
    ys = np.clip(np.arange(h, dtype=np.float64) + sy, 0, h - 1)
    xs = np.clip(np.arange(w, dtype=np.float64) + sx, 0, w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def synthesize_burst(gt, drift_px, frames=1024, gain=2.7, seed=566, angle_deg=0.0):
    """Binary cube [frames, H, W, 3] uint8 whose content drifts by drift_px over the burst.

    gt: float image [H, W, 3] in [0, 1], the pose at the last frame.
    angle_deg: direction of motion, 0 = along +x.
    """
    gt = np.asarray(gt, dtype=np.float32)
    if gt.ndim != 3 or gt.shape[2] != 3 or gt.min() < 0 or gt.max() > 1:
        raise ValueError("gt must be an RGB image [H,W,3] in [0,1]")
    if frames < 2 or drift_px < 0 or gain <= 0:
        raise ValueError("need frames >= 2, drift_px >= 0 and gain > 0")
    radiance = np.clip(gt / gain, 0, 1)
    rng = np.random.default_rng([seed, int(round(drift_px * 100)), int(round(angle_deg * 100))])
    direction = (np.sin(np.deg2rad(angle_deg)), np.cos(np.deg2rad(angle_deg)))   # (dy, dx)
    cube = np.empty((frames,) + gt.shape, dtype=np.uint8)
    for t in range(frames):
        behind = drift_px * (frames - 1 - t) / (frames - 1)    # 0 at the last frame
        frame = shift_image(radiance, behind * direction[0], behind * direction[1])
        cube[t] = rng.random(gt.shape, dtype=np.float32) < frame
    return cube
