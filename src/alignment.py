"""Coarse-to-fine block alignment of photon frame groups using NumPy.

Align averaged groups, rather than individual noisy binary frames. Both the
baseline and TAA pipeline should call this same implementation.
"""

import numpy as np


def _downsample(image):
    """Average 2x2 cells, retaining odd edges through edge padding."""
    image = np.pad(image, ((0, image.shape[0] % 2),
                           (0, image.shape[1] % 2)), mode="edge")
    return (image[::2, ::2] + image[1::2, ::2]
            + image[::2, 1::2] + image[1::2, 1::2]) * 0.25


def _match(source, reference):
    """Return piecewise-constant backward offsets from reference to source."""
    sources, references = [source], [reference]
    while len(sources) < 3 and min(sources[-1].shape) >= 32:
        sources.append(_downsample(sources[-1]))
        references.append(_downsample(references[-1]))

    field = None
    for level in range(len(sources) - 1, -1, -1):
        moving, fixed = sources[level], references[level]
        h, w = fixed.shape
        if field is None:
            prediction = np.zeros((h, w, 2), dtype=np.float32)
            radius = 4
        else:
            prediction = np.repeat(np.repeat(field, 2, axis=0), 2, axis=1)[:h, :w] * 2
            radius = 2
        field = np.empty((h, w, 2), dtype=np.float32)
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                bh, bw = min(8, h - y), min(8, w - x)
                target = fixed[y:y + bh, x:x + bw]
                py, px = np.rint(np.median(
                    prediction[y:y + bh, x:x + bw], axis=(0, 1))).astype(int)
                # Search nearest the prediction first so ties favor continuity.
                candidates = sorted(
                    ((dy, dx) for dy in range(py - radius, py + radius + 1)
                     for dx in range(px - radius, px + radius + 1)),
                    key=lambda shift: ((shift[0] - py) ** 2 + (shift[1] - px) ** 2,
                                       shift[0] ** 2 + shift[1] ** 2))
                # If no candidate has enough valid overlap, retain the coarse
                # prediction. Resetting to zero would make missing pixels look
                # valid and feed unrelated edge content into the merge.
                best, offset = float("inf"), (py, px)
                for dy, dx in candidates:
                    sy, sx = y + dy, x + dx
                    ty0, tx0 = max(0, -sy), max(0, -sx)
                    ty1, tx1 = min(bh, h - sy), min(bw, w - sx)
                    if (ty1 - ty0) * (tx1 - tx0) < bh * bw / 2 or ty1 <= ty0 or tx1 <= tx0:
                        continue
                    error = float(np.mean((moving[sy + ty0:sy + ty1, sx + tx0:sx + tx1]
                                           - target[ty0:ty1, tx0:tx1]) ** 2))
                    if error < best:
                        best, offset = error, (dy, dx)
                field[y:y + bh, x:x + bw] = offset
    return field


def align_warp(burst, history):
    """Align a binary frame group to an RGB reference image.

    Args:
        burst: Nonempty uint8/bool array [T,H,W,3] containing only 0 and 1.
            A contiguous group of roughly 32-128 frames is a starting point.
        history: Finite RGB image [H,W,3] in [0,1], in the same intensity
            domain as the group mean. Its coordinates define the output grid.

    Returns:
        warped_frame: float32 [H,W,3], the group mean sampled in reference
            coordinates. Pixels outside the source are NaN; callers must exclude
            them from accumulation and maintain per-pixel contribution counts.
        motion_field: float32 [H,W,2], backward sampling offsets (dy,dx).
            warped[y,x] = group_mean[y+dy,x+dx]. These are pixel displacements,
            not velocities. Positive dx samples to the right in the source.

    Uses at most three pyramid levels, 8x8 blocks, integer SSD matching on
    channel means, and nearest-neighbor warping. Motion is constant per block;
    subpixel motion and motion within each group are not modeled. No history
    rejection or adaptive blending is performed here.
    """
    burst, history = np.asarray(burst), np.asarray(history)
    if burst.ndim != 4 or burst.shape[-1] != 3 or min(burst.shape) == 0:
        raise ValueError("burst must have nonempty shape [T,H,W,3]")
    if burst.dtype not in (np.dtype("uint8"), np.dtype("bool")) or burst.max() > 1:
        raise ValueError("burst must be uint8/bool containing only 0 and 1")
    if history.shape != burst.shape[1:]:
        raise ValueError("history must have shape [H,W,3] matching burst")
    if (history.dtype.kind not in "buif" or not np.all(np.isfinite(history))
            or np.any(history < 0) or np.any(history > 1)):
        raise ValueError("history must be finite and in [0,1]")
    source = burst.mean(axis=0, dtype=np.float32)
    reference = history.astype(np.float32)
    field = _match(source.mean(axis=2), reference.mean(axis=2))
    yy, xx = np.indices(source.shape[:2])
    sy = yy + field[..., 0].astype(int)
    sx = xx + field[..., 1].astype(int)
    valid = (sy >= 0) & (sy < source.shape[0]) & (sx >= 0) & (sx < source.shape[1])
    warped = np.full_like(source, np.nan)
    warped[valid] = source[sy[valid], sx[valid]]
    return warped, field
