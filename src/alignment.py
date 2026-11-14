"""Coarse-to-fine block alignment of photon frame groups using NumPy.

Align averaged groups, rather than individual noisy binary frames. Both the
baseline and TAA pipeline should call this same implementation.
"""

import numpy as np


def _fractional_patch(image, y, x, height, width, dy, dx):
    """Bilinearly sample a patch whose required neighbors are all in bounds."""
    iy, ix = int(np.floor(dy)), int(np.floor(dx))
    fy, fx = dy - iy, dx - ix
    sy, sx = y + iy, x + ix
    patch = image[sy:sy + height, sx:sx + width]
    if fx:
        patch = (1 - fx) * patch + fx * image[sy:sy + height, sx + 1:sx + width + 1]
    if fy:
        below = image[sy + 1:sy + height + 1, sx:sx + width]
        if fx:
            below = ((1 - fx) * below
                     + fx * image[sy + 1:sy + height + 1, sx + 1:sx + width + 1])
        patch = (1 - fy) * patch + fy * below
    return patch


def _refine(source, reference, y, x, height, width, offset, integer_error):
    """Search half/quarter/eighth pixels near the integer winner on fixed support.

    All fractional candidates use exactly the same reference pixels. Require
    at least half a block with a one-pixel source halo, otherwise keep integer
    motion. Exact integer matches are retained without interpolation.
    """
    dy, dx = offset
    if integer_error == 0:
        return offset
    h, w = source.shape
    y0, x0 = max(0, 1 - y - dy), max(0, 1 - x - dx)
    y1, x1 = min(height, h - 1 - y - dy), min(width, w - 1 - x - dx)
    if y1 <= y0 or x1 <= x0 or (y1 - y0) * (x1 - x0) < height * width / 2:
        return offset
    target = reference[y + y0:y + y1, x + x0:x + x1]

    def error(candidate):
        patch = _fractional_patch(source, y + y0, x + x0, y1 - y0, x1 - x0, *candidate)
        return float(np.mean((patch - target) ** 2))

    best = error(offset)
    refined = (float(dy), float(dx))
    for step in (0.5, 0.25, 0.125):
        center = refined
        # Prefer smaller adjustments for exact ties; never worsen the score
        # relative to the integer winner on this common support.
        adjustments = sorted(((a, b) for a in (-step, 0, step) for b in (-step, 0, step)),
                             key=lambda pair: pair[0] ** 2 + pair[1] ** 2)
        for ay, ax in adjustments:
            candidate = (center[0] + ay, center[1] + ax)
            if abs(candidate[0] - dy) > 0.5 or abs(candidate[1] - dx) > 0.5:
                continue
            score = error(candidate)
            if score < best:
                best, refined = score, candidate
    return refined


def _block_confidence(source, reference, y, x, height, width, offset, error, candidates):
    """Heuristic quality from integer match ambiguity, residual, and texture."""
    if not np.isfinite(error):
        return 0.0
    dy, dx = offset
    h, w = source.shape
    y0, x0 = max(0, -y - dy), max(0, -x - dx)
    y1, x1 = min(height, h - y - dy), min(width, w - x - dx)
    target = reference[y + y0:y + y1, x + x0:x + x1]
    patch = source[y + dy + y0:y + dy + y1, x + dx + x0:x + dx + x1]
    alternatives = []
    for cy, cx, _ in candidates:
        if max(abs(cy - dy), abs(cx - dx)) <= 1:
            continue
        sy, sx = y + cy + y0, x + cx + x0
        if sy < 0 or sx < 0 or sy + target.shape[0] > h or sx + target.shape[1] > w:
            continue
        # Compare alternatives on the winner's identical reference support;
        # a smaller/easier border patch must not inflate match separation.
        candidate = source[sy:sy + target.shape[0], sx:sx + target.shape[1]]
        alternatives.append(float(np.mean((candidate - target) ** 2)))
    if not alternatives:
        return 0.0
    second = min(alternatives)
    # A near-tied distant match is ambiguous, including repetitive patterns.
    separation = max(0.0, second - error) / max(second, 1e-12)
    texture = min(float(np.var(target)), float(np.var(patch)))
    agreement = texture / (texture + error + 1e-12)
    coverage = target.size / (height * width)
    return float(np.clip(separation * agreement * coverage, 0, 1))


def _downsample(image):
    """Average 2x2 cells, retaining odd edges through edge padding."""
    image = np.pad(image, ((0, image.shape[0] % 2),
                           (0, image.shape[1] % 2)), mode="edge")
    return (image[::2, ::2] + image[1::2, ::2]
            + image[::2, 1::2] + image[1::2, 1::2]) * 0.25


def _match(source, reference):
    """Return backward offsets and block confidence on the reference grid."""
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
        confidence = np.zeros((h, w), dtype=np.float32)
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                bh, bw = min(8, h - y), min(8, w - x)
                target = fixed[y:y + bh, x:x + bw]
                py, px = np.rint(np.median(
                    prediction[y:y + bh, x:x + bw], axis=(0, 1))).astype(int)
                # Search nearest the prediction first so ties favor continuity.
                search = {(dy, dx) for dy in range(py - radius, py + radius + 1)
                          for dx in range(px - radius, px + radius + 1)}
                predicted_height = max(0, min(bh, h - y - py) - max(0, -y - py))
                predicted_width = max(0, min(bw, w - x - px) - max(0, -x - px))
                if level == 0 and predicted_height * predicted_width >= bh * bw / 2:
                    # Coarse aliasing can predict several pixels of motion
                    # for a subpixel shift. Keep small motions searchable,
                    # but never replace a geometrically missing prediction
                    # with unrelated in-bounds edge content.
                    search.update((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3))
                candidates = sorted(
                    search,
                    key=lambda shift: ((shift[0] - py) ** 2 + (shift[1] - px) ** 2,
                                       shift[0] ** 2 + shift[1] ** 2, shift[0], shift[1]))
                # If no candidate has enough valid overlap, retain the coarse
                # prediction. Resetting to zero would make missing pixels look
                # valid and feed unrelated edge content into the merge.
                best, offset = float("inf"), (py, px)
                scores = []
                for dy, dx in candidates:
                    sy, sx = y + dy, x + dx
                    ty0, tx0 = max(0, -sy), max(0, -sx)
                    ty1, tx1 = min(bh, h - sy), min(bw, w - sx)
                    if (ty1 - ty0) * (tx1 - tx0) < bh * bw / 2 or ty1 <= ty0 or tx1 <= tx0:
                        continue
                    error = float(np.mean((moving[sy + ty0:sy + ty1, sx + tx0:sx + tx1]
                                           - target[ty0:ty1, tx0:tx1]) ** 2))
                    scores.append((dy, dx, error))
                    if error < best:
                        best, offset = error, (dy, dx)
                if level == 0:
                    confidence[y:y + bh, x:x + bw] = _block_confidence(
                        moving, fixed, y, x, bh, bw, offset, best, scores)
                    if np.isfinite(best):
                        offset = _refine(moving, fixed, y, x, bh, bw, offset, best)
                field[y:y + bh, x:x + bw] = offset
    return field, confidence


def _warp(source, field):
    """Backward bilinear sampling; exact last-row/column integer samples are valid."""
    h, w = source.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    sy, sx = yy + field[..., 0], xx + field[..., 1]
    valid = (sy >= 0) & (sy <= h - 1) & (sx >= 0) & (sx <= w - 1)
    warped = np.full_like(source, np.nan)
    y0 = np.floor(sy[valid]).astype(int)
    x0 = np.floor(sx[valid]).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (sy[valid] - y0).astype(np.float32)[:, None], (sx[valid] - x0).astype(np.float32)[:, None]
    top = (1 - fx) * source[y0, x0] + fx * source[y0, x1]
    bottom = (1 - fx) * source[y1, x0] + fx * source[y1, x1]
    warped[valid] = (1 - fy) * top + fy * bottom
    return warped, valid


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
        motion_field: float32 [H,W,2], fractional backward offsets (dy,dx).
            warped[y,x] samples group_mean at (y+dy,x+dx) bilinearly.
            These are pixel displacements,
            not velocities. Positive dx samples to the right in the source.

    Uses at most three pyramid levels, 8x8 integer block matching on channel
    means, local subpixel SSD refinement down to 1/8 pixel, and bilinear warping.
    Motion is constant per block; motion within each group is not modeled.
    No history rejection or adaptive blending is performed here. For optional
    block-match diagnostics, use align_warp_with_confidence with the same inputs.
    """
    warped, field, _ = align_warp_with_confidence(burst, history)
    return warped, field


def align_warp_with_confidence(burst, history):
    """Return (warped_frame, motion_field, alignment_confidence).

    Inputs and the first two arrays have exactly the align_warp contract and
    arithmetic. alignment_confidence is float32 [H,W] in [0,1], constant per
    finest-level block except invalid warped pixels are zero. It combines the
    integer best/distant-alternative error gap, texture versus residual, and
    overlap fraction. It does not depend explicitly on displacement magnitude.

    This is a heuristic match-quality score, not calibrated probability,
    visibility, statistical sample size, or Track 3's history-retention weight.
    Zero also marks textureless/ambiguous or unmatchable blocks. A confident
    local optimum can still be wrong. No score threshold rejects observations.
    Track 3 must explicitly choose how to consume this optional diagnostic.
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
    field, confidence = _match(source.mean(axis=2), reference.mean(axis=2))
    warped, valid = _warp(source, field)
    confidence[~valid] = 0
    return warped, field, confidence
