"""Baseline reconstruction using the shared alignment backbone."""

import numpy as np

from .alignment import align_warp
from .burst_io import naive_average


def aligned_average(cube, group_size=64, *, reference="last"):
    """Align consecutive groups to a fixed first/last group mean and accumulate.

    reference: "last" (default) or "first". Groups always start at frame 0;
    the reference is the first or last of these existing groups. Thus a short
    final group remains short, and input frames are never reversed/regrouped.

    Returns (reconstruction, frame_counts), both float32. Frame counts have
    shape [H,W] and record valid binary-frame contributions per output pixel.
    The final short group receives its actual frame count as weight. The
    reference stays fixed; there is no history rejection or adaptive blending.
    The reference group contributes once without resampling, and other groups
    are accumulated in their original order. "first" reproduces the original
    reference policy and is required for comparisons with first-reference TAA.
    """
    cube = np.asarray(cube)
    if cube.ndim != 4 or cube.shape[-1] != 3 or min(cube.shape) == 0:
        raise ValueError("cube must have nonempty shape [T,H,W,3]")
    if cube.dtype not in (np.dtype("uint8"), np.dtype("bool")) or cube.max() > 1:
        raise ValueError("cube must be uint8/bool containing only 0 and 1")
    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if not isinstance(reference, str) or reference not in ("first", "last"):
        raise ValueError('reference must be "first" or "last"')
    reference_start = 0 if reference == "first" else ((len(cube) - 1) // group_size) * group_size
    reference_group = cube[reference_start:reference_start + group_size]
    reference_image = naive_average(reference_group)
    reference_count = len(reference_group)
    total = reference_image * reference_count
    counts = np.full(cube.shape[1:3], reference_count, dtype=np.float32)
    for start in range(0, len(cube), group_size):
        if start == reference_start:
            continue
        group = cube[start:start + group_size]
        warped, _ = align_warp(group, reference_image)
        valid = np.all(np.isfinite(warped), axis=2)
        total[valid] += warped[valid] * len(group)
        counts[valid] += len(group)
    return (total / counts[..., None]).astype(np.float32), counts
