"""Baseline reconstruction using the shared alignment backbone."""

import numpy as np

from .alignment import align_warp
from .burst_io import naive_average


def aligned_average(cube, group_size=64):
    """Align consecutive groups to the first group's mean and accumulate.

    Returns (reconstruction, frame_counts), both float32. Frame counts have
    shape [H,W] and record valid binary-frame contributions per output pixel.
    The final short group receives its actual frame count as weight. The
    reference stays fixed; there is no history rejection or adaptive blending.
    """
    cube = np.asarray(cube)
    if cube.ndim != 4 or cube.shape[-1] != 3 or min(cube.shape) == 0:
        raise ValueError("cube must have nonempty shape [T,H,W,3]")
    if cube.dtype not in (np.dtype("uint8"), np.dtype("bool")) or cube.max() > 1:
        raise ValueError("cube must be uint8/bool containing only 0 and 1")
    if isinstance(group_size, bool) or not isinstance(group_size, (int, np.integer)) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    first_count = min(group_size, len(cube))
    reference = naive_average(cube[:first_count])
    total = reference * first_count
    counts = np.full(cube.shape[1:3], first_count, dtype=np.float32)
    for start in range(group_size, len(cube), group_size):
        group = cube[start:start + group_size]
        warped, _ = align_warp(group, reference)
        valid = np.all(np.isfinite(warped), axis=2)
        total[valid] += warped[valid] * len(group)
        counts[valid] += len(group)
    return (total / counts[..., None]).astype(np.float32), counts
