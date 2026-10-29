"""Track 3: neighborhood history clamping and motion-adaptive blending.

All images stay in photon-detection-probability space. Confidence is a
heuristic fraction of history to retain, not a calibrated probability that
the alignment is correct. No input arrays are modified.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .alignment import align_warp
from .burst_io import naive_average


@dataclass(frozen=True)
class TAAConfig:
    """Prototype settings, fixed before evaluation rather than tuned to GT.

    Neighborhood bounds are per-channel min/max, expanded by clamp_margin.
    Adaptive confidence combines disagreement outside those bounds with
    backward displacement magnitude (pixels, not velocity).
    Disable both clamp and adaptive for a frame-weighted baseline control.
    """

    radius: int = 1
    clamp_margin: float = 0.05
    rejection_scale: float = 0.1
    motion_scale: float = 8.0
    clamp: bool = True
    adaptive: bool = True

    def __post_init__(self):
        if (isinstance(self.radius, (bool, np.bool_))
                or not isinstance(self.radius, (int, np.integer))
                or self.radius < 0):
            raise ValueError("radius must be a nonnegative integer")
        for name in ("clamp_margin", "rejection_scale", "motion_scale"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite real number")
        if not 0 <= self.clamp_margin <= 1:
            raise ValueError("clamp_margin must be in [0,1]")
        if self.rejection_scale <= 0 or self.motion_scale <= 0:
            raise ValueError("rejection_scale and motion_scale must be positive")
        if not isinstance(self.clamp, bool) or not isinstance(self.adaptive, bool):
            raise ValueError("clamp and adaptive must be bool")


def _validate_inputs(warped_frame, history, motion_field):
    frame, history, motion = map(np.asarray, (warped_frame, history, motion_field))
    if history.ndim != 3 or history.shape[-1] != 3 or min(history.shape) == 0:
        raise ValueError("history must have nonempty shape [H,W,3]")
    if (history.dtype.kind not in "buif" or not np.all(np.isfinite(history))
            or np.any(history < 0) or np.any(history > 1)):
        raise ValueError("history must be finite and in [0,1]")
    if frame.shape != history.shape or frame.dtype.kind not in "buif":
        raise ValueError("warped_frame must be a real RGB array matching history")
    finite = np.isfinite(frame)
    if np.any(frame[finite] < 0) or np.any(frame[finite] > 1):
        raise ValueError("finite warped_frame values must be in [0,1]")
    if motion.shape != (*history.shape[:2], 2) or motion.dtype.kind not in "buif":
        raise ValueError("motion_field must be a real array of shape [H,W,2]")
    valid = np.all(finite, axis=2) & np.all(np.isfinite(motion), axis=2)
    return frame.astype(np.float32), history.astype(np.float32), motion, valid


def _neighborhood_bounds(frame, valid, radius, margin):
    """Ignore invalid RGB pixels and truncate neighborhoods at image edges."""
    h, w = valid.shape
    padding = ((radius, radius), (radius, radius), (0, 0))
    low_source = np.pad(np.where(valid[..., None], frame, np.inf), padding,
                        constant_values=np.inf)
    high_source = np.pad(np.where(valid[..., None], frame, -np.inf), padding,
                         constant_values=-np.inf)
    lower = np.full_like(frame, np.inf)
    upper = np.full_like(frame, -np.inf)
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            np.minimum(lower, low_source[y:y + h, x:x + w], out=lower)
            np.maximum(upper, high_source[y:y + h, x:x + w], out=upper)
    return np.maximum(lower - margin, 0), np.minimum(upper + margin, 1)


def _blend_with_weights(warped_frame, history, motion_field, history_weight,
                        current_weight, config):
    """Shared kernel; the burst caller owns accumulation weights explicitly."""
    frame, history, motion, valid = _validate_inputs(warped_frame, history, motion_field)
    clamped = history.copy()
    if config.clamp or config.adaptive:
        lower, upper = _neighborhood_bounds(frame, valid, config.radius, config.clamp_margin)
        # At a valid center the neighborhood always contains at least itself.
        clamped[valid] = np.clip(history[valid], lower[valid], upper[valid])

    confidence = np.zeros(history.shape[:2], dtype=np.float32)
    if config.adaptive:
        disagreement = np.max(np.abs(history[valid] - clamped[valid]), axis=1)
        displacement = np.hypot(motion[valid, 0].astype(np.float64),
                                motion[valid, 1].astype(np.float64))
        # Use float64 for intermediate ratios; extreme finite inputs may
        # underflow confidence to zero, which is a complete history reset.
        with np.errstate(over="ignore", under="ignore"):
            confidence[valid] = (np.exp(-disagreement.astype(np.float64)
                                        / config.rejection_scale)
                                 / (1 + displacement / config.motion_scale))
    else:
        confidence[valid] = 1

    weights = np.broadcast_to(np.asarray(history_weight, dtype=np.float32), valid.shape).copy()
    merged = history.copy()
    retained = weights[valid] * confidence[valid]
    weights[valid] = retained + current_weight
    previous = clamped if config.clamp else history
    merged[valid] = ((previous[valid] * retained[:, None]
                      + frame[valid] * current_weight) / weights[valid, None])
    # Invalid updates preserve history and its weight exactly, with confidence
    # zero to mark absent evidence. Never evaluate 0 * NaN in the blend.
    return merged, confidence, weights, valid


def reject_and_blend(warped_frame, history, motion_field):
    """Merge two equal nominal-weight estimates using the fixed Track 3 API.

    warped_frame: RGB [H,W,3] in [0,1], already in history coordinates.
        Nonfinite pixels are missing observations, including partial RGB NaNs.
    history: finite RGB [H,W,3] in [0,1]. No implicit mutable state is kept.
    motion_field: [H,W,2] backward offsets (dy,dx) in pixels. Nonfinite
        offsets also invalidate the corresponding observation.

    Returns float32 (merged_frame [H,W,3], confidence_map [H,W]).
    Confidence is the fraction of prior history weight retained in [0,1].
    It is NOT the final blend coefficient: for these equal nominal weights,
    the history coefficient is confidence / (1 + confidence).
    Missing observations preserve history and return confidence zero.
    Use reconstruct_taa for a burst with unequal or accumulated frame weights.
    """
    merged, confidence, _, _ = _blend_with_weights(
        warped_frame, history, motion_field, 1.0, 1.0, TAAConfig())
    return merged, confidence


@dataclass
class TAAResult:
    """Burst result and diagnostics, all on the first group's reference grid.

    confidence_map describes the LAST update only; zero means absent evidence
    or rejected history. valid_update_mask disambiguates those cases. For a
    single group both are zero because no temporal comparison was made.
    frame_counts counts geometrically valid frames BEFORE history rejection,
    so it matches the baseline's coverage. history_weights are discounted
    accumulation weights, not statistical effective sample sizes.
    """

    reconstruction: np.ndarray
    confidence_map: np.ndarray
    frame_counts: np.ndarray
    history_weights: np.ndarray
    valid_update_mask: np.ndarray


def reconstruct_taa(cube, group_size=64, config=None):
    """Align frame groups to a fixed reference, then update TAA history.

    Uses the unmodified Track 2 aligner and exactly its baseline's first-group
    reference and group partition. Only the rejection/blending stage differs.
    The three-argument reject_and_blend and this burst wrapper use the same
    kernel; explicit per-pixel history weights let the wrapper handle partial
    groups and missing observations without changing the agreed API.

    A valid group of n frames retains confidence * previous_weight units of
    history, adds n units of current evidence, and updates the weighted mean.
    Missing observations change neither history nor its weight. Both effects
    disabled (TAAConfig(clamp=False, adaptive=False)) recover aligned_average
    within float32 rounding, including partial groups and invalid borders.
    """
    cube = np.asarray(cube)
    if cube.ndim != 4 or cube.shape[-1] != 3 or min(cube.shape) == 0:
        raise ValueError("cube must have nonempty shape [T,H,W,3]")
    if cube.dtype not in (np.dtype("uint8"), np.dtype("bool")) or cube.max() > 1:
        raise ValueError("cube must be uint8/bool containing only 0 and 1")
    if (isinstance(group_size, (bool, np.bool_))
            or not isinstance(group_size, (int, np.integer)) or group_size <= 0):
        raise ValueError("group_size must be a positive integer")
    if config is None:
        config = TAAConfig()
    if not isinstance(config, TAAConfig):
        raise ValueError("config must be a TAAConfig")

    first_count = min(group_size, len(cube))
    reference = naive_average(cube[:first_count])
    history = reference.copy()
    counts = np.full(cube.shape[1:3], first_count, dtype=np.float32)
    weights = counts.copy()
    confidence = np.zeros(cube.shape[1:3], dtype=np.float32)
    valid = np.zeros(cube.shape[1:3], dtype=bool)
    for start in range(group_size, len(cube), group_size):
        group = cube[start:start + group_size]
        warped, motion = align_warp(group, reference)
        history, confidence, weights, valid = _blend_with_weights(
            warped, history, motion, weights, len(group), config)
        counts[valid] += len(group)
    return TAAResult(history, confidence, counts, weights, valid)
