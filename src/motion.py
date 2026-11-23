"""Global motion measurement and speed binning for photon bursts (issues #23, #25).

Motion is the global translation between 256-frame window means, found by
phase correlation with sub-pixel refinement. A single binary frame is too noisy
to correlate, so windows are used. This is the logic of
scripts/measure_burst_motion.py, moved here so the binning, the full-scale
runner and the analysis share one definition of "motion speed".

Definitions (all in pixels of the measured region):
  drift_px          length of the shift between the last window and the first
  speed_px_per_256  drift_px divided by the number of 256-frame steps between them
  path_px           sum of the window-to-window shift lengths
  min_peak          smallest correlation peak; below about 0.05 the estimate is noisy
"""
import numpy as np

WINDOW = 256
# Central region used for real 800x800 bursts. x bounds are multiples of 8.
REAL_CROP = (150, 650, 152, 648)
LABELS3 = ("slow", "medium", "fast")


def phase_corr(a, b):
    """Shift (dy, dx, peak) of b relative to a, with parabolic sub-pixel refinement."""
    a = a - a.mean()
    b = b - b.mean()
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


def window_means(cube, window=WINDOW):
    """Grayscale mean image of each full window of `window` frames."""
    count = len(cube) // window
    if count < 2:
        raise ValueError(f"need at least {2 * window} frames, got {len(cube)}")
    return [cube[i * window:(i + 1) * window].mean(axis=(0, 3), dtype=np.float32)
            for i in range(count)]


def summarize(windows, window=WINDOW):
    """Motion summary dict from a list of window-mean images."""
    vs_first = [phase_corr(windows[0], w) for w in windows[1:]]
    steps = [phase_corr(windows[i], windows[i + 1]) for i in range(len(windows) - 1)]
    dy, dx, _ = vs_first[-1]
    drift = float(np.hypot(dy, dx))
    return {
        "window": window,
        "windows": len(windows),
        "dy": dy,
        "dx": dx,
        "drift_px": round(drift, 2),
        "speed_px_per_256": round(drift / (len(windows) - 1) * (256 / window), 2),
        "path_px": round(float(sum(np.hypot(s[0], s[1]) for s in steps)), 2),
        "min_peak": min(s[2] for s in vs_first),
        "reliable": bool(min(s[2] for s in vs_first) >= 0.05),
    }


def default_region(cube):
    """REAL_CROP for full-size 800x800 bursts, otherwise the whole image."""
    y0, y1, x0, x1 = REAL_CROP
    return REAL_CROP if cube.shape[1] >= y1 and cube.shape[2] >= x1 else None


def measure_cube_motion(cube, crop=None, window=WINDOW):
    """Motion summary of an in-memory cube [T,H,W,3]; crop=(y0,y1,x0,x1) in pixels."""
    if crop is not None:
        y0, y1, x0, x1 = crop
        cube = cube[:, y0:y1, x0:x1]
    return summarize(window_means(cube, window), window)


def measure_burst_motion(scene_dir, index, frames=1024, crop=REAL_CROP, window=WINDOW):
    """Motion summary of one on-disk burst, loading one window at a time."""
    from src.burst_io import load_burst
    means = []
    for start in range(0, frames - window + 1, window):
        cube = load_burst(scene_dir, index, frames=slice(start, start + window), crop=crop)
        means.append(cube.mean(axis=(0, 3), dtype=np.float32))
    if len(means) < 2:
        raise ValueError(f"need at least {2 * window} frames, got {frames}")
    return summarize(means, window)


def tertile_edges(values, n_bins=3):
    """Quantile bin edges (n_bins - 1 values) so each bin holds about the same count."""
    values = np.asarray(values, dtype=float)
    if len(values) < n_bins:
        raise ValueError(f"need at least {n_bins} measurements to form {n_bins} bins, got {len(values)}")
    return [round(float(q), 2) for q in np.quantile(values, np.arange(1, n_bins) / n_bins)]


def assign_bin(value, edges, labels=LABELS3):
    """Label for value: edges[i-1] <= value < edges[i]; len(labels) == len(edges) + 1."""
    if len(labels) != len(edges) + 1:
        raise ValueError("need one more label than edges")
    return labels[int(np.searchsorted(np.asarray(edges, dtype=float), value, side="right"))]
