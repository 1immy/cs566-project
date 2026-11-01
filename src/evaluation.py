"""Track 4 reconstruction metrics, preserving evaluate(reconstruction, ground_truth).

Both arguments are matching RGB float images in [0,1], already mapped into
one common intensity domain by the caller. SSIM uses a 7x7 uniform window,
reflect padding, population moments, and C1=(0.01)^2, C2=(0.03)^2.
"""
import numpy as np


# Fixed for Track 4 comparisons: never fit gain to each method or scene.
DEFAULT_EVALUATION_GAIN = 2.7


def apply_intensity_rule(reconstruction, gain=DEFAULT_EVALUATION_GAIN):
    """Map photon-domain RGB into [0,1] with one shared fixed gain.
    Apply to each method separately using the *same* gain. Ground truth is
    already in [0,1] and must not be gain-adjusted. Inputs are not modified.
    """
    if isinstance(gain, (bool, np.bool_)) or not np.isscalar(gain):
        raise ValueError("gain must be a finite positive scalar")
    try:
        gain = float(gain)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("gain must be a finite positive scalar") from error
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be a finite positive scalar")
    image = np.asarray(reconstruction)
    if (image.ndim != 3 or image.shape[-1] != 3 or not all(image.shape[:2])
            or image.dtype.kind not in "buif" or not np.isfinite(image).all()
            or np.any(image < 0) or np.any(image > 1)):
        raise ValueError("reconstruction must be finite RGB [H,W,3] in [0,1]")
    return np.clip(image.astype(np.float64) * gain, 0, 1).astype(np.float32)


def _local_mean(image, window=7):
    radius = window // 2
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0, dtype=np.float64), axis=1),
                      ((1, 0), (1, 0)))
    return (integral[window:, window:] - integral[:-window, window:]
            - integral[window:, :-window] + integral[:-window, :-window]) / (window * window)


def evaluate_channels(reconstruction, ground_truth):
    """Return per-RGB-channel scores and their arithmetic means."""
    rec = np.asarray(reconstruction, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if rec.shape != gt.shape or rec.ndim != 3 or rec.shape[2] != 3 or min(rec.shape[:2]) < 2:
        raise ValueError("images must have matching nonempty RGB [H,W,3] shapes, H,W>=2")
    if not np.isfinite(rec).all() or not np.isfinite(gt).all():
        raise ValueError("images must contain only finite values")
    if np.any((rec < 0) | (rec > 1)) or np.any((gt < 0) | (gt > 1)):
        raise ValueError("images must be in [0,1]; apply common exposure mapping before scoring")
    psnr, ssim = [], []
    for channel in range(3):
        a, b = rec[..., channel], gt[..., channel]
        mse = float(np.mean((a - b)**2))
        psnr.append(float('inf') if mse == 0 else float(-10 * np.log10(mse)))
        mu_a, mu_b = _local_mean(a), _local_mean(b)
        var_a = np.maximum(0, _local_mean(a*a) - mu_a*mu_a)
        var_b = np.maximum(0, _local_mean(b*b) - mu_b*mu_b)
        cov = _local_mean(a*b) - mu_a*mu_b
        c1, c2 = .01**2, .03**2
        value = ((2*mu_a*mu_b + c1)*(2*cov + c2)
                 / ((mu_a*mu_a + mu_b*mu_b + c1)*(var_a + var_b + c2)))
        ssim.append(float(np.mean(value)))
    return {"psnr": dict(zip(("r", "g", "b", "mean"), [*psnr, float(np.mean(psnr))])),
            "ssim": dict(zip(("r", "g", "b", "mean"), [*ssim, float(np.mean(ssim))]))}


def evaluate(reconstruction, ground_truth):
    """Return (mean_per_channel_PSNR_dB, mean_per_channel_SSIM)."""
    result = evaluate_channels(reconstruction, ground_truth)
    return result["psnr"]["mean"], result["ssim"]["mean"]
