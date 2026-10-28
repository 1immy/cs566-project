"""Seeded Track 3 characterization, not a held-out benchmark or GT tuning.

Run from the repository root: venv/bin/python scripts/characterize_taa.py
Uses known Bernoulli probabilities and shared cached warps for every method.
Oracle identity/translation removes registration error to isolate rejection;
the real-aligner experiment separately measures registration error. Production
alignment, grouping, and reconstruction functions are not modified.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.alignment import align_warp
from src.baseline import aligned_average
from src.burst_io import naive_average
from src.taa import TAAConfig, reconstruct_taa


CONFIGS = {
    "disabled": TAAConfig(clamp=False, adaptive=False),
    "clamp_only": TAAConfig(adaptive=False),
    # This still uses neighborhood disagreement AND displacement weighting.
    "adaptive_no_clip": TAAConfig(clamp=False),
    "full": TAAConfig(),
}


def oracle_warp(group, shift):
    """Exact backward integer sampling, with missing borders (no wraparound)."""
    mean = naive_average(group)
    y, x = np.indices(mean.shape[:2])
    sy, sx = y + shift[0], x + shift[1]
    valid = (sy >= 0) & (sy < mean.shape[0]) & (sx >= 0) & (sx < mean.shape[1])
    warped = np.full_like(mean, np.nan)
    warped[valid] = mean[sy[valid], sx[valid]]
    motion = np.broadcast_to(np.array(shift, np.float32), (*mean.shape[:2], 2)).copy()
    return warped, motion


def reconstruct_methods(cube, group_size, shifts, configs, real_alignment=False):
    """Feed identical warps to baseline and TAA, including identical support."""
    reference = naive_average(cube[:group_size])
    groups = [cube[start:start + group_size] for start in range(0, len(cube), group_size)]
    assert len(groups) == len(shifts) and shifts[0] == (0, 0)
    warps = [(align_warp(group, reference) if real_alignment else oracle_warp(group, shift))
             for group, shift in zip(groups[1:], shifts[1:])]
    with patch("src.baseline.align_warp", side_effect=warps):
        baseline, counts = aligned_average(cube, group_size)
    images = {"naive": naive_average(cube), "aligned": baseline}
    for name, config in configs.items():
        with patch("src.taa.align_warp", side_effect=warps) as align:
            result = reconstruct_taa(cube, group_size, config)
        for call in align.call_args_list:
            np.testing.assert_array_equal(call.args[1], reference)
        np.testing.assert_array_equal(result.frame_counts, counts)
        if name == "disabled":
            tolerance = max(1e-7, (len(groups) - 1) * float(np.finfo(np.float32).eps))
            np.testing.assert_allclose(result.reconstruction, baseline, atol=tolerance, rtol=0)
            np.testing.assert_array_equal(result.history_weights, counts)
        images[name] = result.reconstruction
    return images, warps


def summarize(errors):
    """MSE uncertainty across independent trials, plus empirical decomposition.

    ddof=0 makes MSE = empirical squared bias + empirical variance exactly.
    Finite-trial squared bias contains sampling error; it is not an unbiased
    estimate of population squared bias. SE uses independent per-trial MSEs.
    """
    errors = np.asarray(errors, dtype=np.float64)
    trial_mse = np.mean(errors ** 2, axis=tuple(range(1, errors.ndim)))
    return {
        "mse": float(trial_mse.mean()),
        "mse_standard_error": float(trial_mse.std(ddof=1) / np.sqrt(len(errors))),
        "mean_signed_error": float(errors.mean()),
        "empirical_squared_bias": float(np.mean(errors.mean(axis=0) ** 2)),
        "empirical_variance": float(np.mean(errors.var(axis=0))),
        "trial_mse": trial_mse.tolist(),
    }


def scene_probability(kind, rate, shape=(16, 24)):
    y, x = np.indices(shape)
    if kind == "uniform":
        pattern = np.zeros(shape)
    elif kind == "edge":
        pattern = np.where(x < shape[1] // 2, -1, 1)
    elif kind == "checkerboard":
        pattern = 2 * ((x + y) % 2) - 1
    else:
        raise ValueError(kind)
    contrast = min(rate, 1 - rate) / 2
    return np.repeat((rate + contrast * pattern)[..., None], 3, axis=2)


def static_trials(seed, trials):
    records = []
    configs = dict(CONFIGS)
    for name, changes in (
            ("radius_0", {"radius": 0}), ("radius_2", {"radius": 2}),
            ("margin_0", {"clamp_margin": 0}), ("margin_0.1", {"clamp_margin": 0.1}),
            ("rejection_0.03", {"rejection_scale": 0.03}),
            ("rejection_0.3", {"rejection_scale": 0.3})):
        configs[name] = replace(CONFIGS["full"], **changes)
    for scene_id, kind in enumerate(("uniform", "edge", "checkerboard")):
        for rate_id, rate in enumerate((0.02, 0.25, 0.75)):
            probability = scene_probability(kind, rate)
            errors = {}
            for trial in range(trials):
                rng = np.random.default_rng(np.random.SeedSequence([seed, 0, scene_id, rate_id, trial]))
                cube = (rng.random((256, *probability.shape)) < probability).astype(np.uint8)
                for group_size in (8, 32, 64):
                    images, _ = reconstruct_methods(
                        cube, group_size, [(0, 0)] * (256 // group_size), configs)
                    for name, image in images.items():
                        errors.setdefault((group_size, name), []).append(image - probability)
            for (group_size, name), samples in errors.items():
                records.append({"experiment": "static_oracle", "scene": kind, "rate": rate,
                                "group_size": group_size, "method": name,
                                "config": asdict(configs[name]) if name in configs else None,
                                **summarize(samples)})
            print(f"static: {kind}, rate={rate}, {trials} trials", flush=True)
    return records


def translation_trials(seed, trials, real_alignment=False):
    records = []
    group_size = 64 if real_alignment else 32
    shape = (96, 96) if real_alignment else (32, 64)
    # Same interior for all directions, displacements, configurations, and trials.
    support = np.s_[24:72, 24:72] if real_alignment else np.s_[2:30, 18:46]
    texture_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    probability = 0.05 + 0.6 * texture_rng.random((*shape, 3))
    errors, residuals = {}, {}
    configs = dict(CONFIGS)
    # With margin=1, disagreement is exactly zero. This isolates motion weighting
    # without adding a production switch or approximating infinity with a scale.
    configs["motion_only"] = TAAConfig(clamp=False, clamp_margin=1)
    for trial in range(trials):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2, trial]))
        registered = (rng.random((256, *probability.shape)) < probability).astype(np.uint8)
        for displacement in (0, 8, -8, 16, -16):
            offsets = np.rint(np.linspace(0, displacement, 256 // group_size)).astype(int)
            shifts = [(0, int(dx)) for dx in offsets]
            cube = np.concatenate([np.roll(registered[start:start + group_size], shift, axis=(1, 2))
                                   for start, shift in zip(range(0, 256, group_size), shifts)])
            images, warps = reconstruct_methods(cube, group_size, shifts, configs, real_alignment)
            for name, image in images.items():
                errors.setdefault((displacement, name), []).append((image - probability)[support])
            for (_, motion), shift in zip(warps, shifts[1:]):
                residual = motion[support] - np.asarray(shift)
                residuals.setdefault(displacement, []).append(float(np.mean(np.any(residual != 0, axis=2))))
    for (displacement, name), samples in errors.items():
        records.append({"experiment": "translation_real" if real_alignment else "translation_oracle",
                        "last_dx": displacement, "group_size": group_size, "method": name,
                        "config": asdict(configs[name]) if name in configs else None,
                        "wrong_motion_fraction": float(np.mean(residuals[displacement])),
                        **summarize(samples)})
    print(f"translation ({'real' if real_alignment else 'oracle'}): {trials} trials", flush=True)
    return records


def stale_history_trials(seed, trials):
    """A favorable control: deliberately wrong first group, correct current data.

    The target is current p=0.2; first-group p=0.8 is injected contamination.
    This is NOT a static-scene or first-reference-time benchmark.
    """
    errors = {}
    for trial in range(trials):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3, trial]))
        probability = np.full((128, 16, 24, 3), 0.2)
        probability[:64] = 0.8
        cube = (rng.random(probability.shape) < probability).astype(np.uint8)
        images, _ = reconstruct_methods(cube, 64, [(0, 0), (0, 0)], CONFIGS)
        for name, image in images.items():
            errors.setdefault(name, []).append(image.astype(np.float64) - 0.2)
    return [{"experiment": "stale_history", "method": name,
             "config": asdict(CONFIGS[name]) if name in CONFIGS else None,
             **summarize(samples)} for name, samples in errors.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=5662026)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("results/sample_taa/peer_review/synthetic.json"))
    args = parser.parse_args(argv)
    if args.seed < 0 or args.trials < 2:
        parser.error("--seed must be nonnegative and --trials must be at least 2")
    records = static_trials(args.seed, args.trials)
    records += translation_trials(args.seed, args.trials)
    records += translation_trials(args.seed, args.trials, real_alignment=True)
    records += stale_history_trials(args.seed, args.trials)
    report = {
        "purpose": "Track 3 characterization only; no SPC GT, PSNR/SSIM, or parameter fitting",
        "seed": args.seed, "trials": args.trials,
        "seed_sequences": {"static": "[seed,0,scene_id,rate_id,trial]",
                           "texture": "[seed,1]", "translation": "[seed,2,trial]",
                           "stale": "[seed,3,trial]"},
        "static": "256 Bernoulli frames, 16x24 RGB, identity oracle, all pixels; paired inputs across settings",
        "translation_oracle": "256 frames, 32x64 RGB, groups of 32, crop y=2:30,x=18:46; exact integer warp",
        "translation_real": "256 frames, 96x96 RGB, groups of 64, crop y=24:72,x=24:72; unchanged Track 2 aligner",
        "translation": "rounded linear offsets from 0 to last_dx; paired registered noise across displacements; rolled source, no wrapped warp support",
        "stale": "128 frames, groups of 64, 16x24 RGB; first-group p=0.8 is contamination, current target p=0.2",
        "reference": "fixed first group for all methods; cached warps shared by aligned baseline and TAA",
        "control_atol": "max(1e-7, (group_count-1) * float32 epsilon), rtol=0; counts and weights exact",
        "adaptive_no_clip": "neighborhood-disagreement AND displacement weighting, without image clipping",
        "decomposition": "empirical variance uses ddof=0; empirical squared bias includes finite-trial noise",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "scripts/characterize_taa.py", "src/taa.py", "src/baseline.py", "src/alignment.py", "src/burst_io.py")},
        "results": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} measurements to {args.out}")


if __name__ == "__main__":
    main()
