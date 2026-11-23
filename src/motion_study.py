"""Shared reconstruction, scoring and record format for the motion-speed study (#6, #24, #25).

One record (a JSON file) per scene and burst, written by scripts/run_full_scale.py
(real bursts) and scripts/run_synthetic_sweep.py (synthetic bursts), read by
scripts/analyze_motion_bins.py. Every method sees the same cube, the same
reference group and the same fixed gain; nothing is fitted to the ground truth.
"""
import hashlib
import json
import os

from src.baseline import aligned_average
from src.burst_io import naive_average
from src.evaluation import DEFAULT_EVALUATION_GAIN, apply_intensity_rule, evaluate_channels
from src.taa import TAAConfig, reconstruct_taa

SCHEMA = 1
METHODS = ("naive", "aligned", "taa")


def reconstruct_methods(cube, group_size=64, reference="last"):
    """Naive average, block-aligned average and default TAA on one cube."""
    return {
        "naive": naive_average(cube),
        "aligned": aligned_average(cube, group_size, reference=reference)[0],
        "taa": reconstruct_taa(cube, group_size, TAAConfig(), reference=reference).reconstruction,
    }


def score_methods(methods, gt, gain=DEFAULT_EVALUATION_GAIN):
    """Per-method PSNR/SSIM after the shared fixed-gain intensity rule."""
    scores = {}
    for name, image in methods.items():
        result = evaluate_channels(apply_intensity_rule(image, gain), gt)
        scores[name] = {f"{metric}_{channel}": value
                        for metric in ("psnr", "ssim")
                        for channel, value in result[metric].items()}
    return scores


def config_key(frames, crop, group_size, reference, gain):
    """Hash of everything that changes the numbers, used to decide if a record is reusable."""
    payload = json.dumps({"schema": SCHEMA, "frames": frames, "crop": list(crop) if crop else None,
                          "group_size": group_size, "reference": reference, "gain": gain},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_record(source, scene, burst, motion, scores, *, frames, crop, group_size,
                reference, gain, seconds, **extra):
    record = {"schema": SCHEMA, "source": source, "scene": scene, "burst": int(burst),
              "config_key": config_key(frames, crop, group_size, reference, gain),
              "frames": frames, "crop": list(crop) if crop else None, "group_size": group_size,
              "reference": reference, "gain": gain, "motion": motion, "scores": scores,
              "seconds": round(float(seconds), 1)}
    record.update(extra)
    return record


def write_json_atomic(path, data):
    """Write via a temporary file and rename, so an interrupted run never leaves a half file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    os.replace(tmp, path)


def read_record(path):
    """The record at path, or None if missing, unreadable or from another schema."""
    try:
        with open(path, encoding="utf-8") as file:
            record = json.load(file)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return None
    return record


def load_records(directory):
    """All valid records under directory (recursive), sorted by source, scene, burst."""
    records = []
    for root, _, names in os.walk(directory):
        for name in names:
            if name.endswith(".json") and not name.endswith(".tmp"):
                record = read_record(os.path.join(root, name))
                if record and "scores" in record and "motion" in record:
                    records.append(record)
    return sorted(records, key=lambda r: (r["source"], r["scene"], r["burst"]))
