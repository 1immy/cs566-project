"""Shared protocol, provenance, and parity checks for the Track 3 study."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import run_taa_study as study
from src.burst_io import load_burst, load_gt, naive_average
from src.evaluation import evaluate_channels
from src.taa import TAAResult, reconstruct_taa


class TAAStudyTests(unittest.TestCase):
    def _sample(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "test_scene_000002.npz"
        cube = np.random.default_rng(566).integers(0, 2, (5, 8, 8, 3), dtype=np.uint8)
        np.savez_compressed(path, packed=np.packbits(cube, axis=2), width=8,
                            crop=np.array([16, 24, 24, 32]))
        gt = np.random.default_rng(567).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        Image.fromarray(gt).save(path.with_name(path.stem + "_gt.png"))
        return path

    def test_sample_scores_use_shared_gain_and_record_all_component_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._sample(directory / "inputs")
            out = directory / "outputs"
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = study.main(["--samples", str(path), "--out", str(out),
                                       "--group-size", "2", "--save-arrays"])
            self.assertEqual(manifest, json.loads((out / "manifest.json").read_text()))
            self.assertIn("full kitchen1 0,1,10 excluded from this run", manifest["data_status"])
            self.assertIn("does not evaluate full kitchen1", (out / "findings.md").read_text())
            self.assertNotIn(str(directory), json.dumps(manifest))
            record = manifest["samples"][0]
            self.assertEqual((record["scene"], record["burst"]), ("test_scene", 2))
            self.assertEqual(record["crop_y0_y1_x0_x1"], [16, 24, 24, 32])
            self.assertEqual(record["frame_indices_inclusive"], [0, 4])
            self.assertEqual(record["frame_index_scope"], "sample-local")
            self.assertEqual(record["reference_frames_inclusive"], [4, 4])
            self.assertEqual(record["last_update_frames_inclusive"], [2, 3])
            self.assertEqual(record["input_sha256"][path.name],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(record["baseline_control"]["coverage_equal"])
            for name, digest in record["artifacts_sha256"].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
            with (out / "results.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["method"] for r in rows}, {"naive", "aligned", *study.CONFIGS})
            self.assertEqual(len(rows), 7)
            with np.load(out / "test_scene_000002_reconstructions.npz") as archive:
                for row in rows:
                    self.assertEqual(float(row["gain"]), 2.7)
                    self.assertEqual(row["reference"], "last")
                    expected = evaluate_channels(
                        np.clip(2.7 * archive[row["method"]], 0, 1), archive["ground_truth"])
                    for metric in ("psnr", "ssim"):
                        for channel in ("r", "g", "b", "mean"):
                            self.assertAlmostEqual(float(row[f"{metric}_{channel}"]),
                                                   expected[metric][channel])
                self.assertEqual(archive["valid_update_mask"].dtype, bool)
                self.assertEqual(archive["confidence_map"].shape, (8, 8))
            self.assertFalse(manifest["configs"]["blend_only"]["disagreement_weighting"])
            self.assertFalse(manifest["configs"]["confidence_only"]["motion_weighting"])

    def test_full_scene_uses_all_frames_and_passes_crop_to_both_loaders(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene = directory / "kitchen1"
            scene.mkdir()
            path = scene / "000010.npy"
            # A sparse file provides a real full-burst header without a large allocation.
            packed = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                               shape=(1024, 800, 100, 3))
            del packed
            Image.new("RGB", (800, 800), (128, 64, 192)).save(scene / "000010.png")
            out = directory / "outputs"

            def fixture_reconstruction(cube, group_size, reference):
                self.assertEqual(cube.shape, (1024, 8, 8, 3))
                self.assertEqual((group_size, reference), (64, "last"))
                image = np.full(cube.shape[1:], 0.2, dtype=np.float32)
                counts = np.full(cube.shape[1:3], 1024, dtype=np.float32)
                result = TAAResult(image, np.ones_like(counts), counts, counts,
                                   np.ones_like(counts, dtype=bool))
                return ({name: image for name in ("naive", "aligned", *study.CONFIGS)},
                        {name: result for name in study.CONFIGS}, counts, {"coverage_equal": True})

            with (patch.object(study, "load_burst", wraps=load_burst) as burst_loader,
                  patch.object(study, "load_gt", wraps=load_gt) as gt_loader,
                  patch.object(study, "reconstruct", side_effect=fixture_reconstruction),
                  contextlib.redirect_stdout(io.StringIO())):
                manifest = study.main(["--scene-dir", str(scene), "--indices", "10",
                                       "--crop", "16", "24", "24", "32", "--out", str(out)])
            burst_loader.assert_called_once_with(scene, index=10, crop=[16, 24, 24, 32])
            gt_loader.assert_called_once_with(scene, index=10, crop=[16, 24, 24, 32])
            self.assertEqual(manifest["data_status"], "full-burst development study; not held out")
            record = manifest["samples"][0]
            self.assertEqual(record["shape"], [1024, 8, 8, 3])
            self.assertEqual(record["frame_index_scope"], "original burst")
            self.assertEqual(record["frame_indices_inclusive"], [0, 1023])
            self.assertEqual(record["reference_frames_inclusive"], [960, 1023])
            self.assertEqual(record["last_update_frames_inclusive"], [896, 959])
            self.assertEqual(record["crop_y0_y1_x0_x1"], [16, 24, 24, 32])
            self.assertFalse((out / "kitchen1_000010_reconstructions.npz").exists())

    def test_incomplete_packed_burst_is_rejected_before_loading_or_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene = directory / "kitchen1"
            scene.mkdir()
            np.save(scene / "000000.npy", np.zeros((256, 8, 1, 3), np.uint8))
            Image.new("RGB", (8, 8)).save(scene / "000000.png")
            out = directory / "outputs"
            with patch.object(study, "load_burst") as loader:
                with self.assertRaisesRegex(ValueError, "expected full packed shape"):
                    study.main(["--scene-dir", str(scene), "--indices", "0", "--out", str(out)])
                loader.assert_not_called()
            self.assertFalse(out.exists())

    def test_invalid_cli_and_symlinked_outputs_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._sample(directory / "inputs")
            out = directory / "outputs"
            for extra in (["--gain", "nan"], ["--group-size", "0"],
                          ["--crop", "0", "8", "0", "8"],
                          ["--samples", str(path), str(path)]):
                with (self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()),
                      self.assertRaises(SystemExit)):
                    study.main(["--samples", str(path), "--out", str(out), *extra])
                self.assertFalse(out.exists())
            out.mkdir()
            gt_path = path.with_name(path.stem + "_gt.png")
            original = gt_path.read_bytes()
            (out / f"{path.stem}_comparison.png").symlink_to(gt_path)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                study.main(["--samples", str(path), "--out", str(out)])
            self.assertEqual(gt_path.read_bytes(), original)
            self.assertFalse((out / "manifest.json").exists())

    def test_reconstruction_parity_checks_detect_coverage_and_control_corruption(self):
        cube = np.random.default_rng(566).integers(0, 2, (5, 8, 8, 3), dtype=np.uint8)

        def identity(group, reference):
            return naive_average(group), np.zeros((*reference.shape[:2], 2), np.float32)

        with (patch("src.taa.align_warp", side_effect=identity),
              patch("src.baseline.align_warp", side_effect=identity)):
            for field, disabled in (("reconstruction", True), ("history_weights", True),
                                    ("frame_counts", True), ("frame_counts", False)):
                def corrupt(cube, group_size, config, *, reference):
                    result = reconstruct_taa(cube, group_size, config, reference=reference)
                    is_disabled = not config.clamp and not config.adaptive
                    if is_disabled == disabled:
                        getattr(result, field)[:] += 0.01
                    return result

                message = "coverage mismatch" if field == "frame_counts" else "disabled TAA"
                with (self.subTest(field=field, disabled=disabled),
                      patch.object(study, "reconstruct_taa", side_effect=corrupt),
                      self.assertRaisesRegex(RuntimeError, message)):
                    study.reconstruct(cube, 2, "last")


if __name__ == "__main__":
    unittest.main()
