import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.run_sample_taa import main
from src.burst_io import naive_average
from src.taa import reconstruct_taa


class TAARunnerTests(unittest.TestCase):
    def _sample(self, directory, stem="scene_000002"):
        directory.mkdir(parents=True, exist_ok=True)
        cube = np.random.default_rng(566).integers(0, 2, (5, 8, 8, 3), dtype=np.uint8)
        path = directory / f"{stem}.npz"
        np.savez_compressed(path, packed=np.packbits(cube, axis=2), width=8,
                            crop=np.array([0, 8, 0, 8]))
        Image.fromarray(cube[0] * 255).save(directory / f"{stem}_gt.png")
        return path

    def test_outputs_provenance_and_baseline_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._sample(directory / "inputs")
            out = directory / "outputs"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--samples", str(path), "--out", str(out), "--group-size", "2"])
            with np.load(out / "scene_000002_reconstructions.npz") as data:
                np.testing.assert_allclose(data["aligned"], data["taa_disabled"], atol=1e-7)
                for key in ("naive", "aligned", "taa", "ground_truth"):
                    self.assertEqual(data[key].shape, (8, 8, 3))
                    self.assertEqual(data[key].dtype, np.float32)
                    self.assertTrue(np.isfinite(data[key]).all())
                self.assertEqual(data["confidence_map"].shape, (8, 8))
                self.assertEqual(data["valid_update_mask"].dtype, bool)
                self.assertTrue(np.all(data["history_weights"] <= data["frame_counts"]))
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertNotIn(str(directory), json.dumps(manifest))
            record = manifest["samples"][0]
            self.assertEqual(record["scene_from_filename"], "scene")
            self.assertEqual(record["burst_index_from_filename"], 2)
            self.assertEqual(record["sample_local_frame_indices"], [0, 4])
            self.assertEqual(record["group_frame_counts"], [2, 2, 1])
            self.assertEqual(record["sample_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertLessEqual(record["baseline_control_max_abs_error"], 1e-7)
            self.assertEqual(len(manifest["source_sha256"]), 6)
            for suffix, digest in record["artifacts_sha256"].items():
                artifact = out / f"scene_000002_{suffix}"
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), digest)
            with Image.open(out / "scene_000002_confidence.png") as image:
                self.assertEqual(image.size, (8, 8))
            with Image.open(out / "scene_000002_comparison.png") as image:
                self.assertEqual(image.size, (800, 44))

    def test_ablation_flags_and_single_group_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._sample(directory / "inputs")
            out = directory / "outputs"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--samples", str(path), "--out", str(out), "--no-clamp", "--no-adaptive"])
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertFalse(manifest["taa"]["clamp"])
            self.assertFalse(manifest["taa"]["adaptive"])
            self.assertIsNone(manifest["samples"][0]["last_update_confidence_mean_on_valid"])
            with np.load(out / "scene_000002_reconstructions.npz") as data:
                np.testing.assert_array_equal(data["taa"], data["naive"])
                np.testing.assert_array_equal(data["confidence_map"], 0)

    def test_invalid_arguments_do_not_write_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._sample(directory / "inputs")
            duplicate = self._sample(directory / "duplicate")
            out = directory / "outputs"
            for extra in (["--group-size", "0"], ["--radius", "-1"],
                          ["--motion-scale", "nan"],
                          ["--samples", str(path), str(duplicate)],
                          ["--samples", str(directory / "missing.npz")]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main(["--samples", str(path), "--out", str(out), *extra])
                self.assertFalse(out.exists())

    def test_inputs_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self._sample(directory)
            gt = directory / "scene_000002_gt.png"
            original = gt.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--samples", str(path), "--out", str(directory)])
            self.assertEqual(gt.read_bytes(), original)
            self.assertFalse((directory / "manifest.json").exists())

    def test_many_small_groups_allow_roundoff_but_reject_control_regressions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cube = (np.random.default_rng(5662026).random((256, 16, 24, 3)) < 0.25).astype(np.uint8)
            path = directory / "synthetic_000000.npz"
            np.savez_compressed(path, packed=np.packbits(cube, axis=2), width=24)
            Image.new("RGB", (24, 16)).save(directory / "synthetic_000000_gt.png")
            out = directory / "outputs"
            arguments = ["--samples", str(path), "--out", str(out), "--group-size", "8"]

            def identity(group, reference):
                return naive_average(group), np.zeros((*reference.shape[:2], 2), np.float32)

            with (patch("src.taa.align_warp", side_effect=identity),
                  patch("src.baseline.align_warp", side_effect=identity),
                  contextlib.redirect_stdout(io.StringIO())):
                main(arguments)
                record = json.loads((out / "manifest.json").read_text())["samples"][0]
                self.assertGreater(record["baseline_control_max_abs_error"], 1e-7)
                self.assertLess(record["baseline_control_max_abs_error"], record["baseline_control_atol"])
                for field in ("reconstruction", "frame_counts", "history_weights"):
                    def corrupted_control(cube, group_size, config):
                        result = reconstruct_taa(cube, group_size, config)
                        if not config.clamp and not config.adaptive:
                            getattr(result, field)[:] += 1e-3
                        return result

                    with (self.subTest(field=field),
                          patch("scripts.run_sample_taa.reconstruct_taa", side_effect=corrupted_control),
                          self.assertRaisesRegex(RuntimeError, "alignment/coverage control failed")):
                        main(arguments)
