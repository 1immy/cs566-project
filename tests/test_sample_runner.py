import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.run_sample_baselines import main


class SampleRunnerTests(unittest.TestCase):
    def _make_sample(self, directory, stem="sample"):
        directory.mkdir(parents=True, exist_ok=True)
        cube = np.random.default_rng(566).integers(0, 2, (3, 8, 8, 3), dtype=np.uint8)
        path = directory / f"{stem}.npz"
        np.savez_compressed(path, packed=np.packbits(cube, axis=2), width=8,
                            crop=np.array([0, 8, 0, 8]))
        Image.fromarray(cube[0] * 255).save(directory / f"{stem}_gt.png")
        return path, cube

    def test_runner_outputs_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, cube = self._make_sample(directory / "inputs")
            out = directory / "outputs"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--samples", str(path), "--out", str(out), "--group-size", "64"])
            with np.load(out / "sample_reconstructions.npz") as result:
                np.testing.assert_array_equal(result["naive"], result["aligned"])
                np.testing.assert_array_equal(result["frame_counts"], len(cube))
                self.assertEqual(result["aligned"].dtype, np.float32)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["samples"][0]["sample_local_frame_indices"], [0, 2])
            self.assertEqual(len(manifest["samples"][0]["sample_sha256"]), 64)
            self.assertEqual(len(manifest["source_sha256"]), 4)
            self.assertNotIn(str(directory), json.dumps(manifest))
            for suffix in ("naive", "aligned", "gt"):
                with Image.open(out / f"sample_{suffix}.png") as image:
                    self.assertEqual(image.size, (8, 8))

    def test_duplicate_stems_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, _ = self._make_sample(directory / "one")
            second, _ = self._make_sample(directory / "two")
            out = directory / "outputs"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--samples", str(first), str(second), "--out", str(out)])
            self.assertFalse(out.exists())

    def test_input_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, _ = self._make_sample(directory)
            gt = directory / "sample_gt.png"
            original = gt.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--samples", str(path), "--out", str(directory)])
            self.assertEqual(gt.read_bytes(), original)
            self.assertFalse((directory / "manifest.json").exists())
