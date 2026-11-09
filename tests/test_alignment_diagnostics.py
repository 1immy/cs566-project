import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.run_alignment_diagnostics import main


class AlignmentDiagnosticsTests(unittest.TestCase):
    def test_partial_reference_and_exported_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cube = np.random.default_rng(566).integers(0, 2, (3, 8, 8, 3), dtype=np.uint8)
            path = directory / "scene_000000.npz"
            np.savez_compressed(path, packed=np.packbits(cube, axis=2), width=8)
            out = directory / "outputs"
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = main(["--samples", str(path), "--group-size", "2", "--out", str(out)])
            record = manifest["samples"][0]
            self.assertEqual(record["reference_sample_local_frame_indices"], [2, 2])
            self.assertEqual(len(record["groups"]), 1)
            group = record["groups"][0]
            self.assertEqual(group["sample_local_frame_indices"], [0, 1])
            archive = out / group["archive"]
            self.assertEqual(group["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            with np.load(archive) as data:
                self.assertEqual(data["motion_field"].shape, (8, 8, 2))
                confidence = data["alignment_confidence"]
                self.assertEqual(confidence.shape, (8, 8))
                self.assertEqual(confidence.dtype, np.float32)
                self.assertTrue(((confidence >= 0) & (confidence <= 1)).all())
                np.testing.assert_array_equal(confidence[~np.isfinite(data["warped_frame"]).all(axis=2)], 0)
                np.testing.assert_array_equal(data["reference_frame_indices"], [2, 2])
            self.assertNotIn(str(directory), str(manifest))

    def test_single_group_exports_no_self_confidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "scene_000000.npz"
            np.savez_compressed(path, packed=np.zeros((1, 8, 1, 3), dtype=np.uint8), width=8)
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = main(["--samples", str(path), "--out", str(directory / "outputs")])
            self.assertEqual(manifest["samples"][0]["groups"], [])

    def test_empty_sample_rejected_without_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "empty.npz"
            np.savez_compressed(path, packed=np.zeros((0, 8, 1, 3), dtype=np.uint8), width=8)
            out = directory / "outputs"
            with self.assertRaises(ValueError):
                main(["--samples", str(path), "--out", str(out)])
            self.assertFalse(out.exists())
