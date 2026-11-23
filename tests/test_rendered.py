"""Rendered-frame bursts and the rendered sweep (issue #6)."""
import contextlib
import io
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import run_rendered_sweep
from src.rendered import burst_from_frames, load_rendered_frames, read_linear_png


def write_frames(folder, speed, frames=512, size=48):
    """Linear 16-bit gray frames of a blocky texture panning `speed` px, last frame = GT pose."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(3)
    base = np.kron(rng.random((size // 4, size // 4)), np.ones((4, 4)))
    wide = np.concatenate([base, base], axis=1)
    for t in range(frames):
        shift = int(round(speed * (frames - 1 - t) / (frames - 1)))
        frame = wide[:, shift:shift + size] * 0.9
        Image.fromarray((frame * 65535).astype(np.uint16)).save(os.path.join(folder, f"f_{t + 1:04d}.png"))


class RenderedTests(unittest.TestCase):
    def test_read_png_is_linear_float(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "a.png")
            Image.fromarray(np.full((4, 4), 32768, dtype=np.uint16)).save(path)
            self.assertAlmostEqual(float(read_linear_png(path)[0, 0]), 32768 / 65535, places=5)

    def test_burst_matches_expected_mean_and_gt(self):
        frames = np.full((600, 16, 16), 0.54, dtype=np.float32)
        frames[-1] = 0.9
        cube, gt = burst_from_frames(frames, gain=2.7, seed=1)
        self.assertEqual(cube.shape, (600, 16, 16, 3))
        self.assertAlmostEqual(float(cube[:-1].mean()), 0.54 / 2.7, delta=0.01)
        self.assertTrue(np.allclose(gt, 0.9))
        again, _ = burst_from_frames(frames, gain=2.7, seed=1)
        self.assertTrue(np.array_equal(cube, again))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            burst_from_frames(np.full((1, 4, 4), 0.5))
        with self.assertRaises(ValueError):
            burst_from_frames(np.full((5, 4, 4), 1.5))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                load_rendered_frames(root)

    def test_sweep_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:
            for speed in (0, 8):
                write_frames(os.path.join(root, "data", f"frames_v{speed}"), speed)
            out = os.path.join(root, "out")
            with contextlib.redirect_stdout(io.StringIO()):
                records = run_rendered_sweep.main(["--frames-root", os.path.join(root, "data"),
                                                   "--speeds", "0", "8", "--out", out])
            self.assertEqual([r["source"] for r in records], ["rendered", "rendered"])
            self.assertTrue(os.path.isfile(os.path.join(out, "rendered_pan", "000001.json")))
            self.assertGreater(records[1]["motion"]["drift_px"], records[0]["motion"]["drift_px"])

    def test_missing_folder_is_an_error(self):
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_rendered_sweep.main(["--frames-root", root, "--speeds", "5"])


if __name__ == "__main__":
    unittest.main()
