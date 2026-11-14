import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.make_taa_failure_gallery import (
    CROPS, FRAMES, GROUP_SIZE, SHAPE, _crop, main, make_scene,
    probability_sequence, render_case, retention_rgb,
)


class TAAFailureGalleryTests(unittest.TestCase):
    def test_fast_motion_has_no_wrap_and_target_is_terminal_instant(self):
        probability, details = probability_sequence("fast_motion")
        self.assertEqual(probability.shape, (FRAMES, *SHAPE, 3))
        self.assertEqual(details["positions_per_frame"][0], 0)
        self.assertEqual(details["positions_per_frame"][-1], 48)
        np.testing.assert_array_equal(probability[0, :, 48:], probability[-1, :, :-48])
        target = _crop(probability[-1], CROPS["fast_motion"])
        exposure = _crop(probability[-GROUP_SIZE:].mean(axis=0), CROPS["fast_motion"])
        self.assertGreater(float(np.mean((exposure - target) ** 2)), 0)

    def test_disocclusion_reveals_fixed_crop_instead_of_wrapping_foreground(self):
        probability, details = probability_sequence("disocclusion")
        np.testing.assert_allclose(probability[0, 24, 32], (0.68, 0.16, 0.13))
        self.assertFalse(np.allclose(probability[-1, 24, 32], (0.68, 0.16, 0.13)))
        # Unoccluded top/bottom rows remain static through the full sequence.
        np.testing.assert_array_equal(probability[0, :12], probability[-1, :12])
        self.assertEqual(details["foreground_left_x"], [24, 96])

    def test_noise_is_seeded_and_does_not_change_latent_target(self):
        first, target, exposure, _ = make_scene("fast_motion", seed=17)
        second, second_target, second_exposure, _ = make_scene("fast_motion", seed=17)
        changed, changed_target, _, _ = make_scene("fast_motion", seed=18)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, changed))
        np.testing.assert_array_equal(target, second_target)
        np.testing.assert_array_equal(target, changed_target)
        np.testing.assert_array_equal(exposure, second_exposure)
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.isin(first, (0, 1)).all())

    def test_invalid_and_rejected_history_have_distinct_display_colors(self):
        confidence = np.array([[0, 0.5, 1], [0, 0.5, 1]], np.float32)
        valid = np.array([[True, True, True], [False, False, False]])
        display = retention_rgb(confidence, valid)
        np.testing.assert_array_equal(display[0], [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
        np.testing.assert_array_equal(display[1], [[1, 0, 1]] * 3)

    def test_panel_renderer_and_invalid_seed(self):
        image = np.zeros((*SHAPE, 3), np.float32)
        confidence = np.zeros(SHAPE, np.float32)
        valid = np.ones(SHAPE, bool)
        images = {name: image for name in ("naive", "aligned", "taa")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "gallery.png"
            render_case("fast_motion", image, images, confidence, valid, output, seed=17)
            with Image.open(output) as rendered:
                self.assertEqual(rendered.size, (1308, 310))
            out = Path(temporary) / "invalid"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--seed", "-1", "--out", str(out)])
            self.assertFalse(out.exists())
