import unittest

import numpy as np

from scripts.characterize_taa import oracle_warp, summarize


class TAACharacterizationTests(unittest.TestCase):
    def test_oracle_translation_uses_backward_offsets_and_excludes_wrapped_border(self):
        registered = np.random.default_rng(566).integers(0, 2, (4, 5, 7, 3), dtype=np.uint8)
        reference = registered.mean(axis=0, dtype=np.float32)
        for shift in ((0, 2), (0, -2), (1, -2), (-1, 2)):
            with self.subTest(shift=shift):
                source = np.roll(registered, shift, axis=(1, 2))
                warped, motion = oracle_warp(source, shift)
                y, x = np.indices((5, 7))
                valid = ((y + shift[0] >= 0) & (y + shift[0] < 5)
                         & (x + shift[1] >= 0) & (x + shift[1] < 7))
                np.testing.assert_array_equal(warped[valid], reference[valid])
                self.assertTrue(np.isnan(warped[~valid]).all())
                np.testing.assert_array_equal(motion[..., 0], shift[0])
                np.testing.assert_array_equal(motion[..., 1], shift[1])

    def test_statistics_separate_trial_uncertainty_from_spatial_bias_and_variance(self):
        result = summarize([[1, 3], [-1, 1]])
        self.assertEqual(result["trial_mse"], [5, 1])
        self.assertEqual(result["mse"], 3)
        self.assertAlmostEqual(result["mse_standard_error"], 2)
        self.assertEqual(result["mean_signed_error"], 1)
        self.assertEqual(result["empirical_squared_bias"], 2)
        self.assertEqual(result["empirical_variance"], 1)
