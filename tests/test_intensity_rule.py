"""Regression checks for the shared Track 4 intensity rule."""
import unittest
import numpy as np
from src.evaluation import (DEFAULT_EVALUATION_GAIN, apply_intensity_rule,
                            evaluate_channels)
from scripts.run_evaluation import _map_intensity

class IntensityRuleTests(unittest.TestCase):
    def test_default_and_clipping(self):
        self.assertEqual(DEFAULT_EVALUATION_GAIN, 2.7)
        image = np.array([[[0.0, 0.2, 0.5], [0.1, 0.3, 1.0]]], dtype=np.float32)
        np.testing.assert_allclose(apply_intensity_rule(image),
                                   np.clip(image * 2.7, 0, 1), atol=1e-7)

    def test_same_mapping_for_each_method(self):
        # No method-specific adjustment: the same value maps identically.
        value = np.full((8, 8, 3), 0.2, dtype=np.float32)
        outputs = [value.copy() for _ in ('naive', 'aligned', 'taa')]
        mapped = [_map_intensity(output, DEFAULT_EVALUATION_GAIN) for output in outputs]
        for result in mapped[1:]:
            np.testing.assert_array_equal(result, mapped[0])
        gt = np.full_like(value, 0.54)
        for result in mapped:
            self.assertAlmostEqual(evaluate_channels(result, gt)['psnr']['mean'],
                                   evaluate_channels(mapped[0], gt)['psnr']['mean'])

    def test_inputs_unmodified(self):
        image = np.full((4, 5, 3), 0.4, dtype=np.float32)
        original = image.copy()
        image.flags.writeable = False
        result = apply_intensity_rule(image)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(result.dtype, np.float32)

    def test_rejects_invalid_gains_and_inputs(self):
        image = np.ones((3, 3, 3), dtype=np.float32) * .2
        for gain in (-1, 0, float('nan'), float('inf'), True, [2.7]):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                apply_intensity_rule(image, gain)
        for invalid in (image[..., :2], image + 2, image * float('nan')):
            with self.assertRaises(ValueError):
                apply_intensity_rule(invalid)

if __name__ == '__main__':
    unittest.main()
