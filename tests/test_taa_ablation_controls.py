"""Check that the three ablation ingredients have distinct behavior."""

import unittest
from unittest.mock import patch

import numpy as np

from src.taa import TAAConfig, _blend_with_weights


class TAAComponentTests(unittest.TestCase):
    def setUp(self):
        self.current = np.full((3, 5, 3), 0.2, dtype=np.float32)
        self.history = np.full_like(self.current, 0.75)
        self.motion = np.zeros((3, 5, 2), dtype=np.float32)
        self.motion[..., 1] = 8

    def blend(self, config):
        return _blend_with_weights(
            self.current, self.history, self.motion, 128, 64, config)

    def test_isolated_component_formulas(self):
        # Current bounds are [0.15,0.25], disagreement=0.5, displacement=8.
        cases = (
            (TAAConfig(adaptive=False), 0.25, 1),
            (TAAConfig(clamp=False, disagreement_weighting=False), 0.75, 0.5),
            (TAAConfig(clamp=False, motion_weighting=False), 0.75, np.exp(-5)),
            (TAAConfig(), 0.25, np.exp(-5) / 2),
        )
        for config, previous, retention in cases:
            with self.subTest(config=config):
                merged, confidence, weights, valid = self.blend(config)
                expected_weight = 128 * retention + 64
                expected = (previous * 128 * retention + 0.2 * 64) / expected_weight
                np.testing.assert_allclose(confidence, retention, atol=1e-7, rtol=0)
                np.testing.assert_allclose(weights, expected_weight, rtol=1e-7)
                np.testing.assert_allclose(merged, expected, atol=1e-7, rtol=0)
                self.assertTrue(valid.all())

    def test_confidence_only_ignores_finite_motion_magnitude(self):
        config = TAAConfig(clamp=False, motion_weighting=False)
        expected = self.blend(config)
        self.motion *= -1000
        actual = self.blend(config)
        for before, after in zip(expected, actual):
            np.testing.assert_array_equal(before, after)

    def test_blend_only_does_not_compute_neighborhood_bounds(self):
        # Even completely implausible history must not engage disagreement
        # rejection or image clipping in the isolated displacement ablation.
        with patch("src.taa._neighborhood_bounds", side_effect=AssertionError("bounds used")):
            merged, confidence, _, _ = self.blend(
                TAAConfig(clamp=False, disagreement_weighting=False))
        np.testing.assert_array_equal(confidence, 0.5)
        np.testing.assert_allclose(merged, 0.475, atol=1e-7, rtol=0)

    def test_adaptive_master_switch_overrides_component_flags(self):
        expected = self.blend(TAAConfig(clamp=False, adaptive=False))
        for motion in (True, False):
            for disagreement in (True, False):
                with self.subTest(motion=motion, disagreement=disagreement):
                    actual = self.blend(TAAConfig(
                        clamp=False, adaptive=False, motion_weighting=motion,
                        disagreement_weighting=disagreement))
                    for before, after in zip(expected, actual):
                        np.testing.assert_array_equal(before, after)
        independently_disabled = self.blend(TAAConfig(
            clamp=False, motion_weighting=False, disagreement_weighting=False))
        for before, after in zip(expected, independently_disabled):
            np.testing.assert_array_equal(before, after)

    def test_missing_pixels_remain_missing_in_each_ablation(self):
        self.current[0, 0] = np.nan
        self.motion[0, 1] = np.inf
        for config in (TAAConfig(adaptive=False),
                       TAAConfig(clamp=False, disagreement_weighting=False),
                       TAAConfig(clamp=False, motion_weighting=False), TAAConfig()):
            with self.subTest(config=config), np.errstate(all="raise"):
                merged, confidence, weights, valid = self.blend(config)
                np.testing.assert_array_equal(merged[~valid], self.history[~valid])
                np.testing.assert_array_equal(confidence[~valid], 0)
                np.testing.assert_array_equal(weights[~valid], 128)

    def test_component_switches_require_booleans(self):
        for name in ("motion_weighting", "disagreement_weighting"):
            for value in (None, 0, 1, "false", np.bool_(True)):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    TAAConfig(**{name: value})
