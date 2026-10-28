import inspect
import unittest

import numpy as np

from src.taa import TAAConfig, _blend_with_weights, reject_and_blend


class HistoryRejectionTests(unittest.TestCase):
    def setUp(self):
        self.current = np.full((5, 7, 3), 0.2, dtype=np.float32)
        self.history = self.current.copy()
        self.motion = np.zeros((5, 7, 2), dtype=np.float32)

    def test_fixed_api_and_consistent_history(self):
        self.assertEqual(list(inspect.signature(reject_and_blend).parameters),
                         ["warped_frame", "history", "motion_field"])
        merged, confidence = reject_and_blend(self.current, self.history, self.motion)
        np.testing.assert_allclose(merged, self.history)
        np.testing.assert_array_equal(confidence, 1)
        self.assertEqual(merged.dtype, np.float32)
        self.assertEqual(confidence.dtype, np.float32)
        self.assertEqual(confidence.shape, (5, 7))

    def test_clamp_removes_stale_bright_history(self):
        self.history[:] = 0.9
        merged, confidence = reject_and_blend(self.current, self.history, self.motion)
        self.assertTrue(np.all((merged >= 0.2) & (merged <= 0.25)))
        self.assertLess(float(confidence.max()), 0.01)
        self.assertLess(np.mean((merged - self.current) ** 2),
                        np.mean(((self.history + self.current) / 2 - self.current) ** 2))

    def test_neighborhood_is_spatial_and_per_channel(self):
        self.current[:] = [0.2, 0.4, 0.6]
        self.current[2, 2, 0] = 0.8
        self.history[:] = [0.7, 0.1, 0.9]
        result, _, _, _ = _blend_with_weights(
            self.current, self.history, self.motion, 1, 1,
            TAAConfig(clamp_margin=0, adaptive=False))
        # Red history is plausible beside the bright red neighbor. Green and
        # blue use their own bounds; a distant red pixel is clamped to 0.2.
        np.testing.assert_allclose(result[2, 3], [0.45, 0.4, 0.6])
        np.testing.assert_allclose(result[0, 6], [0.2, 0.4, 0.6])

    def test_motion_reduces_history_weight_symmetrically(self):
        self.history[:] = 0.24  # Within the default margin, so no rejection.
        stationary, stable_confidence = reject_and_blend(self.current, self.history, self.motion)
        for dy, dx in ((0, 8), (0, -8), (8, 0)):
            with self.subTest(offset=(dy, dx)):
                self.motion[:] = [dy, dx]
                moving, confidence = reject_and_blend(self.current, self.history, self.motion)
                self.assertTrue(np.all(confidence < stable_confidence))
                self.assertTrue(np.all(moving < stationary))
                self.assertTrue(np.all(moving >= self.current))

    def test_correctly_registered_displacement_still_discards_history_weight(self):
        # Equal images isolate the displacement penalty from disagreement.
        for dy, dx, retention in ((0, 0, 1), (0, 8, 0.5), (0, -8, 0.5),
                                   (16, 0, 1 / 3), (-16, 0, 1 / 3), (0, 16, 1 / 3)):
            for old_weight, new_weight in ((1, 1), (128, 64)):
                with self.subTest(offset=(dy, dx), weights=(old_weight, new_weight)):
                    self.motion[:] = [dy, dx]
                    merged, confidence, weights, valid = _blend_with_weights(
                        self.current, self.history, self.motion, old_weight, new_weight,
                        TAAConfig())
                    np.testing.assert_allclose(merged, self.history, atol=1e-7, rtol=0)
                    np.testing.assert_allclose(confidence, retention, atol=1e-7, rtol=0)
                    expected_weight = old_weight * retention + new_weight
                    np.testing.assert_allclose(weights, expected_weight, rtol=1e-7)
                    self.assertTrue(valid.all())

    def test_noisy_current_group_can_bias_correct_equal_weight_average(self):
        self.current[:] = 0.30
        self.history[:] = 0.20
        merged, confidence = reject_and_blend(self.current, self.history, self.motion)
        np.testing.assert_allclose(confidence, 0.60653066, atol=1e-7, rtol=0)
        np.testing.assert_allclose(merged, 0.28112297, atol=1e-7, rtol=0)
        np.testing.assert_array_equal((self.current + self.history) / 2, 0.25)

    def test_invalid_neighbors_and_opposite_border_do_not_expand_bounds(self):
        self.history[:] = 0.7
        self.current[2, 3] = [0.9, np.nan, 0.9]
        self.current[3, 2] = 0.9
        self.motion[3, 2, 0] = np.nan
        self.current[-1, -1] = 0.9
        merged, _, _, valid = _blend_with_weights(
            self.current, self.history, self.motion, 1, 1,
            TAAConfig(clamp_margin=0, adaptive=False))
        # Partially missing RGB and finite RGB with missing motion are both
        # excluded as whole pixels. A distant corner cannot wrap into (0,0).
        np.testing.assert_allclose(merged[2, 2], 0.2)
        np.testing.assert_allclose(merged[0, 0], 0.2)
        np.testing.assert_array_equal(merged[~valid], self.history[~valid])

    def test_missing_observations_preserve_history_and_weights(self):
        self.history[:] = 0.8
        self.current[0] = np.nan
        self.current[1, 0, 1] = np.nan
        self.current[2, 0] = np.inf
        self.motion[3, 0] = np.nan
        self.motion[4, 0] = np.inf
        with np.errstate(all="raise"):
            merged, confidence, weights, valid = _blend_with_weights(
                self.current, self.history, self.motion, 64, 32, TAAConfig())
        np.testing.assert_array_equal(merged[~valid], self.history[~valid])
        np.testing.assert_array_equal(weights[~valid], 64)
        np.testing.assert_array_equal(confidence[~valid], 0)
        self.assertTrue(np.isfinite(merged).all())
        self.assertTrue(np.all(merged[valid] <= 0.25))

    def test_all_missing_and_single_pixel_neighborhood(self):
        for shape in ((1, 1, 3), (1, 5, 3), (5, 1, 3)):
            with self.subTest(shape=shape), np.errstate(all="raise"):
                history = np.full(shape, 0.7, dtype=np.float32)
                motion = np.zeros((*shape[:2], 2), dtype=np.float32)
                merged, confidence = reject_and_blend(np.full(shape, np.nan), history, motion)
                np.testing.assert_array_equal(merged, history)
                np.testing.assert_array_equal(confidence, 0)
                merged, _ = reject_and_blend(np.full(shape, 0.2), history, motion)
                self.assertTrue(np.all(merged <= 0.25))

    def test_inputs_are_not_modified(self):
        self.current[0] = np.nan
        arrays = (self.current, self.history, self.motion)
        copies = [array.copy() for array in arrays]
        for array in arrays:
            array.flags.writeable = False
        reject_and_blend(*arrays)
        for array, original in zip(arrays, copies):
            np.testing.assert_array_equal(array, original)

    def test_invalid_inputs(self):
        for bad in (self.history[:-1], self.history[..., 0], self.history[:0],
                    self.history + np.nan, self.history + np.inf,
                    self.history - 1, self.history + 1,
                    self.history.astype(complex), self.history.astype(str)):
            with self.subTest(history_shape=bad.shape, dtype=bad.dtype):
                with self.assertRaises(ValueError):
                    reject_and_blend(self.current, bad, self.motion)
        for bad in (self.current[:-1], self.current - 1, self.current + 1,
                    self.current.astype(complex), self.current.astype(str)):
            with self.assertRaises(ValueError):
                reject_and_blend(bad, self.history, self.motion)
        for bad in (self.motion[..., 0], self.motion[:-1],
                    self.motion.astype(complex), self.motion.astype(str)):
            with self.assertRaises(ValueError):
                reject_and_blend(self.current, self.history, bad)

    def test_invalid_configuration(self):
        for kwargs in ({"radius": -1}, {"radius": 1.5}, {"radius": True},
                       {"clamp_margin": -0.1}, {"clamp_margin": 2},
                       {"motion_scale": 0}, {"rejection_scale": -1},
                       {"motion_scale": np.inf}, {"rejection_scale": np.nan},
                       {"clamp_margin": "0.1"}, {"clamp_margin": True},
                       {"clamp": 1}, {"adaptive": "false"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TAAConfig(**kwargs)

    def test_disabling_rejection_is_weighted_average(self):
        self.history[:] = 0.8
        merged, confidence, weights, _ = _blend_with_weights(
            self.current, self.history, self.motion, 128, 32,
            TAAConfig(clamp=False, adaptive=False))
        np.testing.assert_allclose(merged, 0.68)
        np.testing.assert_array_equal(weights, 160)
        np.testing.assert_array_equal(confidence, 1)
