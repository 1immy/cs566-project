import unittest
from unittest.mock import patch

import numpy as np

from src.baseline import aligned_average
from src.burst_io import load_sample, naive_average
from src.taa import TAAConfig, reconstruct_taa, reject_and_blend


class TAAPipelineTests(unittest.TestCase):
    def test_single_group_initializes_without_claiming_confidence(self):
        cube = np.random.default_rng(566).integers(0, 2, (5, 8, 8, 3), dtype=np.uint8)
        result = reconstruct_taa(cube)
        np.testing.assert_array_equal(result.reconstruction, naive_average(cube))
        np.testing.assert_array_equal(result.history_weights, 5)
        np.testing.assert_array_equal(result.frame_counts, 5)
        np.testing.assert_array_equal(result.confidence_map, 0)
        self.assertFalse(result.valid_update_mask.any())

    def test_first_equal_group_update_matches_public_api(self):
        cube = np.zeros((4, 5, 7, 3), dtype=np.uint8)
        cube[2:] = 1
        current = naive_average(cube[2:])
        motion = np.full((5, 7, 2), 4, dtype=np.float32)
        expected, confidence = reject_and_blend(current, naive_average(cube[:2]), motion)
        with patch("src.taa.align_warp", return_value=(current, motion)):
            result = reconstruct_taa(cube, group_size=2)
        np.testing.assert_allclose(result.reconstruction, expected)
        np.testing.assert_array_equal(result.confidence_map, confidence)

    def test_fixed_reference_partial_group_and_missing_border(self):
        cube = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        cube[2:] = 1
        observed_references = []

        def warp(group, reference):
            observed_references.append(reference.copy())
            current = naive_average(group)
            current[0] = np.nan
            return current, np.zeros((8, 8, 2), dtype=np.float32)

        with patch("src.taa.align_warp", side_effect=warp):
            result = reconstruct_taa(cube, 2, TAAConfig(clamp=False, adaptive=False))
        np.testing.assert_allclose(result.reconstruction[1:], 3 / 5)
        np.testing.assert_array_equal(result.frame_counts[1:], 5)
        np.testing.assert_array_equal(result.history_weights[1:], 5)
        np.testing.assert_array_equal(result.reconstruction[0], 0)
        np.testing.assert_array_equal(result.frame_counts[0], 2)
        np.testing.assert_array_equal(result.history_weights[0], 2)
        np.testing.assert_array_equal(result.confidence_map[0], 0)
        self.assertEqual(len(observed_references), 2)
        for reference in observed_references:
            np.testing.assert_array_equal(reference, 0)
        # Even when clamping changes the merged history drastically, alignment
        # must continue to see the original reference rather than that history.
        observed_references.clear()
        with patch("src.taa.align_warp", side_effect=warp):
            result = reconstruct_taa(cube, 2)
        self.assertTrue(np.all(result.reconstruction[1:] > 0.9))
        for reference in observed_references:
            np.testing.assert_array_equal(reference, 0)
        self.assertTrue(np.all(result.history_weights <= result.frame_counts))

    def test_partial_group_has_actual_weight_with_adaptive_blending(self):
        cube = np.zeros((10, 4, 4, 3), dtype=np.uint8)
        # Means 0.25, 0.25, 0.5 for groups of 4, 4, 2 frames.
        cube[0] = cube[4] = cube[8] = 1

        def warp(group, reference):
            return naive_average(group), np.zeros((4, 4, 2), dtype=np.float32)

        with patch("src.taa.align_warp", side_effect=warp):
            result = reconstruct_taa(cube, 4, TAAConfig(clamp_margin=1))
        np.testing.assert_allclose(result.reconstruction, 0.3)
        np.testing.assert_array_equal(result.history_weights, 10)

    def test_reappearing_pixels_keep_their_own_accumulation_weights(self):
        cube = np.zeros((6, 4, 4, 3), dtype=np.uint8)
        cube[4:] = 1
        call_count = 0

        def warp(group, reference):
            nonlocal call_count
            current = naive_average(group)
            if call_count == 0:
                current[0] = np.nan
            call_count += 1
            return current, np.zeros((4, 4, 2), dtype=np.float32)

        with patch("src.taa.align_warp", side_effect=warp):
            result = reconstruct_taa(cube, 2, TAAConfig(clamp=False, adaptive=False))
        np.testing.assert_allclose(result.reconstruction[0], 0.5)
        np.testing.assert_allclose(result.reconstruction[1:], 1 / 3)
        np.testing.assert_array_equal(result.history_weights[0], 4)
        np.testing.assert_array_equal(result.history_weights[1:], 6)

    def test_real_translation_preserves_geometry_and_missing_edges(self):
        reference = np.random.default_rng(566).integers(0, 2, (96, 96, 3), dtype=np.uint8)
        cube = np.stack([reference, reference, np.roll(reference, 8, axis=1)])
        result = reconstruct_taa(cube, 2)
        np.testing.assert_allclose(result.reconstruction[24:-24, 24:-24],
                                   reference[24:-24, 24:-24], atol=1e-7)
        np.testing.assert_array_equal(result.reconstruction[:, -8:], reference[:, -8:])
        np.testing.assert_array_equal(result.frame_counts[:, -8:], 2)
        np.testing.assert_array_equal(result.history_weights[:, -8:], 2)
        self.assertFalse(result.valid_update_mask[:, -8:].any())

    def test_missing_final_update_resets_diagnostic_but_preserves_useful_history(self):
        cube = np.zeros((12, 4, 4, 3), dtype=np.uint8)
        cube[::4] = 1
        current = naive_average(cube[4:8])
        final = current.copy()
        final[0, 0] = np.nan
        motion = np.zeros((4, 4, 2), dtype=np.float32)
        motion[..., 1] = 8
        with patch("src.taa.align_warp", side_effect=[(current, motion), (final, 2 * motion)]):
            result = reconstruct_taa(cube, 4)
        np.testing.assert_array_equal(result.reconstruction, 0.25)
        # Second group retained half of four frames and added four: weight=6.
        self.assertEqual(result.history_weights[0, 0], 6)
        self.assertEqual(result.frame_counts[0, 0], 8)
        self.assertEqual(result.confidence_map[0, 0], 0)
        self.assertFalse(result.valid_update_mask[0, 0])
        self.assertAlmostEqual(float(result.confidence_map[1, 1]), 1 / 3)
        self.assertEqual(result.frame_counts[1, 1], 12)

    def test_disabled_taa_matches_baseline_on_committed_samples(self):
        for index in (0, 1):
            with self.subTest(sample=index):
                cube = load_sample(f"data/samples/kitchen1_{index:06d}.npz")
                # 96 ensures a short final group as well as real invalid edges.
                baseline, counts = aligned_average(cube, 96)
                result = reconstruct_taa(cube, 96, TAAConfig(clamp=False, adaptive=False))
                np.testing.assert_allclose(result.reconstruction, baseline, atol=1e-7, rtol=0)
                np.testing.assert_array_equal(result.frame_counts, counts)
                np.testing.assert_array_equal(result.history_weights, counts)
                taa = reconstruct_taa(cube, 96)
                self.assertTrue(np.isfinite(taa.reconstruction).all())
                self.assertTrue(np.all((taa.reconstruction >= 0) & (taa.reconstruction <= 1)))
                self.assertTrue(np.all((taa.confidence_map >= 0) & (taa.confidence_map <= 1)))
                np.testing.assert_array_equal(taa.frame_counts, counts)

    def test_invalid_bursts_and_group_sizes(self):
        cube = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        for bad in (cube[:0], cube[..., 0], cube.astype(float), cube + 2):
            with self.assertRaises(ValueError):
                reconstruct_taa(bad)
        for size in (0, -1, True, np.bool_(True), 2.5):
            with self.assertRaises(ValueError):
                reconstruct_taa(cube, size)
        with self.assertRaises(ValueError):
            reconstruct_taa(cube, config={})
