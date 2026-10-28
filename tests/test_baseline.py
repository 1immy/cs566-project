import unittest
from unittest.mock import patch

import numpy as np

from src.baseline import aligned_average
from src.burst_io import naive_average


class BaselineTests(unittest.TestCase):
    def test_single_group(self):
        cube = np.random.default_rng(566).integers(0, 2, (5, 8, 8, 3), dtype=np.uint8)
        result, counts = aligned_average(cube, group_size=64)
        np.testing.assert_allclose(result, naive_average(cube))
        np.testing.assert_array_equal(counts, 5)

    def test_partial_group_and_invalid_border(self):
        cube = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        cube[2:] = 1

        def identity_with_border(group, history):
            warped = naive_average(group)
            warped[0] = np.nan
            return warped, np.zeros((8, 8, 2), dtype=np.float32)

        with patch("src.baseline.align_warp", side_effect=identity_with_border):
            result, counts = aligned_average(cube, group_size=2)
        np.testing.assert_allclose(result[1:], 3 / 5)
        np.testing.assert_array_equal(counts[1:], 5)
        np.testing.assert_array_equal(result[0], 0)
        np.testing.assert_array_equal(counts[0], 2)

    def test_invalid_group_size(self):
        cube = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                aligned_average(cube, size)

    def test_translated_groups_end_to_end(self):
        rng = np.random.default_rng(566)
        reference = rng.integers(0, 2, (96, 96, 3), dtype=np.uint8)
        shifted = np.roll(reference, (0, 8), axis=(0, 1))
        # A short final group exercises real alignment and frame weighting.
        cube = np.stack([reference, reference, shifted])
        result, counts = aligned_average(cube, group_size=2)
        interior = np.s_[24:-24, 24:-24]
        np.testing.assert_array_equal(result[interior], reference[interior])
        np.testing.assert_array_equal(counts[interior], 3)
        np.testing.assert_array_equal(counts[:, -8:], 2)
        np.testing.assert_array_equal(result[:, -8:], reference[:, -8:])
        self.assertLess(np.mean((result[interior] - reference[interior]) ** 2),
                        np.mean((naive_average(cube)[interior] - reference[interior]) ** 2))
