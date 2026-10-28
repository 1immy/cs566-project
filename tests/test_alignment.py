"""Run from the repo root: py -m unittest discover -s tests."""

import unittest

import numpy as np

from src.alignment import align_warp
from src.burst_io import load_sample


class AlignmentTests(unittest.TestCase):
    def test_identity(self):
        rng = np.random.default_rng(566)
        burst = rng.binomial(1, 0.2, (32, 65, 67, 3)).astype(np.uint8)
        reference = burst.mean(axis=0, dtype=np.float32)
        warped, motion = align_warp(burst, reference)
        np.testing.assert_array_equal(motion, 0)
        np.testing.assert_array_equal(warped, reference)
        self.assertEqual(warped.dtype, np.float32)

    def test_known_shift(self):
        rng = np.random.default_rng(566)
        # Repeated binary texture isolates geometry from photon sampling noise.
        reference = rng.integers(0, 2, (96, 96, 3), dtype=np.uint8)
        for dy, dx in ((4, -8), (-4, 8)):
            source = np.roll(reference, (dy, dx), axis=(0, 1))
            warped, motion = align_warp(source[None], reference)
            interior = np.s_[24:-24, 24:-24]
            np.testing.assert_array_equal(motion[interior][..., 0], dy)
            np.testing.assert_array_equal(motion[interior][..., 1], dx)
            np.testing.assert_array_equal(warped[interior], reference[interior])
            self.assertLess(np.mean((warped[interior] - reference[interior]) ** 2),
                            np.mean((source[interior].astype(float) - reference[interior]) ** 2))

    def test_invalid_inputs(self):
        burst = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        reference = np.zeros((16, 16, 3), dtype=np.float32)
        for bad in (burst[:0], burst.astype(float), burst + 2):
            with self.assertRaises(ValueError):
                align_warp(bad, reference)
        for bad in (reference + np.nan, reference + np.inf, reference - 0.1,
                    reference + 1.1, reference.astype(complex) + 1j,
                    reference.astype(str), reference[:-1]):
            with self.subTest(history_dtype=bad.dtype, history_shape=bad.shape):
                with self.assertRaises(ValueError):
                    align_warp(burst, bad)

    def test_missing_edge_preserves_predicted_motion(self):
        rng = np.random.default_rng(566)
        reference = rng.integers(0, 2, (96, 96, 3), dtype=np.uint8)
        yy, xx = np.indices(reference.shape[:2])
        for dy, dx in ((-8, 0), (8, 0), (0, -8), (0, 8)):
            with self.subTest(shift=(dy, dx)):
                source = np.roll(reference, (dy, dx), axis=(0, 1))
                warped, motion = align_warp(source[None], reference)
                missing = ((yy + dy < 0) | (yy + dy >= 96)
                           | (xx + dx < 0) | (xx + dx >= 96))
                self.assertTrue(np.all(np.isnan(warped[missing])))
                np.testing.assert_array_equal(motion[missing],
                                              np.tile((dy, dx), (missing.sum(), 1)))

    def test_independent_photon_groups(self):
        rng = np.random.default_rng(566)
        probability = 0.05 + 0.3 * rng.random((96, 96, 3))
        shift = (3, -5)
        first = rng.binomial(1, probability, (64, *probability.shape)).astype(np.uint8)
        second = rng.binomial(1, np.roll(probability, shift, axis=(0, 1)),
                              (64, *probability.shape)).astype(np.uint8)
        reference = first.mean(axis=0, dtype=np.float32)
        warped, motion = align_warp(second, reference)
        interior = np.s_[24:-24, 24:-24]
        # Photon samples differ between groups; matching need not be exact.
        np.testing.assert_array_equal(np.median(motion[interior], axis=(0, 1)), shift)
        unaligned = second.mean(axis=0, dtype=np.float32)
        aligned_error = np.mean((warped[interior] - probability[interior]) ** 2)
        unaligned_error = np.mean((unaligned[interior] - probability[interior]) ** 2)
        self.assertLess(aligned_error, unaligned_error)

    def test_opposite_local_motion(self):
        rng = np.random.default_rng(566)
        reference = rng.integers(0, 2, (96, 128, 3), dtype=np.uint8)
        source = reference.copy()
        source[:, :64] = np.roll(reference[:, :64], 3, axis=1)
        source[:, 64:] = np.roll(reference[:, 64:], -3, axis=1)
        warped, motion = align_warp(source[None], reference)
        for columns, dx in ((slice(24, 40), 3), (slice(88, 104), -3)):
            interior = np.s_[24:-24, columns]
            np.testing.assert_array_equal(motion[interior][..., 0], 0)
            np.testing.assert_array_equal(motion[interior][..., 1], dx)
            np.testing.assert_array_equal(warped[interior], reference[interior])

    def test_committed_samples(self):
        for index in (0, 1):
            cube = load_sample(f"data/samples/kitchen1_{index:06d}.npz")
            reference = cube[:64].mean(axis=0, dtype=np.float32)
            warped, motion = align_warp(cube[64:128], reference)
            self.assertEqual(warped.shape, reference.shape)
            self.assertEqual(motion.shape, (*reference.shape[:2], 2))
            self.assertTrue(np.all(np.isfinite(motion)))
            values = warped[np.isfinite(warped)]
            self.assertGreater(values.size, 0)
            self.assertTrue(np.all((values >= 0) & (values <= 1)))


if __name__ == "__main__":
    unittest.main()
