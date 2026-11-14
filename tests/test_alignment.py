"""Run from the repo root: py -m unittest discover -s tests."""

import unittest

import numpy as np

from src.alignment import _warp, align_warp, align_warp_with_confidence
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
        # Independently sampled noise can move the continuous SSD minimum
        # slightly away from the known integer displacement.
        np.testing.assert_allclose(np.median(motion[interior], axis=(0, 1)), shift, atol=0.25)
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

    def test_fractional_shifts(self):
        y, x = np.indices((96, 96), dtype=np.float32)

        def probability(y, x):
            return np.stack([
                0.4 + 0.12 * np.sin(0.29 * x + 0.07 * y + channel * 0.4)
                + 0.1 * np.cos(0.23 * y - 0.13 * x + channel)
                + 0.08 * np.sin(0.17 * x - 0.19 * y + 0.3 * channel)
                for channel in range(3)], axis=2)

        reference = probability(y, x)
        interior = np.s_[24:-24, 24:-24]
        for dy, dx in ((0.375, -0.25), (-0.375, 0.25), (0.5, 0.5), (-0.5, -0.5)):
            with self.subTest(shift=(dy, dx)):
                # Deterministic binary counts isolate interpolation geometry;
                # separate tests cover independent photon sampling noise.
                counts = np.rint(256 * probability(y - dy, x - dx)).astype(int)
                burst = (np.arange(256)[:, None, None, None] < counts).astype(np.uint8)
                warped, motion = align_warp(burst, reference)
                np.testing.assert_allclose(np.median(motion[interior], axis=(0, 1)),
                                           (dy, dx), atol=0.125)
                self.assertTrue(np.any(motion[interior] != np.rint(motion[interior])))
                yy, xx = np.indices(reference.shape[:2])
                integer = np.rint(motion).astype(int)
                source = burst.mean(axis=0, dtype=np.float32)
                nearest = source[(yy + integer[..., 0])[interior],
                                 (xx + integer[..., 1])[interior]]
                error = np.mean((warped[interior] - reference[interior]) ** 2)
                integer_error = np.mean((nearest - reference[interior]) ** 2)
                self.assertLess(error, integer_error)

    def test_bilinear_sampling_and_borders(self):
        source = np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 35
        field = np.empty((3, 4, 2), dtype=np.float32)
        field[:] = (0.25, 0.5)
        warped, valid = _warp(source, field)
        expected = ((0.75 * 0.5) * source[0, 0] + (0.75 * 0.5) * source[0, 1]
                    + (0.25 * 0.5) * source[1, 0] + (0.25 * 0.5) * source[1, 1])
        np.testing.assert_allclose(warped[0, 0], expected)
        self.assertFalse(valid[-1].any())
        self.assertFalse(valid[:, -1].any())
        self.assertTrue(np.isnan(warped[~valid]).all())
        field[:] = (-0.25, -0.5)
        warped, valid = _warp(source, field)
        expected = (0.125 * source[1, 2] + 0.125 * source[1, 3]
                    + 0.375 * source[2, 2] + 0.375 * source[2, 3])
        np.testing.assert_allclose(warped[-1, -1], expected)
        self.assertFalse(valid[0].any())
        self.assertFalse(valid[:, 0].any())
        self.assertTrue(np.isnan(warped[~valid]).all())
        exact, valid = _warp(source, np.zeros_like(field))
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(exact, source)

    def test_optional_confidence_preserves_outputs(self):
        cube = load_sample("data/samples/kitchen1_000000.npz")
        reference = cube[:64].mean(axis=0, dtype=np.float32)
        cube.setflags(write=False)
        reference.setflags(write=False)
        warped, motion = align_warp(cube[64:128], reference)
        detailed, detailed_motion, confidence = align_warp_with_confidence(cube[64:128], reference)
        np.testing.assert_array_equal(warped, detailed)
        np.testing.assert_array_equal(motion, detailed_motion)
        self.assertEqual(confidence.shape, reference.shape[:2])
        self.assertEqual(confidence.dtype, np.float32)
        self.assertTrue(np.isfinite(confidence).all())
        self.assertTrue(((confidence >= 0) & (confidence <= 1)).all())
        valid = np.isfinite(warped).all(axis=2)
        np.testing.assert_array_equal(confidence[~valid], 0)
        for y in range(0, len(confidence), 8):
            for x in range(0, confidence.shape[1], 8):
                scores = confidence[y:y + 8, x:x + 8][valid[y:y + 8, x:x + 8]]
                self.assertLessEqual(len(np.unique(scores)), 1)

    def test_confidence_on_unique_flat_and_repetitive_matches(self):
        rng = np.random.default_rng(566)
        reference = rng.integers(0, 2, (96, 96, 3), dtype=np.uint8)
        source = np.roll(reference, (0, 8), axis=(0, 1))
        _, _, unique = align_warp_with_confidence(source[None], reference)
        _, _, stationary = align_warp_with_confidence(reference[None], reference)
        interior = np.s_[24:-24, 24:-24]
        self.assertGreater(float(unique[interior].mean()), 0.9)
        np.testing.assert_allclose(unique[interior], stationary[interior], atol=1e-6)
        np.testing.assert_array_equal(unique[:, -8:], 0)
        flat = np.zeros_like(reference)
        _, _, flat_confidence = align_warp_with_confidence(flat[None], flat)
        np.testing.assert_array_equal(flat_confidence, 0)
        y, x = np.indices(reference.shape[:2])
        periodic = np.repeat(((x + y) % 2)[..., None], 3, axis=2).astype(np.uint8)
        _, _, periodic_confidence = align_warp_with_confidence(periodic[None], periodic)
        np.testing.assert_array_equal(periodic_confidence[interior], 0)


if __name__ == "__main__":
    unittest.main()
