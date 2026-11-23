"""Motion measurement, binning and the synthetic burst simulator (issues #6, #23)."""
import unittest

import numpy as np

from src.motion import (assign_bin, measure_cube_motion, phase_corr, summarize,
                        tertile_edges, window_means)
from src.synthetic import shift_image, synthesize_burst


def texture(size=64, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((size // 4, size // 4, 3)).astype(np.float32)
    return np.kron(base, np.ones((4, 4, 1), dtype=np.float32)) * 0.9


class MotionTests(unittest.TestCase):
    def test_phase_corr_recovers_integer_shift(self):
        a = texture(64)[..., 0]
        b = np.roll(a, 3, axis=1)
        dy, dx, peak = phase_corr(a, b)
        self.assertAlmostEqual(dy, 0, delta=0.3)
        self.assertAlmostEqual(abs(dx), 3, delta=0.5)
        self.assertGreater(peak, 0.1)

    def test_window_means_needs_two_windows(self):
        with self.assertRaises(ValueError):
            window_means(np.zeros((300, 8, 8, 3), dtype=np.uint8))
        self.assertEqual(len(window_means(np.zeros((512, 8, 8, 3), dtype=np.uint8))), 2)

    def test_tertile_edges_and_bins(self):
        edges = tertile_edges([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(len(edges), 2)
        labels = [assign_bin(v, edges) for v in (1, 2, 3, 5, 6, 8, 9)]
        self.assertEqual(labels[0], "slow")
        self.assertEqual(labels[-1], "fast")
        self.assertIn("medium", labels)
        with self.assertRaises(ValueError):
            tertile_edges([1, 2])

    def test_assign_bin_edges_are_lower_inclusive(self):
        self.assertEqual(assign_bin(5.0, [5.0, 10.0]), "medium")
        self.assertEqual(assign_bin(4.99, [5.0, 10.0]), "slow")
        self.assertEqual(assign_bin(10.0, [5.0, 10.0]), "fast")
        with self.assertRaises(ValueError):
            assign_bin(1, [5.0], labels=("a", "b", "c"))

    def test_static_burst_measures_near_zero(self):
        cube = synthesize_burst(texture(), 0, frames=512)
        self.assertLess(measure_cube_motion(cube)["drift_px"], 0.5)

    def test_measured_drift_follows_nominal_and_is_monotonic(self):
        gt = texture()
        measured = [measure_cube_motion(synthesize_burst(gt, d, frames=512))["drift_px"]
                    for d in (0, 4, 8)]
        self.assertLess(measured[0], measured[1])
        self.assertLess(measured[1], measured[2])
        # windows are 256 frames; their centres are 256 of 511 frames apart
        self.assertAlmostEqual(measured[2], 8 * 256 / 511, delta=1.0)

    def test_summary_fields(self):
        wins = window_means(synthesize_burst(texture(), 6, frames=512))
        info = summarize(wins)
        for key in ("dy", "dx", "drift_px", "speed_px_per_256", "path_px", "min_peak", "reliable"):
            self.assertIn(key, info)


class SyntheticTests(unittest.TestCase):
    def test_deterministic_for_a_seed_and_different_across_seeds(self):
        gt = texture(32)
        a = synthesize_burst(gt, 5, frames=16, seed=1)
        b = synthesize_burst(gt, 5, frames=16, seed=1)
        c = synthesize_burst(gt, 5, frames=16, seed=2)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(a.dtype, np.uint8)
        self.assertTrue(set(np.unique(a)) <= {0, 1})

    def test_static_burst_average_matches_radiance(self):
        gt = texture(32)
        cube = synthesize_burst(gt, 0, frames=2000, gain=2.7)
        np.testing.assert_allclose(cube.mean(axis=0), gt / 2.7, atol=0.05)

    def test_last_frame_is_at_the_ground_truth_pose(self):
        gt = np.zeros((16, 32, 3), dtype=np.float32)
        gt[:, 8:12] = 1.0                       # bright bar
        cube = synthesize_burst(gt, 8, frames=300, gain=1.0)
        last = cube[-20:].mean(axis=(0, 3))
        first = cube[:20].mean(axis=(0, 3))
        self.assertGreater(last[:, 8:12].mean(), 0.9)          # bar where the ground truth has it
        self.assertLess(first[:, 8:12].mean(), 0.5)            # earlier frames show it displaced

    def test_shift_image_subpixel_and_clamping(self):
        image = np.zeros((1, 4, 3), dtype=np.float32)
        image[0, 2] = 1.0
        out = shift_image(image, 0, 0.5)
        np.testing.assert_allclose(out[0, 1], 0.5)
        np.testing.assert_allclose(out[0, 2], 0.5)
        np.testing.assert_allclose(shift_image(image, 0, 0), image)

    def test_rejects_bad_input(self):
        gt = texture(16)
        for kwargs in ({"frames": 1}, {"drift_px": -1}, {"gain": 0}):
            args = {"gt": gt, "drift_px": 1, "frames": 8, "gain": 2.7}
            args.update(kwargs)
            with self.assertRaises(ValueError):
                synthesize_burst(**args)
        with self.assertRaises(ValueError):
            synthesize_burst(gt * 2, 1, frames=8)


if __name__ == "__main__":
    unittest.main()
