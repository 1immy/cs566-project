import unittest
import numpy as np
from src.evaluation import evaluate, evaluate_channels


class EvaluationTests(unittest.TestCase):
    def test_identical(self):
        image = np.random.default_rng(566).random((16, 20, 3), dtype=np.float32)
        psnr, ssim = evaluate(image, image.copy())
        self.assertTrue(np.isinf(psnr))
        self.assertAlmostEqual(ssim, 1.0, places=10)

    def test_constant_offset_known_psnr(self):
        gt = np.full((8, 8, 3), 0.5)
        rec = np.full((8, 8, 3), 0.4)
        psnr, ssim = evaluate(rec, gt)
        self.assertAlmostEqual(psnr, 20.0)
        self.assertLess(ssim, 1)

    def test_channels_mean(self):
        gt = np.full((9, 9, 3), 0.6)
        rec = gt.copy()
        rec[..., 0] -= 0.1
        rec[..., 1] -= 0.2
        rec[..., 2] -= 0.4
        metrics = evaluate_channels(rec, gt)
        expected = np.mean([-20*np.log10(0.1), -20*np.log10(0.2), -20*np.log10(0.4)])
        self.assertAlmostEqual(metrics['psnr']['mean'], expected)
        self.assertAlmostEqual(metrics['ssim']['mean'], np.mean([metrics['ssim'][x] for x in 'rgb']))

    def test_invalid(self):
        image = np.zeros((8, 8, 3))
        for other in (image[..., :2], image + np.nan, image + 2):
            with self.assertRaises(ValueError):
                evaluate(other, image)


if __name__ == '__main__':
    unittest.main()
