"""Checks for frame/crop/reference controls in the Track 4 harness."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.run_evaluation import (_check_crop, _full_images, _select_cube, _reconstruct, main)


class HarnessTests(unittest.TestCase):
    def test_frame_selection_keeps_time_order(self):
        cube = np.arange(6, dtype=np.uint8).reshape(6, 1, 1, 1)
        selected = _select_cube(cube, frames=4)
        np.testing.assert_array_equal(selected[:, 0, 0, 0], [0, 1, 2, 3])

    def test_crop_applies_to_cube(self):
        cube = np.arange(4 * 8 * 16 * 3, dtype=np.int32).reshape(4, 8, 16, 3)
        selected = _select_cube(cube, frames=3, crop=(2, 6, 8, 16))
        np.testing.assert_array_equal(selected, cube[:3, 2:6, 8:16])

    def test_full_burst_is_not_reversed(self):
        cube = np.arange(5, dtype=np.uint8)[:, None, None, None]
        cube = np.broadcast_to(cube, (5, 8, 8, 3)).copy()
        gt = np.zeros((8, 8, 3), dtype=np.float32)
        with (patch('scripts.run_evaluation.load_burst', return_value=cube),
              patch('scripts.run_evaluation.load_gt', return_value=gt),
              patch('scripts.run_evaluation._reconstruct', return_value={'naive': gt}) as recon):
            out_gt, _, count = _full_images('unused', 10, 64, [], 5, None, 'last')
        np.testing.assert_array_equal(recon.call_args.args[0][0], 0)
        np.testing.assert_array_equal(recon.call_args.args[0][-1], 4)
        self.assertEqual(recon.call_args.args[3], 'last')
        self.assertEqual(count, 5)
        np.testing.assert_array_equal(out_gt, gt)

    def test_report_table_five_variants(self):
        from src.taa import TAAResult
        cube = np.random.default_rng(7).integers(0, 2, (4, 8, 16, 3), dtype=np.uint8)
        with (patch('scripts.run_evaluation.aligned_average',
                    side_effect=lambda c, g, reference: (c[0 if reference == 'first' else -1].astype(np.float32), None)) as align,
              patch('scripts.run_evaluation.reconstruct_taa',
                    side_effect=lambda c, g, config, reference: TAAResult(c[-1].astype(np.float32), None, None, None, None)) as taa):
            results = _reconstruct(cube, 2, [], report_table=True)
        self.assertEqual(set(results), {'naive', 'aligned_first', 'aligned_last', 'clamp_only_last', 'taa_last'})
        self.assertEqual([c.kwargs['reference'] for c in align.call_args_list], ['first', 'last'])
        self.assertEqual([c.kwargs['reference'] for c in taa.call_args_list], ['last', 'last'])
        np.testing.assert_array_equal(results['aligned_first'], cube[0])
        np.testing.assert_array_equal(results['aligned_last'], cube[-1])
        self.assertFalse(taa.call_args_list[0].args[2].adaptive)
        self.assertTrue(taa.call_args_list[1].args[2].adaptive)

    def test_invalid_crop_and_archive_flags(self):
        import argparse
        parser = argparse.ArgumentParser()
        with self.assertRaises(SystemExit):
            _check_crop((0, 20, 1, 9), parser, packed=True)
        with self.assertRaises(SystemExit):
            _check_crop((5, 5, 0, 16), parser, packed=False)
        with self.assertRaises(SystemExit):
            main(['--archives', 'unused.npz', '--reference', 'last'])

    def test_cli_sample_writes_protocol_and_ssim(self):
        # Synthetic samples and truth to keep this test independent of repo data.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rng = np.random.default_rng(566)
            cube = rng.integers(0, 2, (4, 8, 16, 3), dtype=np.uint8)
            packed = np.packbits(cube, axis=2)
            src = folder / 'kitchen1_000000.npz'
            np.savez(src, packed=packed, width=16)
            from PIL import Image
            Image.fromarray(np.full((8, 16, 3), 128, dtype=np.uint8)).save(
                folder / 'kitchen1_000000_gt.png')
            rows = main(['--samples', str(src), '--frames', '4', '--crop', '0', '8', '0', '16',
                         '--reference', 'last', '--group-size', '2', '--out', str(folder / 'out')])
            self.assertEqual(len(rows), 3)
            self.assertIn('ssim_mean', rows[0])
            protocol = json.loads((folder / 'out' / 'protocol.json').read_text())
            self.assertEqual(protocol['reference_group'], 'last')
            self.assertEqual(protocol['frames_used']['kitchen1_000000'], 4)
            self.assertEqual(protocol['gain'], 2.7)
            self.assertEqual(protocol['crop_y0_y1_x0_x1'], [0, 8, 0, 16])
            self.assertTrue((folder / 'out' / 'results.csv').exists())


if __name__ == '__main__':
    unittest.main()
