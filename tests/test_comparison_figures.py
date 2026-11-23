"""Row selection and tile splitting of the final comparison figures (issue #30)."""
import contextlib
import csv
import io
import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import make_comparison_figures as figures


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["scene", "burst", "bin", "dpsnr_taa_aligned"])
        writer.writeheader()
        writer.writerows(rows)


def comparison_image(tile=60, gap=6, margin=4, title=12):
    width = margin + 4 * tile + 3 * gap + margin
    image = np.full((title + tile + margin, width, 3), 255, dtype=np.uint8)
    for i in range(4):
        x = margin + i * (tile + gap)
        image[title:title + tile, x:x + tile] = 40 + 50 * i
    return image


class FigureTests(unittest.TestCase):
    def test_selection_takes_largest_gain_and_loss_per_bin(self):
        rows = [dict(scene="a", burst=0, bin="slow", dpsnr_taa_aligned=0.1),
                dict(scene="b", burst=1, bin="slow", dpsnr_taa_aligned=0.9),
                dict(scene="c", burst=0, bin="slow", dpsnr_taa_aligned=-0.7),
                dict(scene="d", burst=0, bin="fast", dpsnr_taa_aligned=0.3)]
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "per_burst.csv")
            write_csv(path, rows)
            chosen = figures.select_rows(path)
        self.assertEqual(chosen[:2], [("b", 1, "slow"), ("c", 0, "slow")])
        self.assertEqual([c for c in chosen if c[2] == "fast"], [("d", 0, "fast")])
        self.assertFalse(any(c[2] == "medium" for c in chosen))

    def test_split_tiles_finds_four_equal_tiles(self):
        tiles = figures.split_tiles(comparison_image())
        self.assertEqual(len(tiles), 4)
        self.assertTrue(all(t.shape[:2] == (60, 60) for t in tiles))
        self.assertEqual([int(t[0, 0, 0]) for t in tiles], [40, 90, 140, 190])

    def test_split_tiles_rejects_other_layouts(self):
        with self.assertRaises(ValueError):
            figures.split_tiles(np.full((50, 200, 3), 255, dtype=np.uint8))

    def test_main_writes_both_figures(self):
        with tempfile.TemporaryDirectory() as root:
            results = os.path.join(root, "full_scale")
            os.makedirs(os.path.join(results, "s"))
            base = os.path.join(results, "s", "000000")
            image = comparison_image(tile=320)
            Image.fromarray(image).save(base + "_comparison.png")
            scores = {k: {"psnr_mean": 14.0 + i, "ssim_mean": 0.3} for i, k in enumerate(("naive", "aligned", "taa"))}
            with open(base + ".json", "w", encoding="utf-8") as file:
                json.dump({"scores": scores, "motion": {"drift_px": 4.0}}, file)
            csv_path = os.path.join(root, "per_burst.csv")
            write_csv(csv_path, [dict(scene="s", burst=0, bin="slow", dpsnr_taa_aligned=1.0)])
            out = os.path.join(root, "figs")
            with contextlib.redirect_stdout(io.StringIO()):
                figures.main(["--results", results, "--per-burst", csv_path, "--out", out])
            self.assertTrue(os.path.isfile(os.path.join(out, "comparison_full_frames.png")))
            self.assertTrue(os.path.isfile(os.path.join(out, "comparison_crops.png")))

    def test_missing_inputs_are_an_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            figures.main(["--results", "nope", "--per-burst", "nope.csv"])


if __name__ == "__main__":
    unittest.main()
