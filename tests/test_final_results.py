"""Statistics and table of the final multi-scene results (issue #29)."""
import contextlib
import csv
import io
import os
import tempfile
import unittest

from scripts import final_results_table as table

HEAD = ["scene", "burst", "drift_px", "reliable", "bin", "psnr_naive", "ssim_naive", "psnr_aligned",
        "ssim_aligned", "psnr_taa", "ssim_taa", "dpsnr_taa_aligned", "dssim_taa_aligned"]


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEAD)
        for scene, burst, drift, label, naive, aligned, taa in rows:
            writer.writerow([scene, burst, drift, "False", label, naive, 0.2, aligned, 0.4, taa, 0.3, taa - aligned, -0.1])


class FinalResultsTests(unittest.TestCase):
    def test_sign_test_exact_values(self):
        self.assertEqual(table.sign_test([1] * 10), (10, 0, 0, 2 / 2 ** 10))
        wins, losses, ties, p = table.sign_test([1, 1, -1, -1, 0])
        self.assertEqual((wins, losses, ties), (2, 2, 1))
        self.assertAlmostEqual(p, 1.0)
        self.assertEqual(table.sign_test([0, 0])[3], 1.0)
        self.assertAlmostEqual(table.sign_test([1] * 14 + [-1] * 17)[3], 0.7201, places=4)

    def test_scene_level_averages_bursts_of_one_scene(self):
        rows = [dict(scene="a", x=3.0, y=1.0), dict(scene="a", x=1.0, y=1.0), dict(scene="b", x=0.0, y=1.0)]
        self.assertEqual(sorted(table.scene_means(rows, "x", "y")), [-1.0, 1.0])

    def test_end_to_end_writes_page_and_csv(self):
        data = []
        for i in range(12):
            label = ("slow", "medium", "fast")[i % 3]
            data.append((f"s{i % 10}", i % 2, 3.0 + i, label, 14.0, 15.0 + 0.1 * i, 15.0 + 0.05 * i))
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "per_burst.csv")
            write_rows(source, data)
            out = os.path.join(root, "final.md")
            with contextlib.redirect_stdout(io.StringIO()):
                table.main(["--per-burst", source, "--out", out])
            with open(out, encoding="utf-8") as file:
                text = file.read()
            self.assertIn("12 real bursts from 10 scenes", text)
            self.assertIn("| all | 12 |", text)
            self.assertIn("Exact sign test", text)
            self.assertTrue(os.path.isfile(os.path.join(root, "final.csv")))

    def test_missing_input_is_an_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            table.main(["--per-burst", "nope.csv"])


if __name__ == "__main__":
    unittest.main()
