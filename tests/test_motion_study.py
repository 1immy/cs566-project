"""Resumable full-scale runner, motion binning and the bin analysis (issues #23, #24, #25)."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import analyze_motion_bins, bin_scenes_by_motion, run_full_scale
from src import motion_study
from src.motion import measure_burst_motion
from src.synthetic import synthesize_burst

FRAMES, GROUP = 512, 64


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def make_scene(root, scene, drifts, size=64, seed=0):
    """A scene folder in the real on-disk format: NNNNNN.npy bit-packed along width + NNNNNN.png."""
    folder = os.path.join(root, scene)
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = rng.random((size // 4, size // 4, 3)).astype(np.float32)
    gt = np.kron(base, np.ones((4, 4, 1), dtype=np.float32)) * 0.9
    for index, drift in enumerate(drifts):
        cube = synthesize_burst(gt, drift, frames=FRAMES, seed=index)
        np.save(os.path.join(folder, f"{index:06d}.npy"), np.packbits(cube, axis=2))
        Image.fromarray((gt * 255).astype(np.uint8)).save(os.path.join(folder, f"{index:06d}.png"))
    return folder


def run_quiet(main, argv):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        code = main(argv)
    return code, out.getvalue()


class FullScaleRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "data")
        self.out = os.path.join(self.tmp.name, "out")
        make_scene(self.root, "sceneA", [0, 6])
        self.base = ["--data-root", self.root, "--scenes", "sceneA", "--bursts", "0", "1",
                     "--frames", str(FRAMES), "--group-size", str(GROUP), "--out", self.out]

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_run_writes_records_figures_and_summary(self):
        code, _ = run_quiet(run_full_scale.main, self.base)
        self.assertEqual(code, 0)
        for burst in ("000000", "000001"):
            record = read_json(os.path.join(self.out, "sceneA", f"{burst}.json"))
            self.assertEqual(record["source"], "real")
            self.assertEqual(set(record["scores"]), {"naive", "aligned", "taa"})
            self.assertTrue(os.path.isfile(os.path.join(self.out, "sceneA", f"{burst}_comparison.png")))
        with open(os.path.join(self.out, "summary.csv"), encoding="utf-8") as file:
            rows = file.read().strip().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertFalse([f for _, _, names in os.walk(self.out) for f in names if f.endswith(".tmp")])
        raw = [f for _, _, names in os.walk(self.out) for f in names if f.endswith((".npy", ".npz"))]
        self.assertEqual(raw, [], "no raw data may be written to the output folder")

    def test_rerun_skips_finished_bursts(self):
        run_quiet(run_full_scale.main, self.base)
        with patch("scripts.run_full_scale.reconstruct_methods", side_effect=AssertionError("recomputed")):
            code, text = run_quiet(run_full_scale.main, self.base)
        self.assertEqual(code, 0)
        self.assertIn("2 skipped", text)

    def test_changed_setting_recomputes(self):
        run_quiet(run_full_scale.main, self.base)
        changed = self.base[:-2] + ["--gain", "2.0", "--out", self.out]
        with patch("scripts.run_full_scale.reconstruct_methods",
                   wraps=run_full_scale.reconstruct_methods) as spy:
            run_quiet(run_full_scale.main, changed)
        self.assertEqual(spy.call_count, 2)

    def test_corrupt_record_is_recomputed(self):
        run_quiet(run_full_scale.main, self.base)
        path = os.path.join(self.out, "sceneA", "000001.json")
        write_text(path, "{ not json")
        with patch("scripts.run_full_scale.reconstruct_methods",
                   wraps=run_full_scale.reconstruct_methods) as spy:
            run_quiet(run_full_scale.main, self.base)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(read_json(path)["burst"], 1)

    def test_failure_is_logged_and_retried_without_redoing_finished_work(self):
        os.remove(os.path.join(self.root, "sceneA", "000001.npy"))
        code, _ = run_quiet(run_full_scale.main, self.base)
        self.assertEqual(code, 1)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "sceneA", "000000.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.out, "sceneA", "000001.error.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "sceneA", "000001.json")))
        make_scene(self.root, "sceneA", [0, 6])                 # the missing burst reappears
        with patch("scripts.run_full_scale.reconstruct_methods",
                   wraps=run_full_scale.reconstruct_methods) as spy:
            code, _ = run_quiet(run_full_scale.main, self.base)
        self.assertEqual(code, 0)
        self.assertEqual(spy.call_count, 1)
        self.assertFalse(os.path.exists(os.path.join(self.out, "sceneA", "000001.error.txt")))

    def test_dry_run_writes_nothing(self):
        code, text = run_quiet(run_full_scale.main, self.base + ["--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("would run", text)
        self.assertFalse(os.path.exists(self.out))

    def test_plan_runs_selected_bursts_only(self):
        plan = os.path.join(self.tmp.name, "bins.json")
        write_text(plan, json.dumps({"selected": [{"scene": "sceneA", "burst": 1}], "all": []}))
        argv = ["--data-root", self.root, "--plan", plan, "--frames", str(FRAMES),
                "--group-size", str(GROUP), "--out", self.out]
        run_quiet(run_full_scale.main, argv)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "sceneA", "000001.json")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "sceneA", "000000.json")))

    def test_bad_arguments_exit(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_full_scale.main(self.base + ["--crop", "0", "64", "4", "60"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_full_scale.main(["--data-root", self.root, "--frames", "100"])


class BinScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "data")
        for index, (scene, drifts) in enumerate([("s1", [0, 4]), ("s2", [8, 16]), ("s3", [2, 24])]):
            make_scene(self.root, scene, drifts, seed=index)
        self.out = os.path.join(self.tmp.name, "bins")
        self.argv = ["--data-root", self.root, "--bursts", "0", "1", "--per-bin", "1", "--out", self.out]

    def measure(self, scene_dir, burst, **kwargs):
        return measure_burst_motion(scene_dir, burst, frames=FRAMES, crop=None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bins_are_ordered_selected_and_resumable(self):
        with patch("scripts.bin_scenes_by_motion.measure_burst_motion", side_effect=self.measure) as spy:
            summary = bin_scenes_by_motion.main(self.argv)
        self.assertEqual(spy.call_count, 6)
        self.assertEqual(sum(summary["counts"].values()), 6)
        by_scene = {(e["scene"], e["burst"]): e for e in summary["all"]}
        self.assertEqual(by_scene[("s1", 0)]["bin"], "slow")
        self.assertEqual(by_scene[("s3", 1)]["bin"], "fast")
        self.assertEqual({e["bin"] for e in summary["selected"]}, {"slow", "medium", "fast"})
        self.assertEqual(len(summary["selected"]), 3)
        with patch("scripts.bin_scenes_by_motion.measure_burst_motion", side_effect=AssertionError("remeasured")):
            again = bin_scenes_by_motion.main(self.argv)
        self.assertEqual(again["edges_px"], summary["edges_px"])

    def test_fixed_edges(self):
        with patch("scripts.bin_scenes_by_motion.measure_burst_motion", side_effect=self.measure):
            summary = bin_scenes_by_motion.main(self.argv + ["--edges", "3", "10"])
        self.assertEqual(summary["edges_px"], [3.0, 10.0])
        self.assertEqual(summary["method"], "fixed")

    def test_too_few_bursts_is_an_error(self):
        with patch("scripts.bin_scenes_by_motion.measure_burst_motion", side_effect=self.measure):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                bin_scenes_by_motion.main(["--data-root", self.root, "--scenes", "s1", "--bursts", "0",
                                           "--out", self.out])


class AnalysisTests(unittest.TestCase):
    def record(self, source, scene, burst, drift, aligned, taa, naive=10.0):
        scores = {m: {"psnr_mean": p, "ssim_mean": p / 40} for m, p in
                  (("naive", naive), ("aligned", aligned), ("taa", taa))}
        return motion_study.make_record(source, scene, burst, {"drift_px": drift, "min_peak": 0.2, "reliable": True},
                                        scores, frames=1024, crop=None, group_size=64,
                                        reference="last", gain=2.7, seconds=1)

    def test_bootstrap_needs_three_and_is_seeded(self):
        self.assertIsNone(analyze_motion_bins.bootstrap_ci([1.0, 2.0]))
        a = analyze_motion_bins.bootstrap_ci([1.0, 2.0, 4.0])
        self.assertEqual(a, analyze_motion_bins.bootstrap_ci([1.0, 2.0, 4.0]))
        self.assertLessEqual(a[0], 7 / 3)
        self.assertGreaterEqual(a[1], 7 / 3)

    def test_analyse_computes_bin_means_and_deltas(self):
        records = [self.record("real", "a", i, d, 20.0, 19.0) for i, d in enumerate([1, 2, 5, 6, 12, 14])]
        rows, bins = analyze_motion_bins.analyse(records, [3.0, 10.0])
        self.assertEqual([b["bin"] for b in bins], ["slow", "medium", "fast"])
        self.assertEqual([b["n"] for b in bins], [2, 2, 2])
        self.assertAlmostEqual(bins[0]["dpsnr_taa_aligned_mean"], -1.0)
        self.assertIsNone(bins[0]["dpsnr_taa_aligned_ci"])         # n = 2
        self.assertAlmostEqual(bins[0]["psnr_aligned_mean"], 20.0)

    def test_main_keeps_sources_separate_and_writes_files(self):
        results = tempfile.TemporaryDirectory()
        self.addCleanup(results.cleanup)
        out = os.path.join(results.name, "out")
        for i, d in enumerate([1, 2, 5, 6, 12, 14]):
            motion_study.write_json_atomic(os.path.join(results.name, "real", f"{i}.json"),
                                           self.record("real", "a", i, d, 20.0, 19.5))
            motion_study.write_json_atomic(os.path.join(results.name, "syn", f"{i}.json"),
                                           self.record("synthetic", "s", i, d, 30.0, 29.0))
        written = analyze_motion_bins.main(["--results", results.name, "--out", out])
        self.assertEqual(set(written), {"real", "synthetic"})
        self.assertAlmostEqual(written["real"]["bins"][0]["psnr_aligned_mean"], 20.0)
        self.assertAlmostEqual(written["synthetic"]["bins"][0]["psnr_aligned_mean"], 30.0)
        for name in ("per_burst.csv", "per_bin.csv", "summary.md", "psnr_vs_drift.png",
                     "ssim_vs_drift.png", "delta_by_bin.png"):
            self.assertTrue(os.path.isfile(os.path.join(out, "real", name)), name)

    def test_empty_results_is_an_error(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                analyze_motion_bins.main(["--results", empty, "--out", os.path.join(empty, "o")])


if __name__ == "__main__":
    unittest.main()
