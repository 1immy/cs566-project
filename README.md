# CS 566 Final Project — Temporal Anti-Aliasing for Single-Photon Reconstruction

**Team:** Adrian Gottwein (gottwein@wisc.edu), Limo Kemei (kkemei@wisc.edu), Jack Chen (zchen843@wisc.edu), Ethan Shao (zshao46@wisc.edu)
**Course:** CS 566 Introduction to Computer Vision, Fall 2026 (Prof. Mohit Gupta)
**Project webpage:** https://1immy.github.io/cs566-project/

## Overview
Single-photon (SPAD) cameras capture thousands of binary frames per second, where each pixel only records whether a photon arrived during the exposure — a single frame is meaningless, and a clean image only emerges from aligning and merging a burst. We tested whether history-rejection heuristics from real-time temporal anti-aliasing (TAA) in game engines — neighborhood clamping, motion-adaptive blending — improve classical align-and-merge reconstruction of single-photon bursts, especially under fast motion and disocclusion. We evaluate on the Single Photon Challenge benchmark (co-organized by Prof. Gupta's lab), comparing our TAA-augmented pipeline against naive averaging and hierarchical block-alignment baselines via PSNR/SSIM and a failure analysis across motion speed.

**Status (final):** experiments are complete and written up on the webpage. The remaining work is the 5-minute presentation video (due Dec 8) and the final proofread of the webpage (due Dec 10). See [docs/findings/](docs/findings/) for the write-ups and [docs/presentation/](docs/presentation/) for the video script and Q&A.

## Final findings
Protocol for every number in this section: 1,024-frame bursts, groups of 64, last-group reference, one fixed gain of 2.7 for all methods, and the same aligner and photon cube for naive averaging, aligned averaging and TAA. PSNR and SSIM are computed on full frames.

**Real data.** 31 bursts from 29 training scenes, grouped into slow, medium and fast thirds by measured drift (edges 12.8 and 37.8 px). Mean (standard deviation) over bursts:

| Motion bin | n | PSNR naive | PSNR aligned | PSNR TAA | SSIM naive | SSIM aligned | SSIM TAA |
|---|---:|---|---|---|---|---|---|
| slow | 10 | 15.02 (1.83) | 15.49 (1.71) | 15.45 (1.60) | 0.330 | 0.437 | 0.381 |
| medium | 11 | 14.46 (2.50) | 15.26 (2.31) | 15.26 (2.52) | 0.293 | 0.395 | 0.325 |
| fast | 10 | 13.59 (3.11) | 14.64 (2.82) | 14.93 (3.06) | 0.296 | 0.361 | 0.269 |
| all | 31 | 14.36 (2.52) | 15.14 (2.27) | 15.22 (2.39) | 0.306 | 0.398 | 0.325 |

- **Alignment beats naive averaging, and the gain grows with motion:** +0.48 dB (slow), +0.80 dB (medium), +1.05 dB (fast); +0.78 dB overall [95% bootstrap interval +0.55, +1.03].
- **TAA history rejection adds no consistent PSNR gain over alignment:** +0.08 dB overall [-0.18, +0.34]; TAA is better on 14 of 31 bursts and worse on 17 (exact sign test p = 0.72).
- **TAA costs SSIM:** lower than aligned on all 31 bursts (-0.073 on average, p < 0.001).
- **Reference choice matters more than TAA:** the ground truth is the pose at the end of the burst, so aligning to the last group beats aligning to the first by 1.5 dB or more on `kitchen1`.
- **Controlled sweeps (known motion, one scene, one seed):** alignment holds up to about 20 px of global motion in both the numpy simulator and the Blender-rendered pan; TAA tracks alignment, slightly below it at most speeds ([docs/findings/synthetic-sweep.md](docs/findings/synthetic-sweep.md)).

Caveats: the measured drift is unreliable for most bursts (26 of the 31 have a weak motion estimate), so the bins mean "large measured drift", not a known speed; samples are small (10 to 11 bursts per bin); one fixed gain may fit some scenes poorly; training scenes only (the 5 test scenes have no public ground truth). Full tables, sign tests and figures: [docs/findings/final-results.md](docs/findings/final-results.md), [docs/findings/comparison-figures.md](docs/findings/comparison-figures.md), [docs/findings/motion-bins.md](docs/findings/motion-bins.md).

## Mid-term findings
These were preliminary, single-scene numbers. The final results above replace them. Gain-matched PSNR (dB) on `kitchen1`, bursts 0 / 1 / 10, 300x304 crop, all 1,024 frames, groups of 64. One scene, no held-out data: indicative, not a benchmark.

| Method | b0 | b1 | b10 |
|---|---|---|---|
| Naive average | 16.2 | 14.7 | 16.8 |
| Aligned, first-group reference | 14.0 | 12.5 | 13.9 |
| Aligned, last-group reference | **19.7** | **19.1** | **19.3** |
| Clamp only, last-group reference | 19.7 | 18.5 | 19.3 |
| TAA (default), last-group reference | 17.3 | 15.3 | 18.3 |

- The scene moves inside a single burst, so we align groups of frames, not single binary frames ([docs/findings/burst-motion.md](docs/findings/burst-motion.md)).
- The ground truth is at the pose of the last frames, so the reference group must be the last one.
- Our first TAA configuration was 1.0 to 3.8 dB worse than plain alignment under the per-image gain above: displacement-based confidence discards correct frames.

Evaluation-harness results on the same bursts (fixed gain 2.7, current aligner with sub-pixel refinement). Not comparable with the table above, which fits one gain per image.

| Method | PSNR b0 | b1 | b10 | SSIM b0 | b1 | b10 |
|---|---|---|---|---|---|---|
| Naive average | 15.11 | 14.52 | 14.68 | 0.298 | 0.303 | 0.358 |
| Aligned, first-group reference | 13.94 | 12.40 | 13.43 | 0.248 | 0.218 | 0.293 |
| Aligned, last-group reference | **16.29** | 16.98 | **15.83** | 0.431 | 0.407 | 0.484 |
| Clamp only, last-group reference | **16.29** | 17.01 | **15.83** | **0.431** | **0.415** | **0.485** |
| TAA (default), last-group reference | 16.24 | **17.17** | 15.80 | 0.391 | 0.369 | 0.454 |

- The baseline ordering matches the table above. TAA is within 0.2 dB of last-group alignment in PSNR but 0.03 to 0.04 lower in SSIM. We did not separate whether the gain rule or the improved aligner closes the PSNR gap. Reproduce with the command in [docs/midTerm-evaluation.md](docs/midTerm-evaluation.md).

## Milestones
| Milestone | Due | Status |
|---|---|---|
| Proposal | Sep 29 | Submitted |
| Mid-term report | Oct 29 | Submitted |
| TAA end to end | Nov 8 | Done |
| Motion sweeps + failure analysis | Nov 19 | Done |
| Final evaluation + ablations | Nov 30 | Done (31 real bursts, simulated and rendered sweeps) |
| Final presentation (5-minute pre-recorded video) | Dec 8 | Script and Q&A drafted; recording in progress |
| Project webpage | Dec 10 | Final results, Data and Alignment, Method and Discussion added; video, Q&A and final proofread to come |

Progress is tracked as GitHub issues grouped by these milestones.

## Webpage
The webpage source lives in `webpage/` on `main` for reference, and GitHub Pages serves it from the `gh-pages` branch. Copy changes onto `gh-pages`; never merge `main` into `gh-pages`. The page is dated by report: Proposal (Sep 29), Mid-term (Oct 29) and Final (Nov 23). Still to come: the presentation video and the Q&A text under it.

## Key links
- Benchmark & dataset: https://singlephotonchallenge.com/
- Quanta burst photography (Gupta's lab): https://wisionlab.com/project/quanta-burst-photography/
- TAA survey (NVIDIA): https://research.nvidia.com/labs/rtr/publication/yang2020survey/
- Simulator: visionsim (MIT-licensed, Gupta's lab)

## Repo structure
```
cs566-project/
├── README.md, CONTRIBUTING.md, AI-CONTEXT.md   # CLAUDE.md / AGENTS.md point to AI-CONTEXT.md
├── docs/
│   ├── proposal.*            # proposal sources and PDF
│   ├── findings/             # motion, alignment, TAA, sweeps, final results, figures
│   └── presentation/         # video script and Q&A
├── src/
│   ├── burst_io.py           # load_burst, load_gt, naive_average
│   ├── alignment.py          # align_warp, aligned_average (Track 2)
│   ├── baseline.py           # baseline runners
│   ├── taa.py                # reject_and_blend, reconstruct_taa (Track 3)
│   ├── motion.py             # burst motion measure (phase correlation)
│   ├── synthetic.py          # numpy motion-sweep simulator
│   └── rendered.py           # Blender-rendered sweep loader and burst sampling
├── scripts/                  # run_full_scale, bin_scenes_by_motion, analyze_motion_bins,
│   │                         # final_results_table, make_comparison_figures, sweeps, ...
│   └── blender/              # pan scene generation and frame rendering
├── tests/                    # unit tests (131)
├── data/samples/             # ~1 MB committed sample; full data is not in the repo
├── results/                  # small outputs only
└── webpage/                  # copy of the site; served from gh-pages
```

## Setup
```bash
git clone https://github.com/1immy/cs566-project.git
cd cs566-project
pip install -r requirements.txt
python3 -m unittest discover -s tests          # 131 tests
python3 scripts/run_sample_baselines.py        # naive and aligned baselines on the sample
python3 scripts/run_sample_taa.py              # TAA on the sample
```
On Windows, one test (`test_taa_study`) needs symlink permission and errors unless Developer Mode is on; the rest pass.

The full dataset (about 425 GB) stays on a private NAS and is not in the repo. A 10-scene slice is shared with the team via Drive (link in CONTRIBUTING.md). The real-data results above were produced with `scripts/bin_scenes_by_motion.py` and `scripts/run_full_scale.py` on the full training set; see [docs/findings/motion-bins.md](docs/findings/motion-bins.md).

## Team roles
- Limo Kemei: Track 1, Data & Infra (dataset, burst loader, synthetic sweeps, repo/webpage)
- Ethan Shao: Track 2, Alignment backbone (`align_warp`, naive-averaging merge)
- Adrian Gottwein: Track 3, TAA history rejection (clamping, adaptive blend, confidence map)
- Jack Chen: Track 4, Evaluation & analysis (PSNR/SSIM, motion-speed sweep, ablations); also supporting Track 3 or 4

See [CONTRIBUTING.md](CONTRIBUTING.md) for the interfaces and dependencies between tracks.
