# CS 566 Final Project — Temporal Anti-Aliasing for Single-Photon Reconstruction

**Team:** Adrian Gottwein (gottwein@wisc.edu), Limo Kemei (kkemei@wisc.edu), Jack Chen (zchen843@wisc.edu), Ethan Shao (zshao46@wisc.edu)
**Course:** CS 566 Introduction to Computer Vision, Fall 2026 (Prof. Mohit Gupta)
**Project webpage:** https://1immy.github.io/cs566-project/

## Overview
Single-photon (SPAD) cameras capture thousands of binary frames per second, where each pixel only records whether a photon arrived during the exposure — a single frame is meaningless, and a clean image only emerges from aligning and merging a burst. We're testing whether history-rejection heuristics from real-time temporal anti-aliasing (TAA) in game engines — neighborhood clamping, motion-adaptive blending — improve classical align-and-merge reconstruction of single-photon bursts, especially under fast motion and disocclusion. We evaluate on the Single Photon Challenge benchmark (co-organized by Prof. Gupta's lab), comparing our TAA-augmented pipeline against naive averaging and hierarchical block-alignment baselines via PSNR/SSIM and a failure analysis across motion speed.

**Status (mid-term):** topic approved by Prof. Gupta; scope refined as results came in. Both pipeline stages run on real Single Photon Challenge data. See [docs/findings/](docs/findings/) for the write-ups.

## Mid-term findings
Gain-matched PSNR (dB) on `kitchen1`, bursts 0 / 1 / 10, 300x304 crop, all 1,024 frames, groups of 64. One scene, no held-out data: indicative, not a benchmark.

| Method | b0 | b1 | b10 |
|---|---|---|---|
| Naive average | 16.2 | 14.7 | 16.8 |
| Aligned, first-group reference | 14.0 | 12.5 | 13.9 |
| Aligned, last-group reference | **19.7** | **19.1** | **19.3** |
| Clamp only, last-group reference | 19.7 | 18.5 | 19.3 |
| TAA (default), last-group reference | 17.3 | 15.3 | 18.3 |

- The scene moves inside a single burst (about 2 to 20 px), so we align groups of frames, not single binary frames ([docs/findings/burst-motion.md](docs/findings/burst-motion.md)).
- The ground truth is at the pose of the last frames, so the reference group must be the last one.
- Our first TAA configuration is worse than plain alignment: displacement-based confidence discards correct frames. Next step is a residual-based confidence.

## Milestones
| Milestone | Due | Status |
|---|---|---|
| Proposal | Sep 29 | Submitted |
| Mid-term report | Oct 29 | Submitted |
| TAA end to end | Nov 8 | In progress |
| Motion sweeps + failure analysis | Nov 19 | Not started |
| Final evaluation + ablations | Nov 30 | Not started |
| Final presentation (5-minute pre-recorded video) | Dec 8 | Not started |
| Project webpage | Dec 10 | Updated for mid-term |

Progress is tracked as GitHub issues grouped by these milestones.

## Webpage
The webpage source lives in `webpage/` on `main` for reference, and GitHub Pages serves it from the `gh-pages` branch. Copy changes onto `gh-pages`; never merge `main` into `gh-pages`. Current slots: `result-baseline.png`, `result-groundtruth.png`, `result-compare.png` (five-panel comparison). Still to come: PSNR/SSIM vs. motion speed chart.

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
│   └── findings/             # burst motion, alignment baseline, TAA write-ups
├── src/
│   ├── burst_io.py           # load_burst, load_gt, naive_average
│   ├── alignment.py          # align_warp, aligned_average (Track 2)
│   ├── baseline.py           # baseline runners
│   └── taa.py                # reject_and_blend, reconstruct_taa (Track 3)
├── scripts/                  # make_sample, run_sample_baselines, run_sample_taa, measure_burst_motion, ...
├── tests/                    # unit tests (44)
├── data/samples/             # ~1 MB committed sample; full data is not in the repo
├── results/                  # small outputs only
└── webpage/                  # copy of the site; served from gh-pages
```

## Setup
```bash
git clone https://github.com/1immy/cs566-project.git
cd cs566-project
pip install -r requirements.txt
python3 -m unittest discover -s tests          # 44 tests
python3 scripts/run_sample_baselines.py        # naive and aligned baselines on the sample
python3 scripts/run_sample_taa.py              # TAA on the sample
```
The full dataset (about 425 GB) stays on a private NAS and is not in the repo. A 10-scene slice is shared with the team via Drive (link in CONTRIBUTING.md).

## Team roles
- Limo Kemei: Track 1, Data & Infra (dataset, burst loader, synthetic sweeps, repo/webpage)
- Ethan Shao: Track 2, Alignment backbone (`align_warp`, naive-averaging merge)
- Adrian Gottwein: Track 3, TAA history rejection (clamping, adaptive blend, confidence map)
- Jack Chen: Track 4, Evaluation & analysis (PSNR/SSIM, motion-speed sweep, ablations); also supporting Track 3 or 4

See [CONTRIBUTING.md](CONTRIBUTING.md) for the interfaces and dependencies between tracks.
