# CS 566 Final Project — Temporal Anti-Aliasing for Single-Photon Reconstruction

**Team:** Adrian, Limo, Jack, Ethan
**Course:** CS 566 Introduction to Computer Vision, Fall 2026 (Prof. Mohit Gupta)
**Project webpage:** [link once gh-pages is live]

## Overview
Single-photon (SPAD) cameras capture thousands of binary frames per second, where each pixel only records whether a photon arrived during the exposure — a single frame is meaningless, and a clean image only emerges from aligning and merging a burst. We're testing whether history-rejection heuristics from real-time temporal anti-aliasing (TAA) in game engines — neighborhood clamping, motion-adaptive blending — improve classical align-and-merge reconstruction of single-photon bursts, especially under fast motion and disocclusion. We evaluate on the Single Photon Challenge benchmark (co-organized by Prof. Gupta's lab), comparing our TAA-augmented pipeline against naive averaging and hierarchical block-alignment baselines via PSNR/SSIM and a failure analysis across motion speed.

**Status:** topic approved by Prof. Gupta as a reasonable direction; scope may be refined as implementation progresses. He also offered to connect the team with his lab for data access or implementation guidance if needed.

## Milestones
| Milestone | Due | Status |
|---|---|---|
| Proposal | Sep 29 | Submitted |
| Mid-term report | Oct 29 | Not started |
| Final presentation | Dec 1–10 | Not started |
| Project webpage | Dec 10 | Skeleton live |

## Webpage images
The webpage (`webpage/index.html`) has image slots pre-wired under `webpage/assets/`. Add files with these exact names and they'll render automatically:
- `pipeline-overview.png` — diagram of the reconstruction pipeline (raw frames → alignment → history rejection → merge)
- `result-baseline.png`, `result-taa.png`, `result-groundtruth.png` — side-by-side reconstruction comparison for one scene
- `psnr-ssim-chart.png` — PSNR/SSIM vs. motion speed chart

## Key links
- Benchmark & dataset: https://singlephotonchallenge.com/
- Quanta burst photography (Gupta's lab): https://wisionlab.com/project/quanta-burst-photography/
- TAA survey (NVIDIA): https://research.nvidia.com/labs/rtr/publication/yang2020survey/
- Simulator: visionsim (MIT-licensed, Gupta's lab)

## Repo structure
```
cs566-project/
├── README.md
├── docs/
│   └── proposal_template.md
├── src/               # implementation code
├── data/              # datasets (gitignored if large)
├── results/           # output images/videos/plots
└── webpage/           # served from gh-pages branch
```

## Setup
```bash
git clone <repo-url>
cd cs566-project
pip install -r requirements.txt   # add as dependencies are pinned down
```

## Team roles
- [Teammate]: [role — e.g. data pipeline]
- [Teammate]: [role — e.g. model implementation]
- [Teammate]: [role — e.g. evaluation / webpage]
