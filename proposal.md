# CS 566 Project Proposal

**Team members:** Adrian, Limo, Jack, Ethan
**Date:** [Sep 29, 2026]

## 1. Problem statement
Single-photon avalanche diode (SPAD) cameras capture high-speed bursts of binary frames — each pixel only records whether a photon arrived during an exposure, so a single frame carries almost no usable signal. Reconstructing a clean image requires aligning and merging many frames (quanta burst photography). We address **single-photon image reconstruction under motion**: specifically, whether history-rejection heuristics from real-time temporal anti-aliasing (TAA) — neighborhood clamping, motion-adaptive blending — improve classical align-and-merge reconstruction, especially under fast motion and disocclusion.

## 2. Motivation
- **Scientific/technical interest:** SPAD arrays are an emerging sensor technology (megapixel-scale arrays now exist), and the standard reconstruction pipeline (estimate motion, align, merge) is structurally identical to TAA in real-time rendering — a connection that, as far as we've found, hasn't been explored. Bringing game-engine history-rejection tricks into photon-level reconstruction is a genuinely cross-domain angle.
- **Fit with the course/instructor:** this builds directly on Prof. Gupta's own quanta burst photography work and the Single Photon Challenge his lab co-organized, giving us a public benchmark with ground truth rather than needing to build our own.

## 3. Related work (brief)
1. **Quanta burst photography** (Gupta's lab) — establishes the align-then-accumulate pipeline for SPAD bursts that we take as our baseline.
2. **TAA survey** (NVIDIA, Yang et al. 2020) — catalogs history-rejection techniques (neighborhood clamping, motion-adaptive blend weights) from real-time rendering that we adapt to this domain.
3. **Single Photon Challenge** — provides the benchmark dataset and existing leaderboard submissions as points of comparison.

## 4. Approach
1. Implement baseline: naive frame averaging, then hierarchical block alignment + merging (simplified quanta burst photography).
2. Implement TAA-augmented pipeline: same alignment/merge backbone plus neighborhood-clamping history rejection, motion-adaptive blend weights, and a per-pixel confidence map.
3. Run both on Single Photon Challenge training scenes (held-out subset) and visionsim-generated synthetic sweeps (varying motion speed, light level).
4. Evaluate and analyze where/why each method breaks down under fast motion and disocclusion.

Course techniques used: image alignment/transformations, RANSAC, optical flow, tracking, and the quanta computational imaging lecture.

## 5. Data
**Single Photon Challenge** dataset: 50 simulated training scenes + 5 test scenes (ground truth private), each burst ("photon cube") = 1024 binary frames. Full set ~425 GB (~133 GB compressed, ~8.5 GB/chunk) — plan to use 5–10 chunks, plus the small one-burst-per-scene sample for early prototyping. Supplemented with our own synthetic sweeps via the lab's MIT-licensed **visionsim** simulator.

## 6. Evaluation plan
- PSNR/SSIM on held-out training scenes (confirming with the instructor whether the leaderboard still accepts test-set submissions).
- Failure analysis vs. motion speed using visionsim sweeps — directly answering "when does temporal reconstruction break down?"
- Framing: an interpretable classical method with rigorous failure analysis, not a state-of-the-art claim against neural approaches.

## 7. Scope and feasibility
This topic was discussed with Prof. Gupta, who confirmed it is a reasonable and worthwhile direction to propose. As expected at this stage, the specific scope may be refined as implementation progresses and initial results clarify what's achievable within the semester. He also offered to connect the team with members of his lab for data access or implementation guidance if needed.

## 8. Timeline

| Week | Dates | Milestone |
|---|---|---|
| 1–2 | Sep 29 – Oct 12 | Data pipeline (subset download, visionsim setup) + baseline (naive averaging + block alignment) working |
| 3–4 | Oct 13 – Oct 26 | TAA-augmented pipeline (clamping, adaptive blending, confidence map) implemented |
| 5 | Oct 27 – Oct 29 | Mid-term report: baseline vs. TAA results so far, difficulties |
| 6–8 | Oct 30 – Nov 19 | Motion-speed sweeps via visionsim, failure analysis, iteration |
| 9 | Nov 20 – Nov 30 | Final evaluation, ablations (which TAA tricks help most), results polish |
| 10 | Dec 1 – Dec 10 | Presentation + final webpage |

## 9. Risks / fallback plan
- **Data size:** mitigated by using a 5–10 chunk subset rather than the full 425 GB.
- **Leaderboard may not accept new submissions** (challenge has concluded): fallback is evaluation on held-out training scenes only.
- **Beating the leaderboard is unrealistic** against neural methods: reframe contribution as interpretable classical method + failure analysis, not SOTA.
